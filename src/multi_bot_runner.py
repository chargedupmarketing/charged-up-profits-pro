"""
src/multi_bot_runner.py

Multi-symbol trading bot runner — ONE TopstepX connection, multiple strategies.

Exchange regulations only permit one active WebSocket session per account.
Running separate bot_runner.py processes for ES and NQ breaks the first
connection the moment the second one authenticates.

This script solves the problem by:
  1. Creating ONE TradingSuite with ALL requested symbols in a single call.
  2. Spinning up ONE shared asyncio event loop in a background thread.
  3. Creating one BotRunner instance per symbol and injecting the shared
     suite + loop so each runner uses its own per-symbol InstrumentContext.
  4. Running each BotRunner's strategy loop concurrently in separate threads.

Usage:
    python src/multi_bot_runner.py                        # ES + NQ (default)
    python src/multi_bot_runner.py --symbols ES NQ MNQ    # all three
    python src/multi_bot_runner.py --symbols ES MNQ       # ES + MNQ only

Paper mode (default ON) — set execution.paper_mode = false in settings.yaml
for live trading and make sure PROJECT_X_ACCOUNT_NAME points to a funded account.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
import threading
import time
from pathlib import Path

# Force UTF-8 on Windows to avoid encoding errors from SDK emoji output
os.environ.setdefault("PYTHONUTF8", "1")
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.bot_runner import BotRunner, _apply_symbol_overrides

# Micro (paper) <-> live symbol mapping
_PAPER_SYMBOL = {
    "ES":  "MES",
    "NQ":  "MNQ",
    "MNQ": "MNQ",
    "MES": "MES",
}

# ---------------------------------------------------------------------------
# Shared event-loop helpers
# ---------------------------------------------------------------------------

_MAX_CONNECT_RETRIES = 4
_RETRY_DELAYS = [5, 15, 30, 60]   # seconds between retries


def _create_suite_and_start_loop(
    symbols: list[str],
) -> tuple[object, asyncio.AbstractEventLoop]:
    """
    Create the shared asyncio event loop AND the TradingSuite in one step.

    Architecture:
      1. Create a fresh event loop.
      2. Use loop.run_until_complete() to create the TradingSuite —
         this is critical because the SDK's WebSocket/SignalR connections
         need a properly-driven event loop during initialization (just like
         asyncio.run() provides).
      3. After the suite is created, start loop.run_forever() in a daemon
         thread so the suite's background tasks (bar updates, order fills,
         reconnects) continue running.

    Returns (suite, loop).
    """
    from project_x_py import TradingSuite  # type: ignore

    last_exc: Exception | None = None

    for attempt in range(1, _MAX_CONNECT_RETRIES + 1):
        # Create a FRESH event loop for each attempt.  Reusing the same loop
        # after a failed TradingSuite.create() leaves lingering async tasks
        # that interfere with subsequent retries and cause
        # "Subscription returned False" even when the server accepts the
        # connection (confirmed: asyncio.run() / fresh loop always succeeds).
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            logger.info(
                "Attempt {}/{}: Creating TradingSuite for {} ...",
                attempt, _MAX_CONNECT_RETRIES, symbols,
            )
            suite = loop.run_until_complete(
                TradingSuite.create(
                    symbols,
                    timeframes=["1min"],
                    initial_days=1,  # 3 days caused market-hub timeout on ES+NQ
                )
            )
            # ── Suppress noisy SDK loggers ────────────────────────────────
            import logging as _lg
            # WARNING: suppresses INFO/DEBUG only
            for name in ("SignalRCoreClient", "signalrcore", "urllib3"):
                lgr = _lg.getLogger(name)
                lgr.setLevel(_lg.WARNING)
                lgr.handlers = [h for h in lgr.handlers
                                 if not isinstance(h, _lg.StreamHandler)]
            # project_x_py.position_manager floods with
            # "Error in position processor: no running event loop" (ERROR level)
            # during the loop transition — harmless, the processor self-recovers.
            # Silence to CRITICAL so real SDK errors still surface.
            for name in (
                "project_x_py",
                "project_x_py.position_manager",
                "project_x_py.position_manager.core",
            ):
                lgr = _lg.getLogger(name)
                lgr.setLevel(_lg.CRITICAL)
                lgr.handlers = [h for h in lgr.handlers
                                 if not isinstance(h, _lg.StreamHandler)]

            logger.info(
                "TradingSuite connected on attempt {}/{}",
                attempt, _MAX_CONNECT_RETRIES,
            )

            # ── Start the shared event loop in a background thread ───────
            # IMPORTANT: call asyncio.set_event_loop(loop) inside the thread
            # so that any SDK component that calls asyncio.get_event_loop()
            # from that thread finds the correct loop (fixes the position
            # manager "no running event loop" RuntimeError).
            def _run_loop_forever(lp: asyncio.AbstractEventLoop) -> None:
                asyncio.set_event_loop(lp)
                lp.run_forever()

            t = threading.Thread(
                target=_run_loop_forever,
                args=(loop,),
                name="multi-bot-shared-loop",
                daemon=True,
            )
            t.start()
            return suite, loop

        except Exception as exc:
            last_exc = exc
            delay = _RETRY_DELAYS[min(attempt - 1, len(_RETRY_DELAYS) - 1)]
            logger.warning(
                "Attempt {}/{}: TradingSuite connection failed: {} — retrying in {}s ...",
                attempt, _MAX_CONNECT_RETRIES, exc, delay,
            )
            # Close the failed loop so its lingering tasks don't bleed into
            # the next attempt's fresh loop.
            try:
                loop.close()
            except Exception:
                pass
            time.sleep(delay)

    # All retries exhausted
    raise RuntimeError(
        f"Failed to create TradingSuite after {_MAX_CONNECT_RETRIES} attempts. "
        f"Last error: {last_exc}"
    )


def _disconnect_suite_sync(
    suite: object, loop: asyncio.AbstractEventLoop
) -> None:
    """Disconnect the shared TradingSuite, blocking until done."""
    async def _do() -> None:
        try:
            await suite.disconnect()  # type: ignore[attr-defined]
            logger.info("Shared TradingSuite disconnected")
        except Exception as exc:
            logger.warning("Error disconnecting shared suite: {}", exc)

    fut = asyncio.run_coroutine_threadsafe(_do(), loop)
    try:
        fut.result(timeout=15)
    except Exception:
        pass
    loop.call_soon_threadsafe(loop.stop)


# ---------------------------------------------------------------------------
# Multi-symbol orchestrator
# ---------------------------------------------------------------------------

class MultiBotRunner:
    """
    Orchestrates multiple per-symbol BotRunner instances that share a single
    TopstepX connection (one TradingSuite / one WebSocket session).
    """

    def __init__(
        self,
        symbols: list[str],
        settings_path: str = "config/settings.yaml",
    ) -> None:
        self._settings_path = settings_path
        self._symbols = symbols

        base_cfg = _load_settings(settings_path)
        paper_mode: bool = base_cfg.get("execution", {}).get("paper_mode", True)

        # Determine the SDK symbols to include in TradingSuite.create()
        # paper mode → micro symbols (MES/MNQ), live → full symbols (ES/NQ)
        self._sdk_symbols: list[str] = [
            (_PAPER_SYMBOL.get(s, s) if paper_mode else s)
            for s in symbols
        ]
        logger.info(
            "MultiBotRunner: symbols={} sdk_symbols={} paper_mode={}",
            symbols, self._sdk_symbols, paper_mode,
        )

        # One BotRunner per symbol
        self._runners: list[BotRunner] = []
        for sym in symbols:
            runner = BotRunner(
                settings_path=settings_path,
                symbol_override=sym,
            )
            self._runners.append(runner)

        # Shared asyncio infrastructure (created in run())
        self._loop: asyncio.AbstractEventLoop | None = None
        self._suite: object | None = None

        # Graceful shutdown
        self._stop_event = threading.Event()
        signal.signal(signal.SIGINT,  self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        # 1+2. Create the event loop and TradingSuite together.
        #       The suite is created with loop.run_until_complete() (like asyncio.run),
        #       then the loop is moved to a background thread for ongoing events.
        logger.info("Creating shared TradingSuite for {}", self._sdk_symbols)
        self._suite, self._loop = _create_suite_and_start_loop(self._sdk_symbols)
        logger.info("Shared TradingSuite ready — background loop running")

        # 3. Inject shared suite + loop into each runner's ExecutionEngine
        for runner, sym in zip(self._runners, self._sdk_symbols):
            runner._engine.inject_suite_and_loop(
                suite=self._suite,
                symbol=sym,
                loop=self._loop,
            )
            logger.info("Injected shared suite into {} engine (sdk_symbol={})",
                        runner._symbol, sym)

        # 4. Run each strategy in its own thread
        threads: list[threading.Thread] = []
        for runner in self._runners:
            t = threading.Thread(
                target=self._run_one,
                args=(runner,),
                name=f"bot-{runner._symbol}",
                daemon=True,
            )
            threads.append(t)

        for t in threads:
            t.start()

        # 5. Wait for all strategies to finish (or Ctrl-C)
        try:
            while any(t.is_alive() for t in threads):
                time.sleep(1)
                if self._stop_event.is_set():
                    logger.info("Stop event received — waiting for runners to wind down")
                    break
        finally:
            for t in threads:
                t.join(timeout=30)
            self._cleanup()

    def _run_one(self, runner: BotRunner) -> None:
        """Run a single BotRunner in this thread; catch and log exceptions."""
        try:
            runner.run()
        except Exception as exc:
            import traceback as _tb
            logger.error(
                "BotRunner {} crashed: {}\n{}",
                runner._symbol, exc, _tb.format_exc(),
            )

    def _shutdown(self, *_args: object) -> None:
        """SIGINT / SIGTERM handler — signals all runners to stop."""
        logger.warning("MultiBotRunner: shutdown signal received")
        self._stop_event.set()
        for runner in self._runners:
            try:
                runner._engine.flatten_all("shutdown")
            except Exception:
                pass

    def _cleanup(self) -> None:
        """Disconnect the shared suite and stop the event loop."""
        if self._suite and self._loop:
            _disconnect_suite_sync(self._suite, self._loop)


# ---------------------------------------------------------------------------
# Helpers (thin wrappers from bot_runner)
# ---------------------------------------------------------------------------

def _load_settings(path: str = "config/settings.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ChargedUp Profits Bot — multi-symbol runner (one connection)"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["ES", "NQ"],
        help="Symbols to trade simultaneously (e.g. ES NQ MNQ). Default: ES NQ",
    )
    parser.add_argument(
        "--settings",
        default="config/settings.yaml",
        help="Path to settings.yaml",
    )
    args = parser.parse_args()

    runner = MultiBotRunner(symbols=args.symbols, settings_path=args.settings)
    runner.run()


if __name__ == "__main__":
    main()
