"""
execution_engine.py

TopstepX execution engine via the ProjectX API (project-x-py SDK).

Responsibilities:
  - Authenticate with TopstepX REST API using API key + username
  - Subscribe to real-time 1m market data bars via SignalR WebSocket
  - Accumulate a rolling history of 1m and 15m bars (pandas DataFrames)
  - Place bracket orders (entry limit + stop + take-profit)
  - Monitor order fills via real-time user-hub events
  - Auto-flatten + halt on connection loss or daily-loss breach
  - Fire on_bar / on_kill_switch callbacks for the strategy layer

Paper mode (config: execution.paper_mode = true):
  Uses the Combine / Eval account selected by PROJECT_X_ACCOUNT_NAME.

Live mode (config: execution.paper_mode = false):
  Uses a funded TopstepX account.  Set PROJECT_X_ACCOUNT_NAME in .env to
  the exact account name shown on your TopstepX dashboard.

Environment variables (.env):
    PROJECT_X_USERNAME          Your TopstepX login email / username
    PROJECT_X_API_KEY           API key from TopstepX dashboard
    PROJECT_X_ACCOUNT_NAME      (optional) Account name to trade; uses the
                                first active account when not set.
"""

from __future__ import annotations

import asyncio
import datetime
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Optional

import pandas as pd
import yaml
from dotenv import load_dotenv
from loguru import logger

from src.setup_detector import Direction, Signal

load_dotenv()

# ---------------------------------------------------------------------------
# Suppress noisy SignalR INFO logs from project-x-py internals.
# The signalrcore library prints "on_reconnect not defined" and
# "Close message received from server" on every reconnect cycle at INFO level,
# which floods the terminal.  We keep WARNING+ so real errors still show.
# ---------------------------------------------------------------------------
import logging as _logging
_logging.getLogger("SignalRCoreClient").setLevel(_logging.WARNING)
_logging.getLogger("signalrcore").setLevel(_logging.WARNING)
_logging.getLogger("urllib3").setLevel(_logging.WARNING)  # noisy HTTP library too
# position_manager floods with "no running event loop" ERRORs during startup
# transition — harmless, processor self-recovers. Silence to CRITICAL.
_logging.getLogger("project_x_py").setLevel(_logging.CRITICAL)
_logging.getLogger("project_x_py.position_manager").setLevel(_logging.CRITICAL)
_logging.getLogger("project_x_py.position_manager.core").setLevel(_logging.CRITICAL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_settings(path: str = "config/settings.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# History depth — keep last 5 trading days of 1m bars (~2 000 bars)
_MAX_1M_BARS = 3000


# ---------------------------------------------------------------------------
# Order state
# ---------------------------------------------------------------------------

class OrderStatus(str, Enum):
    PENDING    = "PENDING"     # submitted, not yet acknowledged
    OPEN       = "OPEN"        # resting in the market
    FILLED     = "FILLED"      # fully filled
    CANCELLED  = "CANCELLED"   # cancelled
    REJECTED   = "REJECTED"    # rejected by exchange
    PARTIAL    = "PARTIAL"     # partially filled


@dataclass
class LiveOrder:
    """Tracks one leg of an active bracket."""
    order_id:    int | None             = None
    status:      OrderStatus            = OrderStatus.PENDING
    fill_price:  float | None           = None
    exit_price:  float | None           = None   # set when the closing leg fills
    fill_time:   datetime.datetime | None = None


@dataclass
class ActiveBracket:
    """Holds IDs and state for a live bracket trade."""
    signal:           Signal
    entry_order:      LiveOrder = field(default_factory=LiveOrder)
    stop_order:       LiveOrder = field(default_factory=LiveOrder)
    target_order:     LiveOrder = field(default_factory=LiveOrder)
    contract_id:      str = ""
    n_contracts:      int = 1
    placed_at:        datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    closed:           bool = False

    # Trailing stop tracking — once set, stop won't move again for this trade
    stop_moved_to_breakeven: bool = False

    # Convenience: expose fill/exit so bot_runner can use a single object
    @property
    def fill_price(self) -> float | None:
        return self.entry_order.fill_price

    @property
    def exit_price(self) -> float | None:
        return self.entry_order.exit_price

    def at_1r(self, current_price: float) -> bool:
        """
        Returns True if price has moved at least 1× the stop distance
        in the favorable direction from the fill price (1R profit reached).
        """
        fp = self.fill_price
        if fp is None:
            return False
        stop = self.signal.stop_price
        stop_dist = abs(fp - stop)
        if stop_dist < 0.01:
            return False
        if self.signal.direction.value.upper() == "LONG":
            return current_price >= fp + stop_dist
        else:
            return current_price <= fp - stop_dist


# ---------------------------------------------------------------------------
# Bar accumulator — converts 1m bars into 15m bars + pandas history
# ---------------------------------------------------------------------------

class BarStore:
    """
    Accumulates 1-minute bars into:
      - a rolling deque of raw 1m dicts  (→ get_df_1m)
      - a rolling deque of completed 15m dicts (→ get_df_15m)

    Fires on_bar_15m(bar_series) each time a 15m bar closes.
    Fires on_bar(df_1m, df_15m) after updating both DataFrames.
    """

    def __init__(
        self,
        on_bar_15m:  Callable[[pd.Series], None] | None = None,
        on_bar:      Callable[[pd.DataFrame, pd.DataFrame], None] | None = None,
        max_1m_bars: int = _MAX_1M_BARS,
    ) -> None:
        self._on_bar_15m = on_bar_15m
        self._on_bar     = on_bar

        self._buf_1m:    Deque[dict] = deque(maxlen=max_1m_bars)
        self._buf_15m:   Deque[dict] = deque(maxlen=500)
        self._window_buf: list[dict] = []    # 1m bars in current 15m window
        self._current_window: datetime.datetime | None = None

    # ------------------------------------------------------------------
    def ingest(self, bar: dict) -> None:
        """Accept one completed 1-minute bar dict."""
        ts = bar.get("timestamp")
        if ts is None:
            return
        # Make timezone-aware (assume UTC if naive)
        if isinstance(ts, datetime.datetime) and ts.tzinfo is None:
            ts = ts.replace(tzinfo=datetime.timezone.utc)
        bar = dict(bar)
        bar["timestamp"] = ts

        self._buf_1m.append(bar)

        # 15m window management
        window = ts.replace(minute=(ts.minute // 15) * 15,
                            second=0, microsecond=0)
        if self._current_window is None:
            self._current_window = window

        if window == self._current_window:
            self._window_buf.append(bar)
        else:
            # New window — emit the just-completed 15m bar
            if self._window_buf:
                self._emit_15m()
            self._window_buf = [bar]
            self._current_window = window

        # Fire the combined bar callback
        if self._on_bar:
            try:
                self._on_bar(self.get_df_1m(), self.get_df_15m())
            except Exception as e:
                logger.error("on_bar callback error: {}", e)

    # ------------------------------------------------------------------
    def _emit_15m(self) -> None:
        if not self._window_buf:
            return
        bar_15m = {
            "timestamp": self._window_buf[0]["timestamp"],
            "open":      self._window_buf[0]["open"],
            "high":      max(b["high"]   for b in self._window_buf),
            "low":       min(b["low"]    for b in self._window_buf),
            "close":     self._window_buf[-1]["close"],
            "volume":    sum(b["volume"] for b in self._window_buf),
        }
        self._buf_15m.append(bar_15m)
        logger.trace("15m bar emitted: {}", bar_15m["timestamp"])
        if self._on_bar_15m:
            try:
                self._on_bar_15m(pd.Series(bar_15m))
            except Exception as e:
                logger.error("on_bar_15m callback error: {}", e)

    # ------------------------------------------------------------------
    def get_df_1m(self) -> pd.DataFrame:
        if not self._buf_1m:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
        df = pd.DataFrame(list(self._buf_1m))
        df = df.set_index("timestamp").sort_index()
        return df

    def get_df_15m(self) -> pd.DataFrame:
        if not self._buf_15m:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
        df = pd.DataFrame(list(self._buf_15m))
        df = df.set_index("timestamp").sort_index()
        return df


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class ExecutionEngine:
    """
    TopstepX execution engine.

    Usage
    -----
    engine = ExecutionEngine(settings_path="config/settings.yaml")
    engine.on_bar(my_bar_callback)          # called (df_1m, df_15m) each minute
    engine.on_kill_switch(my_ks_callback)   # called when connection dies
    engine.start()                          # blocks until connected

    engine.submit_signal(signal)            # async-safe, non-blocking
    engine.flatten_all("reason")            # cancel all + close position
    engine.stop()                           # clean disconnect
    """

    def __init__(self, settings_path: str = "config/settings.yaml") -> None:
        self._cfg        = _load_settings(settings_path)
        self._bar_store  = BarStore(
            on_bar_15m=self._fire_on_bar_15m,
            on_bar=self._fire_on_bar,
        )

        # Registered callbacks
        self._bar_callbacks:        list[Callable] = []
        self._bar_15m_callbacks:    list[Callable] = []
        self._kill_switch_callbacks: list[Callable] = []

        # asyncio infra (background thread)
        self._loop:   asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None           = None

        # TradingSuite instance
        self._suite: Any = None
        self._ctx: Any = None            # per-symbol InstrumentContext (multi-symbol mode)
        self._suite_is_shared: bool = False  # True when suite was injected externally
        self._injected_symbol: str = ""      # symbol requested via inject_suite_and_loop()
        self._contract_id: str = ""

        # Active bracket (one trade at a time)
        self._active_bracket: ActiveBracket | None = None
        self._bracket_lock = threading.Lock()

        # State
        self._killed             = False
        self._connected          = False
        self._daily_pnl_dollars  = 0.0
        self._trades_today       = 0

        exc_cfg = self._cfg.get("execution", {})
        self._order_timeout_sec        = exc_cfg.get("order_timeout_seconds", 90)
        self._paper_mode               = exc_cfg.get("paper_mode", True)
        self._limit_to_market_enabled  = exc_cfg.get("limit_to_market_on_timeout", True)

        inst_cfg                 = self._cfg.get("instrument", {})
        self._live_symbol        = inst_cfg.get("symbol",       "ES")
        self._micro_symbol       = inst_cfg.get("micro_symbol", "MES")
        self._point_value        = float(inst_cfg.get("point_value", 50))

        risk_cfg                 = self._cfg.get("risk", {})
        self._max_daily_loss     = float(risk_cfg.get("max_daily_loss_dollars", -200))
        self._max_trades         = int(risk_cfg.get("max_trades_per_day", 3))
        self._n_contracts        = int(risk_cfg.get("initial_contracts", 1))

    # ------------------------------------------------------------------
    # Multi-symbol injection (call before start())
    # ------------------------------------------------------------------

    def inject_suite_and_loop(
        self,
        suite: Any,
        symbol: str,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """
        Inject a pre-created multi-symbol TradingSuite and shared asyncio loop.
        MUST be called before start().

        Exchange regulations permit only ONE active WebSocket session per account.
        When trading multiple symbols simultaneously, create ONE TradingSuite with
        all symbols (e.g. TradingSuite.create(["ES","NQ"])) then inject it into
        each ExecutionEngine so they all share the same connection.

        Args:
            suite:  TradingSuite created with multiple symbols
            symbol: The specific symbol this engine handles (e.g. "ES", "MNQ")
            loop:   The already-running asyncio event loop that owns the suite
        """
        self._suite = suite
        self._injected_symbol = symbol
        self._suite_is_shared = True
        self._loop = loop
        self._thread = None   # no thread management in shared-loop mode

    # ------------------------------------------------------------------
    # Callback registration (called before start())
    # ------------------------------------------------------------------

    def on_bar(
        self,
        callback: Callable[[pd.DataFrame, pd.DataFrame], None],
    ) -> None:
        """
        Register a callback fired every completed 1-minute bar.
        Signature: callback(df_1m: pd.DataFrame, df_15m: pd.DataFrame)
        Both DataFrames are indexed by timezone-aware timestamp.
        """
        self._bar_callbacks.append(callback)

    def on_bar_15m(self, callback: Callable[[pd.Series], None]) -> None:
        """
        Register a callback fired every completed 15-minute bar.
        Signature: callback(bar: pd.Series)
        """
        self._bar_15m_callbacks.append(callback)

    def on_kill_switch(self, callback: Callable[[], None]) -> None:
        """Register a callback fired when the connection kill-switch triggers."""
        self._kill_switch_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Bar history access
    # ------------------------------------------------------------------

    def get_df_1m(self) -> pd.DataFrame:
        """Return rolling history of 1-minute bars as a pandas DataFrame."""
        return self._bar_store.get_df_1m()

    def get_df_15m(self) -> pd.DataFrame:
        """Return rolling history of 15-minute bars as a pandas DataFrame."""
        return self._bar_store.get_df_15m()

    # ------------------------------------------------------------------
    # Active order access
    # ------------------------------------------------------------------

    @property
    def active_order(self) -> ActiveBracket | None:
        """The currently open bracket, or None if flat."""
        with self._bracket_lock:
            b = self._active_bracket
            if b is None or b.closed:
                return None
            return b

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Start the engine.  Connects to TopstepX and begins streaming.
        Blocks until the initial connection succeeds (timeout = 60s).

        In multi-symbol mode (inject_suite_and_loop was called) the shared
        asyncio loop is already running — we simply schedule _async_start()
        on it without creating a new loop or thread.
        """
        ready_event  = threading.Event()
        error_holder: list[Exception] = []

        async def _boot():
            try:
                await self._async_start()
                ready_event.set()
            except Exception as exc:
                error_holder.append(exc)
                ready_event.set()

        if self._suite_is_shared:
            # Multi-symbol: loop is already running externally — just enqueue startup
            assert self._loop is not None, "inject_suite_and_loop() must be called first"
            asyncio.run_coroutine_threadsafe(_boot(), self._loop)
        else:
            # Single-symbol: spin up our own loop in a background thread
            if self._thread and self._thread.is_alive():
                logger.warning("ExecutionEngine.start() called while already running")
                return
            self._loop   = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._loop.run_forever,
                name="execution-engine-loop",
                daemon=True,
            )
            self._thread.start()
            logger.info("Asyncio event loop started in background thread")
            asyncio.run_coroutine_threadsafe(_boot(), self._loop)

        # Multi-symbol mode: both engines share one loop, so they seed historical
        # data sequentially — allow up to 3 minutes for both to finish.
        _timeout = 180 if self._suite_is_shared else 90
        if not ready_event.wait(timeout=_timeout):
            raise RuntimeError(
                "ExecutionEngine: timeout waiting for TopstepX connection"
            )
        if error_holder:
            raise error_holder[0]

    def stop(self) -> None:
        """Gracefully disconnect and stop the background loop.

        In multi-symbol (shared-loop) mode only the per-symbol subscriptions
        are torn down; the shared TradingSuite and event loop are left running
        for the other symbols and must be stopped by the multi_bot_runner.
        """
        # Mark disconnected immediately so bar callbacks and order logic stop
        # processing right away, even if async cleanup is still pending.
        self._connected = False

        if self._loop and not self._loop.is_closed():
            # In shared-loop mode the loop may be busy seeding historical bars
            # for the other symbol (1 000+ bars, several minutes).  Allow up to
            # 3 minutes; catch TimeoutError so the bot can finish cleanly even
            # if cleanup is delayed.
            _timeout = 180 if self._suite_is_shared else 15
            try:
                asyncio.run_coroutine_threadsafe(
                    self._async_stop(), self._loop
                ).result(timeout=_timeout)
            except TimeoutError:
                logger.warning(
                    "ExecutionEngine.stop() timed out after {}s — "
                    "loop still running for other symbols; continuing.",
                    _timeout,
                )
            if not self._suite_is_shared:
                # Only stop the loop when we own it
                self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("ExecutionEngine stopped")

    def submit_signal(self, signal: Signal) -> None:
        """
        Submit a trading signal (non-blocking).
        Internally calls place_bracket_order — returns None immediately;
        use on_bar callback to detect fills.
        """
        self.place_bracket_order(signal, contracts=self._n_contracts)

    def place_bracket_order(
        self,
        signal: Signal,
        contracts: int = 1,
    ) -> ActiveBracket | None:
        """
        Place a bracket order for the given signal.
        Returns the ActiveBracket immediately (with PENDING status).
        Returns None if rejected by pre-flight checks.
        """
        if self._killed:
            logger.warning("Order rejected — kill-switch is active")
            return None
        if not self._connected:
            logger.warning("Order rejected — not connected to TopstepX")
            return None
        with self._bracket_lock:
            if self._active_bracket and not self._active_bracket.closed:
                logger.warning("Order rejected — existing position open")
                return None

        bracket = ActiveBracket(
            signal=signal,
            contract_id=self._contract_id,
            n_contracts=contracts,
        )
        with self._bracket_lock:
            self._active_bracket = bracket

        asyncio.run_coroutine_threadsafe(
            self._async_place_bracket(bracket), self._loop
        )
        return bracket

    def flatten_all(self, reason: str = "manual") -> None:
        """Cancel all open orders and close any open position immediately."""
        logger.warning("flatten_all called ({})", reason)
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._async_flatten(reason), self._loop
            )

    def reset_daily(self) -> None:
        """Call at the start of each trading day."""
        self._daily_pnl_dollars = 0.0
        self._trades_today      = 0
        self._killed            = False
        logger.info("Daily counters reset")

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_killed(self) -> bool:
        return self._killed

    # ------------------------------------------------------------------
    # Internal callback dispatchers
    # ------------------------------------------------------------------

    def _fire_on_bar(
        self, df_1m: pd.DataFrame, df_15m: pd.DataFrame
    ) -> None:
        # Trailing stop to breakeven: once a trade reaches 1R profit, move
        # the stop to the fill price so the trade cannot turn into a loss.
        self._check_trailing_stop_to_be(df_1m)

        for cb in self._bar_callbacks:
            try:
                cb(df_1m, df_15m)
            except Exception as e:
                logger.error("on_bar callback error: {}", e)

    def _check_trailing_stop_to_be(self, df_1m: pd.DataFrame) -> None:
        """
        Called on every new bar. When a position has reached 1R profit
        and the stop hasn't yet been moved, request a stop modification
        to the entry (fill) price — this eliminates the risk of a winner
        turning into a loser.

        Stop modification is sent as a cancel+replace via the async loop.
        If the exchange rejects it, the original stop stays in place.
        """
        with self._bracket_lock:
            bracket = self._active_bracket

        if bracket is None or bracket.closed:
            return
        if bracket.stop_moved_to_breakeven:
            return
        if bracket.entry_order.status != OrderStatus.FILLED:
            return
        if bracket.fill_price is None:
            return
        if df_1m.empty:
            return

        current_price = float(df_1m["close"].iloc[-1])

        if bracket.at_1r(current_price):
            fill_price = bracket.fill_price
            logger.info(
                "Trailing stop: 1R reached (price={:.2f}, fill={:.2f}) — "
                "moving stop to breakeven",
                current_price, fill_price,
            )
            bracket.stop_moved_to_breakeven = True  # Set before async call to prevent double-trigger

            # Modify the stop order on the exchange
            asyncio.run_coroutine_threadsafe(
                self._async_move_stop_to_breakeven(bracket, fill_price),
                self._loop,
            )

    async def _async_move_stop_to_breakeven(
        self, bracket: ActiveBracket, new_stop_price: float
    ) -> None:
        """
        Cancel the existing stop order and replace it with a new stop at
        new_stop_price (entry price / breakeven).
        """
        try:
            _orders = self._ctx.orders if self._ctx is not None else self._suite.orders

            # Cancel current stop
            stop_id = bracket.stop_order.order_id
            if stop_id:
                await _orders.cancel_order(stop_id)
                logger.debug("Cancelled original stop order {} for breakeven move", stop_id)

            # Place new stop at breakeven
            direction = bracket.signal.direction.value.upper()
            stop_side = "Sell" if direction == "LONG" else "Buy"

            new_stop = await _orders.place_order(
                contract_id  = bracket.contract_id,
                size         = bracket.n_contracts,
                side         = stop_side,
                order_type   = "Stop",
                stop_price   = new_stop_price,
            )
            new_id = getattr(new_stop, "id", None) or (new_stop.get("id") if isinstance(new_stop, dict) else None)
            if new_id:
                bracket.stop_order.order_id = str(new_id)
            logger.info(
                "Stop moved to breakeven @ {:.2f} (order_id={})",
                new_stop_price, new_id,
            )
        except Exception as e:
            logger.warning(
                "Could not move stop to breakeven ({}). "
                "Original stop remains active.", e
            )

    def _fire_on_bar_15m(self, bar: pd.Series) -> None:
        for cb in self._bar_15m_callbacks:
            try:
                cb(bar)
            except Exception as e:
                logger.error("on_bar_15m callback error: {}", e)

    def _fire_kill_switch(self) -> None:
        self._killed = True
        for cb in self._kill_switch_callbacks:
            try:
                cb()
            except Exception as e:
                logger.error("on_kill_switch callback error: {}", e)

    # ------------------------------------------------------------------
    # Async internals
    # ------------------------------------------------------------------

    async def _async_start(self) -> None:
        """Boot TradingSuite and attach event handlers.

        Supports two modes:
          • Single-symbol  — creates its own TradingSuite for one symbol.
          • Multi-symbol   — uses an injected TradingSuite (shared connection);
                             subscribes to the per-symbol InstrumentContext so
                             each ExecutionEngine only sees its own symbol's events.
        """
        from project_x_py import TradingSuite, EventType

        if self._suite_is_shared:
            # --- Multi-symbol: suite was pre-created by multi_bot_runner ---
            symbol = self._injected_symbol
            logger.info("Using shared TradingSuite for symbol={}", symbol)
        else:
            # --- Single-symbol: create our own suite ---
            symbol = self._micro_symbol if self._paper_mode else self._live_symbol
            logger.info(
                "Connecting to TopstepX (paper_mode={}, symbol={})",
                self._paper_mode, symbol,
            )
            self._suite = await TradingSuite.create(
                symbol,
                timeframes=["1min"],
                initial_days=1,  # 3 days caused market-hub timeout on ES+NQ
            )
            # TradingSuite.create() resets loggers to INFO — re-apply suppressions.
            import logging as _lg
            for _n, _lvl in (
                ("SignalRCoreClient",                         _lg.WARNING),
                ("signalrcore",                               _lg.WARNING),
                ("urllib3",                                   _lg.WARNING),
                ("project_x_py",                             _lg.CRITICAL),
                ("project_x_py.position_manager",            _lg.CRITICAL),
                ("project_x_py.position_manager.core",       _lg.CRITICAL),
            ):
                _l = _lg.getLogger(_n)
                _l.setLevel(_lvl)
                _l.handlers = [h for h in _l.handlers
                                if not isinstance(h, _lg.StreamHandler)]

        # Resolve per-symbol InstrumentContext (SDK v3.5+)
        try:
            self._ctx = self._suite[symbol]
        except (KeyError, TypeError):
            self._ctx = None  # Fallback — single-instrument compat shim

        # Resolve contract ID
        if self._ctx and self._ctx.instrument_info:
            self._contract_id = self._ctx.instrument_info.id or symbol
        else:
            self._contract_id = self._suite.instrument_id or symbol

        # Account info (shared REST client)
        acct_name = "?"
        try:
            _ai = self._suite.client.account_info
            if _ai:
                acct_name = f"{_ai.name} (id={_ai.id})"
        except Exception:
            pass
        logger.info(
            "Connected to TopstepX | account='{}' | symbol='{}' | contract='{}'",
            acct_name, symbol, self._contract_id,
        )

        # Attach real-time callbacks.
        # In multi-symbol mode subscribe to the per-symbol event bus so we only
        # receive events for our symbol.  In single-symbol mode use the suite-
        # level bus (backward compatible).
        if self._ctx is not None:
            await self._ctx.event_bus.on(EventType.NEW_BAR,       self._on_bar_event)
            await self._ctx.event_bus.on(EventType.ORDER_FILLED,  self._on_order_filled)
            await self._ctx.event_bus.on(EventType.ORDER_MODIFIED, self._on_order_updated)
        else:
            await self._suite.on("new_bar",       self._on_bar_event)
            await self._suite.on("order_filled",  self._on_order_filled)
            await self._suite.on("order_modified", self._on_order_updated)

        # Seed bar store from historical data.
        # Yield to the event loop every 100 bars so that other coroutines
        # queued on the shared loop (e.g. _async_stop from the other symbol)
        # are never blocked for more than a fraction of a second.
        data_mgr = self._ctx.data if self._ctx is not None else self._suite.data
        try:
            hist = await data_mgr.get_data("1min")
            if hist is not None and not hist.is_empty():
                rows = list(hist.iter_rows(named=True))
                for i, row in enumerate(rows):
                    self._bar_store.ingest(row)
                    if i % 100 == 0:
                        await asyncio.sleep(0)   # yield to loop between batches
                logger.info(
                    "Seeded bar store with {} historical 1m bars", len(rows)
                )
        except Exception as e:
            logger.warning("Could not seed bar store: {}", e)

        self._connected = True
        logger.info("ExecutionEngine ready (symbol={})", symbol)

    async def _async_stop(self) -> None:
        self._connected = False
        if self._suite and not self._suite_is_shared:
            # Only disconnect when we own the suite; in multi-symbol mode the
            # suite is shared and must be disconnected by multi_bot_runner.
            try:
                await self._suite.disconnect()
                logger.info("Disconnected from TopstepX")
            except Exception as e:
                logger.warning("Error during disconnect: {}", e)

    # ------------------------------------------------------------------
    # Real-time event handlers
    # ------------------------------------------------------------------

    async def _on_bar_event(self, event: Any) -> None:
        """Called by TradingSuite when a new 1m bar is completed.

        SDK v3.5+ delivers events as an event object:
          event.data = {"timeframe": "1min", "data": {bar fields}}
        Older builds delivered the bar dict directly.
        """
        try:
            # --- unwrap SDK v3.5+ envelope ---
            if hasattr(event, "data"):
                payload = event.data
                if isinstance(payload, dict):
                    timeframe = payload.get("timeframe", "1min")
                    if "1min" not in timeframe and "1m" not in timeframe:
                        return  # ignore 5m, 15m etc.
                    bar_data = payload.get("data", payload)
                else:
                    bar_data = payload
            else:
                # Legacy: event IS the bar dict
                bar_data = event

            if isinstance(bar_data, dict):
                self._bar_store.ingest(bar_data)
            elif hasattr(bar_data, "__iter__"):
                self._bar_store.ingest(dict(bar_data))
        except Exception as e:
            logger.error("_on_bar_event error: {}", e)

    async def _on_order_filled(self, event: Any) -> None:
        """Handle order fill events.

        SDK v3.5+: event.data is an order object.  Older builds passed the
        order dict directly.  We handle both via _get_field().
        """
        try:
            # Unwrap SDK v3.5+ event envelope
            raw = event.data if hasattr(event, "data") else event
            order_id   = _get_field(raw, "orderId") or _get_field(raw, "id")
            fill_price = (_get_field(raw, "filledPrice")
                          or _get_field(raw, "fillPrice")
                          or _get_field(raw, "averagePrice"))
            logger.info("Order filled | id={} price={}", order_id, fill_price)

            with self._bracket_lock:
                bracket = self._active_bracket
                if bracket is None or bracket.closed:
                    return

                now = datetime.datetime.now(datetime.timezone.utc)

                if bracket.entry_order.order_id == order_id:
                    bracket.entry_order.status     = OrderStatus.FILLED
                    bracket.entry_order.fill_price = fill_price
                    bracket.entry_order.fill_time  = now
                    self._trades_today += 1

                elif bracket.stop_order.order_id == order_id:
                    bracket.stop_order.status      = OrderStatus.FILLED
                    bracket.stop_order.fill_price  = fill_price
                    # Expose exit_price on the entry_order for bot_runner
                    bracket.entry_order.exit_price = fill_price
                    self._close_bracket(bracket, "stop_hit", fill_price)

                elif bracket.target_order.order_id == order_id:
                    bracket.target_order.status     = OrderStatus.FILLED
                    bracket.target_order.fill_price = fill_price
                    bracket.entry_order.exit_price  = fill_price
                    self._close_bracket(bracket, "target_hit", fill_price)

        except Exception as e:
            logger.error("_on_order_filled error: {}", e)

    async def _on_order_updated(self, event: Any) -> None:
        try:
            raw        = event.data if hasattr(event, "data") else event
            order_id   = _get_field(raw, "orderId") or _get_field(raw, "id")
            status_raw = _get_field(raw, "status")
            if status_raw:
                logger.debug("Order updated | id={} status={}", order_id, status_raw)
        except Exception as e:
            logger.error("_on_order_updated error: {}", e)

    # ------------------------------------------------------------------
    # Bracket placement
    # ------------------------------------------------------------------

    async def _async_place_bracket(self, bracket: ActiveBracket) -> None:
        """Execute the actual API call inside the asyncio loop."""
        signal = bracket.signal
        side   = 0 if signal.direction == Direction.LONG else 1

        try:
            logger.info(
                "Placing bracket | {} {}x @ entry={:.2f} sl={:.2f} tp={:.2f}",
                "BUY" if side == 0 else "SELL",
                bracket.n_contracts,
                signal.entry_price,
                signal.stop_price,
                signal.target_price,
            )

            _orders = self._ctx.orders if self._ctx is not None else self._suite.orders
            result = await _orders.place_bracket_order(
                contract_id       = self._contract_id,
                side              = side,
                size              = bracket.n_contracts,
                entry_price       = signal.entry_price,
                stop_loss_price   = signal.stop_price,
                take_profit_price = signal.target_price,
                entry_type        = "limit",
            )

            if not result.success:
                logger.error("Bracket order rejected: {}", result.error_message)
                with self._bracket_lock:
                    bracket.closed = True
                return

            with self._bracket_lock:
                bracket.entry_order.order_id  = result.entry_order_id
                bracket.stop_order.order_id   = result.stop_order_id
                bracket.target_order.order_id = result.target_order_id
                bracket.entry_order.status    = OrderStatus.OPEN
                bracket.stop_order.status     = OrderStatus.OPEN
                bracket.target_order.status   = OrderStatus.OPEN

            logger.info(
                "Bracket open | entry={} stop={} target={}",
                result.entry_order_id,
                result.stop_order_id,
                result.target_order_id,
            )

            asyncio.create_task(
                self._monitor_entry_timeout(bracket, self._order_timeout_sec)
            )

        except Exception as exc:
            logger.error("_async_place_bracket error: {}", exc)
            with self._bracket_lock:
                if bracket:
                    bracket.closed = True

    async def _monitor_entry_timeout(
        self, bracket: ActiveBracket, timeout_sec: int
    ) -> None:
        half = max(10, timeout_sec // 2)
        await asyncio.sleep(half)

        with self._bracket_lock:
            already_done = bracket.closed or bracket.entry_order.status == OrderStatus.FILLED
        if already_done:
            return

        if self._limit_to_market_enabled:
            logger.info(
                "Entry order {} unfilled after {}s — converting limit → market",
                bracket.entry_order.order_id, half,
            )
            await self._async_convert_limit_to_market(bracket)
            # Allow remaining time for market fill
            await asyncio.sleep(timeout_sec - half)
            with self._bracket_lock:
                if bracket.closed or bracket.entry_order.status == OrderStatus.FILLED:
                    return
            logger.warning("Market entry still unfilled after {}s — flattening", timeout_sec)
            await self._async_flatten("entry_timeout_after_market")
        else:
            await asyncio.sleep(timeout_sec - half)
            with self._bracket_lock:
                if bracket.closed:
                    return
                if bracket.entry_order.status != OrderStatus.FILLED:
                    logger.warning(
                        "Entry order {} timed out after {}s — cancelling",
                        bracket.entry_order.order_id, timeout_sec,
                    )
            await self._async_flatten("entry_timeout")

    async def _async_convert_limit_to_market(self, bracket: ActiveBracket) -> None:
        """Cancel existing limit bracket and immediately re-place with a market entry."""
        try:
            _orders = self._ctx.orders if self._ctx is not None else self._suite.orders
            await _orders.cancel_all_orders(contract_id=self._contract_id)

            signal = bracket.signal
            side = "BUY" if signal.direction.value.upper() == "LONG" else "SELL"

            # Reset bracket order IDs so fill callbacks can re-attach
            with self._bracket_lock:
                bracket.entry_order.order_id = ""
                bracket.entry_order.status   = OrderStatus.PENDING
                bracket.stop_order.order_id  = ""
                bracket.stop_order.status    = OrderStatus.PENDING
                bracket.target_order.order_id = ""
                bracket.target_order.status  = OrderStatus.PENDING

            result = await _orders.place_bracket_order(
                contract_id       = self._contract_id,
                side              = side,
                size              = bracket.n_contracts,
                entry_price       = signal.entry_price,
                stop_loss_price   = signal.stop_price,
                take_profit_price = signal.target_price,
                entry_type        = "market",
            )

            if result.success:
                with self._bracket_lock:
                    bracket.entry_order.order_id   = result.entry_order_id
                    bracket.stop_order.order_id    = result.stop_order_id
                    bracket.target_order.order_id  = result.target_order_id
                    bracket.entry_order.status     = OrderStatus.OPEN
                    bracket.stop_order.status      = OrderStatus.OPEN
                    bracket.target_order.status    = OrderStatus.OPEN
                logger.info(
                    "Market bracket placed: entry={} stop={} target={}",
                    result.entry_order_id, result.stop_order_id, result.target_order_id,
                )
            else:
                logger.warning("Market bracket rejected: {} — flattening", result.error_message)
                await self._async_flatten("market_entry_failed")
        except Exception as exc:
            logger.error("_async_convert_limit_to_market error: {}", exc)
            await self._async_flatten("market_entry_error")

    async def _async_flatten(self, reason: str = "manual") -> None:
        logger.warning("Flattening all positions ({})", reason)
        try:
            if self._suite and self._contract_id:
                _orders = self._ctx.orders if self._ctx is not None else self._suite.orders
                await _orders.cancel_all_orders(
                    contract_id=self._contract_id
                )
                await _orders.close_position(
                    contract_id=self._contract_id,
                )
        except Exception as e:
            logger.error("_async_flatten error: {}", e)
        finally:
            with self._bracket_lock:
                if self._active_bracket:
                    self._active_bracket.closed = True

    # ------------------------------------------------------------------
    # P&L and kill-switch
    # ------------------------------------------------------------------

    def _close_bracket(
        self,
        bracket: ActiveBracket,
        reason: str,
        fill_price: float | None,
    ) -> None:
        bracket.closed = True
        if fill_price and bracket.entry_order.fill_price:
            raw_pts = fill_price - bracket.entry_order.fill_price
            if bracket.signal.direction == Direction.SHORT:
                raw_pts = -raw_pts
            pnl = raw_pts * self._point_value * bracket.n_contracts
            self._daily_pnl_dollars += pnl
            logger.info(
                "Trade closed ({}) | fill={:.2f} pnl=${:.2f} | daily=${:.2f}",
                reason, fill_price, pnl, self._daily_pnl_dollars,
            )
            self._check_daily_loss()

    def _check_daily_loss(self) -> None:
        if self._daily_pnl_dollars <= self._max_daily_loss:
            logger.critical(
                "Daily loss limit hit (${:.2f} <= ${:.2f}) — kill-switch",
                self._daily_pnl_dollars, self._max_daily_loss,
            )
            self._fire_kill_switch()
            asyncio.run_coroutine_threadsafe(
                self._async_flatten("daily_loss_limit"), self._loop
            )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _get_field(obj: Any, field: str) -> Any:
    """Safely extract a field from an event object or dict."""
    if isinstance(obj, dict):
        return obj.get(field)
    return getattr(obj, field, None)
