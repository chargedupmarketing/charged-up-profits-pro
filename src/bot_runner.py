"""
src/bot_runner.py

Main live-trading orchestrator.  Ties together all modules:
  SessionEngine → LevelBuilder → SetupDetector → RiskManager
  → ExecutionEngine → AuditLogger → AccountManager

Run with:
    python src/bot_runner.py                    # uses settings.yaml default symbol (ES)
    python src/bot_runner.py --symbol NQ        # trade NQ with NQ-specific overrides
    python src/bot_runner.py --symbol ES        # explicitly trade ES

The bot will:
  1. Wait until 7:45am EST to build today's levels
  2. Start listening for signals at 9:00am EST
  3. Enforce all risk rules + funded account limits before placing any order
  4. Log every decision to the audit database
  5. Write data/bot_state_<SYMBOL>.json every 30s (dashboard reads this)
  6. Shut down cleanly at 13:00 ET (or earlier on kill-switch)

Paper mode is ON by default (config: execution.paper_mode = true).
Change to false only after a full paper trading review period.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from monitoring.audit_log import AuditLogger
from src.account_manager import AccountManager
from src.data_integrity import DataIntegrityChecker, clean_bars
from src.drift_monitor import DriftMonitor
from src.execution_engine import ExecutionEngine, OrderStatus
from src.feature_builder import compute_features
from src.level_builder import LevelBuilder
from src.market_calendar import MarketCalendar
from src.notifier import Notifier
from src.order_state import OrderStateTracker
from src.portfolio_risk import get_portfolio_risk
from src.risk_manager import KillSwitchReason, RiskManager
from src.session_engine import SessionEngine
from src.setup_detector import SetupDetector


def _load_settings(path: str = "config/settings.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _config_hash(cfg: dict) -> str:
    """SHA-256 of the config dict for reproducibility audit. Stored per trade."""
    import hashlib, json
    canon = json.dumps(cfg, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(canon.encode()).hexdigest()[:12]


def _apply_symbol_overrides(cfg: dict, symbol: str) -> dict:
    """
    Merge symbol-specific overrides from cfg['symbols'][symbol] into the
    instrument / risk / levels sections of cfg.  Returns the merged config.
    """
    import copy
    cfg = copy.deepcopy(cfg)
    symbols_cfg = cfg.get("symbols", {})
    if symbol not in symbols_cfg:
        return cfg  # no overrides for this symbol

    sym = symbols_cfg[symbol]
    # Instrument overrides
    for key in ("symbol", "micro_symbol", "point_value", "topstepx_symbol", "databento_dataset"):
        if key in sym:
            cfg["instrument"][key] = sym[key]

    # Risk overrides
    for key, val in sym.get("risk_overrides", {}).items():
        cfg["risk"][key] = val

    # Levels overrides
    for key, val in sym.get("levels_overrides", {}).items():
        cfg["levels"][key] = val

    return cfg


class BotRunner:
    def __init__(
        self,
        settings_path: str = "config/settings.yaml",
        symbol_override: str | None = None,
    ) -> None:
        self._cfg = _load_settings(settings_path)
        self._settings_path = settings_path

        # Apply per-symbol overrides (NQ uses different point_value, range limits, etc.)
        if symbol_override:
            self._cfg = _apply_symbol_overrides(self._cfg, symbol_override)
            logger.info("Symbol override applied: {}", symbol_override)
        self._symbol = self._cfg["instrument"]["symbol"]

        # State file path — one file per symbol so dashboard can track both
        self._state_file = ROOT / "data" / f"bot_state_{self._symbol}.json"
        self._state_file.parent.mkdir(parents=True, exist_ok=True)

        self._session = SessionEngine(settings_path)
        self._level_builder = LevelBuilder(settings_path)
        # Pass symbol to SetupDetector so enabled_setups filter is applied from init
        self._setup_detector = SetupDetector(settings_path, symbol=self._symbol)
        self._risk = RiskManager(settings_path)
        self._engine = ExecutionEngine(settings_path=settings_path)
        self._audit = AuditLogger(settings_path)

        # CRITICAL: propagate per-symbol overrides into sub-components that load
        # their own config from the YAML file (LevelBuilder, SetupDetector).
        # Without this, NQ/MNQ would use ES's max_8am_range_points=20 instead
        # of the symbol-specific value (e.g. 80 for NQ), causing a false NO-TRADE day.
        if symbol_override:
            self._level_builder.configure_for_symbol(symbol_override)
            self._setup_detector.configure_for_symbol(symbol_override)

        # Per-symbol ML bypass (symbols.ES.ml_bypass: true disables ML gate for ES)
        _sym_cfg = self._cfg.get("symbols", {}).get(self._symbol, {})
        self._ml_bypass: bool = bool(_sym_cfg.get("ml_bypass", False))

        # AccountManager — monitors funded account health
        self._account_mgr = AccountManager(settings_path, risk_manager=self._risk)
        self._risk.set_account_manager(self._account_mgr)

        self._day_levels = None
        self._levels_built = False
        self._current_signal = None
        self._entry_fill_price = None
        self._entry_ts = None
        self._last_state_write = 0.0   # unix timestamp of last state file write
        self._paper_mode: bool = bool(self._cfg.get("execution", {}).get("paper_mode", True))
        self._premarket_assessment: dict = {}   # written by _run_premarket_analysis()
        self._bar_write_counter: int = 0       # throttle bar file writes

        # Shadow mode: simulate decisions without placing orders
        self._shadow_mode: bool = bool(
            self._cfg.get("execution", {}).get("shadow_mode", False)
        )

        # Section B: Execution safety settings
        exec_cfg = self._cfg.get("execution", {})
        self._max_fill_slippage_ticks: float = float(
            exec_cfg.get("max_fill_slippage_ticks", 4)
        )
        self._connection_watchdog_secs: int = int(
            exec_cfg.get("connection_watchdog_seconds", 60)
        )
        self._last_bar_received: float = time.time()  # updated on each bar

        # Section A: data integrity checker
        self._integrity_checker = DataIntegrityChecker()
        self._data_healthy: bool = True

        # Section B2: idempotent order state tracker
        self._order_state = OrderStateTracker()

        # Section C: drift monitor
        self._drift_monitor = DriftMonitor(settings_path=settings_path)

        # Section A: market calendar (log daily summary at startup)
        _cal_summary = MarketCalendar.day_summary()
        logger.info("MarketCalendar: {}", _cal_summary)
        if _cal_summary.get("is_roll_week") and _cal_summary.get("roll_liquidity_warning"):
            logger.warning(
                "ROLL WEEK — liquidity migrating to next contract. "
                "Front month: ES={} NQ={}",
                _cal_summary.get("front_month_ES"),
                _cal_summary.get("front_month_NQ"),
            )
        if _cal_summary.get("is_dst_week"):
            logger.warning("DST TRANSITION WEEK — verify all scheduled times are correct")

        # Email notifications (fire-and-forget via Resend)
        self._notifier = Notifier()

        # Wire up callbacks
        self._engine.on_bar(self._on_new_bar)           # (df_1m, df_15m) each 1m bar
        self._engine.on_kill_switch(self._on_connection_kill)

        # Graceful shutdown on Ctrl+C
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        # Register our PID in bot_pids.json so the dashboard detects us as running
        # even when the bot is started manually from the terminal (not via the panel).
        self._pids_file = ROOT / "data" / "bot_pids.json"
        self._register_pid()

        # Write initial state file so dashboard sees the process starting
        self._write_state_file()

        self._config_hash: str = _config_hash(self._cfg)

        logger.info(
            "BotRunner initialised — symbol={} paper_mode={} config_hash={}",
            self._symbol,
            self._paper_mode,
            self._config_hash,
        )

        # ── SAFETY: loud warning when running live ────────────────────────
        if not self._paper_mode:
            _warn = (
                "\n" + "!" * 70 +
                "\n  LIVE TRADING IS ENABLED (paper_mode = false)"
                "\n  Real orders will be placed with real money."
                "\n  Ensure you have completed paper trading evaluation"
                "\n  and set paper_mode deliberately in config/settings.yaml"
                "\n" + "!" * 70
            )
            logger.warning(_warn)
            # Write a marker so the dashboard can surface this warning
            _live_flag = ROOT / "data" / "live_mode_active.flag"
            _live_flag.write_text(
                json.dumps({"symbol": self._symbol,
                            "started_at": datetime.datetime.now().isoformat()}),
                encoding="utf-8",
            )
        else:
            _live_flag = ROOT / "data" / "live_mode_active.flag"
            if _live_flag.exists():
                try:
                    _live_flag.unlink()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        today = datetime.date.today()

        # ── Section A: Market calendar checks ────────────────────────────
        if MarketCalendar.is_holiday(today):
            logger.info("CME holiday — no trading today ({})", today)
            return

        if not self._session.is_trading_day():
            logger.info("Today ({}) is not a trading day. Exiting.", today)
            return

        # Early-close: shorten the execution window
        if MarketCalendar.is_early_close(today):
            exec_end_early = MarketCalendar.exec_end_time(
                today,
                default_end=self._cfg.get("session", {}).get("execution_end", "13:00"),
            )
            logger.warning(
                "EARLY CLOSE DAY — execution window ends at {} ET",
                exec_end_early,
            )

        # Shadow-mode startup notice
        if self._shadow_mode:
            logger.warning(
                "SHADOW MODE — signals will be evaluated but NO orders will be placed. "
                "Useful for side-by-side comparison with live/paper."
            )

        logger.info("Starting trading session for {} — symbol={}", today, self._symbol)
        self._risk.new_day()
        self._audit.log_event(
            "session_start",
            f"date={today} symbol={self._symbol} config_hash={self._config_hash} "
            f"paper={self._paper_mode} shadow={self._shadow_mode}",
        )

        # Start execution engine (connects to TopstepX, starts background thread)
        self._engine.start()

        # Wait for pre-market level build time (7:45am)
        self._wait_for_level_build_time()
        self._build_today_levels()

        # Run pre-market analysis during idle window (7:45am – 9:00am)
        self._run_premarket_analysis()

        # Sleep until 9:00am execution window — state file refreshed every 30s
        self._wait_for_exec_window()

        # Main event loop — strategy runs on each bar callback
        logger.info("Entering main event loop — monitoring for signals...")
        try:
            while True:
                now = self._session.now_est()
                session = self._session.get_session(now.date())

                # Periodically write state file for dashboard (every 30s)
                if time.time() - self._last_state_write >= 30:
                    self._write_state_file()

                # End of session
                if now >= session.exec_end:
                    logger.info("Execution window closed. Shutting down.")
                    break

                # Kill switch
                if self._risk.today.kill_switch_active:
                    reason = self._risk.today.kill_switch_reason.value
                    logger.warning("Kill switch active ({}). Halting.", reason)
                    self._audit.log_event("kill_switch", reason)
                    self._engine.flatten_all()
                    day_pnl = getattr(self._risk.today, "pnl_dollars", 0.0)
                    self._notifier.notify_kill_switch(
                        symbol=self._symbol,
                        reason=reason,
                        day_pnl_dollars=day_pnl,
                        paper_mode=self._paper_mode,
                    )
                    break

                # ── Section B: Connection watchdog ────────────────────────
                # If no new bar received in N seconds → data/connection lost
                secs_since_bar = time.time() - self._last_bar_received
                if secs_since_bar > self._connection_watchdog_secs:
                    logger.error(
                        "CONNECTION WATCHDOG: no bar data for {:.0f}s (limit={}s). "
                        "Flattening and halting.",
                        secs_since_bar, self._connection_watchdog_secs,
                    )
                    self._audit.log_event(
                        "watchdog_halt",
                        f"no_bar_data_{secs_since_bar:.0f}s",
                    )
                    self._engine.flatten_all()
                    self._risk.trigger_kill_switch(KillSwitchReason.CONNECTION_LOST)
                    break

                time.sleep(1)

        finally:
            self._end_of_day()

    # ------------------------------------------------------------------
    # Bar callback — called on every new 1-min bar close
    # ------------------------------------------------------------------

    def _on_new_bar(self, df_1m: pd.DataFrame, df_15m: pd.DataFrame) -> None:
        """
        Called by ExecutionEngine every completed 1-minute bar.
        Runs on the engine's background asyncio thread — keep it fast.
        """
        # ── Section B: Update connection watchdog timestamp ───────────────
        self._last_bar_received = time.time()

        if not self._levels_built or self._day_levels is None:
            return

        now = self._session.now_est()

        # Outside execution window — do nothing
        if not self._session.is_in_execution_window(now):
            return

        # ── Section A: Data integrity gate ───────────────────────────────
        # Run check on first bar of the session, then every 30 bars
        if not hasattr(self, "_integrity_check_count"):
            self._integrity_check_count = 0
        self._integrity_check_count += 1
        if self._integrity_check_count <= 1 or self._integrity_check_count % 30 == 0:
            report = self._integrity_checker.check(df_1m, df_15m, symbol=self._symbol)
            self._data_healthy = report.is_healthy
            if not report.is_healthy:
                logger.error(
                    "DATA INTEGRITY FAILURE — NO TRADING until resolved: {}",
                    report.issues,
                )
                return

        if not self._data_healthy:
            return

        # Update which levels are now tested
        if not df_1m.empty:
            self._level_builder.update_tested_status(
                self._day_levels, float(df_1m["close"].iloc[-1]), df_1m.index[-1]
            )

        # Already in a position — check if the bracket exited via TP/SL
        if self._current_signal and self._engine.active_order:
            bracket = self._engine.active_order
            # exit_price is set by the engine once a stop or target fills
            if (
                bracket.entry_order.status == OrderStatus.FILLED
                and bracket.entry_order.exit_price is not None
            ):
                self._handle_exit(bracket)
            return

        # Position just closed (bracket.closed = True) — clear local state
        if self._current_signal and self._engine.active_order is None:
            # exit already handled by _on_order_filled path; clear signal
            self._current_signal = None

        # Not in position — look for a new signal
        if self._risk.today.open_position:
            return

        if df_1m.empty or df_15m.empty:
            return

        # Write bar data for TradingView chart every 3 bars
        self._bar_write_counter += 1
        if self._bar_write_counter % 3 == 0:
            self._write_bars_file(df_1m, df_15m)

        df_5m = df_1m.resample("5min").agg({
            "open": "first", "high": "max",
            "low": "min", "close": "last", "volume": "sum",
        }).dropna()

        signals = self._setup_detector.detect(
            self._day_levels, df_15m, df_5m, df_1m
        )

        if not signals:
            return

        # Score each signal with the ML filter; pick the highest-confidence one.
        # ml_bypass=True (set per-symbol in config) skips the ML gate entirely —
        # used when sample size is too small for the model to add discriminative value.
        ml_enabled = self._cfg["ml"]["enabled"] and not self._ml_bypass
        if self._ml_bypass:
            logger.debug("ML gate bypassed for {} (ml_bypass=true in config)", self._symbol)
        threshold = self._cfg["ml"]["min_probability_threshold"]

        best_signal = None
        best_prob = -1.0
        best_features: dict = {}

        for sig in signals:
            features = compute_features(sig, self._day_levels, df_15m, df_1m)
            if ml_enabled:
                prob = self._apply_ml_filter(features)
            else:
                prob = 1.0

            # Log all ML-rejected signals
            if ml_enabled and prob < threshold:
                logger.debug(
                    "Signal {} filtered by ML (prob={:.2f} < {:.2f})",
                    sig, prob, threshold,
                )
                self._audit.log_signal(
                    sig, approved=False,
                    rejection_reason=f"ml_filter prob={prob:.2f}",
                    features=features,
                )
                continue

            if prob > best_prob:
                best_prob = prob
                best_signal = sig
                best_features = features

        if best_signal is None:
            return

        signal = best_signal
        features = best_features

        # Feed current volatility regime into risk manager for adaptive sizing
        self._risk.set_volatility_regime(features.get("atr_regime", 1.0))

        # Risk approval
        approved, reason = self._risk.approve_signal(signal)
        self._audit.log_signal(
            signal, approved=approved,
            rejection_reason=reason, features=features,
        )

        if not approved:
            logger.debug("Signal rejected: {}", reason)
            return

        # ── Section C: feed features into drift monitor ───────────────────
        self._drift_monitor.update_live_features(features)

        # ── Section B2: idempotent duplicate-order guard ─────────────────
        if self._order_state.is_duplicate(
            symbol=self._symbol,
            setup_type=signal.setup_type.value,
            direction=signal.direction.value,
            entry_price=signal.entry_price,
        ):
            logger.warning("OrderState: duplicate signal suppressed — already submitted/filled today")
            return

        # ── Section F: portfolio / cross-symbol risk gate ─────────────────
        _port_risk = get_portfolio_risk()
        _port_allowed, _port_reason = _port_risk.allow_new_position(
            self._symbol, signal.direction.value
        )
        if not _port_allowed:
            logger.info(
                "Portfolio risk blocked {} {}: {}",
                self._symbol, signal.direction.value, _port_reason,
            )
            return

        # ── Section G: Shadow mode — log decision but skip order ─────────
        if self._shadow_mode:
            logger.info(
                "SHADOW MODE — would place {} {} at {} stop={} target={}",
                signal.direction.value, signal.setup_type.value,
                signal.entry_price, signal.stop_price, signal.target_price,
            )
            self._audit.log_signal(
                signal, approved=True,
                rejection_reason="shadow_mode_no_order", features=features,
            )
            return

        # Place bracket order on TopstepX
        contracts  = self._risk.get_contract_count()
        live_order = self._engine.place_bracket_order(signal, contracts=contracts)

        if live_order:
            self._current_signal = signal
            self._risk.record_fill(signal, signal.entry_price)
            self._entry_ts = now
            logger.info("Order placed: {}", signal)

            # ── Section B2: record in order state tracker ─────────────────
            _order_id = getattr(live_order, "entry_order", live_order)
            _order_id = getattr(_order_id, "id", str(id(live_order)))
            self._order_state.record_submitted(
                symbol=self._symbol,
                setup_type=signal.setup_type.value,
                direction=signal.direction.value,
                entry_price=signal.entry_price,
                order_id=str(_order_id),
            )

            # ── Section F: record portfolio open position ─────────────────
            get_portfolio_risk().record_open_position(
                self._symbol, signal.direction.value
            )

            # ── Section B: Bracket verification ──────────────────────────
            # After placing bracket, verify stop+target are attached correctly.
            # If verification fails → immediately flatten to prevent naked positions.
            self._verify_bracket_attached(live_order, signal)

            # ── Email: trade entry ──────────────────────────────────────
            self._notifier.notify_entry(
                symbol=self._symbol,
                direction=signal.direction.value,
                entry_price=signal.entry_price,
                stop_price=signal.stop_price,
                target_price=signal.target_price,
                contracts=contracts,
                setup_type=signal.setup_type.value,
                paper_mode=self._paper_mode,
            )
        else:
            logger.error("Order placement failed for signal: {}", signal)

    # ------------------------------------------------------------------
    # Exit handling
    # ------------------------------------------------------------------

    def _handle_exit(self, bracket) -> None:
        """bracket is an ActiveBracket from self._engine.active_order."""
        if self._current_signal is None or self._entry_ts is None:
            return

        exit_price = bracket.entry_order.exit_price or 0.0
        fill_price = bracket.entry_order.fill_price or self._current_signal.entry_price
        exit_ts = datetime.datetime.now()

        direction = self._current_signal.direction
        from src.setup_detector import Direction
        if direction == Direction.LONG:
            pnl_pts = exit_price - fill_price
        else:
            pnl_pts = fill_price - exit_price

        contracts = self._risk.get_contract_count()
        point_val = self._cfg["instrument"]["point_value"]
        commission = self._cfg["backtest"]["commission_per_side"] * 2 * contracts

        pnl_gross = pnl_pts * point_val * contracts
        pnl_net = pnl_gross - commission

        exit_reason = "tp" if pnl_pts > 0 else "sl"

        self._risk.record_exit(
            self._current_signal,
            fill_price,
            exit_price,
            exit_ts,
            self._entry_ts,
        )

        self._audit.log_trade(
            signal=self._current_signal,
            fill_price=fill_price,
            exit_price=exit_price,
            contracts=contracts,
            pnl_points=pnl_pts,
            pnl_dollars=pnl_net,
            commission_dollars=commission,
            exit_reason=exit_reason,
            entry_ts=self._entry_ts,
            exit_ts=exit_ts,
            point_value=point_val,
        )

        # ── Email: trade exit ───────────────────────────────────────────
        day_pnl = self._risk.today.pnl_dollars if hasattr(self._risk, "today") else 0.0
        self._notifier.notify_exit(
            symbol=self._symbol,
            direction=direction.value,
            entry_price=fill_price,
            exit_price=exit_price,
            pnl_points=pnl_pts,
            pnl_dollars=pnl_net,
            exit_reason=exit_reason,
            day_pnl_dollars=day_pnl,
            contracts=contracts,
            paper_mode=self._paper_mode,
        )

        # ── Section F: record portfolio close position ────────────────────
        try:
            get_portfolio_risk().record_close_position(self._symbol, pnl_net)
        except Exception:
            pass

        # ── Section C: update drift monitor with realized_R ──────────────
        try:
            from src.labeling.labeler import compute_realized_R
            slip_cost = self._cfg.get("backtest", {}).get("slippage_per_side", 1.0)
            slip_dollars = slip_cost * point_val * contracts * 2
            r = compute_realized_R(
                entry_price=fill_price,
                exit_price=exit_price,
                stop_price=self._current_signal.stop_price,
                direction=self._current_signal.direction.value,
                point_value=point_val,
                contracts=contracts,
                total_costs=commission + slip_dollars,
            )
            self._drift_monitor.update_live_trade(r)
        except Exception:
            pass

        self._current_signal = None
        self._entry_ts = None

    # ------------------------------------------------------------------
    # Section B: Bracket verification + slippage guard
    # ------------------------------------------------------------------

    def _verify_bracket_attached(self, live_order, signal) -> None:
        """
        After placing a bracket order, verify that stop and target legs
        are attached at approximately the correct prices.
        If not → flatten immediately to prevent a naked position.
        """
        if live_order is None:
            return
        try:
            stop_order  = getattr(live_order, "stop_order",  None)
            target_order = getattr(live_order, "target_order", None)

            if stop_order is None or target_order is None:
                logger.error(
                    "BRACKET VERIFICATION FAILED — stop or target leg not attached! "
                    "Flattening immediately to prevent naked position."
                )
                self._engine.flatten_all()
                self._current_signal = None
                self._audit.log_event(
                    "bracket_verification_fail",
                    f"stop={stop_order is not None} target={target_order is not None}",
                )
                return

            # Verify prices are within 2 ticks of expected
            tick = self._cfg.get("instrument", {}).get("tick_size", 0.25)
            tol = 2 * tick

            stop_price  = getattr(stop_order,   "price", None) or getattr(stop_order,   "stop_price", None)
            target_price = getattr(target_order, "price", None) or getattr(target_order, "limit_price", None)

            if stop_price and abs(stop_price - signal.stop_price) > tol:
                logger.error(
                    "BRACKET VERIFICATION: stop price mismatch: "
                    "submitted={} expected={} — flattening",
                    stop_price, signal.stop_price,
                )
                self._engine.flatten_all()
                self._current_signal = None
                return

            if target_price and abs(target_price - signal.target_price) > tol:
                logger.error(
                    "BRACKET VERIFICATION: target price mismatch: "
                    "submitted={} expected={} — flattening",
                    target_price, signal.target_price,
                )
                self._engine.flatten_all()
                self._current_signal = None
                return

            logger.debug(
                "Bracket verified: stop={} target={}",
                stop_price, target_price,
            )
        except Exception as e:
            logger.warning("Bracket verification exception (non-fatal): {}", e)

    def _check_fill_slippage(self, fill_price: float, signal_price: float) -> bool:
        """
        Returns True if slippage is within the allowed limit.
        If slippage > max_fill_slippage_ticks → returns False and logs error.
        """
        tick = self._cfg.get("instrument", {}).get("tick_size", 0.25)
        slip_ticks = abs(fill_price - signal_price) / tick
        if slip_ticks > self._max_fill_slippage_ticks:
            logger.error(
                "EXCESSIVE SLIPPAGE: fill={} signal={} diff={:.1f} ticks "
                "(limit={} ticks) — reducing size flag set",
                fill_price, signal_price, slip_ticks, self._max_fill_slippage_ticks,
            )
            self._audit.log_event(
                "excessive_slippage",
                f"fill={fill_price} expected={signal_price} ticks={slip_ticks:.1f}",
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Level building
    # ------------------------------------------------------------------

    def _wait_for_level_build_time(self) -> None:
        pre_scan = self._cfg["session"]["pre_market_scan_start"]
        h, m = map(int, pre_scan.split(":"))
        now = self._session.now_est()
        target = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if now < target:
            wait_sec = (target - now).total_seconds()
            logger.info("Waiting {:.0f}s until {} for level build...", wait_sec, pre_scan)
            time.sleep(wait_sec)

    def _wait_for_exec_window(self) -> None:
        """
        Sleep until the 9:00am execution window opens, refreshing the state file
        every 30s so the dashboard shows a live countdown.
        """
        exec_start = self._cfg["session"]["execution_start"]
        h, m = map(int, exec_start.split(":"))
        now = self._session.now_est()
        target = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if now >= target:
            return
        remaining = (target - now).total_seconds()
        logger.info(
            "Pre-market complete — waiting {:.0f}s until {} execution window opens",
            remaining, exec_start,
        )
        while True:
            now = self._session.now_est()
            remaining = (target - now.replace(
                hour=h, minute=m, second=0, microsecond=0
            )).total_seconds()
            # Re-compute target relative to now (avoids DST drift)
            target_now = now.replace(hour=h, minute=m, second=0, microsecond=0)
            remaining = (target_now - now).total_seconds()
            if remaining <= 0:
                break
            self._write_state_file()
            time.sleep(min(30, remaining))
        logger.info("Execution window open — ready to trade")

    # ------------------------------------------------------------------
    # Pre-market analysis (7:45am – 9:00am idle window)
    # ------------------------------------------------------------------

    def _run_premarket_analysis(self) -> None:
        """
        Performs 5 tasks during the idle pre-market window:
          1. News / economic calendar check
          2. Day-type classification (gap up/down, inside day, vol regime)
          3. Level quality scoring
          4. Pre-market volatility regime assessment
          5. Kick off AI retrain as background process

        Results written to data/premarket_<SYMBOL>.json (dashboard reads this).
        """
        from src.news_filter import NewsFilter

        today = self._session.now_est().date()
        notes: list[str] = []

        assessment: dict = {
            "date": str(today),
            "symbol": self._symbol,
            "news_day": False,
            "day_type": "NORMAL",
            "gap_direction": "FLAT",
            "gap_size_atr": 0.0,
            "overnight_range_atr": 0.0,
            "vol_regime": "NORMAL",
            "atr_ratio": 1.0,
            "current_atr": 0.0,
            "level_quality": "UNKNOWN",
            "level_count": 0,
            "session_outlook": "TRADE",
            "session_notes": [],
            "learner_retrain": "NOT_RUN",
            "timestamp": datetime.datetime.now().isoformat(),
        }

        # ── 1. News / economic calendar check ─────────────────────────
        news_filter = NewsFilter(settings=self._cfg)
        if news_filter.is_news_day(today):
            assessment["news_day"] = True
            assessment["session_outlook"] = "NO_TRADE"
            notes.append("⚠️ High-impact news day — trading blocked")
            logger.warning("Pre-market: HIGH-IMPACT NEWS DAY — bot will not trade today")

        # ── 2 + 4. Day-type + volatility regime from bar data ─────────
        df_15m = self._engine.get_df_15m()
        current_atr = 0.0

        if not df_15m.empty and len(df_15m) >= 20:
            hi = df_15m["high"]
            lo = df_15m["low"]
            cl = df_15m["close"].shift(1)
            tr = pd.concat([hi - lo, (hi - cl).abs(), (lo - cl).abs()], axis=1).max(axis=1)
            atr_series = tr.rolling(14).mean().dropna()

            if len(atr_series) >= 20:
                current_atr = float(atr_series.iloc[-1])
                rolling_avg = float(atr_series.tail(20).mean())
                atr_ratio = current_atr / rolling_avg if rolling_avg > 0 else 1.0
                assessment["atr_ratio"] = round(atr_ratio, 2)
                assessment["current_atr"] = round(current_atr, 2)

                if atr_ratio > 1.8:
                    assessment["vol_regime"] = "HIGH"
                    if assessment["session_outlook"] == "TRADE":
                        assessment["session_outlook"] = "CAUTION"
                    notes.append(
                        f"⚡ High volatility: ATR is {atr_ratio:.1f}× its 20-day avg "
                        f"— sizing will be reduced"
                    )
                elif atr_ratio < 0.7:
                    assessment["vol_regime"] = "LOW"
                    notes.append(
                        f"😴 Low volatility: ATR ratio={atr_ratio:.2f} "
                        f"— may be a slow/narrow day"
                    )
                else:
                    notes.append(f"✅ Volatility normal (ATR ratio={atr_ratio:.2f})")

            # Gap analysis: today's anchor open vs prior close
            anchor = self._day_levels.anchor if self._day_levels else None
            if anchor and current_atr > 0:
                prev_bars = df_15m[df_15m.index.date < today]
                if len(prev_bars) >= 1:
                    prev_close = float(prev_bars["close"].iloc[-1])
                    today_open = float(getattr(anchor, "open", prev_close))
                    gap = today_open - prev_close
                    gap_atr = gap / current_atr
                    assessment["gap_size_atr"] = round(abs(gap_atr), 2)
                    if gap_atr > 0.5:
                        assessment["gap_direction"] = "UP"
                        assessment["day_type"] = "GAP_UP"
                        if assessment["session_outlook"] == "TRADE":
                            assessment["session_outlook"] = "CAUTION"
                        notes.append(
                            f"📈 Gap UP: +{abs(gap):.1f}pts above prior close "
                            f"({abs(gap_atr):.1f}× ATR) — watch for gap fill"
                        )
                    elif gap_atr < -0.5:
                        assessment["gap_direction"] = "DOWN"
                        assessment["day_type"] = "GAP_DOWN"
                        if assessment["session_outlook"] == "TRADE":
                            assessment["session_outlook"] = "CAUTION"
                        notes.append(
                            f"📉 Gap DOWN: -{abs(gap):.1f}pts below prior close "
                            f"({abs(gap_atr):.1f}× ATR) — watch for gap fill"
                        )
                    else:
                        notes.append(
                            f"➡️ Flat open (gap={gap:+.1f}pts, {abs(gap_atr):.2f}× ATR)"
                        )

            # Overnight range
            try:
                overnight = df_15m[
                    (df_15m.index.date == today) & (df_15m.index.hour < 9)
                ]
                if len(overnight) >= 2 and current_atr > 0:
                    ovn_range = float(overnight["high"].max() - overnight["low"].min())
                    ovn_atr = ovn_range / current_atr
                    assessment["overnight_range_atr"] = round(ovn_atr, 2)
                    if ovn_atr > 2.0:
                        notes.append(
                            f"🌙 Wide overnight range: {ovn_range:.1f}pts "
                            f"({ovn_atr:.1f}× ATR) — choppy overnight"
                        )
                    else:
                        notes.append(
                            f"🌙 Overnight range: {ovn_range:.1f}pts ({ovn_atr:.1f}× ATR)"
                        )
            except Exception:
                pass

        # ── 3. Level quality scoring ───────────────────────────────────
        if self._day_levels:
            anchor = self._day_levels.anchor
            active_highs = self._day_levels.active_highs() \
                if hasattr(self._day_levels, "active_highs") else []
            active_lows = self._day_levels.active_lows() \
                if hasattr(self._day_levels, "active_lows") else []
            level_count = len(active_highs) + len(active_lows)
            assessment["level_count"] = level_count

            if self._day_levels.no_trade_day:
                assessment["level_quality"] = "NO_TRADE"
                assessment["session_outlook"] = "NO_TRADE"
                notes.append("🚫 No-trade day: anchor candle outside valid range")
            elif anchor is None:
                assessment["level_quality"] = "POOR"
                if assessment["session_outlook"] == "TRADE":
                    assessment["session_outlook"] = "CAUTION"
                notes.append("⚠️ No anchor candle found — levels may be unreliable")
            elif level_count >= 4:
                assessment["level_quality"] = "GREAT"
                notes.append(f"✅ {level_count} price levels — excellent setup")
            elif level_count >= 2:
                assessment["level_quality"] = "GOOD"
                notes.append(f"✅ {level_count} price levels — clean setup")
            elif level_count == 1:
                assessment["level_quality"] = "OK"
                notes.append("✅ 1 price level — limited but workable")
            else:
                assessment["level_quality"] = "POOR"
                if assessment["session_outlook"] == "TRADE":
                    assessment["session_outlook"] = "CAUTION"
                notes.append("⚠️ No price levels — limited trade opportunities today")

        # ── Final outlook summary note (prepended) ─────────────────────
        if assessment["session_outlook"] == "TRADE":
            notes.insert(0, "🟢 Conditions look GOOD — ready to trade at 9:00am")
        elif assessment["session_outlook"] == "CAUTION":
            notes.insert(0, "🟡 CAUTION: Abnormal conditions — trade with reduced size")
        else:
            notes.insert(0, "🔴 NO-TRADE today — bot will not place orders")

        assessment["session_notes"] = notes

        # Log the assessment
        logger.info(
            "Pre-market assessment [{}]: outlook={} day_type={} vol={} levels={} atr_ratio={}",
            self._symbol,
            assessment["session_outlook"],
            assessment["day_type"],
            assessment["vol_regime"],
            assessment["level_count"],
            assessment["atr_ratio"],
        )
        for note in notes:
            clean = note.encode("ascii", "ignore").decode("ascii").strip()
            logger.info("  {}", clean)

        # ── 5. Kick off AI retrain as background process ───────────────
        try:
            retrain_log = ROOT / "data" / "run_logs" / f"premarket_retrain_{self._symbol}.log"
            retrain_log.parent.mkdir(parents=True, exist_ok=True)
            with open(retrain_log, "w") as _rf:
                proc = subprocess.Popen(
                    [sys.executable, "src/ml_retrain_all.py", "--symbols", self._symbol],
                    stdout=_rf,
                    stderr=subprocess.STDOUT,
                    cwd=str(ROOT),
                    stdin=subprocess.DEVNULL,
                )
            assessment["learner_retrain"] = f"RUNNING (pid={proc.pid})"
            logger.info(
                "Pre-market: AI retrain started in background (pid={}) — "
                "log: {}",
                proc.pid, retrain_log.name,
            )
        except Exception as exc:
            assessment["learner_retrain"] = f"FAILED: {exc}"
            logger.warning("Pre-market: could not start AI retrain: {}", exc)

        # Persist assessment for the dashboard
        self._premarket_assessment = assessment
        premarket_file = ROOT / "data" / f"premarket_{self._symbol}.json"
        try:
            premarket_file.write_text(json.dumps(assessment, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.debug("Could not write premarket file: {}", exc)

        self._write_state_file()

    def _build_today_levels(self) -> None:
        """Fetch recent bar history from the engine and build today's levels."""
        df_15m = self._engine.get_df_15m()
        df_1m  = self._engine.get_df_1m()

        if df_15m.empty:
            logger.warning("No 15m bars available yet for level build — retrying in 30s")
            time.sleep(30)
            df_15m = self._engine.get_df_15m()

        today = self._session.now_est().date()
        self._day_levels = self._level_builder.build(today, df_15m, df_1m)
        self._levels_built = True

        if self._day_levels.no_trade_day:
            # Previously this marked a hard kill-switch:
            #   self._risk.today.kill_switch_active = True
            #   self._risk.today.kill_switch_reason = KillSwitchReason.NO_TRADE_DAY
            # Option A: disable only the NO_TRADE_DAY kill-switch while
            # still logging and exposing the no-trade assessment.
            logger.warning("No-trade day declared for {}", today)

        logger.info("Levels built for {}: anchor={}", today, self._day_levels.anchor)

    # ------------------------------------------------------------------
    # ML filter (Phase 2 stub)
    # ------------------------------------------------------------------

    def _apply_ml_filter(self, features: dict) -> float:
        """
        Returns predicted win probability.
        Stub — loads a trained XGBoost model from disk.
        Activate after Phase 1 is profitable.
        """
        model_path = ROOT / "data" / "ml_filter_model.pkl"
        if not model_path.exists():
            logger.warning("ML model not found at {}. Returning probability=1.0", model_path)
            return 1.0

        try:
            import pickle
            import numpy as np
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            X = pd.DataFrame([features])
            prob = model.predict_proba(X)[0][1]
            return float(prob)
        except Exception as e:
            logger.error("ML filter error: {}. Bypassing.", e)
            return 1.0

    # ------------------------------------------------------------------
    # End of day
    # ------------------------------------------------------------------

    def _end_of_day(self) -> None:
        self._engine.stop()
        summary = self._risk.daily_summary()
        self._audit.log_daily_summary(
            date=summary["date"],
            total_trades=summary["trades"],
            wins=summary["wins"],
            losses=summary["losses"],
            win_rate=summary["win_rate"],
            pnl_dollars=summary["pnl_dollars"],
            kill_switch_triggered=summary["kill_switch"],
            kill_reason=summary["kill_reason"],
        )
        self._audit.log_event("session_end", str(summary))
        logger.info("End of day: {}", summary)
        self._write_state_file(running=False)
        self._unregister_pid()

    # ------------------------------------------------------------------
    # PID registration — so dashboard detects bot regardless of start method
    # ------------------------------------------------------------------

    def _register_pid(self) -> None:
        """Write our PID to bot_pids.json so the dashboard shows us as Running."""
        try:
            pids: dict = {}
            if self._pids_file.exists():
                try:
                    pids = json.loads(self._pids_file.read_text(encoding="utf-8"))
                except Exception:
                    pids = {}
            pids[self._symbol] = os.getpid()
            self._pids_file.write_text(json.dumps(pids, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug("Could not register PID: {}", e)

    def _unregister_pid(self) -> None:
        """Remove our PID from bot_pids.json on clean shutdown."""
        try:
            pids: dict = {}
            if self._pids_file.exists():
                try:
                    pids = json.loads(self._pids_file.read_text(encoding="utf-8"))
                except Exception:
                    pids = {}
            pids.pop(self._symbol, None)
            self._pids_file.write_text(json.dumps(pids, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug("Could not unregister PID: {}", e)

    # ------------------------------------------------------------------
    # Bar data file — written every ~3 bars for TradingView chart
    # ------------------------------------------------------------------

    def _write_bars_file(self, df_1m: pd.DataFrame, df_15m: pd.DataFrame) -> None:
        """
        Serialize recent OHLCV bars and today's price levels to JSON so the
        dashboard TradingView Lightweight Chart can render them live.
        """
        try:
            def _to_tv(df: pd.DataFrame, n: int) -> list:
                rows = []
                for ts, row in df.tail(n).iterrows():
                    try:
                        t = int(pd.Timestamp(ts).timestamp())
                    except Exception:
                        continue
                    rows.append({
                        "time": t,
                        "open":   round(float(row["open"]),  4),
                        "high":   round(float(row["high"]),  4),
                        "low":    round(float(row["low"]),   4),
                        "close":  round(float(row["close"]), 4),
                        "volume": int(row.get("volume", 0)),
                    })
                return rows

            # Build level lines
            levels = []
            if self._day_levels and self._day_levels.anchor:
                a = self._day_levels.anchor
                levels += [
                    {"price": float(a.high), "label": "OR High", "color": "#26a69a"},
                    {"price": float(a.low),  "label": "OR Low",  "color": "#ef5350"},
                    {"price": float(a.mid),  "label": "OR Mid",  "color": "#ff9800"},
                ]
            if self._day_levels:
                for lvl in (self._day_levels.active_highs()
                            if hasattr(self._day_levels, "active_highs") else []):
                    levels.append({
                        "price": float(lvl.price),
                        "label": f"H/{lvl.session_origin}",
                        "color": "#4dd0e1",
                    })
                for lvl in (self._day_levels.active_lows()
                            if hasattr(self._day_levels, "active_lows") else []):
                    levels.append({
                        "price": float(lvl.price),
                        "label": f"L/{lvl.session_origin}",
                        "color": "#ce93d8",
                    })

            bars_file = ROOT / "data" / f"bars_{self._symbol}.json"
            bars_file.write_text(
                json.dumps({
                    "symbol":      self._symbol,
                    "bars_1m":     _to_tv(df_1m,  300),
                    "bars_15m":    _to_tv(df_15m, 100),
                    "levels":      levels,
                    "last_update": datetime.datetime.now().isoformat(),
                }),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.debug("Could not write bars file: {}", exc)

    # ------------------------------------------------------------------
    # State file — written every 30s and on shutdown for the dashboard
    # ------------------------------------------------------------------

    def _write_state_file(self, running: bool = True) -> None:
        """Write a JSON snapshot of bot state to data/bot_state_<SYMBOL>.json."""
        try:
            summary = self._risk.daily_summary()
            acct_summary = self._account_mgr.summary() if self._account_mgr else {}
            state = {
                "symbol": self._symbol,
                "running": running,
                "pid": os.getpid(),
                "paper_mode": self._cfg["execution"]["paper_mode"],
                "date": str(datetime.date.today()),
                "today_pnl": summary.get("pnl_dollars", 0.0),
                "today_trades": summary.get("trades", 0),
                "today_wins": summary.get("wins", 0),
                "today_losses": summary.get("losses", 0),
                "today_win_rate": summary.get("win_rate", 0.0),
                "open_position": self._risk.today.open_position,
                "kill_switch_active": summary.get("kill_switch", False),
                "kill_switch_reason": summary.get("kill_reason", "none"),
                "levels_built": self._levels_built,
                "premarket": self._premarket_assessment,
                "account": acct_summary,
                "last_update": datetime.datetime.now().isoformat(),
            }
            self._state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
            self._last_state_write = time.time()
        except Exception as e:
            logger.debug("Could not write state file: {}", e)

    # ------------------------------------------------------------------
    # Connection kill callback
    # ------------------------------------------------------------------

    def _on_connection_kill(self) -> None:
        self._risk.trigger_connection_kill()
        self._audit.log_event("kill_switch", "connection_lost")

    # ------------------------------------------------------------------
    # Graceful shutdown
    # ------------------------------------------------------------------

    def _shutdown(self, signum, frame) -> None:
        logger.info("Shutdown signal received — cleaning up...")
        self._risk.trigger_manual_kill()
        self._engine.stop()
        self._unregister_pid()
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChargedUp Profits Bot — Live Bot")
    parser.add_argument(
        "--symbol", type=str, default=None,
        help="Symbol to trade (ES or NQ). Overrides settings.yaml instrument.symbol.",
    )
    parser.add_argument(
        "--settings", type=str, default="config/settings.yaml",
        help="Path to settings YAML file.",
    )
    args = parser.parse_args()

    bot = BotRunner(settings_path=args.settings, symbol_override=args.symbol)
    bot.run()
