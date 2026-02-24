"""
backtest/harness.py

Event-driven replay backtester for the ChargedUp Profits Bot strategy.

Key design choices that prevent "cheating":
  - Levels are only built using bars that existed BEFORE the anchor candle time
    (no lookahead leakage).
  - Entries are simulated at the OPEN of the next 1-min bar after signal fires
    (not the close of the signal bar).
  - Conservative fill model: entry slippage applied against direction.
  - Commission applied per side.
  - Daily loss limit and max-trades rules are enforced exactly as in live trading.

Usage:
    python backtest/harness.py --symbol ES --start 2022-01-03 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.feature_builder import compute_features
from src.labeling.labeler import compute_realized_R
from src.level_builder import LevelBuilder
from src.ml_filter import InferenceFilter
from src.session_engine import SessionEngine
from src.setup_detector import Direction, SetupDetector, Signal


def _load_settings(path: str = "config/settings.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Trade result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BacktestTrade:
    date: str
    setup_type: str
    direction: str
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    contracts: int
    commission_dollars: float
    slippage_dollars: float
    pnl_points: float
    pnl_gross: float         # Before costs
    pnl_net: float           # After costs
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    exit_reason: str         # "tp", "sl", "eod"
    point_value: float
    features_json: str = ""  # JSON-encoded feature vector at signal time (for ML training)
    # ── New fields (Section 2-3) ─────────────────────────────────────────────
    realized_R: float = 0.0  # net_pnl / initial_risk  (R-multiple achieved)
    initial_risk_dollars: float = 0.0
    filter_states: str = ""  # JSON: {"trend":true,"volume":true,"regime":true,"news":true}


# ---------------------------------------------------------------------------
# Backtesting engine
# ---------------------------------------------------------------------------

class BacktestHarness:
    """
    Replays historical 1-minute and 15-minute OHLCV data bar-by-bar,
    feeding each bar into the strategy engine and simulating fills.
    """

    def __init__(self, settings_path: str = "config/settings.yaml") -> None:
        self._cfg = _load_settings(settings_path)
        self._session = SessionEngine(settings_path)
        self._level_builder = LevelBuilder(settings_path)
        self._setup_detector = SetupDetector(settings_path)
        self._ml_bypass: bool = False   # updated in configure_for_symbol

        bt = self._cfg["backtest"]
        risk = self._cfg["risk"]
        instr = self._cfg["instrument"]

        # Defaults from root config — overridden per symbol in configure_for_symbol()
        self._commission_per_side: float = bt["commission_per_side"]
        self._slippage_per_side: float = bt["slippage_per_side_points"]
        self._point_value: float = instr["point_value"]
        self._max_daily_loss: float = risk["max_daily_loss_dollars"]
        self._max_trades_per_day: int = risk["max_trades_per_day"]
        self._exec_start_time: str = self._cfg["session"]["execution_start"]
        self._exec_end_time: str = self._cfg["session"]["execution_end"]
        self._initial_contracts: int = risk["initial_contracts"]

        # Partial profit taking config (Phase 2d)
        self._partial_tp_enabled: bool = risk.get("partial_tp_enabled", False)
        self._partial_tp_ratio: float = risk.get("partial_tp_ratio", 1.5)

        # ML filter — loaded per-symbol in configure_for_symbol()
        # Gracefully no-ops when model file doesn't exist or ml.enabled=false
        self._ml_filter: Optional[InferenceFilter] = None
        self._symbol: str = "ES"  # updated in configure_for_symbol

        self.trades: list[BacktestTrade] = []

    def configure_for_symbol(self, symbol: str) -> None:
        """
        Apply per-symbol overrides from the ``symbols:`` block in settings.yaml.

        Called automatically by ``load_data()``.  Handles:
          - ``point_value``  — dollar value per point (e.g. MNQ=$2, NQ=$20, ES=$50)
          - ``backtest_overrides.commission_per_side`` — micro contracts are cheaper
          - ``data_source_symbol`` — resolved internally; not applied here

        Also propagates symbol config to ``LevelBuilder`` and ``SetupDetector``
        so NQ/MNQ use correct range thresholds, setup tolerances, stop clamping,
        and regime multiplier (the root-cause bugs for poor NQ performance).
        """
        sym_cfg: dict = self._cfg.get("symbols", {}).get(symbol, {})
        if not sym_cfg:
            return  # Unknown symbol — use root defaults

        if "point_value" in sym_cfg:
            self._point_value = float(sym_cfg["point_value"])
            logger.debug("Symbol {}: point_value={}", symbol, self._point_value)

        bt_overrides: dict = sym_cfg.get("backtest_overrides", {})
        if "commission_per_side" in bt_overrides:
            self._commission_per_side = float(bt_overrides["commission_per_side"])
            logger.debug(
                "Symbol {}: commission_per_side={}", symbol, self._commission_per_side
            )

        # ── Propagate to sub-components (critical for NQ/MNQ correctness) ────
        self._level_builder.configure_for_symbol(symbol)
        self._setup_detector.configure_for_symbol(symbol)

        # ── Per-symbol ML bypass and enabled setups ───────────────────────────
        self._ml_bypass = bool(sym_cfg.get("ml_bypass", False))
        if self._ml_bypass:
            logger.info(
                "Symbol {}: ml_bypass=true — ML gate disabled, rules run clean",
                symbol,
            )
        # Re-init SetupDetector with symbol so enabled_setups filter is applied
        self._setup_detector = SetupDetector("config/settings.yaml", symbol=symbol)
        self._level_builder.configure_for_symbol(symbol)

        # ── Load per-symbol ML inference filter ──────────────────────────────
        self._symbol = symbol
        self._ml_filter = InferenceFilter(settings_path="config/settings.yaml", symbol=symbol)

    def set_ml_filter(self, ml_filter: "InferenceFilter | None") -> None:
        """
        Inject a pre-built InferenceFilter (or None to disable ML gating).
        Used by clean_eval.py and the per-window WFA to avoid disk-based leakage:
        each evaluation window trains its own filter on ONLY its training data.
        Note: if ml_bypass=true for this symbol (config), the filter is stored
        but never consulted — the bypass flag takes precedence.
        """
        self._ml_filter = ml_filter

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------

    def load_data(self, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Apply per-symbol config overrides (point_value, commission, etc.)
        self.configure_for_symbol(symbol)

        hist_dir = ROOT / "data" / "historical"

        # Some micro contracts (MNQ, MES) share price data with their full-size
        # counterparts — resolve the actual data file to load.
        sym_cfg: dict = self._cfg.get("symbols", {}).get(symbol, {})
        data_symbol = sym_cfg.get("data_source_symbol", symbol)
        if data_symbol != symbol:
            logger.info(
                "{} uses {} price data (same chart, different contract size)",
                symbol, data_symbol,
            )

        p1m  = hist_dir / f"{data_symbol}_ohlcv-1m.parquet"
        p15m = hist_dir / f"{data_symbol}_ohlcv-15m.parquet"

        if not p1m.exists() or not p15m.exists():
            logger.error(
                "Missing data files for {}. Run: "
                "python data/download_historical.py --symbols {}",
                data_symbol, data_symbol,
            )
            sys.exit(1)

        df_1m  = pd.read_parquet(p1m)
        df_15m = pd.read_parquet(p15m)

        df_1m  = self._session.localize_bars(df_1m)
        df_15m = self._session.localize_bars(df_15m)

        logger.info(
            "Loaded {} 1m bars and {} 15m bars for {} (data_source={})",
            len(df_1m), len(df_15m), symbol, data_symbol,
        )
        return df_1m, df_15m

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(
        self,
        df_1m: pd.DataFrame,
        df_15m: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Run the full backtest.  Returns a DataFrame of all trades.

        Performance pre-processing (done once here, not repeated per day):
          • Sort both DataFrames so searchsorted works correctly.
          • Pre-group the 1m DataFrame by calendar date into a dict for O(1) daily
            lookup — avoids scanning 1M+ rows with .index.date on every day.
          • Pre-build a numpy int64 array of the 15m timestamps for fast
            searchsorted O(log n) slicing — replaces per-bar boolean masks.
        """
        self.trades = []

        # ── Ensure sorted index ─────────────────────────────────────────────
        df_1m  = df_1m.sort_index()
        df_15m = df_15m.sort_index()

        # ── Pre-group 1m bars by date (FIX #1) ─────────────────────────────
        # Scanning df_1m_full.index.date == date_obj is O(1 053 599) per day.
        # Building this dict once makes each lookup O(1).
        _1m_by_date: dict = {}
        for _d, _g in df_1m.groupby(df_1m.index.date):
            _1m_by_date[_d] = _g

        # ── FIX #5: pre-compute all 15m rolling indicators ONCE ─────────────
        # RegimeFilter.classify() was recomputing ATR(14)+rolling(20) on the full
        # 70k-row 15m DataFrame on every 1m bar.  TrendFilter was recomputing
        # EWM20 + EWM50 on every bar.  Combined that was ~60 minutes of CPU.
        # Pre-computing these as columns means per-bar access is just .iloc[-1].
        df_15m = df_15m.copy()
        _hi, _lo = df_15m["high"], df_15m["low"]
        _cl_prev  = df_15m["close"].shift(1)
        _tr = pd.concat(
            [_hi - _lo, (_hi - _cl_prev).abs(), (_lo - _cl_prev).abs()],
            axis=1,
        ).max(axis=1)
        df_15m["_atr14"]        = _tr.rolling(14).mean()
        df_15m["_atr14_roll20"] = df_15m["_atr14"].rolling(20).mean()
        df_15m["_ema20"]        = df_15m["close"].ewm(span=20, adjust=False).mean()
        df_15m["_ema50"]        = df_15m["close"].ewm(span=50, adjust=False).mean()

        # ── Pre-build 15m numpy index for searchsorted (FIX #2 & #3) ───────
        # Converts each per-bar boolean mask (O(70 393)) into a binary search (O(17)).
        _15m_idx_np = df_15m.index.values.astype("int64")  # nanoseconds since epoch

        trading_days = self._get_trading_days(df_1m, start_date, end_date)
        total_days = len(trading_days)
        logger.info("Running backtest over {} trading days", total_days)

        for i, date in enumerate(trading_days):
            self._setup_detector.reset_day()
            self._run_day(date, df_1m, df_15m, _1m_by_date, _15m_idx_np)
            # Progress beacon for the web-panel progress bar
            if (i + 1) % 10 == 0 or (i + 1) == total_days:
                logger.info("PROGRESS {}/{}", i + 1, total_days)

        results = pd.DataFrame([vars(t) for t in self.trades])
        if results.empty:
            logger.warning("No trades generated in backtest period")
            return results

        logger.info(
            "Backtest complete: {} trades | Total P&L net={:.2f}$ | Win rate={:.1f}%",
            len(results),
            results["pnl_net"].sum(),
            (results["pnl_net"] > 0).mean() * 100,
        )
        return results

    # ------------------------------------------------------------------
    # Single-day simulation
    # ------------------------------------------------------------------

    def _run_day(
        self,
        date: pd.Timestamp,
        df_1m_full: pd.DataFrame,
        df_15m_full: pd.DataFrame,
        _1m_by_date: Optional[dict] = None,
        _15m_idx_np: Optional[np.ndarray] = None,
    ) -> None:
        date_obj = date.date()

        # ── FIX #1: O(1) daily 1m lookup ────────────────────────────────────
        if _1m_by_date is not None:
            day_1m = _1m_by_date.get(date_obj, pd.DataFrame())
        else:
            day_1m = df_1m_full[df_1m_full.index.date == date_obj]  # slow fallback

        if len(day_1m) < 30:
            return  # Insufficient data (holiday/early close/missing)

        # ── FIX #2: searchsorted for 15m lookback window ────────────────────
        # Old: two O(70k) boolean masks.  New: two O(log 70k) binary searches.
        lookback_start = date - pd.Timedelta(days=6)
        next_day      = pd.Timestamp(date_obj) + pd.Timedelta(days=1)
        # Align timezone to match df_15m_full.index
        tz = df_15m_full.index.tz
        if tz is not None:
            if lookback_start.tzinfo is None:
                lookback_start = lookback_start.tz_localize(tz)
            if next_day.tzinfo is None:
                next_day = next_day.tz_localize(tz)
        ls_ns = np.int64(lookback_start.value)
        nd_ns = np.int64(next_day.value)
        if _15m_idx_np is not None:
            lo_pos = int(np.searchsorted(_15m_idx_np, ls_ns, side="left"))
            hi_pos = int(np.searchsorted(_15m_idx_np, nd_ns, side="left"))
            df_15m_prior = df_15m_full.iloc[lo_pos:hi_pos]
        else:
            df_15m_prior = df_15m_full.loc[
                (df_15m_full.index >= lookback_start)
                & (df_15m_full.index.date <= date_obj)
            ]

        # ── FIX #4: pre-compute 5m bars for the whole day once ──────────────
        # Old: resample growing slice inside the per-bar loop (~200× per day).
        # New: resample once here, then just slice the small result.
        day_5m_full = day_1m.resample("5min").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        ).dropna()

        # Build levels (uses only pre-anchor bars internally)
        day_levels = self._level_builder.build(date_obj, df_15m_prior, day_1m)

        if day_levels.no_trade_day:
            return

        # Define execution window for this day
        exec_start = pd.Timestamp(date_obj).tz_localize("America/New_York").replace(
            hour=int(self._exec_start_time.split(":")[0]),
            minute=int(self._exec_start_time.split(":")[1]),
        )
        exec_end = pd.Timestamp(date_obj).tz_localize("America/New_York").replace(
            hour=int(self._exec_end_time.split(":")[0]),
            minute=int(self._exec_end_time.split(":")[1]),
        )

        # Daily state
        daily_pnl = 0.0
        trades_today = 0
        in_trade = False
        active_signal: Optional[Signal] = None
        active_features: str = ""
        entry_price = 0.0
        entry_ts: Optional[pd.Timestamp] = None
        # Partial TP tracking (Phase 2d)
        partial_tp_hit: bool = False
        effective_stop: float = 0.0   # Trails to B/E after partial TP

        # Bar-by-bar replay
        for i, (ts, bar) in enumerate(day_1m.iterrows()):
            # End of day: flat everything
            if ts > exec_end:
                if in_trade and active_signal is not None and entry_ts is not None:
                    remaining = (
                        self._initial_contracts - max(1, self._initial_contracts // 2)
                        if (self._partial_tp_enabled and partial_tp_hit and self._initial_contracts > 1)
                        else self._initial_contracts
                    )
                    self._record_trade(
                        active_signal, entry_price, float(bar["close"]),
                        entry_ts, ts, "eod",
                        daily_pnl_ref=[daily_pnl],
                        features_json=active_features,
                        contracts_override=remaining,
                        filter_states=_passed_filters,
                    )
                    daily_pnl += self.trades[-1].pnl_net
                break

            # Not yet in execution window
            if ts < exec_start:
                # Still update tested-level status
                self._level_builder.update_tested_status(
                    day_levels, float(bar["close"]), ts
                )
                continue

            # Risk checks (before entering new trade)
            if not in_trade:
                if trades_today >= self._max_trades_per_day:
                    break
                if daily_pnl <= self._max_daily_loss:
                    break

                # ── FIX #3: O(log n) 15m lookup via searchsorted ──────────
                # Old: df_15m_full.loc[df_15m_full.index <= ts]  → O(70 393)
                # New: binary search on pre-built int64 array     → O(17)
                df_1m_so_far = day_1m.loc[:ts]
                if _15m_idx_np is not None:
                    ts_ns = np.int64(ts.value)
                    pos   = int(np.searchsorted(_15m_idx_np, ts_ns, side="right"))
                    df_15m_so_far = df_15m_full.iloc[:pos]
                else:
                    df_15m_so_far = df_15m_full.loc[df_15m_full.index <= ts]

                # ── FIX #4: slice pre-built 5m bars (no resample per bar) ──
                df_5m_so_far = day_5m_full.loc[:ts]

                raw_signals = self._setup_detector.detect(
                    day_levels, df_15m_so_far, df_5m_so_far, df_1m_so_far
                )

                # Score all signals with ML; pick the highest-confidence one
                signal = None
                feat_dict: dict = {}
                captured_features = ""
                best_prob = -1.0
                # ── Section 4: filter state logging ──────────────────────────
                # All signals that reach ML gate have already passed trend /
                # volume / regime / news filters inside SetupDetector.detect().
                # Record that here so training can confirm Mode A (train only on
                # ML-eligible signals = post-rule-filter population).
                _passed_filters = {
                    "trend": True, "volume": True,
                    "regime": True, "news": True,
                }

                for _sig in raw_signals:
                    _feat: dict = {}
                    _feat_json = ""
                    try:
                        _feat = compute_features(_sig, day_levels, df_15m_so_far, df_1m_so_far)
                        _feat_json = json.dumps({
                            k: round(v, 6) if isinstance(v, float) else v
                            for k, v in _feat.items()
                        })
                    except Exception:
                        pass

                    # ML filter gate — pick highest-confidence passing signal.
                    # ml_bypass skips the gate entirely (set for symbols where model
                    # has insufficient data to add discriminative value, e.g. ES).
                    _prob = 1.0
                    if not self._ml_bypass and self._ml_filter is not None and _feat:
                        ml_allow, _prob = self._ml_filter.allows_trade(_feat)
                        if not ml_allow:
                            logger.debug(
                                "ML filter rejected signal {} (prob={:.3f})",
                                _sig.setup_type.value, _prob,
                            )
                            continue

                    if _prob > best_prob:
                        best_prob = _prob
                        signal = _sig
                        feat_dict = _feat
                        captured_features = _feat_json

                if signal is not None:
                    # Simulate fill on NEXT bar open (no lookahead fill)
                    if i + 1 < len(day_1m):
                        next_bar = day_1m.iloc[i + 1]
                        raw_fill = float(next_bar["open"])
                        # Apply adverse slippage
                        if signal.direction == Direction.LONG:
                            fill = raw_fill + self._slippage_per_side
                        else:
                            fill = raw_fill - self._slippage_per_side

                        in_trade = True
                        active_signal = signal
                        active_features = captured_features
                        entry_price = fill
                        entry_ts = day_1m.index[i + 1]
                        partial_tp_hit = False
                        effective_stop = signal.stop_price

            # Manage open position (check TP and SL with optional partial exit)
            elif in_trade and active_signal is not None and entry_ts is not None:
                hi = float(bar["high"])
                lo = float(bar["low"])

                # Compute partial TP price (1.5R by default) if not yet hit
                stop_dist = abs(entry_price - effective_stop)
                if active_signal.direction == Direction.LONG:
                    partial_tp_price = entry_price + stop_dist * self._partial_tp_ratio
                else:
                    partial_tp_price = entry_price - stop_dist * self._partial_tp_ratio

                # Check partial TP hit (Phase 2d)
                if (
                    self._partial_tp_enabled
                    and not partial_tp_hit
                ):
                    hit_partial = (
                        hi >= partial_tp_price
                        if active_signal.direction == Direction.LONG
                        else lo <= partial_tp_price
                    )
                    if hit_partial:
                        partial_tp_hit = True
                        # Record partial exit trade (50% of position at partial TP)
                        self._record_trade(
                            active_signal, entry_price, partial_tp_price,
                            entry_ts, ts, "partial_tp",
                            daily_pnl_ref=[daily_pnl],
                            features_json=active_features,
                            contracts_override=max(1, self._initial_contracts // 2),
                            filter_states=_passed_filters,
                        )
                        daily_pnl += self.trades[-1].pnl_net
                        # Trail stop to breakeven for remainder
                        effective_stop = entry_price

                hit_tp = False
                hit_sl = False

                if active_signal.direction == Direction.LONG:
                    hit_tp = hi >= active_signal.target_price
                    hit_sl = lo <= effective_stop
                else:
                    hit_tp = lo <= active_signal.target_price
                    hit_sl = hi >= effective_stop

                # SL takes priority if both triggered on same bar
                if hit_sl or hit_tp:
                    exit_price = (
                        effective_stop if hit_sl
                        else active_signal.target_price
                    )
                    exit_reason = "sl" if hit_sl else "tp"

                    # Remaining contracts after partial TP
                    remaining = (
                        self._initial_contracts - max(1, self._initial_contracts // 2)
                        if (self._partial_tp_enabled and partial_tp_hit and self._initial_contracts > 1)
                        else self._initial_contracts
                    )
                    self._record_trade(
                        active_signal, entry_price, exit_price,
                        entry_ts, ts, exit_reason,
                        daily_pnl_ref=[daily_pnl],
                        features_json=active_features,
                        contracts_override=remaining,
                        filter_states=_passed_filters,
                    )
                    daily_pnl += self.trades[-1].pnl_net
                    trades_today += 1
                    in_trade = False
                    active_signal = None
                    active_features = ""
                    partial_tp_hit = False

            # Update level tested status after each bar
            self._level_builder.update_tested_status(
                day_levels, float(bar["close"]), ts
            )

    # ------------------------------------------------------------------
    # Fill / cost model
    # ------------------------------------------------------------------

    def _session_phase_slippage(self, ts: pd.Timestamp) -> float:
        """
        Session-phase aware slippage model.
        Open drive (9:00–10:00) and around key news windows are harder to fill.
        Mid-morning is most liquid.  Afternoon reverts to base.

        Returns a slippage multiplier (1.0 = base, 2.0 = double base).
        """
        try:
            h = ts.hour
            m = ts.minute
            minute_of_day = h * 60 + m
            if 9 * 60 <= minute_of_day < 10 * 60:      # open drive
                return 2.0
            if 10 * 60 <= minute_of_day < 11 * 60 + 30:  # liquid mid-morning
                return 0.8
            return 1.0
        except Exception:
            return 1.0

    def _record_trade(
        self,
        signal: Signal,
        entry_price: float,
        exit_price: float,
        entry_ts: pd.Timestamp,
        exit_ts: pd.Timestamp,
        exit_reason: str,
        daily_pnl_ref: list,
        features_json: str = "",
        contracts_override: Optional[int] = None,
        filter_states: Optional[dict] = None,
    ) -> None:
        contracts = contracts_override if contracts_override is not None else self._initial_contracts
        contracts = max(1, contracts)  # always at least 1
        commission = self._commission_per_side * 2 * contracts  # round trip

        # Session-phase slippage — worse at open, better mid-morning
        slip_mult = self._session_phase_slippage(entry_ts)
        slippage_per_side = self._slippage_per_side * slip_mult
        slippage_cost = slippage_per_side * self._point_value * contracts * 2

        if signal.direction == Direction.LONG:
            pnl_pts = exit_price - entry_price
        else:
            pnl_pts = entry_price - exit_price

        pnl_gross = pnl_pts * self._point_value * contracts
        pnl_net = pnl_gross - commission - slippage_cost

        total_costs = commission + slippage_cost
        initial_risk_pts = abs(entry_price - signal.stop_price)
        initial_risk_dollars = initial_risk_pts * self._point_value * contracts

        r_multiple = compute_realized_R(
            entry_price=entry_price,
            exit_price=exit_price,
            stop_price=signal.stop_price,
            direction=signal.direction.value,
            point_value=self._point_value,
            contracts=contracts,
            total_costs=total_costs,
        )

        filter_json = json.dumps(filter_states) if filter_states else ""

        self.trades.append(BacktestTrade(
            date=str(entry_ts.date()),
            setup_type=signal.setup_type.value,
            direction=signal.direction.value,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_price=signal.stop_price,
            target_price=signal.target_price,
            contracts=contracts,
            commission_dollars=round(commission, 4),
            slippage_dollars=round(slippage_cost, 4),
            pnl_points=round(pnl_pts, 2),
            pnl_gross=round(pnl_gross, 2),
            pnl_net=round(pnl_net, 2),
            entry_ts=entry_ts,
            exit_ts=exit_ts,
            exit_reason=exit_reason,
            point_value=self._point_value,
            features_json=features_json,
            realized_R=r_multiple,
            initial_risk_dollars=round(initial_risk_dollars, 2),
            filter_states=filter_json,
        ))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_trading_days(
        df_1m: pd.DataFrame,
        start: Optional[str],
        end: Optional[str],
    ) -> pd.DatetimeIndex:
        dates = pd.DatetimeIndex(sorted(set(df_1m.index.normalize())))
        if start:
            dates = dates[dates >= pd.Timestamp(start, tz="America/New_York")]
        if end:
            dates = dates[dates <= pd.Timestamp(end, tz="America/New_York")]
        return dates

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_metrics(results: pd.DataFrame) -> dict:
        """Compute all key performance metrics from the trade results DataFrame."""
        if results.empty:
            return {"error": "No trades"}

        n = len(results)
        wins = results[results["pnl_net"] > 0]
        losses = results[results["pnl_net"] <= 0]

        gross_profit = wins["pnl_net"].sum()
        gross_loss = abs(losses["pnl_net"].sum())

        # Equity curve
        results = results.sort_values("entry_ts").copy()
        results["cumulative_pnl"] = results["pnl_net"].cumsum()
        peak = results["cumulative_pnl"].cummax()
        drawdown = results["cumulative_pnl"] - peak
        max_drawdown = drawdown.min()

        # Daily aggregation
        daily = results.groupby("date")["pnl_net"].sum()
        sharpe = (daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0

        # ── R-multiple stats (if realized_R column present) ─────────────────
        r_stats: dict = {}
        if "realized_R" in results.columns and results["realized_R"].notna().any():
            r_col = results["realized_R"]
            r_wins  = r_col[r_col > 0]
            r_losses = r_col[r_col <= 0]
            r_stats = {
                "mean_realized_R": round(float(r_col.mean()), 4),
                "median_realized_R": round(float(r_col.median()), 4),
                "avg_R_win": round(float(r_wins.mean()), 4) if len(r_wins) else 0.0,
                "avg_R_loss": round(float(r_losses.mean()), 4) if len(r_losses) else 0.0,
                "expectancy_R": round(float(r_col.mean()), 4),
                "win_rate_by_R": round((r_col > 0).mean() * 100, 1),
            }

        metrics = {
            "total_trades": n,
            "win_rate": round(len(wins) / n * 100, 1),
            "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf"),
            "total_pnl_net": round(results["pnl_net"].sum(), 2),
            "total_pnl_gross": round(results["pnl_gross"].sum(), 2),
            "total_commission": round(results["commission_dollars"].sum(), 2),
            "total_slippage": round(results["slippage_dollars"].sum(), 2),
            "avg_win": round(wins["pnl_net"].mean(), 2) if len(wins) else 0,
            "avg_loss": round(losses["pnl_net"].mean(), 2) if len(losses) else 0,
            "expectancy_per_trade": round(results["pnl_net"].mean(), 2),
            "max_drawdown": round(max_drawdown, 2),
            "sharpe_ratio": round(sharpe, 3),
            "avg_rr_achieved": round(
                (results["pnl_points"] / results.apply(
                    lambda r: abs(r["entry_price"] - r["stop_price"]), axis=1
                )).mean(), 2
            ),
            "trades_per_day": round(
                n / results["date"].nunique(), 2
            ),
            "exit_tp_pct": round(
                (results["exit_reason"] == "tp").mean() * 100, 1
            ),
            "exit_sl_pct": round(
                (results["exit_reason"] == "sl").mean() * 100, 1
            ),
            "exit_eod_pct": round(
                (results["exit_reason"] == "eod").mean() * 100, 1
            ),
        }
        metrics.update(r_stats)
        return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run ChargedUp Profits Bot backtest")
    parser.add_argument("--symbol", default="ES", help="Instrument symbol (ES, NQ, MNQ)")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--save", action="store_true", help="Save trade log to CSV")
    args = parser.parse_args()

    harness = BacktestHarness()
    df_1m, df_15m = harness.load_data(args.symbol)
    results = harness.run(df_1m, df_15m, args.start, args.end)

    if results.empty:
        logger.warning("No trades generated. Check data files and date range.")
        return

    metrics = BacktestHarness.compute_metrics(results)

    print("\n" + "=" * 55)
    print(f"  BACKTEST RESULTS -- {args.symbol}  {args.start or '(all)'} to {args.end or '(all)'}")
    print("=" * 55)
    for k, v in metrics.items():
        print(f"  {k:<30} {v}")
    print("=" * 55 + "\n")

    if args.save:
        out = ROOT / "data" / f"backtest_{args.symbol}_{args.start}_{args.end}.csv"
        results.to_csv(out, index=False)
        logger.info("Saved trade log to {}", out)


if __name__ == "__main__":
    main()
