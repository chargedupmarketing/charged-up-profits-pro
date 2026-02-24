"""
setup_detector.py

Implements the three ChargedUp Profits Bot setups with precise, deterministic rules.
All discretionary language from the original strategy is resolved here:

  DISCRETIONARY TERM         →  CODED DEFINITION
  ─────────────────────────────────────────────────────────────────────
  "break"                    →  candle CLOSE >= 5pts beyond midpoint
  "retest"                   →  price touches within 1.5pts of midpoint
  "rejection candle"         →  1+ candles fail to close through level
  "failure to close above"   →  2+ consecutive 5-min closes below level
  "strength / weakness"      →  reclaim candle: close back inside level
  "trend"                    →  price vs 20-period EMA on 15-min chart
  "volume confirmation"      →  current 1-min volume > 20-bar median
  "regime"                   →  ATR(14) vs 20-bar rolling ATR average

Each setup returns a Signal or None.

Phase 2 enhancements:
  - RegimeFilter: ATR vol state + day-of-week filter
  - Break & Retest: candle body quality + tighter staleness window
  - REJECTION: relaxed 5-min fails_required (fires more often)
  - BOUNCE: relaxed reclaim tolerance (fires more often)
"""

from __future__ import annotations

import copy
import datetime
import functools
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from src.level_builder import AnchorCandle, DayLevels, PriceLevel
from src.news_filter import NewsFilter


@functools.lru_cache(maxsize=4)
def _load_settings(path: str = "config/settings.yaml") -> dict:
    """Load and cache YAML settings — parsed once, not once per bar."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class SetupType(str, Enum):
    BREAK_RETEST = "BREAK_RETEST"
    REJECTION = "REJECTION"
    BOUNCE = "BOUNCE"
    SWEEP_REVERSE = "SWEEP_REVERSE"


@dataclass
class Signal:
    """A trade signal produced by one of the three setups."""
    setup_type: SetupType
    direction: Direction
    entry_price: float
    stop_price: float
    target_price: float
    formed_at: pd.Timestamp
    level_ref: Optional[PriceLevel]   # Which level triggered this signal

    @property
    def stop_distance(self) -> float:
        if self.direction == Direction.LONG:
            return self.entry_price - self.stop_price
        return self.stop_price - self.entry_price

    @property
    def target_distance(self) -> float:
        if self.direction == Direction.LONG:
            return self.target_price - self.entry_price
        return self.entry_price - self.target_price

    @property
    def reward_risk(self) -> float:
        if self.stop_distance <= 0:
            return 0.0
        return round(self.target_distance / self.stop_distance, 2)

    def is_valid_rr(self, min_rr: float = 3.0) -> bool:
        return self.reward_risk >= min_rr

    def __str__(self) -> str:
        return (
            f"Signal({self.setup_type.value} {self.direction.value} "
            f"entry={self.entry_price:.2f} stop={self.stop_price:.2f} "
            f"target={self.target_price:.2f} R:R={self.reward_risk:.1f})"
        )


# ---------------------------------------------------------------------------
# Regime Filter (Phase 2a)
# ---------------------------------------------------------------------------

class RegimeFilter:
    """
    Classifies the current market regime before allowing a trade.

    Conditions that block trading:
      - High-vol regime: ATR(14) on 15m > atr_high_vol_multiplier × rolling avg ATR
        → signals become unreliable; too many stop-outs
      - Banned day-of-week: configurable list (e.g. skip Mondays/Fridays)

    Returns regime state: 'normal', 'high_vol', 'banned_day'
    """

    def __init__(self, cfg: dict) -> None:
        regime_cfg = cfg.get("regime_filter", {})
        self._enabled = regime_cfg.get("enabled", True)
        self._atr_high_vol_mult = regime_cfg.get("atr_high_vol_multiplier", 1.8)
        self._atr_period = regime_cfg.get("atr_period", 14)
        self._atr_rolling_period = regime_cfg.get("atr_rolling_period", 20)
        # Day-of-week: 0=Monday … 4=Friday; default: no day blocked (empty list)
        self._banned_dow: list[int] = regime_cfg.get("banned_days_of_week", [])

    def classify(self, df_15m: pd.DataFrame, current_ts: pd.Timestamp) -> str:
        """Returns 'normal', 'high_vol', or 'banned_day'.

        Fast path: if df_15m contains pre-computed '_atr14' and '_atr14_roll20'
        columns (added by the backtest harness), use them directly — O(1) iloc
        instead of recomputing ATR from scratch on the entire DataFrame every bar.
        """
        if not self._enabled:
            return "normal"

        # Day-of-week check
        if hasattr(current_ts, "dayofweek") and current_ts.dayofweek in self._banned_dow:
            return "banned_day"

        # ── Fast path: use pre-computed cached columns ──────────────────────
        if "_atr14" in df_15m.columns and "_atr14_roll20" in df_15m.columns:
            if len(df_15m) < 1:
                return "normal"
            current_atr    = df_15m["_atr14"].iloc[-1]
            rolling_avg_atr = df_15m["_atr14_roll20"].iloc[-1]
            if pd.isna(current_atr) or pd.isna(rolling_avg_atr):
                return "normal"
            current_atr    = float(current_atr)
            rolling_avg_atr = float(rolling_avg_atr)

        # ── Slow path: compute on-the-fly (live trading / tests without cache) ─
        else:
            if len(df_15m) < self._atr_period + self._atr_rolling_period:
                return "normal"
            hi = df_15m["high"]
            lo = df_15m["low"]
            cl = df_15m["close"].shift(1)
            tr = pd.concat([hi - lo, (hi - cl).abs(), (lo - cl).abs()], axis=1).max(axis=1)
            atr_series = tr.rolling(self._atr_period).mean().dropna()
            if len(atr_series) < self._atr_rolling_period:
                return "normal"
            current_atr    = float(atr_series.iloc[-1])
            rolling_avg_atr = float(atr_series.tail(self._atr_rolling_period).mean())

        if rolling_avg_atr > 0 and current_atr > self._atr_high_vol_mult * rolling_avg_atr:
            return "high_vol"
        return "normal"

    def allows_trade(self, df_15m: pd.DataFrame, current_ts: pd.Timestamp) -> bool:
        state = self.classify(df_15m, current_ts)
        if state != "normal":
            logger.debug("RegimeFilter blocked trade: regime={}", state)
            return False
        return True


# ---------------------------------------------------------------------------
# Trend Filter
# ---------------------------------------------------------------------------

class TrendFilter:
    """
    Trend direction based on the 20-period EMA of 15-min closing prices.
    BULLISH  = current price > EMA → only take LONG setups
    BEARISH  = current price < EMA → only take SHORT setups
    NEUTRAL  = EMA period not available yet

    Phase 2 enhancement: also checks 50-EMA for macro alignment.
    """

    def __init__(self, ema_period: int = 20, macro_ema_period: int = 50) -> None:
        self._period = ema_period
        self._macro_period = macro_ema_period

    def get_trend(self, df_15m: pd.DataFrame, current_price: float) -> str:
        """Returns 'bullish', 'bearish', or 'neutral'.

        Fast path: uses pre-computed '_ema20' column when available (added by the
        backtest harness before the main loop to avoid recomputing EWM every bar).
        """
        # ── Fast path ───────────────────────────────────────────────────────
        if "_ema20" in df_15m.columns:
            val = df_15m["_ema20"].iloc[-1]
            if pd.isna(val):
                return "neutral"
            ema = float(val)
        # ── Slow path ───────────────────────────────────────────────────────
        elif len(df_15m) >= self._period:
            ema = float(df_15m["close"].ewm(span=self._period, adjust=False).mean().iloc[-1])
        else:
            return "neutral"

        if current_price > ema:
            return "bullish"
        elif current_price < ema:
            return "bearish"
        return "neutral"

    def get_macro_trend(self, df_15m: pd.DataFrame, current_price: float) -> str:
        """Returns macro trend using 50-EMA. 'bullish', 'bearish', or 'neutral'.

        Fast path: uses pre-computed '_ema50' column when available.
        """
        # ── Fast path ───────────────────────────────────────────────────────
        if "_ema50" in df_15m.columns:
            val = df_15m["_ema50"].iloc[-1]
            if pd.isna(val):
                return "neutral"
            ema50 = float(val)
        # ── Slow path ───────────────────────────────────────────────────────
        elif len(df_15m) >= self._macro_period:
            ema50 = float(df_15m["close"].ewm(span=self._macro_period, adjust=False).mean().iloc[-1])
        else:
            return "neutral"

        if current_price > ema50:
            return "bullish"
        elif current_price < ema50:
            return "bearish"
        return "neutral"

    def allows_direction(
        self,
        df_15m: pd.DataFrame,
        current_price: float,
        direction: Direction,
        macro_hard_block: bool = False,
    ) -> bool:
        """
        Returns True if the trend filter allows a trade in the given direction.

        Args:
            macro_hard_block: If True, block trades when EMA50 (macro trend) conflicts
                with the trade direction.  Enabled per-symbol via trend_overrides in
                settings.yaml.  NQ/MNQ data shows: winners EMA50 dist=+6.9pt,
                losers EMA50 dist=-3.0pt — a strong directional predictor.
        """
        trend = self.get_trend(df_15m, current_price)
        macro = self.get_macro_trend(df_15m, current_price)

        if trend == "neutral":
            return True  # No filter when insufficient data

        # Micro trend (EMA20) must align
        if direction == Direction.LONG and trend != "bullish":
            return False
        if direction == Direction.SHORT and trend != "bearish":
            return False

        # Macro trend (EMA50) — hard block or soft warning depending on config
        if macro != "neutral":
            if direction == Direction.LONG and macro == "bearish":
                if macro_hard_block:
                    logger.debug("Macro trend hard block: LONG in bearish EMA50 macro")
                    return False
                logger.debug("Macro trend conflict (soft): LONG in bearish macro")
            if direction == Direction.SHORT and macro == "bullish":
                if macro_hard_block:
                    logger.debug("Macro trend hard block: SHORT in bullish EMA50 macro")
                    return False
                logger.debug("Macro trend conflict (soft): SHORT in bullish macro")

        return True


# ---------------------------------------------------------------------------
# Volume filter
# ---------------------------------------------------------------------------

def _volume_confirmed(df_1m: pd.DataFrame, lookback: int = 20, min_ratio: float = 0.8) -> bool:
    """True if the most recent 1-min bar's volume meets the minimum ratio vs rolling median.

    Set to 0.8 (80% of median) — the volume_ratio feature is already fed to the ML model
    which learns the optimal threshold itself.  A hard 1.0 cut was too blunt and reduced
    trade count by ~46% without net benefit once the ML filter is active.
    """
    if len(df_1m) < 2:
        return True  # Can't check; don't filter
    recent_vol = df_1m["volume"].iloc[-1]
    median_vol = df_1m["volume"].iloc[-lookback - 1:-1].median()
    if median_vol <= 0:
        return True
    return recent_vol >= median_vol * min_ratio


# ---------------------------------------------------------------------------
# Stop / target computation
# ---------------------------------------------------------------------------

def _compute_stop_and_target(
    entry: float,
    direction: Direction,
    cfg: dict,
    invalidation_level: Optional[float] = None,
    df_15m: Optional[pd.DataFrame] = None,
    anchor=None,   # Optional[AnchorCandle] — imported at call sites; avoids circular ref
) -> tuple[float, float]:
    """
    Compute stop and target given entry, direction, and config.

    Phase 2d: ATR-based stop sizing.
    Phase 4:  OR-range-based stop sizing (use_or_range_stops=True). When both are
              enabled, OR-range takes precedence over ATR.
    Phase 3:  max_reward_risk_ratio — allow targets up to 4:1 R:R (not just 3:1).

    If an invalidation_level is provided (e.g. swing low for a bounce),
    use that as the stop (overrides ATR/OR-range sizing).
    """
    sl_min = cfg["risk"]["stop_loss_points_min"]
    sl_max = cfg["risk"]["stop_loss_points_max"]
    tp_min = cfg["risk"]["take_profit_points_min"]
    tp_max = cfg["risk"]["take_profit_points_max"]
    min_rr = cfg["risk"]["min_reward_risk_ratio"]
    max_rr = cfg["risk"].get("max_reward_risk_ratio", min_rr)   # Phase 3

    risk_cfg = cfg.get("risk", {})

    # ── Phase 4: OR-range-based stop sizing (takes precedence over ATR) ──────
    use_or_stops = risk_cfg.get("use_or_range_stops", False)
    or_stop_frac = risk_cfg.get("stop_or_fraction", 0.5)
    base_stop: float = sl_min  # fallback

    if use_or_stops and anchor is not None and hasattr(anchor, "range_pts") and anchor.range_pts > 0:
        base_stop = max(sl_min, min(sl_max, anchor.range_pts * or_stop_frac))
    # ── Phase 2d: ATR-based stop sizing (used when OR stops disabled) ─────────
    elif risk_cfg.get("use_atr_stops", False) and df_15m is not None and len(df_15m) >= 15:
        hi = df_15m["high"]
        lo = df_15m["low"]
        cl = df_15m["close"].shift(1)
        tr = pd.concat([hi - lo, (hi - cl).abs(), (lo - cl).abs()], axis=1).max(axis=1)
        atr_val = float(tr.tail(14).mean())
        base_stop = max(sl_min, min(sl_max, atr_val * risk_cfg.get("atr_stop_multiplier", 0.5)))

    if direction == Direction.LONG:
        if invalidation_level is not None:
            stop = invalidation_level - 0.25
            stop_dist = max(sl_min, min(sl_max, entry - stop))
            stop = entry - stop_dist
        else:
            stop = entry - base_stop

        stop_dist = entry - stop
        # Phase 3: allow target up to max_rr × stop_dist, capped by tp_max
        target_dist = max(tp_min, min(tp_max, stop_dist * max_rr))
        # Enforce minimum R:R — if max_rr target is below min_rr, use min_rr
        target_dist = max(target_dist, min(tp_max, stop_dist * min_rr))
        target = entry + target_dist

    else:  # SHORT
        if invalidation_level is not None:
            stop = invalidation_level + 0.25
            stop_dist = max(sl_min, min(sl_max, stop - entry))
            stop = entry + stop_dist
        else:
            stop = entry + base_stop

        stop_dist = stop - entry
        target_dist = max(tp_min, min(tp_max, stop_dist * max_rr))
        target_dist = max(target_dist, min(tp_max, stop_dist * min_rr))
        target = entry - target_dist

    return round(stop, 2), round(target, 2)


# ---------------------------------------------------------------------------
# Candle quality helpers (Phase 2b)
# ---------------------------------------------------------------------------

def _candle_body_ratio(bar: pd.Series) -> float:
    """Returns body size / total range (0–1). 1 = all body, 0 = doji/wick-only."""
    total = float(bar["high"]) - float(bar["low"])
    if total < 0.01:
        return 0.0
    body = abs(float(bar["close"]) - float(bar["open"]))
    return body / total


def _is_strong_candle(bar: pd.Series, min_body_ratio: float = 0.5) -> bool:
    """Returns True if the candle body is at least min_body_ratio of its total range."""
    return _candle_body_ratio(bar) >= min_body_ratio


# ---------------------------------------------------------------------------
# Main SetupDetector
# ---------------------------------------------------------------------------

class SetupDetector:
    """
    Scans incoming bars for the three ChargedUp Profits Bot setups.

    Call `detect(day_levels, df_15m, df_5m, df_1m)` on every new 1-minute bar.
    Returns a Signal if a valid setup is found, otherwise None.

    Phase 2 enhancements:
      - RegimeFilter applied before any setup check
      - BREAK_RETEST: candle body quality check + tighter staleness
      - REJECTION: relaxed to fire on 1-candle fail on 5m OR 2 on 1m
      - BOUNCE: relaxed reclaim tolerance
    """

    def __init__(
        self,
        settings_path: str = "config/settings.yaml",
        symbol: str | None = None,
    ) -> None:
        self._cfg = _load_settings(settings_path)
        self._trend = TrendFilter(
            ema_period=self._cfg["trend_filter"]["ema_period"],
            macro_ema_period=self._cfg["trend_filter"].get("macro_ema_period", 50),
        )
        self._regime = RegimeFilter(self._cfg)
        # Phase 2: News calendar filter
        self._news_filter = NewsFilter(settings=self._cfg)
        self._br_cfg = self._cfg["setups"]["break_retest"]
        self._rej_cfg = self._cfg["setups"]["rejection"]
        self._bounce_cfg = self._cfg["setups"]["bounce"]
        self._sr_cfg = self._cfg["setups"].get("sweep_reverse", {})
        self._tolerance = self._cfg["levels"]["level_tolerance_points"]

        # Macro trend hard block (False by default; overridden per-symbol via trend_overrides)
        self._macro_hard_block: bool = self._cfg.get("trend_filter", {}).get(
            "macro_trend_hard_block", False
        )

        # Per-symbol enabled setups (from symbols.ES.enabled_setups / symbols.NQ.enabled_setups)
        # An empty list means ALL setups are enabled (default).
        _sym = (symbol or self._cfg.get("instrument", {}).get("symbol", "")).upper()
        _sym_cfg = self._cfg.get("symbols", {}).get(_sym, {})
        _raw_enabled = _sym_cfg.get("enabled_setups", [])
        self._enabled_setups: set[str] = (
            {s.upper() for s in _raw_enabled} if _raw_enabled else set()
        )
        if self._enabled_setups:
            from loguru import logger as _log
            _log.info(
                "SetupDetector[{}]: enabled_setups filter = {}",
                _sym, sorted(self._enabled_setups),
            )

        # Cooldown tracking: {(setup_type, level_price_rounded): last_signal_timestamp}
        # Per-level so two Rejections at different levels don't block each other.
        self._last_signal_ts: dict[tuple, Optional[pd.Timestamp]] = {}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def detect(
        self,
        day_levels: DayLevels,
        df_15m: pd.DataFrame,
        df_5m: pd.DataFrame,
        df_1m: pd.DataFrame,
    ) -> list[Signal]:
        """
        Run all enabled setup detectors and return ALL valid signals found.

        Returns a list so the caller (bot runner / backtest harness) can score
        each signal with the ML filter and pick the highest-confidence one.
        Priority ordering (BREAK_RETEST first) is preserved for tie-breaking.

        Phase upgrade: per-level cooldowns mean two Rejection signals at
        different price levels can both fire in the same bar — both are
        returned and the caller selects the best one.
        """
        if day_levels.no_trade_day or day_levels.anchor is None:
            return []

        current_price = float(df_1m["close"].iloc[-1])
        current_ts = df_1m.index[-1]

        # Regime check (Phase 2a): block high-vol or banned-DOW days
        if self._cfg.get("regime_filter", {}).get("enabled", True):
            if not self._regime.allows_trade(df_15m, current_ts):
                return []

        # News filter (Phase 2): block FOMC / CPI / NFP days
        regime_cfg = self._cfg.get("regime_filter", {})
        if regime_cfg.get("news_filter_enabled", True):
            if self._news_filter.is_news_day(current_ts):
                return []

        raw_signals: list[Signal] = []

        def _setup_enabled(setup_name: str) -> bool:
            """Returns False if the setup is excluded by the per-symbol enabled_setups list."""
            if self._enabled_setups and setup_name.upper() not in self._enabled_setups:
                return False
            return True

        if _setup_enabled("BREAK_RETEST") and self._br_cfg.get("enabled", True):
            sig = self._detect_break_retest(day_levels, df_15m, df_1m, current_price, current_ts)
            if sig:
                raw_signals.append(sig)

        if _setup_enabled("REJECTION") and self._rej_cfg.get("enabled", True):
            sigs = self._detect_rejection_all(day_levels, df_15m, df_5m, df_1m, current_price, current_ts)
            raw_signals.extend(sigs)

        if _setup_enabled("BOUNCE") and self._bounce_cfg.get("enabled", True):
            sigs = self._detect_bounce_all(day_levels, df_15m, df_1m, current_price, current_ts)
            raw_signals.extend(sigs)

        if _setup_enabled("SWEEP_REVERSE") and self._sr_cfg.get("enabled", True):
            sigs = self._detect_sweep_reverse_all(day_levels, df_15m, df_1m, current_price, current_ts)
            raw_signals.extend(sigs)

        # Volume filter is a session-wide gate — applies to all signals equally
        vol_ok = _volume_confirmed(df_1m, min_ratio=0.8)

        valid: list[Signal] = []
        for sig in raw_signals:
            # Per-level cooldown: key = (setup_type, level_price_rounded)
            level_price = sig.level_ref.price if sig.level_ref else sig.entry_price
            cooldown_key = (sig.setup_type.value, round(level_price, 1))
            # Use the setup's own cooldown config
            if sig.setup_type == SetupType.BREAK_RETEST:
                cooldown_mins = self._br_cfg.get("signal_cooldown_minutes", 10)
            elif sig.setup_type == SetupType.REJECTION:
                cooldown_mins = self._rej_cfg.get("signal_cooldown_minutes", 10)
            elif sig.setup_type == SetupType.BOUNCE:
                cooldown_mins = self._bounce_cfg.get("signal_cooldown_minutes", 10)
            else:
                cooldown_mins = self._sr_cfg.get("signal_cooldown_minutes", 10)

            last_ts = self._last_signal_ts.get(cooldown_key)
            if last_ts is not None:
                elapsed = (current_ts - last_ts).total_seconds() / 60
                if elapsed < cooldown_mins:
                    logger.debug(
                        "Signal {} @ {:.1f} in cooldown ({:.1f}/{} mins elapsed)",
                        sig.setup_type.value, level_price, elapsed, cooldown_mins
                    )
                    continue

            # Apply trend filter (with optional per-symbol macro hard block)
            if self._cfg["trend_filter"]["enabled"]:
                if not self._trend.allows_direction(
                    df_15m, current_price, sig.direction,
                    macro_hard_block=self._macro_hard_block,
                ):
                    logger.debug("Signal {} filtered by trend filter", sig)
                    continue

            # Apply volume filter (soft 0.8× threshold)
            if not vol_ok:
                logger.debug("Signal {} filtered by low volume", sig)
                continue

            # Enforce minimum R:R
            if not sig.is_valid_rr(self._cfg["risk"]["min_reward_risk_ratio"]):
                logger.debug("Signal {} has R:R={:.1f} below minimum", sig, sig.reward_risk)
                continue

            # Record cooldown for this level
            self._last_signal_ts[cooldown_key] = current_ts
            logger.info("Valid signal found: {}", sig)
            valid.append(sig)

        return valid

    def configure_for_symbol(self, symbol: str) -> None:
        """
        Apply per-symbol overrides for setup thresholds, level tolerance, risk
        clamping values, and the regime filter multiplier.

        This is the critical fix for NQ/MNQ:
          - ``min_break_points`` 5 → 20  (5pt is noise on NQ)
          - ``retest_tolerance_points`` 1.5 → 6
          - ``stop_loss_points_max`` 10 → 40  (ATR stop was clamped to 10pt — constant stop-outs)
          - ``level_tolerance_points`` 0.5 → 2.5  (levels falsely "tested" on NQ wiggles)
          - ``atr_high_vol_multiplier`` 2.2 → 2.8  (NQ naturally spikier)

        Deep-copies ``self._cfg`` before mutating so the LRU-cached shared dict
        is never modified.
        """
        sym_cfg = self._cfg.get("symbols", {}).get(symbol, {})
        if not sym_cfg:
            return

        # ── Setup parameter overrides ─────────────────────────────────────────
        setups_overrides = sym_cfg.get("setups_overrides", {})
        if "break_retest" in setups_overrides:
            self._br_cfg = {**self._br_cfg, **setups_overrides["break_retest"]}
        if "rejection" in setups_overrides:
            self._rej_cfg = {**self._rej_cfg, **setups_overrides["rejection"]}
        if "bounce" in setups_overrides:
            self._bounce_cfg = {**self._bounce_cfg, **setups_overrides["bounce"]}
        if "sweep_reverse" in setups_overrides:
            self._sr_cfg = {**self._sr_cfg, **setups_overrides["sweep_reverse"]}

        # ── Level tolerance (affects tap detection for rejection/bounce) ──────
        levels_overrides = sym_cfg.get("levels_overrides", {})
        if "level_tolerance_points" in levels_overrides:
            self._tolerance = float(levels_overrides["level_tolerance_points"])

        # ── Risk overrides (stop/target clamping — THE most critical NQ fix) ──
        # Without this, ATR-based stop of ~25-50pt gets clamped to 10pt max → constant SL hits
        risk_overrides = sym_cfg.get("risk_overrides", {})
        if risk_overrides:
            self._cfg = copy.deepcopy(self._cfg)
            self._cfg["risk"].update(risk_overrides)

        # ── Regime filter overrides ───────────────────────────────────────────
        regime_overrides = sym_cfg.get("regime_overrides", {})
        if "atr_high_vol_multiplier" in regime_overrides:
            self._regime._atr_high_vol_mult = float(
                regime_overrides["atr_high_vol_multiplier"]
            )
        if "banned_days_of_week" in regime_overrides:
            self._regime._banned_dow = list(regime_overrides["banned_days_of_week"])

        # ── Trend filter overrides ────────────────────────────────────────────
        # NQ/MNQ: EMA50 alignment is the strongest win predictor — enable hard block
        trend_overrides = sym_cfg.get("trend_overrides", {})
        if "macro_trend_hard_block" in trend_overrides:
            self._macro_hard_block = bool(trend_overrides["macro_trend_hard_block"])

        logger.info(
            "Symbol {}: tolerance={:.1f}pt  sl=[{},{}]pt  tp=[{},{}]pt  "
            "break_min={}pt  retest_tol={}pt  regime_mult={:.1f}  "
            "macro_hard_block={}  banned_dow={}",
            symbol,
            self._tolerance,
            self._cfg["risk"].get("stop_loss_points_min", "?"),
            self._cfg["risk"].get("stop_loss_points_max", "?"),
            self._cfg["risk"].get("take_profit_points_min", "?"),
            self._cfg["risk"].get("take_profit_points_max", "?"),
            self._br_cfg.get("min_break_points", "?"),
            self._br_cfg.get("retest_tolerance_points", "?"),
            self._regime._atr_high_vol_mult,
            self._macro_hard_block,
            self._regime._banned_dow,
        )

    def reset_day(self) -> None:
        """Call at the start of each new trading day to clear all cooldown timers."""
        self._last_signal_ts.clear()

    # ------------------------------------------------------------------
    # Setup 1: Break & Retest (Phase 2b + Phase 3 enhanced)
    # ------------------------------------------------------------------

    def _detect_break_retest(
        self,
        day_levels: DayLevels,
        df_15m: pd.DataFrame,
        df_1m: pd.DataFrame,
        current_price: float,
        current_ts: pd.Timestamp,
    ) -> Optional[Signal]:
        """
        Rules (Phase 2b + Phase 3 enhancements):

        PRIMARY path (use_or_extremes=True — research-spec):
          A. OR_HIGH break + retest (LONG):
             1. A bar within max_bars_since_break CLOSES above anchor.high + or_extreme_break_buffer
             2. Price is now back within retest_tolerance of anchor.high
             3. Current bar closes ABOVE anchor.high (confirming buyers hold the break)
             Entry: at OR_HIGH level

          B. OR_LOW break + retest (SHORT):
             1. A bar within max_bars_since_break CLOSES below anchor.low - or_extreme_break_buffer
             2. Price is now back within retest_tolerance of anchor.low
             3. Current bar closes BELOW anchor.low (confirming sellers hold the break)
             Entry: at OR_LOW level

        SECONDARY path (OR_MID — original rule, always active as fallback):
          1. A bar CLOSED >= min_break_points beyond OR_MID with min body ratio.
          2. Price is now back within retest_tolerance of OR_MID.
          3. Current bar closes on the break side of OR_MID.
          Entry: at OR_MID level

        OR_HIGH/OR_LOW signals are returned first when both paths trigger on the same bar.
        Signal is stale if break happened > max_bars_since_break bars ago.
        """
        anchor = day_levels.anchor
        mid = anchor.mid
        or_high = anchor.high
        or_low = anchor.low
        min_break = self._br_cfg["min_break_points"]
        retest_tol = self._br_cfg["retest_tolerance_points"]
        max_stale = self._br_cfg.get("max_bars_since_break", 12)
        min_body_ratio = self._br_cfg.get("min_break_body_ratio", 0.5)
        use_or_extremes = self._br_cfg.get("use_or_extremes", True)
        or_extreme_buf = self._br_cfg.get("or_extreme_break_buffer", 1.0)

        if len(df_1m) < 4:
            return None

        recent = df_1m.tail(max_stale + 5)
        closes = recent["close"].values
        opens = recent["open"].values
        highs = recent["high"].values
        lows = recent["low"].values
        n = len(closes)
        current_close = float(df_1m["close"].iloc[-1])
        prev_close = float(df_1m["close"].iloc[-2])

        # ── PRIMARY: OR_HIGH / OR_LOW break + retest (research-spec) ──────────
        if use_or_extremes:
            # Look for a break above OR_HIGH
            broke_above_or_high = None
            broke_below_or_low = None
            for i in range(n - 1, max(n - max_stale - 1, -1), -1):
                total_range = highs[i] - lows[i]
                body = abs(closes[i] - opens[i])
                body_ratio = body / total_range if total_range > 0.01 else 0.0
                if broke_above_or_high is None and closes[i] >= or_high + or_extreme_buf:
                    if body_ratio >= min_body_ratio:
                        broke_above_or_high = i
                if broke_below_or_low is None and closes[i] <= or_low - or_extreme_buf:
                    if body_ratio >= min_body_ratio:
                        broke_below_or_low = i

            # Check OR_HIGH break + retest → LONG
            if broke_above_or_high is not None:
                near_or_high = abs(current_price - or_high) <= retest_tol
                if near_or_high and current_close >= or_high and current_close >= prev_close:
                    entry = or_high
                    stop, target = _compute_stop_and_target(
                        entry, Direction.LONG, self._cfg, df_15m=df_15m,
                        anchor=anchor,
                    )
                    return Signal(
                        setup_type=SetupType.BREAK_RETEST,
                        direction=Direction.LONG,
                        entry_price=round(entry, 2),
                        stop_price=stop,
                        target_price=target,
                        formed_at=current_ts,
                        level_ref=PriceLevel(
                            price=or_high, direction="high",
                            formed_at=anchor.timestamp, session_origin="or_range",
                        ),
                    )

            # Check OR_LOW break + retest → SHORT
            if broke_below_or_low is not None:
                near_or_low = abs(current_price - or_low) <= retest_tol
                if near_or_low and current_close <= or_low and current_close <= prev_close:
                    entry = or_low
                    stop, target = _compute_stop_and_target(
                        entry, Direction.SHORT, self._cfg, df_15m=df_15m,
                        anchor=anchor,
                    )
                    return Signal(
                        setup_type=SetupType.BREAK_RETEST,
                        direction=Direction.SHORT,
                        entry_price=round(entry, 2),
                        stop_price=stop,
                        target_price=target,
                        formed_at=current_ts,
                        level_ref=PriceLevel(
                            price=or_low, direction="low",
                            formed_at=anchor.timestamp, session_origin="or_range",
                        ),
                    )

        # ── SECONDARY: OR_MID break + retest (original rule) ──────────────────
        broke_up_bar = None
        broke_dn_bar = None
        for i in range(n - 1, max(n - max_stale - 1, -1), -1):
            total_range = highs[i] - lows[i]
            body = abs(closes[i] - opens[i])
            body_ratio = body / total_range if total_range > 0.01 else 0.0

            if closes[i] >= mid + min_break and broke_up_bar is None:
                if body_ratio >= min_body_ratio:
                    broke_up_bar = i
            if closes[i] <= mid - min_break and broke_dn_bar is None:
                if body_ratio >= min_body_ratio:
                    broke_dn_bar = i

        if broke_up_bar is None and broke_dn_bar is None:
            return None

        near_mid = abs(current_price - mid) <= retest_tol
        if not near_mid:
            return None

        use_up = broke_up_bar is not None
        use_dn = broke_dn_bar is not None
        if use_up and use_dn:
            use_up = broke_up_bar > broke_dn_bar
            use_dn = not use_up

        if use_dn:
            if current_close >= mid:
                return None
            if current_close >= prev_close:
                return None
            direction = Direction.SHORT
        else:
            if current_close <= mid:
                return None
            if current_close <= prev_close:
                return None
            direction = Direction.LONG

        entry = mid
        stop, target = _compute_stop_and_target(
            entry, direction, self._cfg, df_15m=df_15m, anchor=anchor,
        )

        return Signal(
            setup_type=SetupType.BREAK_RETEST,
            direction=direction,
            entry_price=round(entry, 2),
            stop_price=stop,
            target_price=target,
            formed_at=current_ts,
            level_ref=None,
        )

    # ------------------------------------------------------------------
    # Setup 2: Rejection off untested high/low (Phase 2c enhanced)
    # Returns ALL valid rejection signals across all active levels.
    # ------------------------------------------------------------------

    def _detect_rejection_all(
        self,
        day_levels: DayLevels,
        df_15m: pd.DataFrame,
        df_5m: pd.DataFrame,
        df_1m: pd.DataFrame,
        current_price: float,
        current_ts: pd.Timestamp,
    ) -> list[Signal]:
        """
        Rules (Phase 2c enhancements — relaxed to fire more often):
          1. Price taps an untested high/low (within tolerance * 2)
          2. Either:
             a) The last fails_required_5m consecutive 5-min candles fail to close
                fully beyond the level (original rule, now 1 candle default), OR
             b) The last fails_required_1m consecutive 1-min candles fail to close
                beyond the level (new secondary rule)
          3. Entry on current bar in opposite direction

        SHORT: taps untested high, fails to close above → short
        LONG:  taps untested low, fails to close below → long

        Returns a list — all levels that qualify simultaneously are returned.
        """
        fails_5m = self._rej_cfg.get("fails_required", 2)
        fails_1m_req = self._rej_cfg.get("fails_required_1m", 3)
        results: list[Signal] = []

        # Check against each active untested HIGH (potential short)
        for lvl in day_levels.active_highs():
            if abs(current_price - lvl.price) <= self._tolerance * 2:
                passed = False

                check_5m = df_5m.tail(fails_5m + 2)
                recent_5m_closes = check_5m["close"].tail(fails_5m)
                if len(recent_5m_closes) >= fails_5m:
                    if (recent_5m_closes < lvl.price - self._tolerance).all():
                        passed = True

                if not passed:
                    check_1m = df_1m.tail(fails_1m_req + 2)
                    recent_1m_closes = check_1m["close"].tail(fails_1m_req)
                    if len(recent_1m_closes) >= fails_1m_req:
                        if (recent_1m_closes < lvl.price - self._tolerance).all():
                            passed = True

                if passed:
                    entry = lvl.price - self._tolerance
                    stop, target = _compute_stop_and_target(
                        entry, Direction.SHORT, self._cfg,
                        invalidation_level=lvl.price,
                        df_15m=df_15m,
                        anchor=day_levels.anchor,
                    )
                    results.append(Signal(
                        setup_type=SetupType.REJECTION,
                        direction=Direction.SHORT,
                        entry_price=round(entry, 2),
                        stop_price=stop,
                        target_price=target,
                        formed_at=current_ts,
                        level_ref=lvl,
                    ))

        # Check against each active untested LOW (potential long)
        for lvl in day_levels.active_lows():
            if abs(current_price - lvl.price) <= self._tolerance * 2:
                passed = False

                check_5m = df_5m.tail(fails_5m + 2)
                recent_5m_closes = check_5m["close"].tail(fails_5m)
                if len(recent_5m_closes) >= fails_5m:
                    if (recent_5m_closes > lvl.price + self._tolerance).all():
                        passed = True

                if not passed:
                    check_1m = df_1m.tail(fails_1m_req + 2)
                    recent_1m_closes = check_1m["close"].tail(fails_1m_req)
                    if len(recent_1m_closes) >= fails_1m_req:
                        if (recent_1m_closes > lvl.price + self._tolerance).all():
                            passed = True

                if passed:
                    entry = lvl.price + self._tolerance
                    stop, target = _compute_stop_and_target(
                        entry, Direction.LONG, self._cfg,
                        invalidation_level=lvl.price,
                        df_15m=df_15m,
                        anchor=day_levels.anchor,
                    )
                    results.append(Signal(
                        setup_type=SetupType.REJECTION,
                        direction=Direction.LONG,
                        entry_price=round(entry, 2),
                        stop_price=stop,
                        target_price=target,
                        formed_at=current_ts,
                        level_ref=lvl,
                    ))

        return results

    # ------------------------------------------------------------------
    # Setup 3: Bounce off untested low/high — returns all valid levels
    # ------------------------------------------------------------------

    def _detect_bounce_all(
        self,
        day_levels: DayLevels,
        df_15m: pd.DataFrame,
        df_1m: pd.DataFrame,
        current_price: float,
        current_ts: pd.Timestamp,
    ) -> list[Signal]:
        """
        Rules (Phase 2c enhancements — relaxed reclaim tolerance):
          1. Price taps an untested low/high (within tolerance * 1.5, relaxed from 1.0)
          2. A 1-min candle CLOSES back inside/above the level (reclaim)
             — tolerance for reclaim is now tolerance * 0.5 (was tolerance, too strict)
          3. Entry on that reclaim candle; stop behind the swing low/high

        LONG:  taps untested low → 1m close back above level → long
        SHORT: taps untested high → 1m close back below level → short

        Returns a list — all qualifying levels are returned.
        """
        reclaim_candles = self._bounce_cfg.get("reclaim_candles", 1)
        touch_tolerance_mult = self._bounce_cfg.get("touch_tolerance_mult", 1.5)
        reclaim_tolerance_mult = self._bounce_cfg.get("reclaim_tolerance_mult", 0.5)
        recent_1m = df_1m.tail(reclaim_candles + 3)
        results: list[Signal] = []

        # Bounce off LOW → LONG
        for lvl in day_levels.active_lows():
            touch_tol = self._tolerance * touch_tolerance_mult
            touched_level = (recent_1m["low"].min() <= lvl.price + touch_tol)
            if touched_level:
                reclaim_price = lvl.price + self._tolerance * reclaim_tolerance_mult
                recent_closes = recent_1m["close"].tail(reclaim_candles)
                if (recent_closes > reclaim_price).all():
                    swing_low = float(recent_1m["low"].min())
                    entry = float(recent_1m["high"].iloc[-1])
                    stop, target = _compute_stop_and_target(
                        entry, Direction.LONG, self._cfg,
                        invalidation_level=swing_low,
                        df_15m=df_15m,
                        anchor=day_levels.anchor,
                    )
                    results.append(Signal(
                        setup_type=SetupType.BOUNCE,
                        direction=Direction.LONG,
                        entry_price=round(entry, 2),
                        stop_price=stop,
                        target_price=target,
                        formed_at=current_ts,
                        level_ref=lvl,
                    ))

        # Bounce (rejection) off HIGH → SHORT
        for lvl in day_levels.active_highs():
            touch_tol = self._tolerance * touch_tolerance_mult
            touched_level = (recent_1m["high"].max() >= lvl.price - touch_tol)
            if touched_level:
                reclaim_price = lvl.price - self._tolerance * reclaim_tolerance_mult
                recent_closes = recent_1m["close"].tail(reclaim_candles)
                if (recent_closes < reclaim_price).all():
                    swing_high = float(recent_1m["high"].max())
                    entry = float(recent_1m["low"].iloc[-1])
                    stop, target = _compute_stop_and_target(
                        entry, Direction.SHORT, self._cfg,
                        invalidation_level=swing_high,
                        df_15m=df_15m,
                        anchor=day_levels.anchor,
                    )
                    results.append(Signal(
                        setup_type=SetupType.BOUNCE,
                        direction=Direction.SHORT,
                        entry_price=round(entry, 2),
                        stop_price=stop,
                        target_price=target,
                        formed_at=current_ts,
                        level_ref=lvl,
                    ))

        return results

    # ------------------------------------------------------------------
    # Setup 4: Sweep & Reverse — liquidity grab + failed breakout reversal
    # ------------------------------------------------------------------

    def _detect_sweep_reverse_all(
        self,
        day_levels: DayLevels,
        df_15m: pd.DataFrame,
        df_1m: pd.DataFrame,
        current_price: float,
        current_ts: pd.Timestamp,
    ) -> list[Signal]:
        """
        A Sweep & Reverse fires when the most recent 1m bar wicks THROUGH a key
        level (liquidity grab / stop hunt) but CLOSES BACK inside the level,
        with a strong-bodied reversal candle.

        SHORT sweep (off highs):
          - bar.high >= level_price + tolerance   (wick pokes above level)
          - bar.close < level_price               (closes back below level)
          - body_ratio >= min_body_ratio           (strong reversal body)
          Entry: bar.close.  Stop: bar.high + 0.25pt buffer.

        LONG sweep (off lows):
          - bar.low  <= level_price - tolerance   (wick pokes below level)
          - bar.close > level_price               (closes back above level)
          - body_ratio >= min_body_ratio
          Entry: bar.close.  Stop: bar.low - 0.25pt buffer.

        Checks: OR_HIGH, OR_LOW, untested highs/lows, prev_day levels.
        Returns all qualifying level sweeps found on the current bar.
        """
        min_body = self._sr_cfg.get("min_body_ratio", 0.45)
        results: list[Signal] = []

        if len(df_1m) < 1:
            return results

        last_bar = df_1m.iloc[-1]
        bar_high = float(last_bar["high"])
        bar_low  = float(last_bar["low"])
        bar_close = float(last_bar["close"])
        bar_open  = float(last_bar["open"])
        bar_range = bar_high - bar_low
        body = abs(bar_close - bar_open)
        body_ratio = body / bar_range if bar_range > 0.01 else 0.0

        if body_ratio < min_body:
            return results  # Weak candle — not a clean reversal

        anchor = day_levels.anchor

        # Collect all candidate levels: OR extremes + untested + prev_day
        candidate_highs: list[PriceLevel] = list(day_levels.active_highs())
        candidate_lows: list[PriceLevel]  = list(day_levels.active_lows())

        if anchor is not None:
            candidate_highs.append(PriceLevel(
                price=anchor.high, direction="high",
                formed_at=anchor.timestamp, session_origin="or_range",
            ))
            candidate_lows.append(PriceLevel(
                price=anchor.low, direction="low",
                formed_at=anchor.timestamp, session_origin="or_range",
            ))

        # SHORT sweep: wick above high level, close back below
        for lvl in candidate_highs:
            if (bar_high >= lvl.price + self._tolerance
                    and bar_close < lvl.price):
                entry = bar_close
                stop_price = bar_high + 0.25
                stop, target = _compute_stop_and_target(
                    entry, Direction.SHORT, self._cfg,
                    invalidation_level=bar_high,
                    df_15m=df_15m,
                    anchor=anchor,
                )
                results.append(Signal(
                    setup_type=SetupType.SWEEP_REVERSE,
                    direction=Direction.SHORT,
                    entry_price=round(entry, 2),
                    stop_price=stop,
                    target_price=target,
                    formed_at=current_ts,
                    level_ref=lvl,
                ))

        # LONG sweep: wick below low level, close back above
        for lvl in candidate_lows:
            if (bar_low <= lvl.price - self._tolerance
                    and bar_close > lvl.price):
                entry = bar_close
                stop, target = _compute_stop_and_target(
                    entry, Direction.LONG, self._cfg,
                    invalidation_level=bar_low,
                    df_15m=df_15m,
                    anchor=anchor,
                )
                results.append(Signal(
                    setup_type=SetupType.SWEEP_REVERSE,
                    direction=Direction.LONG,
                    entry_price=round(entry, 2),
                    stop_price=stop,
                    target_price=target,
                    formed_at=current_ts,
                    level_ref=lvl,
                ))

        return results
