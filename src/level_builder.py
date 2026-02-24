"""
level_builder.py

Builds all key price levels for a trading day before the execution window opens:
  1. 8am anchor candle: high, low, midpoint
  2. Untested session highs/lows from the prior N sessions (Asia + London + prior NY)

"Untested" means: no full 15-min candle close beyond the level in any direction
since the level was formed.

Returns a DayLevels dataclass consumed by setup_detector.py.
"""

from __future__ import annotations

import copy
import datetime
import functools
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from src.session_engine import SessionEngine, SessionInfo


@functools.lru_cache(maxsize=4)
def _load_settings(path: str = "config/settings.yaml") -> dict:
    """Load and cache YAML settings — parsed once per path, not once per bar."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


@dataclass
class AnchorCandle:
    """The 8:00am 15-minute candle — the foundation of every trading day."""
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    # Optional overrides — supplied by LevelBuilder after configure_for_symbol();
    # if None, falls back to global config so standalone usage still works.
    max_range_pts: Optional[float] = field(default=None, repr=False)
    min_range_pts: Optional[float] = field(default=None, repr=False)
    mid: float = field(init=False)
    range_pts: float = field(init=False)
    is_valid: bool = field(init=False)   # False if range > max_8am_range_points

    def __post_init__(self) -> None:
        self.mid = round((self.high + self.low) / 2, 2)
        self.range_pts = round(self.high - self.low, 2)
        # Use caller-supplied limits if available; otherwise fall back to root config
        if self.max_range_pts is None or self.min_range_pts is None:
            cfg = _load_settings()["levels"]
            self.max_range_pts = cfg["max_8am_range_points"]
            self.min_range_pts = cfg.get("min_8am_range_points", 0)
        max_range = self.max_range_pts
        min_range = self.min_range_pts
        self.is_valid = min_range <= self.range_pts <= max_range
        if self.range_pts > max_range:
            logger.warning(
                "8am candle range={:.2f}pts exceeds max={}pts — NO-TRADE day",
                self.range_pts, max_range
            )
        elif self.range_pts < min_range:
            logger.warning(
                "8am candle range={:.2f}pts below min={}pts — NO-TRADE day (too noisy)",
                self.range_pts, min_range
            )


@dataclass
class PriceLevel:
    """A single untested high or low liquidity level."""
    price: float
    direction: str          # "high" or "low"
    formed_at: pd.Timestamp
    session_origin: str     # "asia", "london", "prior_ny", "intraday"
    tested: bool = False    # Becomes True once a full candle closes beyond it

    def __str__(self) -> str:
        return (
            f"PriceLevel({self.direction}={self.price:.2f} "
            f"from {self.session_origin} at {self.formed_at})"
        )


@dataclass
class DayLevels:
    """All levels assembled for a single trading day."""
    date: datetime.date
    anchor: Optional[AnchorCandle]
    untested_highs: list[PriceLevel] = field(default_factory=list)
    untested_lows: list[PriceLevel] = field(default_factory=list)
    no_trade_day: bool = False
    # Cached from config so mark_tested() never reads the YAML file again.
    # Set by LevelBuilder.build(); defaults to typical ES value for safety.
    _tolerance: float = field(default=2.0, repr=False)

    def __post_init__(self) -> None:
        # If the anchor candle was provided but is invalid, mark as no-trade day
        if self.anchor is not None and not self.anchor.is_valid:
            self.no_trade_day = True

    @property
    def all_levels(self) -> list[PriceLevel]:
        return self.untested_highs + self.untested_lows

    def mark_tested(self, price: float, candle_close: float, candle_direction: str) -> None:
        """
        After each candle closes, scan all levels and mark as tested if a full
        candle body has closed beyond them.

        Uses self._tolerance (set once by LevelBuilder.build) — no file I/O.
        """
        tolerance = self._tolerance
        for lvl in self.untested_highs:
            if not lvl.tested and candle_close > lvl.price + tolerance:
                lvl.tested = True
                logger.debug("Level {} TESTED by close={:.2f}", lvl, candle_close)
        for lvl in self.untested_lows:
            if not lvl.tested and candle_close < lvl.price - tolerance:
                lvl.tested = True
                logger.debug("Level {} TESTED by close={:.2f}", lvl, candle_close)

    def active_highs(self) -> list[PriceLevel]:
        return [l for l in self.untested_highs if not l.tested]

    def active_lows(self) -> list[PriceLevel]:
        return [l for l in self.untested_lows if not l.tested]

    def nearest_high_above(self, price: float) -> Optional[PriceLevel]:
        highs = [l for l in self.active_highs() if l.price > price]
        return min(highs, key=lambda l: l.price - price) if highs else None

    def nearest_low_below(self, price: float) -> Optional[PriceLevel]:
        lows = [l for l in self.active_lows() if l.price < price]
        return min(lows, key=lambda l: price - l.price) if lows else None


class LevelBuilder:
    """
    Builds DayLevels for a given trading date using 1m and 15m OHLCV DataFrames.

    Usage (backtesting):
        lb = LevelBuilder()
        day_levels = lb.build(date, df_15m, df_1m)

    Usage (live):
        Same interface — pass the rolling bar cache from the live feed.
    """

    def __init__(self, settings_path: str = "config/settings.yaml") -> None:
        self._cfg = _load_settings(settings_path)
        self._session = SessionEngine(settings_path)
        self._lookback: int = self._cfg["levels"]["lookback_sessions"]
        self._tolerance: float = self._cfg["levels"]["level_tolerance_points"]

    def configure_for_symbol(self, symbol: str) -> None:
        """
        Apply per-symbol level overrides from the ``symbols:`` block in settings.yaml.

        Critical for NQ/MNQ where:
          - ``level_tolerance_points`` must be ~2.5 (not ES's 0.5) — NQ wiggles 3-5pts
          - ``max_8am_range_points`` must be ~80 (not ES's 20)
          - ``min_8am_range_points`` must be ~20 (not ES's 5)

        Deep-copies ``self._cfg`` before mutating so the LRU-cached shared dict
        is never modified.
        """
        sym_cfg = self._cfg.get("symbols", {}).get(symbol, {})
        if not sym_cfg:
            return
        levels_overrides = sym_cfg.get("levels_overrides", {})
        if levels_overrides:
            self._cfg = copy.deepcopy(self._cfg)
            self._cfg["levels"].update(levels_overrides)
            self._tolerance = float(
                self._cfg["levels"].get("level_tolerance_points", self._tolerance)
            )
            logger.info(
                "Symbol {}: level tolerance={:.1f}pt  max_8am={:.0f}pt  min_8am={:.0f}pt",
                symbol,
                self._tolerance,
                self._cfg["levels"]["max_8am_range_points"],
                self._cfg["levels"].get("min_8am_range_points", 0),
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        date: datetime.date,
        df_15m: pd.DataFrame,
        df_1m: Optional[pd.DataFrame] = None,
    ) -> DayLevels:
        """
        Build all levels for `date`.
        df_15m must cover at least `lookback_sessions` days prior to `date`.
        """
        session = self._session.get_session(date)
        anchor_bar = self._session.get_anchor_candle(df_15m, session)

        if anchor_bar is None:
            logger.warning("No anchor candle found for {}; marking no-trade day", date)
            return DayLevels(date=date, anchor=None, no_trade_day=True)

        # Pass symbol-specific range limits so AnchorCandle uses correct thresholds
        # (NQ needs max=80pt, ES uses max=20pt — wrong default would reject all NQ days)
        anchor = AnchorCandle(
            timestamp=anchor_bar.name,
            open=float(anchor_bar["open"]),
            high=float(anchor_bar["high"]),
            low=float(anchor_bar["low"]),
            close=float(anchor_bar["close"]),
            volume=float(anchor_bar.get("volume", 0)),
            max_range_pts=float(self._cfg["levels"]["max_8am_range_points"]),
            min_range_pts=float(self._cfg["levels"].get("min_8am_range_points", 0)),
        )

        day_levels = DayLevels(
            date=date,
            anchor=anchor,
            no_trade_day=not anchor.is_valid,
            _tolerance=self._tolerance,   # cached — never reads YAML again
        )

        if not anchor.is_valid:
            return day_levels

        # Build untested highs/lows from prior sessions
        highs, lows = self._scan_untested_levels(df_15m, session)

        # Phase 5: also add OR_HIGH and OR_LOW as rejection/bounce level candidates
        # when include_or_levels is enabled in settings.yaml (setups.rejection / bounce)
        setups_cfg = self._cfg.get("setups", {})
        include_or = (
            setups_cfg.get("rejection", {}).get("include_or_levels", True)
            or setups_cfg.get("bounce", {}).get("include_or_levels", True)
        )
        if include_or:
            highs = highs + [PriceLevel(
                price=anchor.high,
                direction="high",
                formed_at=anchor.timestamp,
                session_origin="or_range",
            )]
            lows = lows + [PriceLevel(
                price=anchor.low,
                direction="low",
                formed_at=anchor.timestamp,
                session_origin="or_range",
            )]

        # Previous Day High/Low — among the most widely watched levels in ES/NQ.
        # Add as named "prev_day" levels so Sweep+Reverse and Rejection setups
        # can reference them as high-quality liquidity zones.
        prev_day_high, prev_day_low = self._get_prev_day_levels(df_15m, session)
        if prev_day_high is not None:
            highs = highs + [prev_day_high]
        if prev_day_low is not None:
            lows = lows + [prev_day_low]

        # Deduplicate again after adding OR + prev_day levels
        highs = self._deduplicate(highs)
        lows  = self._deduplicate(lows)

        day_levels.untested_highs = highs
        day_levels.untested_lows = lows

        logger.info(
            "{} — anchor={:.2f}/{:.2f} mid={:.2f} range={:.2f}pts | "
            "{} untested highs, {} untested lows{}{} {}",
            date,
            anchor.high, anchor.low, anchor.mid, anchor.range_pts,
            len(highs), len(lows),
            " (+OR extremes)" if include_or else "",
            " (+PDH)" if prev_day_high else "",
            "(+PDL)" if prev_day_low else "",
        )
        return day_levels

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scan_untested_levels(
        self,
        df_15m: pd.DataFrame,
        session: SessionInfo,
    ) -> tuple[list[PriceLevel], list[PriceLevel]]:
        """
        Scan 15-min bars prior to the anchor candle time and identify
        swing highs/lows that have never been fully closed beyond.
        """
        anchor_ts = pd.Timestamp(session.anchor_candle_start)

        # Window: from N sessions back up to (but not including) the anchor candle
        lookback_start = anchor_ts - pd.Timedelta(days=self._lookback + 1)
        prior_bars = df_15m.loc[
            (df_15m.index >= lookback_start) & (df_15m.index < anchor_ts)
        ].copy()

        if len(prior_bars) < 3:
            logger.warning("Insufficient prior bars for level scan on {}", session.date)
            return [], []

        swing_highs = self._find_swing_highs(prior_bars)
        swing_lows = self._find_swing_lows(prior_bars)

        # Filter: keep only untested ones (no full candle close beyond the level
        # at any point after the swing formed, up to the anchor)
        untested_highs = []
        for idx, price in swing_highs:
            if not self._was_closed_above(prior_bars, price, after=idx):
                origin = self._classify_session_origin(idx, session)
                untested_highs.append(PriceLevel(
                    price=price,
                    direction="high",
                    formed_at=idx,
                    session_origin=origin,
                ))

        untested_lows = []
        for idx, price in swing_lows:
            if not self._was_closed_below(prior_bars, price, after=idx):
                origin = self._classify_session_origin(idx, session)
                untested_lows.append(PriceLevel(
                    price=price,
                    direction="low",
                    formed_at=idx,
                    session_origin=origin,
                ))

        # Deduplicate levels that are within tolerance of each other
        untested_highs = self._deduplicate(untested_highs)
        untested_lows = self._deduplicate(untested_lows)

        return untested_highs, untested_lows

    @staticmethod
    def _find_swing_highs(df: pd.DataFrame) -> list[tuple[pd.Timestamp, float]]:
        """
        A swing high is a bar whose 'high' is greater than the high of both
        its immediate left and right neighbours.
        """
        highs = []
        prices = df["high"].values
        timestamps = df.index
        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
                highs.append((timestamps[i], float(prices[i])))
        return highs

    @staticmethod
    def _find_swing_lows(df: pd.DataFrame) -> list[tuple[pd.Timestamp, float]]:
        """A swing low is a bar whose 'low' is lower than both neighbours."""
        lows = []
        prices = df["low"].values
        timestamps = df.index
        for i in range(1, len(prices) - 1):
            if prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
                lows.append((timestamps[i], float(prices[i])))
        return lows

    def _was_closed_above(
        self, df: pd.DataFrame, level: float, after: pd.Timestamp
    ) -> bool:
        """
        Returns True if any 15-min candle *after* the swing high formed has a
        full close strictly above level + tolerance (the 'tested' definition).
        """
        subsequent = df.loc[df.index > after]
        return bool((subsequent["close"] > level + self._tolerance).any())

    def _was_closed_below(
        self, df: pd.DataFrame, level: float, after: pd.Timestamp
    ) -> bool:
        subsequent = df.loc[df.index > after]
        return bool((subsequent["close"] < level - self._tolerance).any())

    def _classify_session_origin(
        self, ts: pd.Timestamp, session: SessionInfo
    ) -> str:
        """Classify a bar timestamp as asia, london, prior_ny, or intraday."""
        t = ts.to_pydatetime().astimezone(session.asia_session_start.tzinfo)
        if session.asia_session_start <= t < session.asia_session_end:
            return "asia"
        elif session.london_session_start <= t < session.london_session_end:
            return "london"
        elif session.ny_open.replace(tzinfo=None) <= t.replace(tzinfo=None):
            return "prior_ny"
        else:
            return "overnight"

    def _get_prev_day_levels(
        self,
        df_15m: pd.DataFrame,
        session: SessionInfo,
    ) -> tuple[Optional[PriceLevel], Optional[PriceLevel]]:
        """
        Extract the absolute high and low of the previous calendar trading day
        from the 15m bar data.

        Returns (PriceLevel or None, PriceLevel or None) for high and low.
        Returns (None, None) if there is insufficient prior data.
        """
        anchor_ts = pd.Timestamp(session.anchor_candle_start)

        # Go back up to 5 calendar days to find the previous trading day
        # (handles weekends/holidays)
        for days_back in range(1, 6):
            prev_date = (anchor_ts - pd.Timedelta(days=days_back)).date()
            mask = df_15m.index.date == prev_date
            prev_day_bars = df_15m.loc[mask]
            if len(prev_day_bars) >= 4:   # At least a few bars — real trading day
                prev_high_price = float(prev_day_bars["high"].max())
                prev_low_price  = float(prev_day_bars["low"].min())
                # Use the last bar's timestamp as "formed_at"
                formed_at = prev_day_bars.index[-1]
                return (
                    PriceLevel(
                        price=round(prev_high_price, 2),
                        direction="high",
                        formed_at=formed_at,
                        session_origin="prev_day",
                    ),
                    PriceLevel(
                        price=round(prev_low_price, 2),
                        direction="low",
                        formed_at=formed_at,
                        session_origin="prev_day",
                    ),
                )

        return None, None

    def _deduplicate(self, levels: list[PriceLevel]) -> list[PriceLevel]:
        """Remove levels within self._tolerance of each other; keep the most recent."""
        if not levels:
            return levels
        sorted_lvls = sorted(levels, key=lambda l: l.price)
        result = [sorted_lvls[0]]
        for lvl in sorted_lvls[1:]:
            if abs(lvl.price - result[-1].price) > self._tolerance:
                result.append(lvl)
            else:
                # Keep more recent one
                if lvl.formed_at > result[-1].formed_at:
                    result[-1] = lvl
        return result

    # ------------------------------------------------------------------
    # Live update: mark levels as tested when new candles close
    # ------------------------------------------------------------------

    def update_tested_status(
        self,
        day_levels: DayLevels,
        latest_close: float,
        candle_timestamp: pd.Timestamp,
    ) -> None:
        """
        Called after each new candle closes during the trading day.
        Marks any levels that now have a full candle close beyond them.
        """
        day_levels.mark_tested(
            price=latest_close,
            candle_close=latest_close,
            candle_direction="up" if latest_close > day_levels.anchor.mid else "down",
        )
