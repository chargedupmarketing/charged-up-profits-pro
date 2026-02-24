"""
session_engine.py

Handles all time/timezone logic:
  - Trading day boundaries
  - Session windows (Asia, London, NY)
  - Whether we are currently inside the allowed execution window
  - Candle alignment for 1m and 15m bars
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal
import yaml
from loguru import logger


def _load_settings(path: str = "config/settings.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


_SETTINGS = _load_settings()
_TZ_STR: str = _SETTINGS["session"]["timezone"]
_TZ = ZoneInfo(_TZ_STR)


@dataclass
class SessionInfo:
    """Describes the trading sessions for a single calendar date."""
    date: datetime.date

    # NY session boundaries (EST/EDT)
    ny_open: datetime.datetime = field(init=False)
    ny_close: datetime.datetime = field(init=False)

    # Anchor candle window
    anchor_candle_start: datetime.datetime = field(init=False)
    anchor_candle_end: datetime.datetime = field(init=False)

    # Allowed execution window
    exec_start: datetime.datetime = field(init=False)
    exec_end: datetime.datetime = field(init=False)

    # Asia session (approximate CME Globex): 6pm prior day → 5am
    asia_session_start: datetime.datetime = field(init=False)
    asia_session_end: datetime.datetime = field(init=False)

    # London session: 3am → 8am (EST)
    london_session_start: datetime.datetime = field(init=False)
    london_session_end: datetime.datetime = field(init=False)

    def __post_init__(self) -> None:
        d = self.date
        anchor_h, anchor_m = map(int, _SETTINGS["session"]["anchor_candle_time"].split(":"))
        exec_sh, exec_sm = map(int, _SETTINGS["session"]["execution_start"].split(":"))
        exec_eh, exec_em = map(int, _SETTINGS["session"]["execution_end"].split(":"))

        def _ts(h: int, m: int = 0, offset_days: int = 0) -> datetime.datetime:
            base = d + datetime.timedelta(days=offset_days)
            return datetime.datetime(base.year, base.month, base.day, h, m, tzinfo=_TZ)

        self.anchor_candle_start = _ts(anchor_h, anchor_m)
        self.anchor_candle_end = self.anchor_candle_start + datetime.timedelta(minutes=15)
        self.exec_start = _ts(exec_sh, exec_sm)
        self.exec_end = _ts(exec_eh, exec_em)
        self.ny_open = _ts(9, 30)
        self.ny_close = _ts(16, 0)

        # Asia: prior-day 6pm → today 5am
        self.asia_session_start = _ts(18, 0, offset_days=-1)
        self.asia_session_end = _ts(5, 0)

        # London: today 3am → today 8am
        self.london_session_start = _ts(3, 0)
        self.london_session_end = _ts(8, 0)


class SessionEngine:
    """
    Central clock/calendar service.  All other modules ask this class
    whether a given timestamp is inside a particular session window.
    """

    def __init__(self, settings_path: str = "config/settings.yaml") -> None:
        self._settings = _load_settings(settings_path)
        self._tz = ZoneInfo(self._settings["session"]["timezone"])
        self._calendar = mcal.get_calendar("CME_Equity")
        logger.info("SessionEngine initialised — timezone={}", self._settings["session"]["timezone"])

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def now_est(self) -> datetime.datetime:
        """Current time in the configured timezone."""
        return datetime.datetime.now(tz=self._tz)

    def is_trading_day(self, date: Optional[datetime.date] = None) -> bool:
        """Returns True if the CME equity calendar considers this a trading day."""
        if date is None:
            date = self.now_est().date()
        schedule = self._calendar.schedule(
            start_date=str(date),
            end_date=str(date),
        )
        return not schedule.empty

    def get_session(self, date: Optional[datetime.date] = None) -> SessionInfo:
        if date is None:
            date = self.now_est().date()
        return SessionInfo(date=date)

    # ------------------------------------------------------------------
    # Live-trading predicates
    # ------------------------------------------------------------------

    def is_in_execution_window(self, ts: Optional[datetime.datetime] = None) -> bool:
        """True if ts is within the 9:00–11:30am execution window."""
        if ts is None:
            ts = self.now_est()
        ts = ts.astimezone(self._tz)
        session = self.get_session(ts.date())
        return session.exec_start <= ts < session.exec_end

    def is_anchor_candle_time(self, ts: Optional[datetime.datetime] = None) -> bool:
        """True if we are in the 8:00–8:15am candle window."""
        if ts is None:
            ts = self.now_est()
        ts = ts.astimezone(self._tz)
        session = self.get_session(ts.date())
        return session.anchor_candle_start <= ts < session.anchor_candle_end

    def minutes_until_exec_start(self) -> float:
        now = self.now_est()
        session = self.get_session(now.date())
        delta = (session.exec_start - now).total_seconds() / 60
        return max(0.0, delta)

    # ------------------------------------------------------------------
    # DataFrame helpers  (align bar DataFrames to EST timezone)
    # ------------------------------------------------------------------

    @staticmethod
    def localize_bars(df: pd.DataFrame, tz_str: str = _TZ_STR) -> pd.DataFrame:
        """
        Ensure a DataFrame with a DatetimeTzIndex is in the target timezone.
        Handles UTC, naive, and already-localised inputs gracefully.
        """
        if df.index.tz is None:
            df = df.copy()
            df.index = df.index.tz_localize("UTC")
        return df.copy().tz_convert(tz_str)

    @staticmethod
    def filter_to_session(df: pd.DataFrame, session: SessionInfo) -> pd.DataFrame:
        """Filter a bar DataFrame to the NY regular session only."""
        return df.loc[
            (df.index >= pd.Timestamp(session.ny_open))
            & (df.index < pd.Timestamp(session.ny_close))
        ]

    @staticmethod
    def get_anchor_candle(df_15min: pd.DataFrame, session: SessionInfo) -> Optional[pd.Series]:
        """
        Extract the 8:00am 15-min bar from a 15-min OHLCV DataFrame.
        Returns None if the candle is not present in the data.
        """
        # Guard: empty frame or non-DatetimeIndex (e.g. bar store not seeded yet)
        if df_15min.empty:
            return None
        if not isinstance(df_15min.index, pd.DatetimeIndex):
            logger.warning("get_anchor_candle: df_15m has no DatetimeIndex — bar store not ready yet")
            return None

        anchor_ts = pd.Timestamp(session.anchor_candle_start)
        if anchor_ts in df_15min.index:
            return df_15min.loc[anchor_ts]
        # Try fuzzy: within ±1 minute
        mask = abs((df_15min.index - anchor_ts).total_seconds()) <= 60
        candidates = df_15min[mask]
        if len(candidates) == 1:
            return candidates.iloc[0]
        if len(candidates) > 1:
            logger.warning("Multiple candidates for anchor candle at {}; taking first", anchor_ts)
            return candidates.iloc[0]
        logger.warning("Anchor candle not found for session date={}", session.date)
        return None

    # ------------------------------------------------------------------
    # Databento data-download helpers
    # ------------------------------------------------------------------

    def trading_days_range(self, start: str, end: str) -> pd.DatetimeIndex:
        """Return all CME trading days between start and end (YYYY-MM-DD strings)."""
        schedule = self._calendar.schedule(start_date=start, end_date=end)
        return pd.DatetimeIndex(schedule.index.date)  # type: ignore[arg-type]
