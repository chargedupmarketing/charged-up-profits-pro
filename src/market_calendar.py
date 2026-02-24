"""
src/market_calendar.py

Central authority for all futures market calendar logic:
  - CME holidays (no trading)
  - Early-close days (shortened session, e.g. day before Thanksgiving)
  - ES/NQ contract roll weeks (avoid illiquid back-month)
  - DST-safe scheduling helpers
  - Front-month symbol routing (ESH/ESM/ESU/ESZ → ES continuous feed)

All times are America/New_York (ET).  Never use datetime.now() without this
module — use MarketCalendar.now_et() or session_engine.now_est() instead.

Usage:
    cal = MarketCalendar()
    if cal.is_holiday(today):
        logger.info("Holiday — no trading")
    if cal.is_early_close(today):
        close_time = cal.early_close_time(today)
    if cal.is_roll_week(today, "ES"):
        logger.warning("Roll week — using front month but watch liquidity")
    front = cal.front_month_code(today, "ES")  # e.g. "ESH25"
"""
from __future__ import annotations

import datetime
from functools import lru_cache
from zoneinfo import ZoneInfo

from loguru import logger

_ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# CME Equity Futures holidays (CME closes these days)
# Source: CME Group holiday calendar
# ---------------------------------------------------------------------------

_CME_HOLIDAYS: frozenset[tuple[int, int, int]] = frozenset([
    # 2024
    (2024, 1, 1),   # New Year's Day
    (2024, 1, 15),  # MLK Day
    (2024, 2, 19),  # Presidents' Day
    (2024, 3, 29),  # Good Friday
    (2024, 5, 27),  # Memorial Day
    (2024, 6, 19),  # Juneteenth
    (2024, 7, 4),   # Independence Day
    (2024, 9, 2),   # Labor Day
    (2024, 11, 28), # Thanksgiving
    (2024, 12, 25), # Christmas
    # 2025
    (2025, 1, 1),   # New Year's Day
    (2025, 1, 20),  # MLK Day
    (2025, 2, 17),  # Presidents' Day
    (2025, 4, 18),  # Good Friday
    (2025, 5, 26),  # Memorial Day
    (2025, 6, 19),  # Juneteenth
    (2025, 7, 4),   # Independence Day
    (2025, 9, 1),   # Labor Day
    (2025, 11, 27), # Thanksgiving
    (2025, 12, 25), # Christmas
    # 2026
    (2026, 1, 1),   # New Year's Day
    (2026, 1, 19),  # MLK Day
    (2026, 2, 16),  # Presidents' Day
    (2026, 4, 3),   # Good Friday
    (2026, 5, 25),  # Memorial Day
    (2026, 6, 19),  # Juneteenth
    (2026, 7, 3),   # Independence Day (observed)
    (2026, 9, 7),   # Labor Day
    (2026, 11, 26), # Thanksgiving
    (2026, 12, 25), # Christmas
])

# Early-close days: futures close at 13:00 ET (1pm)
# Typically the day before: Thanksgiving, Christmas, Independence Day, New Year's
_CME_EARLY_CLOSE: frozenset[tuple[int, int, int]] = frozenset([
    # 2024
    (2024, 7, 3),   # Day before Independence Day
    (2024, 11, 29), # Day after Thanksgiving
    (2024, 12, 24), # Christmas Eve
    # 2025
    (2025, 7, 3),   # Day before Independence Day
    (2025, 11, 28), # Day after Thanksgiving
    (2025, 12, 24), # Christmas Eve
    # 2026
    (2026, 7, 2),   # Day before Independence Day (observed 7/3)
    (2026, 11, 27), # Day after Thanksgiving
    (2026, 12, 24), # Christmas Eve
])

_EARLY_CLOSE_HOUR = 13   # 1:00 PM ET

# ---------------------------------------------------------------------------
# ES/NQ quarterly contract codes: H (Mar), M (Jun), U (Sep), Z (Dec)
# Roll week: Thursday of the week containing the 3rd Friday of expiry month.
# The "roll" is typically executed the Thursday/Friday before expiration.
# We flag the FULL WEEK of expiration as a "roll week" and use reduced confidence.
# ---------------------------------------------------------------------------

_EXPIRY_MONTHS = {1: "H", 4: "M", 7: "U", 10: "Z"}   # quarter month → code
_YEAR_CODES = {25: "25", 26: "26", 27: "27", 28: "28"}


def _third_friday(year: int, month: int) -> datetime.date:
    """Return the 3rd Friday of the given month/year."""
    first = datetime.date(year, month, 1)
    # weekday() 4 = Friday
    offset = (4 - first.weekday()) % 7
    first_friday = first + datetime.timedelta(days=offset)
    return first_friday + datetime.timedelta(weeks=2)


def _expiry_dates(year: int) -> list[datetime.date]:
    """Return ES/NQ expiry dates (3rd Fridays of Mar/Jun/Sep/Dec) for a year."""
    return [_third_friday(year, m) for m in (3, 6, 9, 12)]


@lru_cache(maxsize=32)
def _roll_weeks_for_year(year: int) -> list[tuple[datetime.date, datetime.date]]:
    """
    Return list of (roll_week_start, roll_week_end) date ranges.
    Roll week = Mon–Fri of the week containing the expiry Friday.
    """
    weeks = []
    for exp in _expiry_dates(year):
        # Monday of expiry week
        monday = exp - datetime.timedelta(days=exp.weekday())
        weeks.append((monday, exp))
    return weeks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class MarketCalendar:
    """
    Central calendar for all market schedule decisions.
    Thread-safe (no mutable state).
    """

    @staticmethod
    def now_et() -> datetime.datetime:
        """Current wall-clock time in America/New_York. Use everywhere."""
        return datetime.datetime.now(tz=_ET)

    @staticmethod
    def today_et() -> datetime.date:
        return MarketCalendar.now_et().date()

    # ------------------------------------------------------------------
    # Holiday / close checks
    # ------------------------------------------------------------------

    @staticmethod
    def is_holiday(date: datetime.date) -> bool:
        """True if CME is fully closed on this date."""
        return (date.year, date.month, date.day) in _CME_HOLIDAYS

    @staticmethod
    def is_early_close(date: datetime.date) -> bool:
        """True if CME closes early (1pm ET) on this date."""
        return (date.year, date.month, date.day) in _CME_EARLY_CLOSE

    @staticmethod
    def is_trading_day(date: datetime.date) -> bool:
        """True if the date is a weekday and not a CME holiday."""
        return date.weekday() < 5 and not MarketCalendar.is_holiday(date)

    @staticmethod
    def session_close(date: datetime.date) -> datetime.time:
        """Return the session close time for this date (normal or early)."""
        if MarketCalendar.is_early_close(date):
            return datetime.time(_EARLY_CLOSE_HOUR, 0)
        return datetime.time(16, 0)   # normal close

    @staticmethod
    def exec_end_time(date: datetime.date, default_end: str = "13:00") -> datetime.time:
        """
        Return the effective execution-window end time for this date.
        On early-close days the window ends 30m before close (or at 12:30).
        """
        if MarketCalendar.is_early_close(date):
            return datetime.time(12, 30)
        h, m = map(int, default_end.split(":"))
        return datetime.time(h, m)

    # ------------------------------------------------------------------
    # Contract roll logic
    # ------------------------------------------------------------------

    @staticmethod
    def is_roll_week(date: datetime.date, symbol: str = "ES") -> bool:
        """
        True if the date falls in the expiry week for ES/NQ.
        During roll week:  spreads widen, volume migrates to next contract.
        Recommendation: trade front-month normally Mon/Tue; reduce size Wed–Fri.
        """
        for start, end in _roll_weeks_for_year(date.year):
            if start <= date <= end:
                return True
        return False

    @staticmethod
    def roll_week_day(date: datetime.date) -> int:
        """
        If in roll week, return the day-of-week within it (0=Mon, 4=Fri).
        Returns -1 if not in a roll week.
        """
        for start, end in _roll_weeks_for_year(date.year):
            if start <= date <= end:
                return (date - start).days
        return -1

    @staticmethod
    def front_month_code(date: datetime.date, root: str = "ES") -> str:
        """
        Return the front-month contract code for a given date.
        After roll (Thursday/Friday of expiry week), the NEXT quarter is front month.

        Returns e.g. "ESH25", "NQM26".
        """
        root = root.upper()
        year = date.year
        # Find the next expiry date on or after today
        exps = _expiry_dates(year) + _expiry_dates(year + 1)
        for exp in exps:
            # Roll happens on Thursday of expiry week (2 days before expiry)
            roll_thursday = exp - datetime.timedelta(days=1)
            if date < roll_thursday:
                # This quarter is still front month
                q_month = exp.month   # expiry month = quarter month
                code = _EXPIRY_MONTHS.get(q_month - 0, "H")  # H/M/U/Z
                yr2 = str(exp.year)[-2:]
                return f"{root}{code}{yr2}"
        return f"{root}H{str(year + 1)[-2:]}"  # fallback

    @staticmethod
    def is_rollover_liquidity_warning(date: datetime.date) -> bool:
        """
        True on Wed/Thu/Fri of expiry week when liquidity is migrating.
        In this window: log a warning but allow trading (do not hard block).
        """
        day_in_roll = MarketCalendar.roll_week_day(date)
        return day_in_roll >= 2   # Wed=2, Thu=3, Fri=4

    # ------------------------------------------------------------------
    # DST helpers
    # ------------------------------------------------------------------

    @staticmethod
    def dst_transition_week(date: datetime.date) -> bool:
        """
        True if the date falls within a DST-transition week
        (US spring-forward and fall-back).
        Useful for adding extra caution around scheduled times.
        """
        # US spring forward: 2nd Sunday of March
        # US fall back:      1st Sunday of November
        year = date.year
        # 2nd Sunday of March
        march_1 = datetime.date(year, 3, 1)
        first_sun_march = march_1 + datetime.timedelta(days=(6 - march_1.weekday()) % 7)
        spring_forward = first_sun_march + datetime.timedelta(weeks=1)
        # 1st Sunday of November
        nov_1 = datetime.date(year, 11, 1)
        fall_back = nov_1 + datetime.timedelta(days=(6 - nov_1.weekday()) % 7)

        for anchor in (spring_forward, fall_back):
            week_start = anchor - datetime.timedelta(days=3)
            week_end   = anchor + datetime.timedelta(days=3)
            if week_start <= date <= week_end:
                return True
        return False

    @staticmethod
    def safe_et_time(
        hour: int,
        minute: int = 0,
        date: datetime.date | None = None,
    ) -> datetime.datetime:
        """
        Create a tz-aware ET datetime for today's date (or given date).
        Always uses ZoneInfo("America/New_York") — DST-correct.
        """
        d = date or MarketCalendar.today_et()
        return datetime.datetime(d.year, d.month, d.day, hour, minute,
                                 tzinfo=_ET)

    # ------------------------------------------------------------------
    # Summary (for logging)
    # ------------------------------------------------------------------

    @staticmethod
    def day_summary(date: datetime.date | None = None) -> dict:
        """Return a dict of all calendar flags for a date (for logging/dashboard)."""
        d = date or MarketCalendar.today_et()
        front_es = MarketCalendar.front_month_code(d, "ES")
        front_nq = MarketCalendar.front_month_code(d, "NQ")
        return {
            "date":              str(d),
            "is_trading_day":    MarketCalendar.is_trading_day(d),
            "is_holiday":        MarketCalendar.is_holiday(d),
            "is_early_close":    MarketCalendar.is_early_close(d),
            "session_close":     str(MarketCalendar.session_close(d)),
            "is_roll_week":      MarketCalendar.is_roll_week(d),
            "roll_liquidity_warning": MarketCalendar.is_rollover_liquidity_warning(d),
            "is_dst_week":       MarketCalendar.dst_transition_week(d),
            "front_month_ES":    front_es,
            "front_month_NQ":    front_nq,
        }
