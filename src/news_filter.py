"""
src/news_filter.py

Economic news calendar filter.

Blocks trading on high-impact days: FOMC decisions, CPI releases, NFP
(Non-Farm Payroll) reports.  These events cause unpredictable, large moves
that consistently break the ORB strategy's logic.

Data source: data/high_impact_dates.csv
  - Pre-populated for 2022–2026
  - NFP is auto-generated for future dates (first Friday of each month)
  - Add new events by appending rows: date,event,source

Usage:
    nf = NewsFilter()
    if nf.is_news_day(today):
        return None  # skip all trading today

Config (settings.yaml):
    regime_filter:
      news_filter_enabled: true
      news_filter_file: "data/high_impact_dates.csv"
"""

from __future__ import annotations

import datetime
import functools
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent


@functools.lru_cache(maxsize=1)
def _load_news_dates(csv_path: str) -> frozenset:
    """Load and cache the set of high-impact dates from a CSV file.

    CSV format:
        date,event,source
        2024-06-12,FOMC,manual
        ...
    Lines starting with '#' are treated as comments and skipped.
    """
    path = Path(csv_path)
    if not path.is_absolute():
        path = ROOT / csv_path

    if not path.exists():
        logger.warning("NewsFilter: calendar file not found at {} — filter disabled", path)
        return frozenset()

    dates: set = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 1:
                continue
            date_str = parts[0].strip()
            try:
                dates.add(datetime.date.fromisoformat(date_str))
            except ValueError:
                pass  # Skip header row or malformed lines

    # Auto-generate future NFP dates (first Friday of each month)
    # for up to 12 months beyond today, so the filter keeps working
    today = datetime.date.today()
    for year in range(today.year, today.year + 2):
        for month in range(1, 13):
            first_day = datetime.date(year, month, 1)
            # Find first Friday: weekday() == 4
            days_until_friday = (4 - first_day.weekday()) % 7
            first_friday = first_day + datetime.timedelta(days=days_until_friday)
            # First Friday of month is NFP release day
            dates.add(first_friday)

    logger.info("NewsFilter: loaded {} high-impact dates from {}", len(dates), path.name)
    return frozenset(dates)


def _get_nfp_for_month(year: int, month: int) -> datetime.date:
    """Return the first Friday (NFP release date) of the given month."""
    first_day = datetime.date(year, month, 1)
    days_until_friday = (4 - first_day.weekday()) % 7
    return first_day + datetime.timedelta(days=days_until_friday)


class NewsFilter:
    """
    Checks whether a given date is a high-impact news day.

    Thread-safe: uses a module-level LRU-cached frozenset so multiple
    SetupDetector instances share the same loaded data.
    """

    def __init__(self, settings: dict | None = None, csv_path: str | None = None) -> None:
        """
        Args:
            settings: full settings dict (reads regime_filter.news_filter_file).
            csv_path: direct path override — use if you don't have the full settings dict.
        """
        if csv_path is not None:
            self._csv_path = csv_path
        elif settings is not None:
            regime = settings.get("regime_filter", {})
            self._csv_path = regime.get("news_filter_file", "data/high_impact_dates.csv")
        else:
            self._csv_path = "data/high_impact_dates.csv"

    def is_news_day(self, date: datetime.date | pd.Timestamp | str) -> bool:
        """Return True if `date` is a known high-impact news day."""
        if isinstance(date, str):
            date = datetime.date.fromisoformat(date)
        elif isinstance(date, pd.Timestamp):
            date = date.date()
        elif isinstance(date, datetime.datetime):
            date = date.date()

        news_dates = _load_news_dates(self._csv_path)
        result = date in news_dates
        if result:
            logger.debug("NewsFilter: {} is a high-impact news day — no trade", date)
        return result
