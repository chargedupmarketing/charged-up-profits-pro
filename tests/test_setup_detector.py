"""
tests/test_setup_detector.py

Unit tests for the setup detector and level builder.
Run with: python -m pytest tests/ -v
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.level_builder import AnchorCandle, DayLevels, LevelBuilder, PriceLevel
from src.setup_detector import Direction, SetupDetector, Signal, SetupType, TrendFilter


TZ = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(
    base_price: float = 5000.0,
    n_bars: int = 100,
    start: str = "2024-01-15 03:00",
    freq: str = "1min",
    trend: float = 0.0,
    noise: float = 1.0,
) -> pd.DataFrame:
    """Generate synthetic OHLCV bars."""
    rng = np.random.default_rng(42)
    ts = pd.date_range(start, periods=n_bars, freq=freq, tz="America/New_York")
    closes = base_price + trend * np.arange(n_bars) + noise * rng.standard_normal(n_bars)
    highs = closes + abs(noise * rng.standard_normal(n_bars)) * 0.5
    lows = closes - abs(noise * rng.standard_normal(n_bars)) * 0.5
    opens = closes + noise * rng.standard_normal(n_bars) * 0.3
    volumes = rng.integers(1000, 5000, n_bars).astype(float)
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": volumes,
    }, index=ts)


def _make_anchor_candle(high: float = 5010.0, low: float = 4995.0) -> AnchorCandle:
    ts = pd.Timestamp("2024-01-15 08:00", tz="America/New_York")
    return AnchorCandle(
        timestamp=ts, open=5000.0, high=high, low=low, close=5005.0, volume=1000.0
    )


def _make_day_levels(
    high_levels: list[float] = None,
    low_levels: list[float] = None,
    anchor_high: float = 5010.0,
    anchor_low: float = 4995.0,
) -> DayLevels:
    anchor = _make_anchor_candle(high=anchor_high, low=anchor_low)
    highs = [
        PriceLevel(
            price=p, direction="high",
            formed_at=pd.Timestamp("2024-01-15 06:00", tz="America/New_York"),
            session_origin="london"
        )
        for p in (high_levels or [])
    ]
    lows = [
        PriceLevel(
            price=p, direction="low",
            formed_at=pd.Timestamp("2024-01-15 05:00", tz="America/New_York"),
            session_origin="asia"
        )
        for p in (low_levels or [])
    ]
    return DayLevels(
        date=datetime.date(2024, 1, 15),
        anchor=anchor,
        untested_highs=highs,
        untested_lows=lows,
    )


# ---------------------------------------------------------------------------
# AnchorCandle tests
# ---------------------------------------------------------------------------

class TestAnchorCandle:
    def test_midpoint_calculation(self):
        c = _make_anchor_candle(high=5010.0, low=4990.0)
        assert c.mid == 5000.0

    def test_range_calculation(self):
        c = _make_anchor_candle(high=5010.0, low=4990.0)
        assert c.range_pts == 20.0

    def test_valid_when_range_at_limit(self):
        c = _make_anchor_candle(high=5010.0, low=4990.0)
        assert c.is_valid is True  # 20pts == max

    def test_invalid_when_range_exceeds_limit(self):
        c = _make_anchor_candle(high=5025.0, low=4990.0)
        assert c.is_valid is False  # 35pts > 20

    def test_no_trade_day_flagged(self):
        c = _make_anchor_candle(high=5030.0, low=4990.0)
        dl = DayLevels(date=datetime.date(2024, 1, 15), anchor=c)
        assert dl.no_trade_day is True


# ---------------------------------------------------------------------------
# PriceLevel / DayLevels tests
# ---------------------------------------------------------------------------

class TestDayLevels:
    def test_active_highs_returns_untested(self):
        dl = _make_day_levels(high_levels=[5020.0, 5030.0])
        assert len(dl.active_highs()) == 2

    def test_mark_tested_removes_level(self):
        dl = _make_day_levels(high_levels=[5020.0])
        dl.mark_tested(price=5025.0, candle_close=5021.0, candle_direction="up")
        assert len(dl.active_highs()) == 0

    def test_untested_levels_stay_active(self):
        dl = _make_day_levels(high_levels=[5020.0])
        dl.mark_tested(price=5010.0, candle_close=5015.0, candle_direction="up")
        assert len(dl.active_highs()) == 1  # Close at 5015 doesn't test 5020

    def test_nearest_high_above(self):
        dl = _make_day_levels(high_levels=[5020.0, 5035.0, 5050.0])
        nearest = dl.nearest_high_above(5015.0)
        assert nearest is not None
        assert nearest.price == 5020.0

    def test_nearest_low_below(self):
        dl = _make_day_levels(low_levels=[4970.0, 4985.0, 4990.0])
        nearest = dl.nearest_low_below(4995.0)
        assert nearest is not None
        assert nearest.price == 4990.0


# ---------------------------------------------------------------------------
# TrendFilter tests
# ---------------------------------------------------------------------------

class TestTrendFilter:
    def test_bullish_when_price_above_ema(self):
        tf = TrendFilter(ema_period=5)
        df = _make_bars(base_price=5000.0, trend=2.0, noise=0.1, n_bars=30)
        trend = tf.get_trend(df, current_price=5060.0)
        assert trend == "bullish"

    def test_bearish_when_price_below_ema(self):
        tf = TrendFilter(ema_period=5)
        df = _make_bars(base_price=5000.0, trend=-2.0, noise=0.1, n_bars=30)
        trend = tf.get_trend(df, current_price=4940.0)
        assert trend == "bearish"

    def test_neutral_when_insufficient_data(self):
        tf = TrendFilter(ema_period=20)
        df = _make_bars(n_bars=5)
        trend = tf.get_trend(df, current_price=5000.0)
        assert trend == "neutral"

    def test_allows_long_in_bullish(self):
        tf = TrendFilter(ema_period=5)
        df = _make_bars(base_price=5000.0, trend=2.0, noise=0.1, n_bars=30)
        assert tf.allows_direction(df, 5060.0, Direction.LONG) is True
        assert tf.allows_direction(df, 5060.0, Direction.SHORT) is False


# ---------------------------------------------------------------------------
# Signal validation
# ---------------------------------------------------------------------------

class TestSignal:
    def _make_signal(self, direction=Direction.LONG, entry=5000.0, stop=4993.0, target=5021.0):
        return Signal(
            setup_type=SetupType.BREAK_RETEST,
            direction=direction,
            entry_price=entry,
            stop_price=stop,
            target_price=target,
            formed_at=pd.Timestamp("2024-01-15 09:30", tz="America/New_York"),
            level_ref=None,
        )

    def test_stop_distance_long(self):
        sig = self._make_signal(Direction.LONG, entry=5000.0, stop=4993.0)
        assert sig.stop_distance == pytest.approx(7.0)

    def test_target_distance_long(self):
        sig = self._make_signal(Direction.LONG, entry=5000.0, target=5021.0)
        assert sig.target_distance == pytest.approx(21.0)

    def test_reward_risk_calculation(self):
        sig = self._make_signal(Direction.LONG, entry=5000.0, stop=4993.0, target=5021.0)
        assert sig.reward_risk == pytest.approx(3.0, abs=0.1)

    def test_is_valid_rr_pass(self):
        sig = self._make_signal(Direction.LONG, entry=5000.0, stop=4993.0, target=5021.0)
        assert sig.is_valid_rr(3.0) is True

    def test_is_valid_rr_fail(self):
        sig = self._make_signal(Direction.LONG, entry=5000.0, stop=4993.0, target=5010.0)
        assert sig.is_valid_rr(3.0) is False

    def test_short_stop_distance(self):
        sig = self._make_signal(Direction.SHORT, entry=5000.0, stop=5008.0, target=4976.0)
        assert sig.stop_distance == pytest.approx(8.0)

    def test_short_reward_risk(self):
        sig = self._make_signal(Direction.SHORT, entry=5000.0, stop=5008.0, target=4976.0)
        assert sig.reward_risk == pytest.approx(3.0, abs=0.1)


# ---------------------------------------------------------------------------
# SetupDetector integration test (smoke test with synthetic data)
# ---------------------------------------------------------------------------

class TestSetupDetectorSmoke:
    """
    Smoke tests: verify detect() returns Signal or None without errors.
    These use synthetic data and aren't testing strategy validity.
    """

    def setup_method(self):
        self._detector = SetupDetector()

    def test_returns_none_on_no_trade_day(self):
        dl = _make_day_levels()
        dl.no_trade_day = True
        df = _make_bars(n_bars=60, start="2024-01-15 09:00")
        result = self._detector.detect(dl, df, df, df)
        assert result is None

    def test_returns_none_or_signal_on_valid_day(self):
        dl = _make_day_levels(high_levels=[5025.0], low_levels=[4975.0])
        df_15m = _make_bars(n_bars=50, start="2024-01-15 07:00", freq="15min")
        df_5m = _make_bars(n_bars=60, start="2024-01-15 09:00", freq="5min")
        df_1m = _make_bars(n_bars=60, start="2024-01-15 09:00", freq="1min")
        result = self._detector.detect(dl, df_15m, df_5m, df_1m)
        assert result is None or isinstance(result, Signal)

    def test_signal_has_valid_rr_when_returned(self):
        dl = _make_day_levels(high_levels=[5025.0], low_levels=[4975.0])
        df_15m = _make_bars(n_bars=50, start="2024-01-15 07:00", freq="15min")
        df_5m = _make_bars(n_bars=60, start="2024-01-15 09:00", freq="5min")
        df_1m = _make_bars(n_bars=60, start="2024-01-15 09:00", freq="1min")
        result = self._detector.detect(dl, df_15m, df_5m, df_1m)
        if result is not None:
            assert result.is_valid_rr(3.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
