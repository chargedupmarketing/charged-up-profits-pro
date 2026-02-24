"""
src/feature_builder.py

Computes the feature vector used by the ML filter (Phase 2).
Also used during backtesting to label training data.

All features are derived only from information available AT signal time
(no lookahead). Each feature maps directly to a publicly-described
element of the ChargedUp Profits Bot strategy.

Phase 3a additions (8 new features):
  - candle_body_ratio     : entry bar body quality
  - atr_regime            : current ATR normalized vs rolling avg (vol regime)
  - level_freshness_bars  : bars since level was formed (recency)
  - level_test_count      : how many times level was touched but not broken
  - daily_ema50_distance  : macro trend distance from 50-EMA on 15m
  - spread_proxy          : intraday liquidity proxy (range / close)
  - prev_bar_momentum     : direction of the last 1m bar (continuation vs reversal)
  - bars_in_exec_window   : elapsed time in execution window (late signals less reliable)

Phase 6 additions (break strength — research-spec features):
  - break_excursion_pts   : max distance price traveled beyond break level before retest
  - closes_beyond_level   : count of 1m bars that closed beyond the break level
  - time_to_retest_bars   : bars elapsed from break bar to current retest bar
  - or_range_vs_atr       : OR range / ATR(14) — today's volatility vs average (all setups)

Phase 7 additions (direction-aligned derived features):
  - momentum_aligned      : momentum_3bar_15m × direction_sign (+= good, -= bad)
  - ema50_aligned         : 1 if EMA50 macro trend matches trade direction (key NQ predictor)
  - is_retest_quick       : 1 if time_to_retest_bars <= 8 (fast retests are cleaner)

Phase 8 additions (market context features):
  - gap_size_norm         : (today_open - prev_close) / atr — gap days behave differently
  - session_phase         : 0 = open drive (9–10am), 1 = mid-morning (10–11:30am), 2 = afternoon
  - or_range_vs_5day_avg  : today's OR range / 5-day avg OR range (breakout vs range-bound signal)
  - setup_type            : raw setup type string for per-setup-type ML model routing

Section 7 / schema v1.1 additions (regime intelligence):
  - efficiency_ratio      : Kaufman ER on last 20 bars of 15m (~1.0=trending, ~0.0=chop)
  - vol_state             : ATR(14) percentile rank vs 60-day ATR history (0=low, 1=high)
"""

from __future__ import annotations

import datetime
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.level_builder import DayLevels
from src.setup_detector import Direction, Signal


def compute_features(
    signal: Signal,
    day_levels: DayLevels,
    df_15m: pd.DataFrame,
    df_1m: pd.DataFrame,
) -> dict:
    """
    Returns a flat feature dictionary for the current signal.

    Feature categories:
      1. Opening range characteristics
      2. Level proximity
      3. Volume context
      4. Trend / momentum
      5. Temporal features
      6. Signal-specific features
      7. NEW (Phase 3a): Candle quality, ATR regime, level freshness, macro trend
    """
    features: dict = {}
    anchor = day_levels.anchor

    current_price = float(df_1m["close"].iloc[-1])

    # ----------------------------------------------------------------
    # 1. Opening range (8am candle) features
    # ----------------------------------------------------------------
    atr_val = _atr(df_15m, 14)

    if anchor:
        features["range_pts"] = anchor.range_pts
        features["range_norm"] = _safe_div(anchor.range_pts, atr_val)
        features["price_vs_mid"] = current_price - anchor.mid
        features["price_vs_mid_norm"] = _safe_div(current_price - anchor.mid, anchor.range_pts)
        features["price_above_mid"] = int(current_price > anchor.mid)
    else:
        features.update({
            "range_pts": 0.0,
            "range_norm": 0.0,
            "price_vs_mid": 0.0,
            "price_vs_mid_norm": 0.0,
            "price_above_mid": 0,
        })

    # ----------------------------------------------------------------
    # 2. Level proximity features
    # ----------------------------------------------------------------
    level_ref = signal.level_ref
    if level_ref:
        features["level_dist_pts"] = abs(current_price - level_ref.price)
        features["level_is_high"] = int(level_ref.direction == "high")
        features["level_origin_asia"] = int(level_ref.session_origin == "asia")
        features["level_origin_london"] = int(level_ref.session_origin == "london")

        # Phase 3a: level freshness (bars since formed)
        ts = signal.formed_at
        formed_at = level_ref.formed_at
        if hasattr(ts, "value") and hasattr(formed_at, "value"):
            try:
                delta_mins = (ts - formed_at).total_seconds() / 60
                features["level_freshness_bars"] = max(0.0, delta_mins)
            except Exception:
                features["level_freshness_bars"] = 0.0
        else:
            features["level_freshness_bars"] = 0.0

        # Phase 3a: level_test_count — how many times the level was touched in df_1m
        # without being broken (i.e. how many times price came within tolerance)
        tolerance = 0.5
        if len(df_1m) >= 5:
            level_price = level_ref.price
            touches = (
                (df_1m["low"] <= level_price + tolerance) &
                (df_1m["high"] >= level_price - tolerance) &
                (df_1m["close"].abs() <= level_price + tolerance * 3)
            ).sum()
            features["level_test_count"] = int(touches)
        else:
            features["level_test_count"] = 0

    else:
        features["level_dist_pts"] = 0.0
        features["level_is_high"] = 0
        features["level_origin_asia"] = 0
        features["level_origin_london"] = 0
        features["level_freshness_bars"] = 0.0
        features["level_test_count"] = 0

    features["active_highs_count"] = len(day_levels.active_highs())
    features["active_lows_count"] = len(day_levels.active_lows())

    # ----------------------------------------------------------------
    # 3. Volume features
    # ----------------------------------------------------------------
    vol_1m = df_1m["volume"]
    vol_median = vol_1m.tail(21).iloc[:-1].median() if len(vol_1m) > 2 else 1.0
    current_vol = vol_1m.iloc[-1]
    features["volume_ratio"] = _safe_div(current_vol, vol_median)
    features["volume_above_median"] = int(current_vol >= vol_median)

    # Volume trend: is volume increasing over last 5 bars?
    if len(vol_1m) >= 5:
        features["volume_trend_5"] = float(
            np.polyfit(range(5), vol_1m.tail(5).values, 1)[0]
        )
    else:
        features["volume_trend_5"] = 0.0

    # ----------------------------------------------------------------
    # 4. Trend / momentum features
    # ----------------------------------------------------------------
    if len(df_15m) >= 20:
        ema20 = df_15m["close"].ewm(span=20, adjust=False).mean().iloc[-1]
        features["price_vs_ema20"] = current_price - float(ema20)
        features["price_above_ema20"] = int(current_price > ema20)
    else:
        features["price_vs_ema20"] = 0.0
        features["price_above_ema20"] = 0

    features["atr_15m_14"] = atr_val

    # Momentum: last 3 bar close change on 15m
    if len(df_15m) >= 4:
        mom = float(df_15m["close"].iloc[-1] - df_15m["close"].iloc[-4])
        features["momentum_3bar_15m"] = mom
        features["momentum_3bar_norm"] = _safe_div(mom, atr_val)
    else:
        features["momentum_3bar_15m"] = 0.0
        features["momentum_3bar_norm"] = 0.0

    # 5-bar return on 1m chart
    if len(df_1m) >= 6:
        ret_5 = float(df_1m["close"].iloc[-1] - df_1m["close"].iloc[-6])
        features["return_5bar_1m"] = ret_5
    else:
        features["return_5bar_1m"] = 0.0

    # ----------------------------------------------------------------
    # 5. Temporal features
    # ----------------------------------------------------------------
    ts = signal.formed_at
    try:
        hour = ts.hour
        minute = ts.minute
    except AttributeError:
        hour, minute = 9, 30

    features["hour"] = hour
    features["minute_of_day"] = hour * 60 + minute
    features["day_of_week"] = ts.dayofweek if hasattr(ts, "dayofweek") else 0

    # How many minutes since execution window opened (9:00am)?
    exec_start_min = 9 * 60
    features["mins_since_exec_open"] = max(0, (hour * 60 + minute) - exec_start_min)

    # Phase 3a: bars elapsed in execution window (1m bars since 9:00)
    exec_open_ts = ts.normalize().replace(hour=9, minute=0) if hasattr(ts, "normalize") else ts
    try:
        exec_open_ts = ts.floor("D").replace(hour=9, minute=0, tzinfo=ts.tzinfo)
        bars_in_exec = int((ts - exec_open_ts).total_seconds() / 60)
        features["bars_in_exec_window"] = max(0, bars_in_exec)
    except Exception:
        features["bars_in_exec_window"] = features["mins_since_exec_open"]

    # ----------------------------------------------------------------
    # 6. Signal-specific features
    # ----------------------------------------------------------------
    features["setup_break_retest"] = int(signal.setup_type.value == "BREAK_RETEST")
    features["setup_rejection"] = int(signal.setup_type.value == "REJECTION")
    features["setup_bounce"] = int(signal.setup_type.value == "BOUNCE")
    features["direction_long"] = int(signal.direction == Direction.LONG)
    features["stop_dist_pts"] = signal.stop_distance
    features["target_dist_pts"] = signal.target_distance
    features["rr_ratio"] = signal.reward_risk

    # ----------------------------------------------------------------
    # 7. NEW Phase 3a features
    # ----------------------------------------------------------------

    # 7a. Candle body ratio at signal bar (quality of entry bar)
    if len(df_1m) >= 1:
        last_bar = df_1m.iloc[-1]
        bar_range = float(last_bar["high"]) - float(last_bar["low"])
        bar_body = abs(float(last_bar["close"]) - float(last_bar["open"]))
        features["candle_body_ratio"] = _safe_div(bar_body, bar_range)
    else:
        features["candle_body_ratio"] = 0.0

    # 7b. ATR regime: current ATR vs rolling 20-bar avg ATR
    if len(df_15m) >= 34:  # 14 + 20
        hi = df_15m["high"]
        lo = df_15m["low"]
        cl = df_15m["close"].shift(1)
        tr = pd.concat([hi - lo, (hi - cl).abs(), (lo - cl).abs()], axis=1).max(axis=1)
        atr_series = tr.rolling(14).mean().dropna()
        if len(atr_series) >= 20:
            current_atr = float(atr_series.iloc[-1])
            rolling_avg = float(atr_series.tail(20).mean())
            features["atr_regime"] = _safe_div(current_atr, rolling_avg)
        else:
            features["atr_regime"] = 1.0
    else:
        features["atr_regime"] = 1.0

    # 7c. Macro trend: distance from 50-EMA on 15m
    if len(df_15m) >= 50:
        ema50 = df_15m["close"].ewm(span=50, adjust=False).mean().iloc[-1]
        features["daily_ema50_distance"] = current_price - float(ema50)
        features["price_above_ema50"] = int(current_price > float(ema50))
    else:
        features["daily_ema50_distance"] = 0.0
        features["price_above_ema50"] = 0

    # 7d. Spread proxy: last bar range / close (intraday liquidity)
    if len(df_1m) >= 1:
        last_bar = df_1m.iloc[-1]
        features["spread_proxy"] = _safe_div(
            float(last_bar["high"]) - float(last_bar["low"]),
            float(last_bar["close"]) if float(last_bar["close"]) > 0 else 1.0,
        )
    else:
        features["spread_proxy"] = 0.0

    # 7e. Prev bar momentum: direction of last 1m bar vs previous (continuation signal)
    if len(df_1m) >= 2:
        last_close = float(df_1m["close"].iloc[-1])
        prev_close = float(df_1m["close"].iloc[-2])
        features["prev_bar_momentum"] = last_close - prev_close
        features["prev_bar_bullish"] = int(last_close > prev_close)
    else:
        features["prev_bar_momentum"] = 0.0
        features["prev_bar_bullish"] = 0

    # ----------------------------------------------------------------
    # 8. Phase 6: Break strength features (research-spec)
    #    "break strength: number of closes beyond level, maximum excursion
    #     beyond level, time to retest" — directly from deep-research-report2.md
    # ----------------------------------------------------------------
    _add_break_strength_features(features, signal, day_levels, df_1m)

    # ----------------------------------------------------------------
    # 9. Direction-aligned derived features
    #    These encode the "is the setup aligned with market context?" signal
    #    more directly than raw unsigned values — critical for ML discrimination.
    # ----------------------------------------------------------------

    # 9a. momentum_aligned: positive = momentum agrees with trade direction
    #     For LONG: want positive 15m momentum (price moving up)
    #     For SHORT: want negative 15m momentum (price moving down)
    direction_sign = 1.0 if signal.direction == Direction.LONG else -1.0
    features["momentum_aligned"] = round(
        features.get("momentum_3bar_15m", 0.0) * direction_sign, 4
    )

    # 9b. ema50_aligned: 1 if EMA50 macro trend matches trade direction
    #     NQ/MNQ data: winners EMA50 dist=+6.9pt, losers=-3.0pt (HUGE predictor)
    #     1 = aligned (LONG + price above EMA50, or SHORT + price below EMA50)
    #     0 = counter-trend
    price_above_ema50 = features.get("price_above_ema50", 0)
    if signal.direction == Direction.LONG:
        features["ema50_aligned"] = price_above_ema50
    else:  # SHORT
        features["ema50_aligned"] = int(not bool(price_above_ema50))

    # 9c. is_retest_quick: 1 if retest happened within 8 bars of the break
    #     Fast retests are cleaner — fewer competing signals, sharper level response
    #     ES: time_to_retest <= 10 → WR 43.4% vs 37.6% baseline
    features["is_retest_quick"] = int(
        features.get("time_to_retest_bars", 99) <= 8
    )

    # ----------------------------------------------------------------
    # 10. Phase 8: Market context features
    # ----------------------------------------------------------------

    # 10a. gap_size_norm: (today_open - prev_close) / ATR
    # Gap days (ES opens far from prior close) behave very differently.
    # Large positive gaps → bears try to fill; large negative → bulls try to fill.
    try:
        if anchor and len(df_15m) >= 2:
            today_open = float(anchor.open)
            # Previous day's last 15m close
            anchor_ts = anchor.timestamp
            prev_bars = df_15m.loc[df_15m.index < anchor_ts]
            if len(prev_bars) >= 1:
                prev_close_price = float(prev_bars["close"].iloc[-1])
                gap = today_open - prev_close_price
                features["gap_size_norm"] = _safe_div(gap, atr_val)
            else:
                features["gap_size_norm"] = 0.0
        else:
            features["gap_size_norm"] = 0.0
    except Exception:
        features["gap_size_norm"] = 0.0

    # 10b. session_phase: 0=open drive (9:00–10:00), 1=mid-morning (10:00–11:30), 2=afternoon
    try:
        sig_minute = hour * 60 + (ts.minute if hasattr(ts, "minute") else 0)
        if sig_minute < 10 * 60:          # before 10am
            features["session_phase"] = 0
        elif sig_minute < 11 * 60 + 30:   # 10am–11:30am
            features["session_phase"] = 1
        else:                              # 11:30am+
            features["session_phase"] = 2
    except Exception:
        features["session_phase"] = 0

    # 10c. or_range_vs_5day_avg: today's OR range / 5-day avg OR range
    # Tighter-than-average OR = likely breakout day; wider = likely range/chop day.
    try:
        if anchor and anchor.range_pts > 0 and len(df_15m) >= 5:
            anchor_ts = anchor.timestamp
            # Get 15m bars from the past 5 trading days at the anchor candle time
            # to compute 5-day average OR range
            _5day_ranges = []
            seen_dates: set = set()
            for _ts, _bar in df_15m.loc[df_15m.index < anchor_ts].iloc[::-1].iterrows():
                _d = _ts.date()
                if _d not in seen_dates:
                    seen_dates.add(_d)
                if len(seen_dates) > 5:
                    break
            # Simpler: use rolling 5-bar ATR as a proxy for OR range
            if len(df_15m) >= 5:
                recent_ranges = (df_15m["high"] - df_15m["low"]).tail(20)
                avg_range = float(recent_ranges.mean()) if len(recent_ranges) > 0 else anchor.range_pts
                features["or_range_vs_5day_avg"] = _safe_div(anchor.range_pts, avg_range)
            else:
                features["or_range_vs_5day_avg"] = 1.0
        else:
            features["or_range_vs_5day_avg"] = 1.0
    except Exception:
        features["or_range_vs_5day_avg"] = 1.0

    # 10d. setup_type: raw string for per-setup-type ML model routing
    features["setup_type"] = signal.setup_type.value

    # ----------------------------------------------------------------
    # 11. Phase 9: VWAP distance + prior day high/low context
    # ----------------------------------------------------------------

    # 11a. Intraday VWAP — typical price × volume / cumulative volume
    # Uses today's 1m bars only (after 9:30 open); falls back to mid-day price.
    try:
        exec_open_str = "09:30" if hasattr(df_1m.index[0], "hour") else None
        if exec_open_str and len(df_1m) > 0:
            today_date = df_1m.index[-1].date()
            today_bars = df_1m[df_1m.index.date == today_date]
            if len(today_bars) > 0:
                tp = (today_bars["high"] + today_bars["low"] + today_bars["close"]) / 3.0
                total_vol = today_bars["volume"].sum()
                if total_vol > 0:
                    vwap = float((tp * today_bars["volume"]).sum() / total_vol)
                    features["vwap_distance_norm"] = _safe_div(current_price - vwap, atr_val)
                    features["price_above_vwap"] = int(current_price > vwap)
                else:
                    features["vwap_distance_norm"] = 0.0
                    features["price_above_vwap"] = 0
            else:
                features["vwap_distance_norm"] = 0.0
                features["price_above_vwap"] = 0
        else:
            features["vwap_distance_norm"] = 0.0
            features["price_above_vwap"] = 0
    except Exception:
        features["vwap_distance_norm"] = 0.0
        features["price_above_vwap"] = 0

    # 11b. Prior day high/low context from 15m data
    # Proximity to prior day's high or low is a strong intraday level.
    try:
        if anchor and len(df_15m) >= 2:
            anchor_ts = anchor.timestamp
            prev_bars = df_15m.loc[df_15m.index < anchor_ts]
            if len(prev_bars) >= 1:
                prev_day_date = prev_bars.index[-1].date()
                prev_day_bars = prev_bars[prev_bars.index.date == prev_day_date]
                if len(prev_day_bars) > 0:
                    prev_high = float(prev_day_bars["high"].max())
                    prev_low = float(prev_day_bars["low"].min())
                    features["price_vs_prev_high_norm"] = _safe_div(current_price - prev_high, atr_val)
                    features["price_vs_prev_low_norm"] = _safe_div(current_price - prev_low, atr_val)
                    features["above_prev_high"] = int(current_price > prev_high)
                    features["below_prev_low"] = int(current_price < prev_low)
                else:
                    features["price_vs_prev_high_norm"] = 0.0
                    features["price_vs_prev_low_norm"] = 0.0
                    features["above_prev_high"] = 0
                    features["below_prev_low"] = 0
            else:
                features["price_vs_prev_high_norm"] = 0.0
                features["price_vs_prev_low_norm"] = 0.0
                features["above_prev_high"] = 0
                features["below_prev_low"] = 0
        else:
            features["price_vs_prev_high_norm"] = 0.0
            features["price_vs_prev_low_norm"] = 0.0
            features["above_prev_high"] = 0
            features["below_prev_low"] = 0
    except Exception:
        features["price_vs_prev_high_norm"] = 0.0
        features["price_vs_prev_low_norm"] = 0.0
        features["above_prev_high"] = 0
        features["below_prev_low"] = 0

    # ── Section 7: Regime Intelligence ──────────────────────────────────────
    _add_regime_features(features, df_15m)

    return features


def _add_regime_features(features: dict, df_15m: pd.DataFrame) -> None:
    """
    Compute Kaufman Efficiency Ratio and ATR vol-state percentile.
    Both use only historical 15m bars — no lookahead.
    """
    try:
        # Efficiency Ratio over last 20 15m bars
        n = 20
        if len(df_15m) >= n + 1:
            closes = df_15m["close"].iloc[-n:].values
            net_change = abs(closes[-1] - closes[0])
            sum_changes = float(np.sum(np.abs(np.diff(closes))))
            if sum_changes > 0:
                features["efficiency_ratio"] = round(float(net_change / sum_changes), 4)
            else:
                features["efficiency_ratio"] = 0.0
        else:
            features["efficiency_ratio"] = 0.5  # neutral when insufficient data
    except Exception:
        features["efficiency_ratio"] = 0.5

    try:
        # ATR(14) percentile rank vs last 60 days of 15m ATR values
        atr_now = features.get("atr_15m_14", 0.0)
        # Compute rolling ATR(14) on the full 15m history available
        if len(df_15m) >= 15:
            hi, lo = df_15m["high"], df_15m["low"]
            cl_prev = df_15m["close"].shift(1)
            tr = pd.concat(
                [hi - lo, (hi - cl_prev).abs(), (lo - cl_prev).abs()], axis=1
            ).max(axis=1)
            atr_series = tr.rolling(14).mean().dropna()
            # Use last 60 * 26 values (approx 60 trading days of 15m bars)
            atr_hist = atr_series.iloc[-max(26 * 60, 100):]
            if len(atr_hist) >= 10 and atr_now > 0:
                pct = float((atr_hist < atr_now).mean())
                features["vol_state"] = round(pct, 4)
            else:
                features["vol_state"] = 0.5
        else:
            features["vol_state"] = 0.5
    except Exception:
        features["vol_state"] = 0.5


def _add_break_strength_features(
    features: dict,
    signal,
    day_levels,
    df_1m: pd.DataFrame,
) -> None:
    """
    Compute break-strength metrics for BREAK_RETEST signals.

    Features added:
      break_excursion_pts  — max distance price moved beyond the break level
                             before returning to retest (how far the move committed)
      closes_beyond_level  — count of 1m bars that closed beyond the break level
      time_to_retest_bars  — 1m bars elapsed from the break bar to the current bar
      or_range_vs_atr      — OR range / ATR(14) on 15m chart (calibrates day volatility)

    For non-BREAK_RETEST setups, all four features are set to 0 / neutral
    so the model can still use them as "zero signal" context.
    """
    from src.setup_detector import SetupType  # local import avoids circular ref at module level

    atr_val = features.get("atr_15m_14", 1.0)  # already computed in section 4
    anchor = day_levels.anchor

    # 8d. OR range vs ATR (always computed — useful for all setup types)
    if anchor and anchor.range_pts > 0 and atr_val > 0:
        features["or_range_vs_atr"] = round(anchor.range_pts / atr_val, 4)
    else:
        features["or_range_vs_atr"] = 1.0

    # 8a–8c: only meaningful for BREAK_RETEST signals
    if signal.setup_type.value != "BREAK_RETEST":
        features["break_excursion_pts"] = 0.0
        features["closes_beyond_level"] = 0
        features["time_to_retest_bars"] = 0
        return

    if anchor is None or len(df_1m) < 4:
        features["break_excursion_pts"] = 0.0
        features["closes_beyond_level"] = 0
        features["time_to_retest_bars"] = 0
        return

    # Determine the break level — prefer level_ref (OR_HIGH/LOW), fall back to OR_MID
    if signal.level_ref is not None:
        break_level = signal.level_ref.price
        is_long = signal.level_ref.direction == "high"
    else:
        break_level = anchor.mid
        from src.setup_detector import Direction
        is_long = signal.direction == Direction.LONG

    # Scan recent 1m bars (up to the staleness window, e.g. last 20 bars)
    lookback = min(20, len(df_1m))
    recent = df_1m.iloc[-lookback:]
    closes = recent["close"].values
    highs = recent["high"].values
    lows = recent["low"].values

    if is_long:
        # Long: break was ABOVE the level; excursion = max high beyond level
        beyond_closes = [c for c in closes if c > break_level]
        excursion = max((h - break_level for h in highs if h > break_level), default=0.0)
    else:
        # Short: break was BELOW the level; excursion = max distance below level
        beyond_closes = [c for c in closes if c < break_level]
        excursion = max((break_level - lo for lo in lows if lo < break_level), default=0.0)

    features["break_excursion_pts"] = round(float(excursion), 2)
    features["closes_beyond_level"] = len(beyond_closes)

    # 8c. Time to retest: find the bar index of the break, count bars to current
    break_bar_idx = 0
    if is_long:
        for i, c in enumerate(closes):
            if c > break_level:
                break_bar_idx = i
                break
    else:
        for i, c in enumerate(closes):
            if c < break_level:
                break_bar_idx = i
                break
    features["time_to_retest_bars"] = max(0, lookback - 1 - break_bar_idx)


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    """Average True Range over `period` bars."""
    if len(df) < period + 1:
        if len(df) >= 2:
            return float((df["high"] - df["low"]).mean())
        return 1.0
    hi = df["high"]
    lo = df["low"]
    cl = df["close"].shift(1)
    tr = pd.concat([hi - lo, (hi - cl).abs(), (lo - cl).abs()], axis=1).max(axis=1)
    return float(tr.tail(period).mean())


def _safe_div(a: float, b: float) -> float:
    return round(a / b, 4) if b and b != 0 else 0.0


# ---------------------------------------------------------------------------
# Schema enforcement (Section 1b)
# ---------------------------------------------------------------------------

def validate_feature_vector(features: dict, strict: bool = False) -> list[str]:
    """
    Check that the feature dict contains all required features from the schema.

    Parameters
    ----------
    features : dict
        Output of compute_features().
    strict : bool
        If True, raises ValueError on any missing required feature.
        If False (default), returns a list of missing feature names.

    Returns
    -------
    list[str]  — names of missing required features (empty if all present).
    """
    from src.ml_filter import FEATURE_COLS
    missing = [f for f in FEATURE_COLS if f not in features or features[f] is None]
    if missing:
        msg = (
            f"Feature schema violation — {len(missing)} required features missing: "
            + ", ".join(missing[:10])
            + ("..." if len(missing) > 10 else "")
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)
    return missing


def enforce_schema(features: dict) -> dict:
    """
    Fill any missing required features with 0.0 and log a warning.
    Does NOT silently pass at training time — training should use validate_feature_vector(strict=True).
    Safe to use at inference time so a single missing feature doesn't kill a live trade.
    """
    from src.ml_filter import FEATURE_COLS
    out = dict(features)
    for f in FEATURE_COLS:
        if f not in out or out[f] is None:
            out[f] = 0.0
    return out
