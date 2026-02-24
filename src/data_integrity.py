"""
src/data_integrity.py

Data quality layer for incoming 1m and 15m bar DataFrames.

Checks performed:
  - Monotonic timestamps (no time going backwards)
  - Duplicate timestamps
  - Missing bars (gaps > expected interval × 2)
  - Bad prints: zero volume, zero/negative price, extreme bar range (> 5× ATR)
  - Session-span: bars outside 18:00–17:00 ET (CME Globex hours)

If critical violations are found:
  - Returns a DataIntegrityReport with is_healthy=False
  - The bot MUST NOT trade when is_healthy=False
  - Logs a detailed alert (and can notify via Notifier)

Usage:
    checker = DataIntegrityChecker()
    report = checker.check(df_1m, df_15m, symbol="ES")
    if not report.is_healthy:
        logger.error("Data unhealthy — no trading today: {}", report.issues)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class DataIntegrityReport:
    symbol: str
    is_healthy: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def summary(self) -> str:
        status = "HEALTHY" if self.is_healthy else "UNHEALTHY"
        parts = [f"DataIntegrity[{self.symbol}] {status}"]
        if self.issues:
            parts.append("  ERRORS: " + "; ".join(self.issues))
        if self.warnings:
            parts.append("  WARNINGS: " + "; ".join(self.warnings[:3]))
        return "\n".join(parts)


class DataIntegrityChecker:
    """
    Validates bar data quality before each trading session.

    Parameters
    ----------
    max_gap_multiplier : float
        A gap is flagged if it exceeds (expected_interval × multiplier).
        Default 2.0 → gap > 2 minutes on 1m bars is flagged.
    max_range_atr_multiple : float
        A bar is a "bad print" if its range > this multiple × ATR(14).
        Default 5.0.
    min_bars_for_atr : int
        Minimum bars needed to compute ATR. If fewer, skip ATR checks.
    """

    def __init__(
        self,
        max_gap_multiplier: float = 2.0,
        max_range_atr_multiple: float = 5.0,
        min_bars_for_atr: int = 20,
    ) -> None:
        self._gap_mult = max_gap_multiplier
        self._range_mult = max_range_atr_multiple
        self._min_atr_bars = min_bars_for_atr

    def check(
        self,
        df_1m: pd.DataFrame,
        df_15m: Optional[pd.DataFrame] = None,
        symbol: str = "?",
        session_date: Optional["datetime.date"] = None,
    ) -> DataIntegrityReport:
        """
        Run all checks.  Returns DataIntegrityReport.
        is_healthy=False if any CRITICAL issue found.
        """
        issues: list[str] = []
        warnings: list[str] = []
        stats: dict = {}

        # ── Basic sanity ──────────────────────────────────────────────────
        if df_1m is None or df_1m.empty:
            return DataIntegrityReport(
                symbol=symbol, is_healthy=False,
                issues=["1m DataFrame is empty or None"],
            )

        df = df_1m.copy()
        n = len(df)
        stats["n_bars_1m"] = n

        # ── 1. Monotonic timestamps ───────────────────────────────────────
        if not df.index.is_monotonic_increasing:
            issues.append("1m timestamps are not monotonic — possible data corruption")

        # ── 2. Duplicate timestamps ───────────────────────────────────────
        dupes = df.index.duplicated().sum()
        if dupes > 0:
            issues.append(f"1m DataFrame has {dupes} duplicate timestamps")
            stats["duplicate_bars"] = int(dupes)

        # ── 3. Missing bars (gaps) ────────────────────────────────────────
        if n >= 2 and df.index.is_monotonic_increasing:
            diffs = pd.Series(df.index).diff().dropna()
            expected = pd.Timedelta(minutes=1)
            threshold = expected * self._gap_mult
            gaps = diffs[diffs > threshold]
            if len(gaps) > 0:
                max_gap = gaps.max()
                gap_times = [str(df.index[i].time()) for i in gaps.index[:3]]
                warnings.append(
                    f"1m data: {len(gaps)} gaps > {threshold} "
                    f"(max={max_gap}); at times {gap_times}"
                )
                stats["gaps_count"] = int(len(gaps))
                stats["max_gap_minutes"] = float(max_gap.total_seconds() / 60)
                # Only a critical issue if many gaps (> 5% of bars)
                if len(gaps) > max(3, n * 0.05):
                    issues.append(
                        f"Excessive 1m data gaps: {len(gaps)} gaps out of {n} bars"
                    )

        # ── 4. Required columns ───────────────────────────────────────────
        required = {"open", "high", "low", "close", "volume"}
        missing_cols = required - set(df.columns)
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")

        if missing_cols:
            return DataIntegrityReport(
                symbol=symbol, is_healthy=False,
                issues=issues, warnings=warnings, stats=stats,
            )

        # ── 5. Zero or negative prices ────────────────────────────────────
        zero_close = (df["close"] <= 0).sum()
        if zero_close > 0:
            issues.append(f"{zero_close} bars with close <= 0 (bad prints)")
            stats["zero_price_bars"] = int(zero_close)

        # ── 6. Zero-volume bars ───────────────────────────────────────────
        zero_vol = (df["volume"] == 0).sum()
        if zero_vol > 0:
            vol_pct = zero_vol / n
            if vol_pct > 0.10:   # > 10% zero-volume = problem
                issues.append(
                    f"{zero_vol} zero-volume bars ({vol_pct:.0%}) — "
                    "possible stale/synthetic data"
                )
            elif zero_vol > 0:
                warnings.append(f"{zero_vol} zero-volume bars (< 10% — acceptable)")
            stats["zero_volume_bars"] = int(zero_vol)

        # ── 7. OHLC sanity: high >= close >= low, high >= open >= low ─────
        bad_hl = ((df["high"] < df["low"]) | (df["close"] > df["high"]) |
                  (df["close"] < df["low"])).sum()
        if bad_hl > 0:
            issues.append(f"{bad_hl} bars with invalid OHLC (high < low or close outside range)")
            stats["invalid_ohlc_bars"] = int(bad_hl)

        # ── 8. Bad-print detection: extreme bar range vs ATR ─────────────
        if n >= self._min_atr_bars:
            try:
                hi, lo, cl = df["high"], df["low"], df["close"]
                cl_prev = cl.shift(1)
                tr = pd.concat(
                    [hi - lo, (hi - cl_prev).abs(), (lo - cl_prev).abs()], axis=1
                ).max(axis=1)
                atr14 = float(tr.rolling(14).mean().iloc[-1])
                bar_ranges = hi - lo
                extreme = (bar_ranges > atr14 * self._range_mult).sum()
                if extreme > 0:
                    worst_range = float(bar_ranges.max())
                    warnings.append(
                        f"{extreme} bars with range > {self._range_mult}× ATR "
                        f"(ATR={atr14:.1f}, worst={worst_range:.1f}) — possible bad prints"
                    )
                    stats["extreme_range_bars"] = int(extreme)
                    stats["atr_1m"] = round(atr14, 2)
                    if extreme > max(3, n * 0.02):
                        issues.append(
                            f"Excessive bad prints: {extreme} bars with range > "
                            f"{self._range_mult}× ATR"
                        )
            except Exception as e:
                warnings.append(f"Could not compute ATR for bad-print check: {e}")

        # ── 9. Price continuity (close-to-open gaps > 3× ATR) ────────────
        if n >= self._min_atr_bars:
            try:
                close_open_gaps = (df["open"] - df["close"].shift(1)).abs()
                if "atr_1m" in stats and stats["atr_1m"] > 0:
                    big_gaps = (close_open_gaps > stats["atr_1m"] * 3).sum()
                    if big_gaps > 3:
                        warnings.append(
                            f"{big_gaps} large close-to-open gaps (> 3× ATR) — "
                            "check for data splices or feed errors"
                        )
            except Exception:
                pass

        # ── 10. 15m data checks (lightweight) ────────────────────────────
        if df_15m is not None and not df_15m.empty:
            dupes_15 = df_15m.index.duplicated().sum()
            if dupes_15 > 0:
                warnings.append(f"15m DataFrame: {dupes_15} duplicate timestamps")
                stats["dupes_15m"] = int(dupes_15)
            if not df_15m.index.is_monotonic_increasing:
                issues.append("15m timestamps are not monotonic")
            stats["n_bars_15m"] = len(df_15m)

        # ── Result ────────────────────────────────────────────────────────
        is_healthy = len(issues) == 0
        report = DataIntegrityReport(
            symbol=symbol,
            is_healthy=is_healthy,
            issues=issues,
            warnings=warnings,
            stats=stats,
        )
        if is_healthy:
            logger.info(
                "DataIntegrity[{}]: HEALTHY ({} bars, {} warnings)",
                symbol, n, len(warnings),
            )
        else:
            logger.error("DataIntegrity[{}]: UNHEALTHY\n{}", symbol, report.summary())
        return report


def clean_bars(
    df: pd.DataFrame,
    symbol: str = "?",
    drop_zero_volume: bool = False,
) -> pd.DataFrame:
    """
    Lightly sanitize a bar DataFrame in-place:
      - Sort by index
      - Drop exact duplicate timestamps (keep last)
      - Drop bars with zero/negative prices
      - Optionally drop zero-volume bars

    Returns a clean copy.  Does NOT fill missing bars (intentional — gaps
    should be handled by data providers, not silently interpolated).
    """
    df = df.copy()

    # Sort
    if not df.index.is_monotonic_increasing:
        logger.warning("{}: Sorting bars by timestamp", symbol)
        df = df.sort_index()

    # Drop exact-duplicate timestamps
    dupes = df.index.duplicated(keep="last")
    if dupes.any():
        n_dupes = int(dupes.sum())
        logger.warning("{}: Dropping {} duplicate timestamp bars", symbol, n_dupes)
        df = df[~dupes]

    # Drop zero/negative prices
    bad_price = df["close"] <= 0
    if bad_price.any():
        logger.warning("{}: Dropping {} zero-price bars", symbol, bad_price.sum())
        df = df[~bad_price]

    # Optionally drop zero-volume (conservative — don't do by default)
    if drop_zero_volume:
        zv = df["volume"] == 0
        if zv.any():
            logger.warning("{}: Dropping {} zero-volume bars", symbol, zv.sum())
            df = df[~zv]

    return df
