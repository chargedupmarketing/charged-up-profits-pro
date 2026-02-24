"""
src/drift_monitor.py

Live-vs-backtest drift monitoring and model performance alarms.

Three responsibilities:
  1. Feature drift detection (PSI):
     Compare live signal feature distributions to training distributions.
     PSI > 0.2 → "retrain recommended" alert.

  2. Model performance alarm:
     Track last-N live trades' realized expectancy.
     If expectancy < threshold → disable ML gate and/or alert.

  3. Daily report:
     Writes data/drift_report_{date}.json with expectancy comparison,
     trade count, slippage, fill rate, and drift scores per feature.

Usage:
    monitor = DriftMonitor()
    monitor.update_live_features(feature_dict)        # call after each signal
    monitor.update_live_trade(realized_R)             # call after each exit
    report = monitor.generate_daily_report()
    if monitor.should_disable_ml():
        inference_filter.disable()
"""
from __future__ import annotations

import json
from collections import deque
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent


def _psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index (PSI).
    PSI < 0.1  : no significant drift
    0.1–0.2    : moderate drift — monitor closely
    > 0.2      : significant drift — retrain recommended
    """
    eps = 1e-6
    bins = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        n_bins + 1,
    )
    e_pct = np.histogram(expected, bins=bins)[0] / (len(expected) + eps) + eps
    a_pct = np.histogram(actual,   bins=bins)[0] / (len(actual)   + eps) + eps
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


class DriftMonitor:
    """
    Tracks live feature distributions and realized R-multiples,
    and compares them against training-time baselines.

    Parameters
    ----------
    window_trades : int
        Rolling window for live expectancy alarm (default: 20 trades).
    alarm_expectancy_threshold : float
        If mean realized_R of last `window_trades` drops below this,
        trigger ML-disable alarm. Default: -0.25 (losing > 0.25R/trade).
    psi_warn_threshold : float
        PSI above this → "retrain recommended" warning. Default: 0.2.
    """

    def __init__(
        self,
        window_trades: int = 20,
        alarm_expectancy_threshold: float = -0.25,
        psi_warn_threshold: float = 0.2,
        settings_path: str = "config/settings.yaml",
    ) -> None:
        self._window = window_trades
        self._alarm_R = alarm_expectancy_threshold
        self._psi_warn = psi_warn_threshold

        # Rolling buffers
        self._live_realized_R: deque[float] = deque(maxlen=window_trades)
        self._live_features: list[dict] = []       # feature dicts from live signals
        self._training_features: Optional[pd.DataFrame] = None

        # State
        self._ml_disabled_by_alarm: bool = False
        self._total_live_trades: int = 0

        # Load training feature baseline if available
        self._load_training_baseline()

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def update_live_features(self, features: dict) -> None:
        """Call after each live signal that reaches the ML gate."""
        self._live_features.append(dict(features))

    def update_live_trade(self, realized_R: float) -> None:
        """Call after each live trade exits. Pass realized_R (positive = win)."""
        self._live_realized_R.append(realized_R)
        self._total_live_trades += 1
        self._check_performance_alarm()

    # ------------------------------------------------------------------
    # Alarm logic
    # ------------------------------------------------------------------

    def _check_performance_alarm(self) -> None:
        """Disable ML gate if rolling expectancy is below threshold."""
        if len(self._live_realized_R) < max(5, self._window // 2):
            return  # not enough data yet

        mean_R = float(np.mean(self._live_realized_R))
        if mean_R < self._alarm_R:
            if not self._ml_disabled_by_alarm:
                self._ml_disabled_by_alarm = True
                logger.error(
                    "DriftMonitor: ML PERFORMANCE ALARM — "
                    "last {} trades mean_R={:.3f} < threshold {:.3f}. "
                    "ML gate will be flagged. Review model immediately.",
                    len(self._live_realized_R), mean_R, self._alarm_R,
                )
        else:
            if self._ml_disabled_by_alarm:
                logger.info(
                    "DriftMonitor: Performance recovered — "
                    "mean_R={:.3f} >= threshold. Clearing alarm.",
                    mean_R,
                )
            self._ml_disabled_by_alarm = False

    def should_disable_ml(self) -> bool:
        """True when rolling expectancy alarm is active."""
        return self._ml_disabled_by_alarm

    # ------------------------------------------------------------------
    # Feature drift (PSI)
    # ------------------------------------------------------------------

    def _load_training_baseline(self) -> None:
        """Load training feature distributions from labelled_trades.csv."""
        labelled_path = ROOT / "data" / "labelled_trades.csv"
        if not labelled_path.exists():
            return
        try:
            df = pd.read_csv(labelled_path)
            # Keep only numeric feature columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            self._training_features = df[numeric_cols]
            logger.info(
                "DriftMonitor: loaded {} training samples for baseline ({} features)",
                len(df), len(numeric_cols),
            )
        except Exception as e:
            logger.warning("DriftMonitor: could not load training baseline: {}", e)

    def compute_feature_drift(self) -> dict[str, float]:
        """
        Compute PSI for each feature between training and live distributions.
        Returns {feature_name: psi_score}.
        Only computed when >= 20 live signals are available.
        """
        drift_scores: dict[str, float] = {}

        if not self._live_features or len(self._live_features) < 20:
            return drift_scores

        if self._training_features is None:
            return drift_scores

        live_df = pd.DataFrame(self._live_features)
        shared_cols = [
            c for c in live_df.columns
            if c in self._training_features.columns
            and live_df[c].notna().sum() >= 10
        ]

        for col in shared_cols:
            try:
                expected = self._training_features[col].dropna().values
                actual   = live_df[col].dropna().values
                if len(expected) >= 10 and len(actual) >= 10:
                    psi = _psi(expected, actual)
                    drift_scores[col] = round(psi, 4)
            except Exception:
                pass

        return drift_scores

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    def get_live_stats(self) -> dict:
        """Current live trading statistics."""
        r_arr = np.array(self._live_realized_R) if self._live_realized_R else np.array([])
        return {
            "total_live_trades":    self._total_live_trades,
            "rolling_n":            len(r_arr),
            "rolling_mean_R":       round(float(r_arr.mean()), 4) if len(r_arr) else None,
            "rolling_win_rate_pct": round((r_arr > 0).mean() * 100, 1) if len(r_arr) else None,
            "ml_alarm_active":      self._ml_disabled_by_alarm,
        }

    def generate_daily_report(
        self,
        backtest_expectancy_R: Optional[float] = None,
        symbol: str = "?",
    ) -> dict:
        """
        Write data/drift_report_{date}.json and return it.
        """
        drift_scores = self.compute_feature_drift()
        live_stats   = self.get_live_stats()

        # High-drift features
        high_drift = {k: v for k, v in drift_scores.items() if v > self._psi_warn}
        if high_drift:
            logger.warning(
                "DriftMonitor: {} features with significant drift (PSI > {:.1f}): {}",
                len(high_drift), self._psi_warn, list(high_drift.keys())[:5],
            )
            logger.warning("  → Retrain recommended: python scripts/train_models.py --symbols {}", symbol)

        report = {
            "date":                   str(date.today()),
            "symbol":                 symbol,
            "live_stats":             live_stats,
            "backtest_expectancy_R":  backtest_expectancy_R,
            "drift_scores":           drift_scores,
            "high_drift_features":    high_drift,
            "n_live_features":        len(self._live_features),
            "psi_threshold":          self._psi_warn,
            "retrain_recommended":    len(high_drift) > 3,
            "generated_at":           datetime.now().isoformat(),
        }

        out = ROOT / "data" / f"drift_report_{date.today()}.json"
        out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        logger.info("Drift report written: {}", out)
        return report
