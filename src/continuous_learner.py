"""
src/continuous_learner.py

Autonomous re-learning system that runs during market downtime.

Three trigger conditions (all checked every 5 minutes):
  1. Post-session:      market closed (after 4:15 PM ET on trading days)
  2. High-risk pause:   FOMC / CPI / NFP day detected (retrain before next session)
  3. Accumulation:      >= ACCUMULATION_THRESHOLD new trades since last retrain

For each symbol+setup_type pair, the learner:
  1. Checks if enough new data exists (accumulation threshold)
  2. Loads all available backtest + live trades from the audit DB
  3. Labels trades with TripleBarrierLabeler
  4. Trains a candidate XGBoost + LightGBM ensemble model
  5. Runs the 6-gate safety battery (ModelLifecycle)
  6. If all gates pass: saves model to pending/ and updates manifest.json
  7. Dashboard shows a comparison card; user manually approves

Anti-overfit safeguards (in addition to the 6 gates):
  - max_depth=4 hard limit on all models
  - n_estimators capped at 300
  - Holdout is always chronologically most recent 20% (never random)
  - Minimum class balance check (no training if >85% one class)
  - Feature count is capped: only features present in >= 10% of samples
  - Exponential decay weights (recent trades count more than old ones)

Run as a background subprocess (launched from dashboard):
    python src/continuous_learner.py --symbols ES NQ

The process checks triggers every 5 minutes and retrains when needed.
It writes status to data/learner_state.json for the dashboard to read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_ET = ZoneInfo("America/New_York")

LEARNER_STATE_FILE = ROOT / "data" / "learner_state.json"
MANIFEST_PATH = ROOT / "data" / "models" / "manifest.json"

# Training thresholds
ACCUMULATION_THRESHOLD = 25     # new trades since last retrain that trigger a cycle
MIN_TOTAL_TRADES = 50           # absolute minimum to attempt training
MAX_CLASS_IMBALANCE = 0.85      # if > 85% of labels are one class, skip training
LOOP_INTERVAL_SECONDS = 300     # check triggers every 5 minutes

SETUP_TYPES = ["BREAK_RETEST", "REJECTION", "BOUNCE", "SWEEP_REVERSE"]


# ---------------------------------------------------------------------------
# State file helpers
# ---------------------------------------------------------------------------

def _read_state() -> dict:
    if LEARNER_STATE_FILE.exists():
        try:
            return json.loads(LEARNER_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"last_retrain": {}, "last_trade_count": {}, "running": False, "log": []}


def _write_state(state: dict) -> None:
    LEARNER_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    LEARNER_STATE_FILE.write_text(
        json.dumps(state, indent=2, default=str), encoding="utf-8"
    )


def _log_state(state: dict, message: str, level: str = "INFO") -> None:
    """Append a log line to the state file (dashboard reads it)."""
    entry = {"ts": datetime.utcnow().isoformat(), "level": level, "msg": message}
    state.setdefault("log", [])
    state["log"].append(entry)
    state["log"] = state["log"][-200:]   # Keep last 200 log lines


# ---------------------------------------------------------------------------
# Trigger detection
# ---------------------------------------------------------------------------

def _is_market_closed() -> bool:
    """Returns True if we're outside US Eastern market hours."""
    now = datetime.now(tz=_ET)
    # Weekend
    if now.weekday() >= 5:
        return True
    # Before 4am or after 4:15pm ET
    minutes = now.hour * 60 + now.minute
    if minutes < 4 * 60 or minutes >= 16 * 60 + 15:
        return True
    return False


def _is_news_day() -> bool:
    """Returns True if today is a high-impact news day (FOMC/CPI/NFP)."""
    news_file = ROOT / "data" / "high_impact_dates.csv"
    if not news_file.exists():
        return False
    try:
        df = pd.read_csv(news_file)
        today = str(datetime.now(tz=_ET).date())
        if "date" in df.columns:
            return today in df["date"].astype(str).values
    except Exception:
        pass
    return False


def _count_new_trades_since(symbol: str, last_retrain_ts: str | None) -> int:
    """
    Count trades in the audit DB for this symbol since last_retrain_ts.
    Looks at backtest CSV files and live trade logs.
    """
    count = 0

    # Count from backtest CSVs
    data_dir = ROOT / "data"
    for csv_path in data_dir.glob(f"backtest_{symbol}_*.csv"):
        try:
            df = pd.read_csv(csv_path)
            if last_retrain_ts and "entry_ts" in df.columns:
                df["entry_ts"] = pd.to_datetime(df["entry_ts"], errors="coerce")
                cutoff = pd.Timestamp(last_retrain_ts)
                count += (df["entry_ts"] > cutoff).sum()
            else:
                count += len(df)
        except Exception:
            pass

    # Count from audit DB via AuditLogger
    try:
        from monitoring.audit_log import AuditLogger
        audit = AuditLogger()
        recent = audit.get_recent_trades(days=90)
        df_live = pd.DataFrame(recent)
        if not df_live.empty and "date" in df_live.columns:
            # Filter for this symbol if there's a symbol column
            if last_retrain_ts:
                df_live["entry_ts"] = pd.to_datetime(
                    df_live.get("entry_ts", df_live.get("date", "")), errors="coerce"
                )
                cutoff = pd.Timestamp(last_retrain_ts)
                count += (df_live["entry_ts"] > cutoff).sum()
    except Exception:
        pass

    return int(count)


# ---------------------------------------------------------------------------
# Data loading for training
# ---------------------------------------------------------------------------

def _load_all_trades(symbol: str) -> pd.DataFrame:
    """
    Load all available trade data for a symbol from:
      - Backtest CSV files (data/backtest_{SYMBOL}_*.csv)
      - Walk-forward CSV files (data/wfa_trades_{SYMBOL}.csv)
      - Audit DB recent trades
    Merges and deduplicates by (date, setup_type, entry_price) so that
    overlapping backtest files (e.g. 2020-2025, 2022-2025, 2022-2024)
    don't count the same trades multiple times.
    """
    dfs = []
    data_dir = ROOT / "data"

    for pattern in (f"backtest_{symbol}_*.csv", f"wfa_trades_{symbol}.csv"):
        for p in sorted(data_dir.glob(pattern)):
            try:
                df = pd.read_csv(p)
                if len(df) > 0:
                    dfs.append(df)
                    logger.debug("Loaded {} rows from {}", len(df), p.name)
            except Exception as e:
                logger.warning("Could not load {}: {}", p, e)

    if not dfs:
        logger.warning("No trade CSV files found for {}", symbol)
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    # Deduplicate — overlapping backtest CSVs share the same date ranges
    dedup_cols = [c for c in ("date", "entry_ts", "setup_type", "entry_price")
                  if c in combined.columns]
    if dedup_cols:
        before = len(combined)
        combined = combined.drop_duplicates(subset=dedup_cols).reset_index(drop=True)
        dupes = before - len(combined)
        if dupes:
            logger.info("Dropped {} duplicate rows from overlapping CSVs (kept {})", dupes, len(combined))

    return combined


# ---------------------------------------------------------------------------
# Anti-overfit checks
# ---------------------------------------------------------------------------

def _check_class_balance(y: pd.Series) -> tuple[bool, str]:
    """Returns (ok, reason). Fails if one class > MAX_CLASS_IMBALANCE of samples."""
    if len(y) == 0:
        return False, "No labels"
    pos_rate = float(y.mean())
    if pos_rate > MAX_CLASS_IMBALANCE or pos_rate < (1 - MAX_CLASS_IMBALANCE):
        majority = max(pos_rate, 1 - pos_rate)
        return False, (
            f"Class imbalance too severe: {majority:.1%} majority class "
            f"(limit {MAX_CLASS_IMBALANCE:.0%}) — model would just predict majority"
        )
    return True, f"Class balance OK: {pos_rate:.1%} positive"


def _cap_features(X: pd.DataFrame, min_presence: float = 0.10) -> pd.DataFrame:
    """
    Remove features present in fewer than min_presence of samples.
    Prevents the model from latching onto rare/noisy one-hot columns.
    """
    presence = (X != 0).mean()
    keep = presence[presence >= min_presence].index.tolist()
    removed = len(X.columns) - len(keep)
    if removed > 0:
        logger.debug("Feature cap: removed {} sparse features", removed)
    return X[keep]


# ---------------------------------------------------------------------------
# Main ContinuousLearner class
# ---------------------------------------------------------------------------

class ContinuousLearner:
    """
    Autonomous re-learning orchestrator.

    Call run() to start the blocking main loop (runs as a background subprocess).
    The loop checks all three trigger conditions every LOOP_INTERVAL_SECONDS and
    initiates a retrain cycle when any trigger fires.
    """

    def __init__(self, symbols: list[str]) -> None:
        self._symbols = [s.upper() for s in symbols]
        self._running = True
        self._state = _read_state()
        self._state["running"] = True
        self._state["started_at"] = datetime.utcnow().isoformat()
        self._state["symbols"] = self._symbols
        _write_state(self._state)
        logger.info("ContinuousLearner started — symbols={}", self._symbols)

    def run(self) -> None:
        """Main loop. Runs until stopped or process killed."""
        try:
            while self._running:
                self._state = _read_state()
                self._state["running"] = True
                self._state["last_check"] = datetime.utcnow().isoformat()

                if self._should_retrain():
                    self._run_retrain_cycle()

                _write_state(self._state)
                time.sleep(LOOP_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            logger.info("ContinuousLearner stopped by user")
        finally:
            self._state["running"] = False
            _write_state(self._state)

    def _should_retrain(self) -> bool:
        """Check all three trigger conditions."""
        # Trigger 1: market is closed (good time to retrain without using CPU during trading)
        if not _is_market_closed():
            return False   # Never retrain during market hours

        # Trigger 2: news day — retrain so model is fresh for post-news environment
        if _is_news_day():
            _log_state(self._state, "Trigger: news day detected — initiating retrain")
            return True

        # Trigger 3: enough new trades accumulated since last retrain
        for symbol in self._symbols:
            last_retrain = self._state.get("last_retrain", {}).get(symbol)
            new_count = _count_new_trades_since(symbol, last_retrain)
            if new_count >= ACCUMULATION_THRESHOLD:
                _log_state(
                    self._state,
                    f"Trigger: {new_count} new trades for {symbol} (threshold={ACCUMULATION_THRESHOLD})"
                )
                return True

        return False

    def _run_retrain_cycle(self) -> None:
        """
        Full retrain cycle: for each symbol+setup_type, load data, train,
        run gates, and save pending if all gates pass.
        """
        _log_state(self._state, "=== Starting retrain cycle ===")
        cycle_start = datetime.utcnow()

        from src.ml_filter import MLFilter, TripleBarrierLabeler, FEATURE_COLS
        from src.model_lifecycle import ModelLifecycle
        from sklearn.metrics import brier_score_loss

        lifecycle = ModelLifecycle()

        for symbol in self._symbols:
            _log_state(self._state, f"--- Processing {symbol} ---")

            # Load all trade data
            trades_df = _load_all_trades(symbol)
            if trades_df.empty or len(trades_df) < MIN_TOTAL_TRADES:
                _log_state(
                    self._state,
                    f"{symbol}: only {len(trades_df)} trades (need >= {MIN_TOTAL_TRADES}) — skipping",
                    level="WARN"
                )
                continue

            # Label trades
            try:
                labeler = TripleBarrierLabeler()
                labelled = labeler.label_from_backtest_trades(trades_df)
            except Exception as e:
                _log_state(self._state, f"{symbol}: labeling failed ({e})", level="ERROR")
                continue

            # Train per-setup-type models
            for setup_type in SETUP_TYPES:
                self._retrain_one(
                    symbol, setup_type, labelled, lifecycle
                )

            # Update last retrain timestamp
            if "last_retrain" not in self._state:
                self._state["last_retrain"] = {}
            self._state["last_retrain"][symbol] = datetime.utcnow().isoformat()
            _write_state(self._state)

        elapsed = (datetime.utcnow() - cycle_start).total_seconds()
        _log_state(self._state, f"=== Retrain cycle complete in {elapsed:.0f}s ===")

    def _retrain_one(
        self,
        symbol: str,
        setup_type: str,
        labelled: pd.DataFrame,
        lifecycle: "ModelLifecycle",
    ) -> None:
        """
        Train a candidate model for one symbol+setup_type pair and run safety gates.
        """
        from src.ml_filter import MLFilter, FEATURE_COLS
        from src.model_lifecycle import ModelLifecycle

        key = f"{symbol}_{setup_type}"
        _log_state(self._state, f"  Training {key}...")

        try:
            ml = MLFilter(symbol=symbol, setup_type=setup_type)
            X, y = ml.prepare_dataset(labelled)

            if len(X) == 0:
                _log_state(
                    self._state,
                    f"  {key}: no trades recorded for this setup type yet — skipping",
                )
                return

            if len(X) < MIN_TOTAL_TRADES:
                _log_state(
                    self._state,
                    f"  {key}: only {len(X)} samples after filtering — skipping "
                    f"(need {MIN_TOTAL_TRADES})",
                    level="WARN",
                )
                return

            # Anti-overfit: class balance check
            ok, reason = _check_class_balance(y)
            if not ok:
                _log_state(self._state, f"  {key}: {reason} — skipping", level="WARN")
                return

            # Anti-overfit: cap sparse features
            X = _cap_features(X)

            # Train ensemble (XGBoost + LightGBM with decay weights)
            train_metrics = ml.train(X, y)
            candidate_model = ml._model   # The XGBoost component

            _log_state(
                self._state,
                f"  {key}: trained — AUC={train_metrics.get('mean_cv_auc', 0):.3f} "
                f"({len(X)} samples, LGBM={'YES' if train_metrics.get('lgbm_enabled') else 'NO'})"
            )

            # Run safety gate battery
            battery = lifecycle.evaluate_candidate(
                candidate_model, symbol, setup_type, X, y, labelled
            )

            _log_state(
                self._state,
                f"  {key}: gates {'ALL PASSED' if battery.all_passed else 'FAILED'} "
                f"({sum(g.passed for g in battery.gate_results)}/{len(battery.gate_results)})"
            )

            if battery.all_passed:
                # Compute Brier score for comparison metrics
                try:
                    from sklearn.metrics import brier_score_loss
                    n = len(y)
                    cutoff = int(n * 0.8)
                    X_hold, y_hold = X.iloc[cutoff:], y.iloc[cutoff:]
                    if hasattr(candidate_model, "predict_proba") and len(X_hold) > 0:
                        probs = candidate_model.predict_proba(X_hold)[:, 1]
                        train_metrics["brier_score"] = float(brier_score_loss(y_hold, probs))
                        # Win rate simulation
                        mask = probs >= 0.58
                        if mask.sum() > 0:
                            train_metrics["win_rate_sim"] = float(y_hold[mask].mean())
                        train_metrics["auc"] = train_metrics.get("mean_cv_auc", 0.0)
                except Exception:
                    pass

                pending_path = lifecycle.save_pending(
                    candidate_model, symbol, setup_type, battery, train_metrics
                )
                _log_state(
                    self._state,
                    f"  {key}: PENDING approval — saved to {pending_path.name}"
                )
            else:
                failed_gates = [g.gate_name for g in battery.gate_results if not g.passed]
                _log_state(
                    self._state,
                    f"  {key}: NOT saved — failed gates: {', '.join(failed_gates)}",
                    level="WARN"
                )

        except Exception as e:
            _log_state(self._state, f"  {key}: training error — {e}", level="ERROR")
            logger.exception("Retrain error for {}: {}", key, e)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuous AI re-learning daemon — runs during market downtime"
    )
    parser.add_argument(
        "--symbols", nargs="+", default=["ES", "NQ"],
        help="Symbols to retrain models for (default: ES NQ)"
    )
    args = parser.parse_args()

    learner = ContinuousLearner(symbols=args.symbols)
    learner.run()


if __name__ == "__main__":
    main()
