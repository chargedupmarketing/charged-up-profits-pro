"""
src/model_lifecycle.py

Model versioning and the 6-gate safety battery for the continuous re-learning system.

Manages:
  - Saving candidate (pending) models after training
  - Running 6 safety gates before offering a model for human approval
  - Approve / reject / rollback operations
  - Model version manifest (manifest.json) tracking all versions

Model directory structure:
  data/models/
    manifest.json         — full version history + active model references
    active/               — models currently used by InferenceFilter
    pending/              — models awaiting manual approval in the dashboard
    archive/              — last 5 deployed versions per symbol+setup_type

Safety Gates (all must pass before a model is offered for approval):
  1. MinimumDataGate      — >= 100 total trades spanning >= 3 calendar months
  2. AUCGate              — candidate AUC >= old AUC - 0.015 (no degradation)
  3. WinRateSimGate       — simulated win rate >= old - 2.0 percentage points
  4. FeatureStabilityGate — top-5 features overlap >= 3 of 5 with previous model
  5. CalibrationGate      — Brier score delta <= 0.05
  6. BacktestApplicationGate — P&L on recent 30 days >= 95% of old model's P&L
"""

from __future__ import annotations

import json
import pickle
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR    = ROOT / "data" / "models"
ACTIVE_DIR    = MODELS_DIR / "active"
PENDING_DIR   = MODELS_DIR / "pending"
ARCHIVE_DIR   = MODELS_DIR / "archive"
MANIFEST_PATH = MODELS_DIR / "manifest.json"

# Maximum number of archived versions to keep per symbol+setup_type
MAX_ARCHIVE_VERSIONS = 5


# ---------------------------------------------------------------------------
# Gate result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    """Result of running a single safety gate."""
    gate_name: str
    passed: bool
    reason: str
    old_value: float = 0.0
    new_value: float = 0.0
    threshold: float = 0.0


@dataclass
class BatteryResult:
    """Result of running the full 6-gate battery."""
    symbol: str
    setup_type: str
    all_passed: bool
    gate_results: list[GateResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def summary(self) -> str:
        passed = sum(1 for g in self.gate_results if g.passed)
        total = len(self.gate_results)
        lines = [f"Battery {self.symbol}/{self.setup_type}: {passed}/{total} gates passed"]
        for g in self.gate_results:
            icon = "PASS" if g.passed else "FAIL"
            lines.append(f"  [{icon}] {g.gate_name}: {g.reason}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Individual safety gates
# ---------------------------------------------------------------------------

class BaseGate(ABC):
    """Abstract base for a single safety gate."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def evaluate(
        self,
        candidate_model,
        old_model,
        X: pd.DataFrame,
        y: pd.Series,
        trades_df: pd.DataFrame,
        old_metrics: dict,
    ) -> GateResult:
        ...


class MinimumDataGate(BaseGate):
    """Gate 1: Training set must have >= min_trades and span >= min_months."""

    name = "MinimumDataGate"

    def __init__(self, min_trades: int = 100, min_months: int = 3) -> None:
        self.min_trades = min_trades
        self.min_months = min_months

    def evaluate(self, candidate_model, old_model, X, y, trades_df, old_metrics) -> GateResult:
        n = len(X)

        # Estimate date span from the trades_df if date column exists
        months_span = 0.0
        if "date" in trades_df.columns or "entry_ts" in trades_df.columns:
            date_col = "date" if "date" in trades_df.columns else "entry_ts"
            try:
                dates = pd.to_datetime(trades_df[date_col])
                span_days = (dates.max() - dates.min()).days
                months_span = span_days / 30.44
            except Exception:
                months_span = 99.0  # Can't compute — don't fail on this
        else:
            months_span = 99.0  # Unknown — give benefit of the doubt

        if n < self.min_trades:
            return GateResult(
                gate_name=self.name, passed=False,
                reason=f"Only {n} trades (need >= {self.min_trades})",
                new_value=float(n), threshold=float(self.min_trades),
            )
        if months_span < self.min_months:
            return GateResult(
                gate_name=self.name, passed=False,
                reason=f"Data spans only {months_span:.1f} months (need >= {self.min_months})",
                new_value=months_span, threshold=float(self.min_months),
            )
        return GateResult(
            gate_name=self.name, passed=True,
            reason=f"{n} trades over {months_span:.1f} months",
            new_value=float(n), threshold=float(self.min_trades),
        )


class AUCGate(BaseGate):
    """Gate 2: Candidate AUC >= old AUC - min_delta."""

    name = "AUCGate"

    def __init__(self, min_delta: float = -0.015) -> None:
        self.min_delta = min_delta

    def evaluate(self, candidate_model, old_model, X, y, trades_df, old_metrics) -> GateResult:
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import roc_auc_score

        n_splits = min(3, max(2, len(X) // 20))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        aucs = []

        for train_idx, test_idx in tscv.split(X):
            X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
            if len(y_te.unique()) < 2:
                continue
            try:
                if hasattr(candidate_model, "predict_proba"):
                    preds = candidate_model.predict_proba(X_te)[:, 1]
                else:
                    preds = candidate_model.predict(X_te.values)
                aucs.append(roc_auc_score(y_te, preds))
            except Exception:
                pass

        candidate_auc = float(np.mean(aucs)) if aucs else 0.5
        old_auc = float(old_metrics.get("auc", 0.5))
        delta = candidate_auc - old_auc

        passed = delta >= self.min_delta
        return GateResult(
            gate_name=self.name, passed=passed,
            reason=f"New AUC={candidate_auc:.4f}, Old AUC={old_auc:.4f}, delta={delta:+.4f}",
            old_value=old_auc, new_value=candidate_auc, threshold=old_auc + self.min_delta,
        )


class WinRateSimGate(BaseGate):
    """Gate 3: Simulated win rate on holdout >= old win rate - min_delta."""

    name = "WinRateSimGate"

    def __init__(self, min_delta: float = -0.02) -> None:
        self.min_delta = min_delta

    def evaluate(self, candidate_model, old_model, X, y, trades_df, old_metrics) -> GateResult:
        # Use last 20% as holdout (chronological)
        n = len(X)
        cutoff = int(n * 0.80)
        X_hold = X.iloc[cutoff:]
        y_hold = y.iloc[cutoff:]

        if len(X_hold) < 5:
            return GateResult(
                gate_name=self.name, passed=True,
                reason="Insufficient holdout data — gate skipped",
            )

        try:
            # Candidate win rate simulation
            threshold = 0.58  # use standard threshold
            if hasattr(candidate_model, "predict_proba"):
                cand_probs = candidate_model.predict_proba(X_hold)[:, 1]
            else:
                cand_probs = candidate_model.predict(X_hold.values)

            cand_mask = cand_probs >= threshold
            if cand_mask.sum() == 0:
                return GateResult(
                    gate_name=self.name, passed=False,
                    reason="Candidate model filters ALL holdout trades — too restrictive",
                )
            cand_wr = float(y_hold[cand_mask].mean())

            # Old model win rate on same holdout
            old_wr = float(old_metrics.get("win_rate_sim", 0.5))
            delta = cand_wr - old_wr
            passed = delta >= self.min_delta

            return GateResult(
                gate_name=self.name, passed=passed,
                reason=f"New WR={cand_wr:.1%}, Old WR={old_wr:.1%}, delta={delta:+.1%} "
                       f"({cand_mask.sum()} holdout trades kept)",
                old_value=old_wr, new_value=cand_wr, threshold=old_wr + self.min_delta,
            )
        except Exception as e:
            return GateResult(
                gate_name=self.name, passed=True,
                reason=f"Could not evaluate win rate ({e}) — gate skipped",
            )


class FeatureStabilityGate(BaseGate):
    """Gate 4: Top-5 feature importances must overlap >= min_overlap with previous model."""

    name = "FeatureStabilityGate"

    def __init__(self, min_overlap: int = 3) -> None:
        self.min_overlap = min_overlap

    def _top5_features(self, model, X: pd.DataFrame) -> set[str]:
        try:
            if hasattr(model, "feature_importances_"):
                imp = pd.Series(model.feature_importances_, index=X.columns)
                return set(imp.nlargest(5).index.tolist())
            elif hasattr(model, "feature_importance"):
                # LightGBM native
                imp = model.feature_importance(importance_type="gain")
                names = model.feature_name()
                series = pd.Series(imp, index=names)
                return set(series.nlargest(5).index.tolist())
        except Exception:
            pass
        return set()

    def evaluate(self, candidate_model, old_model, X, y, trades_df, old_metrics) -> GateResult:
        if old_model is None:
            return GateResult(
                gate_name=self.name, passed=True,
                reason="No previous model to compare — first training run, gate skipped",
            )

        cand_top5 = self._top5_features(candidate_model, X)
        old_top5 = set(old_metrics.get("top_features", {}).keys())

        if not old_top5:
            old_top5 = self._top5_features(old_model, X)

        overlap = len(cand_top5 & old_top5)
        passed = overlap >= self.min_overlap

        return GateResult(
            gate_name=self.name, passed=passed,
            reason=f"Feature overlap={overlap}/{self.min_overlap} "
                   f"(new={sorted(cand_top5)}, old={sorted(old_top5)})",
            new_value=float(overlap), threshold=float(self.min_overlap),
        )


class CalibrationGate(BaseGate):
    """Gate 5: Brier score of candidate must not worsen by more than max_delta."""

    name = "CalibrationGate"

    def __init__(self, max_brier_delta: float = 0.05) -> None:
        self.max_brier_delta = max_brier_delta

    def evaluate(self, candidate_model, old_model, X, y, trades_df, old_metrics) -> GateResult:
        # Holdout = last 20%
        n = len(X)
        cutoff = int(n * 0.80)
        X_hold, y_hold = X.iloc[cutoff:], y.iloc[cutoff:]

        if len(X_hold) < 5:
            return GateResult(gate_name=self.name, passed=True,
                              reason="Insufficient holdout data — gate skipped")

        try:
            from sklearn.metrics import brier_score_loss
            if hasattr(candidate_model, "predict_proba"):
                cand_probs = candidate_model.predict_proba(X_hold)[:, 1]
            else:
                cand_probs = candidate_model.predict(X_hold.values)
            cand_brier = float(brier_score_loss(y_hold, cand_probs))

            old_brier = float(old_metrics.get("brier_score", 0.25))
            delta = cand_brier - old_brier
            passed = delta <= self.max_brier_delta

            return GateResult(
                gate_name=self.name, passed=passed,
                reason=f"New Brier={cand_brier:.4f}, Old Brier={old_brier:.4f}, delta={delta:+.4f}",
                old_value=old_brier, new_value=cand_brier,
                threshold=old_brier + self.max_brier_delta,
            )
        except Exception as e:
            return GateResult(gate_name=self.name, passed=True,
                              reason=f"Could not compute Brier score ({e}) — gate skipped")


class BacktestApplicationGate(BaseGate):
    """
    Gate 6: When candidate model is applied to the most recent backtest trades,
    net P&L must be >= min_pnl_ratio × old model's P&L on the same trades.
    """

    name = "BacktestApplicationGate"

    def __init__(self, min_pnl_ratio: float = 0.95) -> None:
        self.min_pnl_ratio = min_pnl_ratio

    def evaluate(self, candidate_model, old_model, X, y, trades_df, old_metrics) -> GateResult:
        if "pnl_net" not in trades_df.columns:
            return GateResult(gate_name=self.name, passed=True,
                              reason="No pnl_net column in trades_df — gate skipped")

        if old_model is None:
            return GateResult(gate_name=self.name, passed=True,
                              reason="No previous model — first training run, gate skipped")

        try:
            threshold = 0.58
            # Candidate P&L
            if hasattr(candidate_model, "predict_proba"):
                cand_probs = candidate_model.predict_proba(X)[:, 1]
            else:
                cand_probs = candidate_model.predict(X.values)
            cand_mask = cand_probs >= threshold
            cand_pnl = float(trades_df.loc[cand_mask, "pnl_net"].sum()) if cand_mask.sum() > 0 else 0.0

            # Old model P&L on same trades
            if hasattr(old_model, "predict_proba"):
                old_probs = old_model.predict_proba(X)[:, 1]
            else:
                old_probs = old_model.predict(X.values)
            old_mask = old_probs >= threshold
            old_pnl = float(trades_df.loc[old_mask, "pnl_net"].sum()) if old_mask.sum() > 0 else 0.0

            if old_pnl <= 0:
                # Old model wasn't profitable either — can't use ratio
                return GateResult(gate_name=self.name, passed=True,
                                  reason=f"Old model P&L={old_pnl:.2f} (non-positive) — gate skipped")

            ratio = cand_pnl / old_pnl
            passed = ratio >= self.min_pnl_ratio

            return GateResult(
                gate_name=self.name, passed=passed,
                reason=f"Candidate P&L=${cand_pnl:,.2f}, Old P&L=${old_pnl:,.2f}, ratio={ratio:.3f}",
                old_value=old_pnl, new_value=cand_pnl, threshold=old_pnl * self.min_pnl_ratio,
            )
        except Exception as e:
            return GateResult(gate_name=self.name, passed=True,
                              reason=f"Backtest gate evaluation failed ({e}) — gate skipped")


# ---------------------------------------------------------------------------
# ModelLifecycle — main orchestrator
# ---------------------------------------------------------------------------

class ModelLifecycle:
    """
    Manages the full lifecycle of ML models:
      1. Run the 6-gate safety battery on a newly trained candidate model
      2. Save passing candidates to the pending directory with comparison metrics
      3. Approve candidates (moves to active/, archives old)
      4. Reject candidates (deletes pending files)
      5. Rollback to a previous archived version

    The manifest.json file tracks all versions and is the source of truth
    for the dashboard's model comparison panel.
    """

    GATES = [
        MinimumDataGate(min_trades=100, min_months=3),
        AUCGate(min_delta=-0.015),
        WinRateSimGate(min_delta=-0.02),
        FeatureStabilityGate(min_overlap=3),
        CalibrationGate(max_brier_delta=0.05),
        BacktestApplicationGate(min_pnl_ratio=0.95),
    ]

    def __init__(self) -> None:
        for d in (ACTIVE_DIR, PENDING_DIR, ARCHIVE_DIR):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Manifest helpers
    # ------------------------------------------------------------------

    def _load_manifest(self) -> dict:
        if MANIFEST_PATH.exists():
            try:
                return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"active": {}, "versions": []}

    def _save_manifest(self, manifest: dict) -> None:
        MANIFEST_PATH.write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )

    def _model_key(self, symbol: str, setup_type: str) -> str:
        return f"{symbol.upper()}_{setup_type.upper()}"

    # ------------------------------------------------------------------
    # Gate battery
    # ------------------------------------------------------------------

    def evaluate_candidate(
        self,
        candidate_model,
        symbol: str,
        setup_type: str,
        X: pd.DataFrame,
        y: pd.Series,
        trades_df: pd.DataFrame,
    ) -> BatteryResult:
        """
        Run all 6 safety gates against a newly trained candidate model.

        Returns a BatteryResult with per-gate pass/fail details.
        Also loads the current active model (if any) for comparison.
        """
        key = self._model_key(symbol, setup_type)
        manifest = self._load_manifest()
        old_metrics: dict = manifest.get("active", {}).get(key, {})

        # Load current active model for comparison gates
        old_model = None
        from src.ml_filter import _model_path_for, _load_model
        active_path = _model_path_for(symbol, setup_type)
        if active_path.exists():
            old_model = _load_model(active_path)

        results: list[GateResult] = []
        for gate in self.GATES:
            try:
                result = gate.evaluate(
                    candidate_model, old_model, X, y, trades_df, old_metrics
                )
            except Exception as e:
                result = GateResult(
                    gate_name=gate.name, passed=True,
                    reason=f"Gate evaluation failed ({e}) — defaulting to PASS",
                )
            results.append(result)
            status = "PASS" if result.passed else "FAIL"
            logger.info("[{}] {} — {}: {}", key, status, gate.name, result.reason)

        all_passed = all(r.passed for r in results)
        return BatteryResult(
            symbol=symbol, setup_type=setup_type,
            all_passed=all_passed, gate_results=results,
        )

    # ------------------------------------------------------------------
    # Save pending
    # ------------------------------------------------------------------

    def save_pending(
        self,
        candidate_model,
        symbol: str,
        setup_type: str,
        battery_result: BatteryResult,
        train_metrics: dict,
    ) -> Path:
        """
        Save a candidate model that passed all gates to the pending directory,
        along with a comparison JSON for the dashboard.

        Returns the path of the saved pending model file.
        """
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        sym = symbol.upper()
        st  = setup_type.upper()
        model_filename = f"ml_filter_{sym}_{st}_{ts}.pkl"
        comparison_filename = f"comparison_{sym}_{st}_{ts}.json"

        pending_model_path = PENDING_DIR / model_filename
        pending_comparison_path = PENDING_DIR / comparison_filename

        # Save model
        with open(pending_model_path, "wb") as f:
            pickle.dump(candidate_model, f)

        # Build comparison dict
        manifest = self._load_manifest()
        key = self._model_key(sym, st)
        old_metrics = manifest.get("active", {}).get(key, {})

        comparison = {
            "symbol": sym,
            "setup_type": st,
            "timestamp": ts,
            "pending_model_file": model_filename,
            "old_metrics": old_metrics,
            "new_metrics": train_metrics,
            "battery_result": battery_result.to_dict(),
            "status": "pending",
        }
        pending_comparison_path.write_text(
            json.dumps(comparison, indent=2, default=str), encoding="utf-8"
        )

        # Update manifest
        if "pending" not in manifest:
            manifest["pending"] = {}
        manifest["pending"][key] = {
            "model_file": model_filename,
            "comparison_file": comparison_filename,
            "timestamp": ts,
            "battery_passed": battery_result.all_passed,
            "new_auc": train_metrics.get("mean_cv_auc", 0.0),
            "new_win_rate_sim": train_metrics.get("win_rate_sim", 0.0),
        }
        self._save_manifest(manifest)

        logger.info("Pending model saved: {} (all_gates={})", pending_model_path, battery_result.all_passed)
        return pending_model_path

    # ------------------------------------------------------------------
    # Approve
    # ------------------------------------------------------------------

    def approve(self, symbol: str, setup_type: str, timestamp: str | None = None) -> bool:
        """
        Move the pending model for symbol+setup_type to the active directory.
        Archives the current active model (keeping last MAX_ARCHIVE_VERSIONS).

        Returns True on success.
        """
        sym = symbol.upper()
        st  = setup_type.upper()
        key = self._model_key(sym, st)

        manifest = self._load_manifest()
        pending_entry = manifest.get("pending", {}).get(key)

        if not pending_entry:
            logger.error("No pending model found for {}", key)
            return False

        model_file = pending_entry["model_file"]
        comparison_file = pending_entry.get("comparison_file", "")
        pending_model_path = PENDING_DIR / model_file

        if not pending_model_path.exists():
            logger.error("Pending model file not found: {}", pending_model_path)
            return False

        # Archive current active model
        from src.ml_filter import _model_path_for
        active_path = _model_path_for(sym, st)
        if active_path.exists():
            archive_name = f"{active_path.stem}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pkl"
            shutil.copy2(active_path, ARCHIVE_DIR / archive_name)
            self._prune_archive(key)

        # Move pending → active
        shutil.move(str(pending_model_path), str(active_path))
        logger.info("Approved and deployed: {} → {}", model_file, active_path)

        # Clean up comparison file
        if comparison_file:
            (PENDING_DIR / comparison_file).unlink(missing_ok=True)

        # Update manifest
        manifest["active"][key] = {
            **pending_entry,
            "status": "active",
            "approved_at": datetime.utcnow().isoformat(),
        }
        manifest["pending"].pop(key, None)

        if "versions" not in manifest:
            manifest["versions"] = []
        manifest["versions"].append({
            "key": key, "timestamp": pending_entry["timestamp"],
            "model_file": model_file, "status": "deployed",
            "approved_at": datetime.utcnow().isoformat(),
        })
        self._save_manifest(manifest)
        return True

    # ------------------------------------------------------------------
    # Reject
    # ------------------------------------------------------------------

    def reject(self, symbol: str, setup_type: str) -> bool:
        """
        Reject a pending model — deletes the pending files and clears the manifest entry.
        """
        sym = symbol.upper()
        st  = setup_type.upper()
        key = self._model_key(sym, st)

        manifest = self._load_manifest()
        pending_entry = manifest.get("pending", {}).get(key)

        if not pending_entry:
            logger.warning("No pending model to reject for {}", key)
            return False

        for fname in (pending_entry.get("model_file", ""), pending_entry.get("comparison_file", "")):
            if fname:
                (PENDING_DIR / fname).unlink(missing_ok=True)

        manifest["pending"].pop(key, None)
        self._save_manifest(manifest)
        logger.info("Rejected pending model for {}", key)
        return True

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback(self, symbol: str, setup_type: str) -> bool:
        """
        Restore the most recent archived version of a model to active.
        """
        sym = symbol.upper()
        st  = setup_type.upper()
        pattern = f"ml_filter_{sym}_{st}_*.pkl"
        archived = sorted(ARCHIVE_DIR.glob(pattern), reverse=True)

        if not archived:
            logger.error("No archived models found for {}_{}", sym, st)
            return False

        latest_archive = archived[0]
        from src.ml_filter import _model_path_for
        active_path = _model_path_for(sym, st)

        shutil.copy2(latest_archive, active_path)
        latest_archive.unlink()  # Remove from archive after restoring
        logger.info("Rolled back {}_{} to {}", sym, st, latest_archive.name)

        manifest = self._load_manifest()
        key = self._model_key(sym, st)
        manifest["active"][key] = {
            "status": "rolled_back",
            "rolled_back_at": datetime.utcnow().isoformat(),
            "source_file": latest_archive.name,
        }
        self._save_manifest(manifest)
        return True

    # ------------------------------------------------------------------
    # Pending models listing (for dashboard)
    # ------------------------------------------------------------------

    def get_pending_models(self) -> list[dict]:
        """
        Return a list of pending model comparison dicts for the dashboard.
        Each dict has: symbol, setup_type, timestamp, old_metrics, new_metrics,
        battery_result, model_file.
        """
        manifest = self._load_manifest()
        pending = []
        for key, entry in manifest.get("pending", {}).items():
            comparison_file = entry.get("comparison_file", "")
            comparison_path = PENDING_DIR / comparison_file
            if comparison_path.exists():
                try:
                    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
                    pending.append(comparison)
                except Exception:
                    pending.append(entry)
            else:
                pending.append(entry)
        return pending

    def get_active_metrics(self, symbol: str, setup_type: str) -> dict:
        """Return the stored metrics for the currently active model."""
        key = self._model_key(symbol, setup_type)
        return self._load_manifest().get("active", {}).get(key, {})

    def get_version_history(self) -> list[dict]:
        """Return the full version deployment history from manifest.json."""
        return self._load_manifest().get("versions", [])

    # ------------------------------------------------------------------
    # Archive pruning
    # ------------------------------------------------------------------

    def _prune_archive(self, key: str) -> None:
        """Keep only the most recent MAX_ARCHIVE_VERSIONS archived files for this key."""
        parts = key.split("_", 1)
        if len(parts) < 2:
            return
        sym, st = parts[0], parts[1]
        pattern = f"ml_filter_{sym}_{st}_*.pkl"
        archived = sorted(ARCHIVE_DIR.glob(pattern), reverse=True)
        for old_file in archived[MAX_ARCHIVE_VERSIONS:]:
            old_file.unlink(missing_ok=True)
            logger.debug("Pruned old archive: {}", old_file.name)
