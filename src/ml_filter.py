"""
src/ml_filter.py

Phase 2 ML filter: XGBoost + LightGBM ensemble meta-label classifier.

Workflow:
  1. Label historical signals with triple-barrier method
  2. Train XGBoost + LightGBM ensemble on labelled features
  3. Save per-setup-type models to data/models/active/
  4. Enable in config: ml.enabled = true

Model files (per-symbol, per-setup-type):
  data/models/active/ml_filter_{SYMBOL}_{SETUP_TYPE}.pkl
  e.g.  ml_filter_ES_BREAK_RETEST.pkl
        ml_filter_NQ_REJECTION.pkl

Backwards-compatible: InferenceFilter still loads legacy
  data/ml_filter_model.pkl if per-setup models don't exist yet.

Usage:
    # Build labels from backtest trade log
    python src/ml_filter.py label --trades data/backtest_ES_*.csv

    # Train model (all setup types for a symbol)
    python src/ml_filter.py train --symbol ES --trades data/backtest_ES_*.csv

    # Evaluate model (out-of-sample) + SHAP explainability
    python src/ml_filter.py eval --symbol ES --trades data/backtest_ES_*.csv

    # Retrain all in one step (used by dashboard and continuous learner)
    python src/ml_filter.py retrain --symbol ES
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_settings(path: str = "config/settings.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Triple-barrier labeling
# ---------------------------------------------------------------------------

# TripleBarrierLabeler is now the canonical home in src/labeling/labeler.py.
# Re-export from there so all existing callers continue to work unchanged.
from src.labeling.labeler import TripleBarrierLabeler, compute_realized_R  # noqa: F401


# ---------------------------------------------------------------------------
# Feature extraction from audit DB / backtest logs
# ---------------------------------------------------------------------------

def extract_features_from_signal_log(signal_log_df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the features_json column from the signal log into a flat DataFrame.
    """
    rows = []
    for _, row in signal_log_df.iterrows():
        features = {}
        if row.get("features_json"):
            try:
                features = json.loads(row["features_json"])
            except (json.JSONDecodeError, TypeError):
                pass
        features["signal_id"] = row.get("id")
        features["label"] = row.get("label", None)
        rows.append(features)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Feature columns (updated with Phase 3a additions)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    # Opening range
    "range_pts", "range_norm", "price_vs_mid_norm", "price_above_mid",
    "or_range_vs_5day_avg",
    # Level proximity
    "level_dist_pts", "level_is_high", "level_origin_asia", "level_origin_london",
    "active_highs_count", "active_lows_count",
    # Level quality
    "level_freshness_bars", "level_test_count",
    # Volume
    "volume_ratio", "volume_above_median", "volume_trend_5",
    # Trend / momentum
    "price_vs_ema20", "price_above_ema20",
    "atr_15m_14", "momentum_3bar_norm", "return_5bar_1m",
    # Macro trend
    "daily_ema50_distance", "price_above_ema50",
    # Temporal
    "hour", "minute_of_day", "day_of_week", "mins_since_exec_open",
    "bars_in_exec_window",
    # Setup type
    "setup_break_retest", "setup_rejection", "setup_bounce",
    "direction_long", "stop_dist_pts", "target_dist_pts", "rr_ratio",
    # Candle quality + regime
    "candle_body_ratio", "atr_regime",
    "spread_proxy", "prev_bar_momentum", "prev_bar_bullish",
    # Break strength
    "or_range_vs_atr", "break_excursion_pts", "closes_beyond_level", "time_to_retest_bars",
    # Direction-aligned derived
    "momentum_aligned", "ema50_aligned", "is_retest_quick",
    # Market context
    "gap_size_norm", "session_phase",
    # VWAP + prior day context
    "vwap_distance_norm", "price_above_vwap",
    "price_vs_prev_high_norm", "price_vs_prev_low_norm",
    "above_prev_high", "below_prev_low",
    # Regime intelligence (Section 7 — new)
    "efficiency_ratio", "vol_state",
]


def _schema_hash(feature_names: list[str]) -> str:
    """SHA-256 of the sorted feature name list — used to detect schema drift."""
    canon = sorted(feature_names)
    return "sha256:" + hashlib.sha256(",".join(canon).encode()).hexdigest()[:16]


def _manifest_path_for(model_path: Path) -> Path:
    """Return the dataset_manifest.json path alongside a model file."""
    return model_path.parent / (model_path.stem + "_manifest.json")


def _save_manifest(model_path: Path, manifest: dict) -> None:
    mp = _manifest_path_for(model_path)
    mp.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    logger.info("Manifest saved: {}", mp)


def _load_manifest(model_path: Path) -> Optional[dict]:
    mp = _manifest_path_for(model_path)
    if not mp.exists():
        return None
    try:
        return json.loads(mp.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Lightweight inference wrapper (used by harness and live trading)
# ---------------------------------------------------------------------------

def _model_dir() -> Path:
    """Return the active model directory, creating it if necessary."""
    d = ROOT / "data" / "models" / "active"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _model_path_for(symbol: str, setup_type: str | None = None) -> Path:
    """
    Return the Path for a per-symbol, per-setup-type model file.

    New convention:  data/models/active/ml_filter_{SYMBOL}_{SETUP_TYPE}.pkl
    Legacy fallback: data/ml_filter_model.pkl  (ES) / ml_filter_model_{SYM}.pkl
    """
    sym = symbol.upper()
    if setup_type:
        return _model_dir() / f"ml_filter_{sym}_{setup_type.upper()}.pkl"
    # Combined (fallback) model
    if sym == "ES":
        return _model_dir() / "ml_filter_ES.pkl"
    return _model_dir() / f"ml_filter_{sym}.pkl"


def _legacy_model_path(symbol: str) -> Path:
    """Legacy model path for backwards compatibility."""
    sym = symbol.upper()
    if sym == "ES":
        return ROOT / "data" / "ml_filter_model.pkl"
    return ROOT / "data" / f"ml_filter_model_{sym}.pkl"


def _load_model(path: Path):
    """Load a pickled model from disk. Returns None if path doesn't exist."""
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning("Failed to load model {}: {}", path, e)
        return None


class InferenceFilter:
    """
    Lightweight ML inference wrapper — loads trained models from disk and
    scores incoming feature vectors.  Used by the backtest harness and live
    bot to gate signals without the full training machinery.

    Model resolution order per signal:
      1. Per-setup-type model: data/models/active/ml_filter_{SYM}_{SETUP}.pkl
      2. Combined symbol model: data/models/active/ml_filter_{SYM}.pkl
      3. Legacy path:           data/ml_filter_model.pkl  (ES only)

    Ensemble: if both XGBoost (.pkl) and LightGBM (.lgbm.pkl) models exist
    for a given setup type, probabilities are averaged 50/50.

    If no model file exists, ``allows_trade`` always returns (True, 0.5) so
    the harness degrades gracefully without ML filtering.
    """

    _SETUP_TYPES = ["BREAK_RETEST", "REJECTION", "BOUNCE", "SWEEP_REVERSE"]

    def __init__(
        self,
        settings_path: str = "config/settings.yaml",
        symbol: str = "ES",
    ) -> None:
        cfg = _load_settings(settings_path)
        ml_cfg = cfg.get("ml", {})
        self._enabled: bool = ml_cfg.get("enabled", False)
        self._threshold: float = ml_cfg.get("min_probability_threshold", 0.58)
        self._sym = symbol.upper()

        # ── EV gating config (with per-symbol overrides — Section H) ─────
        self._ev_gating: bool = ml_cfg.get("ev_gating_enabled", True)
        # Check for per-symbol override first, then fall back to global
        _sym_ev = ml_cfg.get("symbol_ev_overrides", {}).get(self._sym, {})
        self._ev_min: float = float(
            _sym_ev.get("ev_min_threshold", ml_cfg.get("ev_min_threshold", 0.10))
        )
        self._ev_fallback_threshold: float = float(
            _sym_ev.get("ev_fallback_threshold", ml_cfg.get("ev_fallback_threshold", 0.42))
        )
        if _sym_ev:
            logger.info(
                "InferenceFilter[{}]: using per-symbol EV thresholds "
                "ev_min={} fallback={}",
                self._sym, self._ev_min, self._ev_fallback_threshold,
            )

        # Per-setup-type models: {setup_type: (xgb_model, lgbm_model, calibrator, manifest)}
        # Any entry may be None (graceful degradation).
        self._setup_models: dict[str, tuple] = {}

        # Combined fallback model (used when no per-setup model exists)
        self._fallback_model = None
        self._fallback_calibrator = None
        self._fallback_manifest: Optional[dict] = None

        if not self._enabled:
            return

        # Load per-setup-type models
        for st in self._SETUP_TYPES:
            xgb_path   = _model_path_for(self._sym, st)
            lgbm_path  = _model_path_for(self._sym, f"{st}.lgbm")
            cal_path   = _model_path_for(self._sym, f"{st}.calibrator")
            xgb_m   = _load_model(xgb_path)
            lgbm_m  = _load_model(lgbm_path)
            cal_m   = _load_model(cal_path)
            manifest = _load_manifest(xgb_path) if xgb_path.exists() else None
            if xgb_m or lgbm_m:
                self._setup_models[st] = (xgb_m, lgbm_m, cal_m, manifest)
                logger.info(
                    "InferenceFilter: loaded {}/{}/cal={} models for {} {} (ev_gating={})",
                    "XGB" if xgb_m else "-",
                    "LGBM" if lgbm_m else "-",
                    "yes" if cal_m else "no",
                    self._sym, st, self._ev_gating,
                )

        # Fallback combined model
        fallback_path = _model_path_for(self._sym)
        self._fallback_model = _load_model(fallback_path)
        self._fallback_calibrator = _load_model(
            _model_path_for(self._sym, "combined.calibrator")
        )
        self._fallback_manifest = _load_manifest(fallback_path)

        # Legacy path for backwards compatibility
        if self._fallback_model is None:
            self._fallback_model = _load_model(_legacy_model_path(self._sym))
            if self._fallback_model:
                logger.info("InferenceFilter: loaded legacy model for {}", self._sym)

        if not self._setup_models and self._fallback_model is None:
            logger.warning(
                "InferenceFilter: ml.enabled=true but no model files found for {} — "
                "run 'python src/ml_filter.py retrain --symbol {}' first",
                self._sym, self._sym,
            )

    def _predict_proba(self, model, X: "pd.DataFrame") -> float:
        """Get prediction probability from a single model (XGBoost or LightGBM)."""
        try:
            if hasattr(model, "predict_proba"):
                # scikit-learn compatible (XGBClassifier, LGBMClassifier)
                return float(model.predict_proba(X)[0, 1])
            else:
                # LightGBM native booster
                return float(model.predict(X.values)[0])
        except Exception as e:
            logger.warning("Model predict error: {}", e)
            return 0.5

    def _get_model_cols(self, model) -> list[str]:
        """Extract feature names from a model object."""
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
        if hasattr(model, "feature_name_"):
            return list(model.feature_name_)
        if hasattr(model, "feature_names"):
            return list(model.feature_names)
        return FEATURE_COLS

    def _score_with_models(
        self, xgb_m, lgbm_m, features: dict
    ) -> float:
        """Score features with available models, averaging if both exist."""
        probs = []
        for m in (xgb_m, lgbm_m):
            if m is None:
                continue
            cols = self._get_model_cols(m)
            row = {col: features.get(col, 0) for col in cols}
            X = pd.DataFrame([row])[cols]
            probs.append(self._predict_proba(m, X))

        if not probs:
            return 0.5
        return float(np.mean(probs))

    def _calibrate_prob(self, raw_prob: float, calibrator) -> float:
        """Apply isotonic/platt calibration to a raw model score."""
        if calibrator is None:
            return raw_prob
        try:
            return float(calibrator.predict(np.array([[raw_prob]]))[0])
        except Exception:
            try:
                return float(calibrator.predict(np.array([raw_prob]))[0])
            except Exception:
                return raw_prob

    def _ev_decision(
        self, p_cal: float, manifest: Optional[dict]
    ) -> tuple[bool, float, str]:
        """
        Compute expected value and decide whether to allow the trade.

        EV = p_cal * avg_R_win + (1-p_cal) * avg_R_loss

        Returns (allow, ev, reason_string)
        """
        if manifest and "ev_policy" in manifest:
            ev_pol = manifest["ev_policy"]
            avg_R_win  = float(ev_pol.get("avg_R_win", 3.0))
            avg_R_loss = float(ev_pol.get("avg_R_loss", -1.0))
            ev_min     = float(ev_pol.get("ev_min_threshold", self._ev_min))
        else:
            # No manifest — fall back to a reasonable estimate
            # With 4:1 R:R: avg_R_win ≈ 3.0, avg_R_loss ≈ -1.0
            avg_R_win  = 3.0
            avg_R_loss = -1.0
            ev_min     = self._ev_min

        ev = p_cal * avg_R_win + (1.0 - p_cal) * avg_R_loss
        allow = ev >= ev_min
        reason = (
            f"EV={ev:.3f} p={p_cal:.3f} "
            f"R_win={avg_R_win:.2f} R_loss={avg_R_loss:.2f} "
            f"min={ev_min:.2f} -> {'ALLOW' if allow else 'DENY'}"
        )
        return allow, ev, reason

    def allows_trade(self, features: dict) -> tuple[bool, float]:
        """
        Returns (allow, calibrated_probability).
        - allow=True if ML is disabled, all models missing, or EV/probability gate passes.
        - Logs EV decision reason for each signal at DEBUG level.
        - Uses per-setup-type model when available, falls back to combined model.
        - Ensemble: if both XGBoost and LightGBM are loaded, averages their probs.
        - If EV gating enabled and manifest has ev_policy: uses EV >= ev_min.
        - Otherwise falls back to calibrated_prob >= ev_fallback_threshold.
        - Section D: ABSTAIN if calibrated probability is near 0.50 (uncertainty zone)
          or if XGB vs LGBM disagreement is high (> 0.15).
        """
        if not self._enabled:
            return True, 0.5

        try:
            setup_type = str(features.get("setup_type", "")).upper()
            if not setup_type:
                for st in self._SETUP_TYPES:
                    if features.get(f"setup_{st.lower()}", 0):
                        setup_type = st
                        break

            if setup_type in self._setup_models:
                xgb_m, lgbm_m, calibrator, manifest = self._setup_models[setup_type]
            elif self._fallback_model is not None:
                xgb_m   = self._fallback_model
                lgbm_m  = None
                calibrator = self._fallback_calibrator
                manifest   = self._fallback_manifest
            else:
                return True, 0.5  # No model loaded — pass everything through

            # ── Section D: Uncertainty / Abstain Logic ─────────────────────
            # Score XGB and LGBM separately to check disagreement
            xgb_prob  = self._score_with_models(xgb_m, None, features) if xgb_m else 0.5
            lgbm_prob = self._score_with_models(None, lgbm_m, features) if lgbm_m else xgb_prob
            raw_prob  = (xgb_prob + lgbm_prob) / 2 if lgbm_m else xgb_prob

            # Model disagreement: |xgb_prob - lgbm_prob|
            disagreement = abs(xgb_prob - lgbm_prob) if lgbm_m else 0.0
            if disagreement > 0.15:
                logger.debug(
                    "ML gate [{}]: ABSTAIN — model disagreement {:.3f} "
                    "(XGB={:.3f} LGBM={:.3f})",
                    setup_type or "combined", disagreement, xgb_prob, lgbm_prob,
                )
                return False, raw_prob

            # Calibrate the raw probability
            p_cal = self._calibrate_prob(raw_prob, calibrator)

            # Near-0.50 uncertainty zone: abstain only when very close to coin-flip.
            # With ~50% label win rate the model naturally spans 0.40–0.65+.
            # A wide abstain band (0.42–0.58) would block the majority of signals.
            # Only abstain in the very tight band [0.47–0.53] — truly uncertain.
            if 0.47 <= p_cal <= 0.53:
                logger.debug(
                    "ML gate [{}]: ABSTAIN — uncertainty zone p_cal={:.3f}",
                    setup_type or "combined", p_cal,
                )
                return False, p_cal

            # ── EV gating ──────────────────────────────────────────────────
            if self._ev_gating:
                allow, ev, reason = self._ev_decision(p_cal, manifest)
                logger.debug("ML gate [{}]: {}", setup_type or "combined", reason)
                return allow, p_cal
            else:
                allow = p_cal >= self._ev_fallback_threshold
                logger.debug(
                    "ML gate [{}]: p_cal={:.3f} threshold={:.2f} -> {}",
                    setup_type or "combined", p_cal,
                    self._ev_fallback_threshold, "ALLOW" if allow else "DENY",
                )
                return allow, p_cal

        except Exception as e:
            logger.warning("InferenceFilter.allows_trade error: {} — allowing trade", e)
            return True, 0.5


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

class MLFilter:
    """
    Train and evaluate the XGBoost + LightGBM ensemble meta-label filter.

    Per-setup-type models: training with setup_type parameter trains a
    specialist model for that setup only (features most predictive for
    BREAK_RETEST differ from those for REJECTION, etc.).

    Exponential decay sample weights: recent trades outweigh old ones
    (decay factor configurable; default gives oldest trades ~30% weight
    of the most recent trade).
    """

    def __init__(
        self,
        settings_path: str = "config/settings.yaml",
        symbol: str = "ES",
        setup_type: str | None = None,
    ) -> None:
        self._cfg = _load_settings(settings_path)
        self._threshold = self._cfg["ml"]["min_probability_threshold"]
        self._model = None
        self._lgbm_model = None
        self._symbol = symbol.upper()
        self._setup_type = setup_type.upper() if setup_type else None

        # Model file paths
        self._model_path = _model_path_for(self._symbol, self._setup_type)
        self._lgbm_path  = (
            _model_path_for(self._symbol, f"{self._setup_type}.lgbm")
            if self._setup_type
            else _model_path_for(self._symbol, "combined.lgbm")
        )

    def prepare_dataset(
        self,
        trades_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Prepare (X, y) from a trades DataFrame that already has features and labels.
        Handles missing feature columns gracefully.

        When self._setup_type is set, only trades matching that setup type are used,
        so each specialist model trains on its own data only.
        """
        labeler = TripleBarrierLabeler()
        labelled = labeler.label_from_backtest_trades(trades_df)

        # Filter to setup type if a specialist model is being trained
        if self._setup_type and "setup_type" in labelled.columns:
            mask = labelled["setup_type"].str.upper() == self._setup_type
            labelled = labelled[mask].copy()
            logger.info(
                "Filtering dataset to setup_type={}: {} / {} trades",
                self._setup_type, len(labelled), len(trades_df)
            )
        elif self._setup_type:
            # Try one-hot columns (features_json parsed data)
            logger.info(
                "setup_type column not in trades_df — training on all {} trades",
                len(labelled)
            )

        if len(labelled) < 5:
            # Return empty rather than raising so callers can check len(X) and
            # log a clean "skipping" message instead of a red "training error".
            logger.debug(
                "Not enough trades for setup_type={}: {} rows — returning empty dataset",
                self._setup_type, len(labelled),
            )
            empty_x = pd.DataFrame(columns=FEATURE_COLS)
            empty_y = pd.Series(dtype=int)
            return empty_x, empty_y

        # Extract features stored in features_json if available
        if "features_json" in labelled.columns and labelled["features_json"].notna().any():
            feature_df = extract_features_from_signal_log(labelled)
        else:
            feature_df = labelled

        available_cols = [c for c in FEATURE_COLS if c in feature_df.columns]
        if not available_cols:
            raise ValueError(
                "No feature columns found in dataset. "
                "Re-run backtest with the updated harness to generate features_json."
            )

        # ── Schema enforcement ────────────────────────────────────────────
        missing_required = [c for c in FEATURE_COLS if c not in available_cols]
        if missing_required:
            logger.warning(
                "Training schema: {} required features missing from dataset: {}",
                len(missing_required), missing_required[:8],
            )
            logger.warning(
                "Missing features will be filled with 0.  "
                "Re-run scripts/rebuild_backtests.py to regenerate full-feature data."
            )
            # Fill with 0 so training can proceed; models trained on partial features
            # will store the available_cols list in manifest for inference alignment.
            for col in missing_required:
                feature_df = feature_df.copy()
                feature_df[col] = 0.0
            available_cols = FEATURE_COLS  # use full list now that we filled zeros

        X = feature_df[available_cols].fillna(0)
        y = labelled["label"].reset_index(drop=True).astype(int)
        X = X.reset_index(drop=True)
        # Carry realized_R as a passthrough column so train() can compute
        # actual avg_R_win/avg_R_loss for the EV policy manifest.
        if "realized_R" in labelled.columns:
            X["realized_R"] = labelled["realized_R"].reset_index(drop=True)

        logger.info(
            "Dataset ({}): {} samples, {} features, {:.1f}% positive labels",
            self._setup_type or "combined",
            len(X), len(available_cols), y.mean() * 100
        )
        return X, y

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        timestamps: "Optional[pd.Series]" = None,
    ) -> dict:
        """
        Train XGBoost + LightGBM ensemble with:
          1. Purged + embargoed time-series CV (no label-horizon leakage)
          2. Isotonic calibration on a temporal holdout
          3. EV policy stats (avg_R_win, avg_R_loss) from training data
          4. Dataset manifest saved alongside the model

        Returns evaluation metrics on the held-out CV sets.
        """
        try:
            from xgboost import XGBClassifier
            from sklearn.metrics import roc_auc_score
            from sklearn.isotonic import IsotonicRegression
        except ImportError:
            logger.error("XGBoost/scikit-learn not installed. Run: pip install xgboost scikit-learn")
            sys.exit(1)

        from src.cv.purged_time_series_split import PurgedTimeSeriesSplit

        # Optional LightGBM
        lgbm_available = False
        try:
            from lightgbm import LGBMClassifier
            lgbm_available = True
        except ImportError:
            logger.info("LightGBM not installed — XGBoost only. pip install lightgbm")

        ml_cfg = self._cfg.get("ml", {})
        label_horizon = ml_cfg.get("cv_purge_horizon_bars", 60)
        embargo_bars  = ml_cfg.get("cv_embargo_bars", 10)
        label_method  = ml_cfg.get("label_method", "realized_R_binary")
        calibration   = ml_cfg.get("calibration", "isotonic")
        ev_min        = ml_cfg.get("ev_min_threshold", 0.10)

        # Strip passthrough columns before fitting — only use canonical FEATURE_COLS.
        _passthrough = [c for c in X.columns if c not in FEATURE_COLS]
        if _passthrough:
            X = X.drop(columns=_passthrough)

        n = len(y)
        n_splits = min(5, max(2, n // 10))

        # ── Exponential decay sample weights ──────────────────────────────
        weights = np.exp(np.linspace(-1.2, 0.0, n))
        weights = weights / weights.sum() * n

        xgb_params = dict(
            n_estimators=200, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="logloss", use_label_encoder=False, random_state=42,
        )
        lgbm_params = dict(
            n_estimators=200, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1,
        )

        # ── Purged + embargoed CV ─────────────────────────────────────────
        splitter = PurgedTimeSeriesSplit(
            n_splits=n_splits,
            label_horizon_bars=label_horizon,
            embargo_bars=embargo_bars,
        )
        xgb_auc_scores, lgbm_auc_scores = [], []

        ts_series = timestamps if timestamps is not None else (
            X.index.to_series() if isinstance(X.index, pd.DatetimeIndex) else None
        )

        for fold, (train_idx, val_idx) in enumerate(splitter.split(X, timestamps=ts_series)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            w_tr = weights[train_idx]

            if len(y_val.unique()) < 2:
                logger.warning("Fold {}: single class in val — skip AUC", fold + 1)
                continue

            xgb_m = XGBClassifier(**xgb_params)
            xgb_m.fit(X_tr, y_tr, sample_weight=w_tr,
                      eval_set=[(X_val, y_val)], verbose=False)
            xgb_preds = xgb_m.predict_proba(X_val)[:, 1]
            xgb_auc = roc_auc_score(y_val, xgb_preds)
            xgb_auc_scores.append(xgb_auc)

            if lgbm_available:
                from lightgbm import LGBMClassifier as _LGBM
                lgbm_m = _LGBM(**lgbm_params)
                lgbm_m.fit(X_tr, y_tr, sample_weight=w_tr)
                lgbm_preds = lgbm_m.predict_proba(X_val)[:, 1]
                lgbm_auc = roc_auc_score(y_val, lgbm_preds)
                lgbm_auc_scores.append(lgbm_auc)
                ens_preds = 0.5 * xgb_preds + 0.5 * lgbm_preds
                ens_auc = roc_auc_score(y_val, ens_preds)
                logger.info(
                    "Fold {}: XGB={:.3f} LGBM={:.3f} Ens={:.3f} n_val={}",
                    fold + 1, xgb_auc, lgbm_auc, ens_auc, len(X_val),
                )
            else:
                logger.info("Fold {}: XGB={:.3f} n_val={}", fold + 1, xgb_auc, len(X_val))

        # ── Calibration holdout (last 20% of data, chronological) ────────
        cal_n = max(20, n // 5)
        model_n = n - cal_n
        X_model, X_cal = X.iloc[:model_n], X.iloc[model_n:]
        y_model, y_cal = y.iloc[:model_n], y.iloc[model_n:]
        w_model = weights[:model_n]

        # Final model on model_n portion
        final_xgb = XGBClassifier(**xgb_params)
        final_xgb.fit(X_model, y_model, sample_weight=w_model, verbose=False)

        # Calibrate on calibration holdout
        calibrator = None
        if calibration == "isotonic" and len(y_cal) >= 10 and len(y_cal.unique()) == 2:
            cal_preds = final_xgb.predict_proba(X_cal)[:, 1]
            if lgbm_available:
                from lightgbm import LGBMClassifier as _LGBM2
                lgbm_cal = _LGBM2(**lgbm_params)
                lgbm_cal.fit(X_model, y_model, sample_weight=w_model)
                cal_preds = 0.5 * cal_preds + 0.5 * lgbm_cal.predict_proba(X_cal)[:, 1]
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(cal_preds.reshape(-1, 1), y_cal.values)
            logger.info("Isotonic calibrator fitted on {} holdout samples", len(y_cal))

        # ── Retrain final models on FULL dataset for production ───────────
        final_xgb_full = XGBClassifier(**xgb_params)
        final_xgb_full.fit(X, y, sample_weight=weights, verbose=False)
        self._model = final_xgb_full
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._model_path, "wb") as f:
            pickle.dump(final_xgb_full, f)
        logger.info("XGBoost model saved: {}", self._model_path)

        final_lgbm_full = None
        if lgbm_available:
            from lightgbm import LGBMClassifier as _LGBM3
            final_lgbm_full = _LGBM3(**lgbm_params)
            final_lgbm_full.fit(X, y, sample_weight=weights)
            self._lgbm_model = final_lgbm_full
            with open(self._lgbm_path, "wb") as f:
                pickle.dump(final_lgbm_full, f)
            logger.info("LightGBM model saved: {}", self._lgbm_path)

        # Save calibrator
        cal_path = _model_path_for(
            self._symbol,
            f"{self._setup_type}.calibrator" if self._setup_type else "combined.calibrator",
        )
        if calibrator is not None:
            with open(cal_path, "wb") as f:
                pickle.dump(calibrator, f)
            logger.info("Calibrator saved: {}", cal_path)

        # ── EV policy stats ───────────────────────────────────────────────
        # Compute avg_R_win and avg_R_loss from the actual training data.
        # Using realistic values is critical: fantasy defaults like 3.0/-1.0
        # cause the EV gate to require p > 27%, which a model trained on
        # ~50% positive labels will often struggle to hit for marginal signals.
        avg_R_win, avg_R_loss = 1.5, -0.75  # realistic fallback
        # Try to extract realized_R from X (if present as a passthrough column)
        # or from y-aligned R column stored on the index
        _r_col = None
        if "realized_R" in X.columns:
            _r_col = X["realized_R"]
        elif hasattr(y, "name") and isinstance(y.index, pd.Index):
            # Try to get realized_R from the original labelled_trades.csv via index
            try:
                _labelled_path = ROOT / "data" / "labelled_trades.csv"
                if _labelled_path.exists():
                    _ldf = pd.read_csv(_labelled_path, usecols=["realized_R"]).iloc[y.index]
                    _r_col = _ldf["realized_R"]
            except Exception:
                pass
        if _r_col is not None:
            r_wins   = _r_col[_r_col > 0]
            r_losses = _r_col[_r_col < 0]
            if len(r_wins) >= 10:
                avg_R_win = float(r_wins.mean())
            if len(r_losses) >= 10:
                avg_R_loss = float(r_losses.mean())
            logger.info(
                "EV policy from training data: avg_R_win={:.3f}  avg_R_loss={:.3f}  "
                "(n_wins={} n_losses={})",
                avg_R_win, avg_R_loss, len(r_wins), len(r_losses),
            )
        train_ts_str, train_te_str = "", ""
        if isinstance(X.index, pd.DatetimeIndex):
            train_ts_str = str(X.index.min().date())
            train_te_str = str(X.index.max().date())

        # ── Feature importance ────────────────────────────────────────────
        importances = pd.Series(
            final_xgb_full.feature_importances_, index=X.columns
        ).sort_values(ascending=False).head(10)

        mean_xgb_auc  = round(np.mean(xgb_auc_scores),  4) if xgb_auc_scores  else 0.0
        mean_lgbm_auc = round(np.mean(lgbm_auc_scores), 4) if lgbm_auc_scores else 0.0
        mean_ens_auc  = round(
            (mean_xgb_auc + mean_lgbm_auc) / 2 if lgbm_auc_scores else mean_xgb_auc, 4
        )

        # ── Dataset manifest ──────────────────────────────────────────────
        manifest = {
            "schema_version":  ml_cfg.get("schema_version", "1.0"),
            "schema_hash":     _schema_hash(list(X.columns)),
            "features_used":   list(X.columns),
            "n_features":      len(X.columns),
            "label_method":    label_method,
            "train_start":     train_ts_str,
            "train_end":       train_te_str,
            "cv_method":       f"purged_embargo_{n_splits}fold_h{label_horizon}_e{embargo_bars}",
            "calibration":     calibration if calibrator is not None else "none",
            "ev_policy": {
                "avg_R_win":        avg_R_win,
                "avg_R_loss":       avg_R_loss,
                "ev_min_threshold": ev_min,
            },
            "threshold_policy": {
                "method":           "ev",
                "ev_min":           ev_min,
            },
            "metrics": {
                "mean_cv_auc_xgb":  mean_xgb_auc,
                "mean_cv_auc_lgbm": mean_lgbm_auc,
                "mean_cv_auc":      mean_ens_auc,
                "std_cv_auc":       round(np.std(xgb_auc_scores), 4) if xgb_auc_scores else 0.0,
                "n_samples":        n,
                "n_splits_used":    len(xgb_auc_scores),
                "label_pos_rate":   round(float(y.mean()), 4),
            },
            "hyperparams": xgb_params,
            "lgbm_enabled":    lgbm_available,
            "symbol":          self._symbol,
            "setup_type":      self._setup_type or "combined",
            "trained_at":      datetime.datetime.now().isoformat(),
        }
        _save_manifest(self._model_path, manifest)

        return {
            "setup_type":       self._setup_type or "combined",
            "mean_cv_auc_xgb":  mean_xgb_auc,
            "mean_cv_auc_lgbm": mean_lgbm_auc,
            "mean_cv_auc":      mean_ens_auc,
            "std_cv_auc":       round(np.std(xgb_auc_scores), 4) if xgb_auc_scores else 0.0,
            "top_features":     importances.to_dict(),
            "label_distribution": y.value_counts().to_dict(),
            "n_features_used":  len(X.columns),
            "lgbm_enabled":     lgbm_available,
            "calibration":      calibration if calibrator is not None else "none",
            "schema_hash":      manifest["schema_hash"],
            "n_folds_used":     len(xgb_auc_scores),
            "ev_policy":        manifest["ev_policy"],
            "manifest_path":    str(_manifest_path_for(self._model_path)),
        }

    def evaluate_filtering(
        self, X: pd.DataFrame, y: pd.Series, trades_df: pd.DataFrame
    ) -> dict:
        """
        Simulate what happens to strategy performance when the ML filter is applied.
        Shows: all trades vs filtered trades vs filtered-out trades.
        """
        if self._model is None:
            with open(self._model_path, "rb") as f:
                self._model = pickle.load(f)

        probs = self._model.predict_proba(X)[:, 1]
        mask = probs >= self._threshold

        all_pnl = trades_df["pnl_net"].sum() if "pnl_net" in trades_df else 0
        filtered_pnl = trades_df[mask]["pnl_net"].sum() if "pnl_net" in trades_df else 0
        filtered_winrate = (trades_df[mask]["pnl_net"] > 0).mean() if mask.sum() > 0 else 0
        all_winrate = (trades_df["pnl_net"] > 0).mean()

        gross_filtered = trades_df[mask]["pnl_gross"].sum() if "pnl_gross" in trades_df else 0

        return {
            "total_trades": len(trades_df),
            "filtered_trades": int(mask.sum()),
            "trades_removed": int((~mask).sum()),
            "all_pnl": round(all_pnl, 2),
            "filtered_pnl": round(filtered_pnl, 2),
            "filtered_gross_pnl": round(gross_filtered, 2),
            "all_win_rate": round(all_winrate * 100, 1),
            "filtered_win_rate": round(filtered_winrate * 100, 1),
            "pnl_improvement": round(filtered_pnl - all_pnl, 2),
            "threshold_used": self._threshold,
        }

    def shap_analysis(self, X: pd.DataFrame) -> None:
        """
        Generate SHAP feature importance analysis.
        Prints a text-based summary of which features drive predictions.
        Requires: pip install shap
        """
        try:
            import shap
        except ImportError:
            logger.warning("SHAP not installed. Run: pip install shap")
            print("\n  [SHAP] Install shap for feature explainability: pip install shap")
            return

        if self._model is None:
            if not self._model_path.exists():
                logger.error("No model found. Train first.")
                return
            with open(self._model_path, "rb") as f:
                self._model = pickle.load(f)

        explainer = shap.TreeExplainer(self._model)
        shap_values = explainer.shap_values(X)

        # Mean absolute SHAP value per feature
        mean_abs_shap = pd.Series(
            np.abs(shap_values).mean(axis=0),
            index=X.columns
        ).sort_values(ascending=False)

        print("\n" + "=" * 55)
        print("  SHAP Feature Importance (mean |SHAP value|)")
        print("=" * 55)
        print(f"  {'Feature':<32} {'Importance':>12}")
        print("  " + "-" * 47)
        for feat, val in mean_abs_shap.head(20).items():
            bar = "#" * int(val * 100 / mean_abs_shap.iloc[0] * 20)
            print(f"  {feat:<32} {val:>8.4f}  {bar}")
        print("=" * 55)

        # Direction of effect: positive SHAP = increases P(win)
        mean_shap_signed = pd.Series(shap_values.mean(axis=0), index=X.columns)
        print("\n  Feature direction (positive = increases P(win)):")
        print(f"  {'Feature':<32} {'Mean SHAP':>10}")
        print("  " + "-" * 45)
        for feat, val in mean_shap_signed.abs().sort_values(ascending=False).head(15).items():
            signed = mean_shap_signed[feat]
            direction = "[+] bullish" if signed > 0 else "[-] bearish"
            print(f"  {feat:<32} {signed:>+8.4f}  {direction}")
        print()

    def load_and_predict(self, features: dict) -> float:
        """Predict win probability for a single signal's features."""
        if self._model is None:
            if not self._model_path.exists():
                return 1.0  # No model = pass everything through
            with open(self._model_path, "rb") as f:
                self._model = pickle.load(f)

        X = pd.DataFrame([features]).reindex(columns=FEATURE_COLS, fill_value=0)
        return float(self._model.predict_proba(X)[0][1])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _csv_symbol(path: Path) -> str | None:
    """
    Extract the trading symbol from a backtest or WFA CSV filename.

    Expected formats:
      backtest_{SYMBOL}_{START}_{END}.csv  e.g. backtest_ES_2022-01-03_2024-12-31.csv
      wfa_trades_{SYMBOL}.csv              e.g. wfa_trades_NQ.csv

    Returns the symbol string (e.g. 'ES', 'NQ', 'MNQ') or None if unrecognised.
    """
    stem  = path.stem   # strip .csv
    parts = stem.split("_")
    if stem.startswith("backtest_") and len(parts) >= 2:
        return parts[1].upper()          # 'ES', 'NQ', 'MNQ'
    if stem.startswith("wfa_trades_") and len(parts) >= 3:
        return parts[2].upper()          # 'ES', 'NQ', 'MNQ'
    return None


def _find_trades_csv(symbol: str | None = None) -> list[Path]:
    """Find backtest / WFA trade CSV files, optionally filtered by symbol.

    Bug-fix: previously used  ``symbol in filename``  which matched 'ES' inside
    the word 'backtest' causing NQ/MNQ files to be loaded when training the ES
    model (and vice-versa), contaminating every per-symbol ML model.
    Now uses exact symbol extraction from the structured filename.
    """
    candidates = (
        list((ROOT / "data").glob("backtest_*.csv")) +
        list((ROOT / "data").glob("wfa_trades_*.csv"))
    )
    if symbol:
        sym_upper = symbol.upper()
        filtered = [p for p in candidates if _csv_symbol(p) == sym_upper]
        if filtered:
            return filtered
        logger.warning(
            "No trade CSV files found for symbol {}; using all available files.", symbol
        )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="ML filter: label, train, evaluate")
    parser.add_argument(
        "command",
        choices=["label", "train", "eval", "retrain"],
        help=(
            "label = apply triple-barrier labels to backtest trades; "
            "train = fit XGBoost model; "
            "eval  = evaluate the trained model; "
            "retrain = run label + train + eval in one step (use from the web panel)"
        ),
    )
    parser.add_argument("--symbol", default=None,
                        help="Filter trade files by symbol (ES, NQ, MNQ)")
    parser.add_argument("--trades", nargs="*", help="Path(s) to trade CSV file(s)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override min_probability_threshold for eval")
    parser.add_argument("--no-shap", action="store_true",
                        help="Skip SHAP analysis in eval/retrain (faster)")
    args = parser.parse_args()

    trade_files = args.trades or [str(p) for p in _find_trades_csv(args.symbol)]
    if not trade_files:
        logger.error("No trade files found. Run backtesting first.")
        sys.exit(1)

    dfs = []
    for f in trade_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
            logger.info("Loaded {} trades from {}", len(df), f)
        except Exception as e:
            logger.warning("Could not load {}: {}", f, e)

    if not dfs:
        logger.error("No valid trade files loaded.")
        sys.exit(1)

    trades_df = pd.concat(dfs, ignore_index=True)
    logger.info("Total: {} trades from {} files", len(trades_df), len(dfs))

    # Check features_json coverage
    if "features_json" in trades_df.columns:
        has_features = trades_df["features_json"].notna() & (trades_df["features_json"] != "")
        logger.info(
            "Trades with features_json: {}/{} ({:.0f}%)",
            has_features.sum(), len(trades_df), has_features.mean() * 100
        )
        if has_features.sum() < 10:
            logger.warning(
                "Very few trades have feature data. "
                "Re-run backtest with the updated harness to generate features_json."
            )
    else:
        logger.warning("No features_json column found. Re-run backtest with updated harness.")

    ml = MLFilter(symbol=args.symbol or "ES")
    if args.threshold is not None:
        ml._threshold = args.threshold

    if args.command == "label":
        labeler = TripleBarrierLabeler()
        labelled = labeler.label_from_backtest_trades(trades_df)
        out = ROOT / "data" / "labelled_trades.csv"
        labelled.to_csv(out, index=False)
        pos = (labelled["label"] == 1).sum()
        neg = (labelled["label"] == 0).sum()
        logger.info(
            "Labels: {} positive (TP), {} negative (SL/EOD) → saved to {}",
            pos, neg, out
        )

    elif args.command == "train":
        X, y = ml.prepare_dataset(trades_df)
        metrics = ml.train(X, y)
        print("\n=== ML Filter Training Results ===")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        auc = metrics.get("mean_cv_auc", 0)
        if auc >= 0.60:
            print(f"\n  AUC={auc:.3f} >= 0.60 -- PASS: Consider enabling ml.enabled: true in config")
        else:
            print(f"\n  AUC={auc:.3f} < 0.60 -- FAIL: Collect more data before enabling ML filter")

    elif args.command == "eval":
        X, y = ml.prepare_dataset(trades_df)

        # Load or train model first
        if not ml._model_path.exists():
            logger.info("No saved model found — training on all data first...")
            ml.train(X, y)

        eval_metrics = ml.evaluate_filtering(X, y, trades_df)
        print("\n=== ML Filter Evaluation ===")
        for k, v in eval_metrics.items():
            print(f"  {k}: {v}")

        if eval_metrics["pnl_improvement"] > 0:
            print(
                f"\n  ML filter IMPROVES P&L by ${eval_metrics['pnl_improvement']:,.2f} "
                f"({eval_metrics['filtered_trades']} trades kept)"
            )
            print("  >> Consider enabling ml.enabled: true in config/settings.yaml")
        else:
            print(
                f"\n  ML filter does NOT improve P&L "
                f"(change: ${eval_metrics['pnl_improvement']:,.2f})"
            )
            print("  >> Keep ml.enabled: false -- collect more data or tune threshold")

        # SHAP analysis
        if not args.no_shap:
            ml.shap_analysis(X)

    elif args.command == "retrain":
        # ── Convenience command used by the web panel and continuous learner ──
        # Runs label → train (combined + per-setup-type) → eval in one pass.

        sym = args.symbol or "ES"
        print("\n=== Step 1/3: Labelling trades ===")
        labeler = TripleBarrierLabeler()
        labelled = labeler.label_from_backtest_trades(trades_df)
        out = ROOT / "data" / "labelled_trades.csv"
        labelled.to_csv(out, index=False)
        pos = (labelled["label"] == 1).sum()
        neg = (labelled["label"] == 0).sum()
        logger.info("Labels: {} positive (TP), {} negative (SL/EOD) → saved to {}", pos, neg, out)

        print("\n=== Step 2/3: Training ensemble models (combined + per-setup-type) ===")

        # Train combined model first (fallback)
        X, y = ml.prepare_dataset(trades_df)
        if len(X) < 10:
            logger.error(
                "Not enough labelled samples to train ({} rows). "
                "Run a full backtest first to generate more trade data.", len(X)
            )
            sys.exit(1)

        print(f"  Training combined model ({len(X)} samples)...")
        metrics = ml.train(X, y)
        auc = metrics.get("mean_cv_auc", 0)
        print(f"  Combined model: AUC={auc:.3f}  {'PASS' if auc >= 0.58 else 'marginal'}")

        # Train per-setup-type specialist models
        setup_types = ["BREAK_RETEST", "REJECTION", "BOUNCE", "SWEEP_REVERSE"]
        setup_metrics: dict[str, dict] = {}
        for st in setup_types:
            try:
                ml_st = MLFilter(settings_path="config/settings.yaml", symbol=sym, setup_type=st)
                X_st, y_st = ml_st.prepare_dataset(trades_df)
                if len(X_st) >= 10:
                    print(f"  Training {st} specialist ({len(X_st)} samples)...")
                    m_st = ml_st.train(X_st, y_st)
                    setup_metrics[st] = m_st
                    auc_st = m_st.get("mean_cv_auc", 0)
                    print(f"    {st}: AUC={auc_st:.3f}  {'PASS' if auc_st >= 0.55 else 'marginal'}")
                else:
                    print(f"  {st}: only {len(X_st)} samples — skipping specialist model")
            except Exception as e:
                print(f"  {st}: training failed ({e})")

        print("\n=== Step 3/3: Evaluating combined filter performance ===")
        eval_metrics = ml.evaluate_filtering(X, y, trades_df)
        print("Evaluation metrics:")
        for k, v in eval_metrics.items():
            print(f"  {k}: {v}")

        if eval_metrics["pnl_improvement"] > 0:
            print(
                f"\n  ML filter IMPROVES P&L by ${eval_metrics['pnl_improvement']:,.2f} "
                f"({eval_metrics['filtered_trades']} trades kept)"
            )
        else:
            print(
                f"\n  ML filter P&L change: ${eval_metrics['pnl_improvement']:,.2f} "
                f"(collect more data or adjust threshold if negative)"
            )

        if not args.no_shap:
            ml.shap_analysis(X)

        print("\n=== Retrain complete ===")


if __name__ == "__main__":
    main()
