"""
backtest/clean_eval.py

HONEST temporal holdout evaluation — fixes the data-leakage flaw.

The standard backtest + WFA are contaminated: the ML model was trained on
labelled trades drawn from the SAME time period that is then used for
backtest evaluation.  This inflates win-rates to 92-96% and profit factors
into the hundreds — completely unrealistic numbers.

This script does it correctly:
  1. Load all historical labelled trades (labelled_trades.csv).
  2. Apply a hard CUTOFF_DATE: train = before, test = after (strict temporal split).
  3. Train a fresh "clean" ML model using ONLY the pre-cutoff portion.
  4. Run the backtest on the POST-cutoff period using the clean model.
  5. Also run it on the same period with NO ML filter as the baseline.
  6. Report both so we can see the honest ML alpha over the raw rules edge.

Usage:
    python backtest/clean_eval.py --symbol NQ --cutoff 2024-01-01
    python backtest/clean_eval.py --symbol ES --cutoff 2024-01-01 --start 2024-01-01 --end 2025-12-31
"""
from __future__ import annotations

import argparse
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.harness import BacktestHarness
from src.ml_filter import (
    FEATURE_COLS,
    InferenceFilter,
    MLFilter,
    TripleBarrierLabeler,
    extract_features_from_signal_log,
    _load_settings,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_labelled_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.warning("labelled_trades.csv not found at {}", path)
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["entry_ts"])
    return df


def _train_clean_filter(
    train_df: pd.DataFrame,
    symbol: str,
    settings_path: str = "config/settings.yaml",
    tmpdir: Path | None = None,
) -> InferenceFilter | None:
    """
    Train an XGBoost + LightGBM filter on `train_df` and return an InferenceFilter
    whose models live in `tmpdir` (a temp folder, never touching data/models/active/).

    Returns None if there are too few samples to train reliably.
    """
    if tmpdir is None:
        raise ValueError("tmpdir must be provided")

    cfg = _load_settings(settings_path)
    min_samples = cfg.get("ml", {}).get("min_total_trades", 30)

    if len(train_df) < min_samples:
        logger.warning(
            "Only {} labelled trades before cutoff — insufficient to train a clean ML model "
            "(need >= {}). Returning None (no ML filter).",
            len(train_df), min_samples,
        )
        return None

    try:
        from xgboost import XGBClassifier
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import roc_auc_score
    except ImportError:
        logger.error("xgboost / scikit-learn not installed")
        return None

    lgbm_available = False
    try:
        from lightgbm import LGBMClassifier
        lgbm_available = True
    except ImportError:
        pass

    # ── Prepare X, y ────────────────────────────────────────────────────────
    labeler = TripleBarrierLabeler(settings_path)
    labelled = labeler.label_from_backtest_trades(train_df)

    if "features_json" in labelled.columns and labelled["features_json"].notna().any():
        feature_df = extract_features_from_signal_log(labelled)
    else:
        feature_df = labelled

    avail = [c for c in FEATURE_COLS if c in feature_df.columns]
    if not avail:
        logger.error("No feature columns found in training data")
        return None

    X = feature_df[avail].fillna(0).reset_index(drop=True)
    y = labelled["label"].reset_index(drop=True).astype(int)

    if len(X) < min_samples:
        logger.warning("After feature extraction: {} usable samples — skipping", len(X))
        return None

    logger.info(
        "Training clean ML model on {} samples ({:.1f}% wins), {} features",
        len(X), y.mean() * 100, len(avail),
    )

    # ── Exponential decay weights ────────────────────────────────────────────
    n = len(y)
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

    xgb_m = XGBClassifier(**xgb_params)
    xgb_m.fit(X, y, sample_weight=weights)

    lgbm_m = None
    if lgbm_available:
        from lightgbm import LGBMClassifier
        lgbm_m = LGBMClassifier(**lgbm_params)
        lgbm_m.fit(X, y, sample_weight=weights)

    # Save to temp dir so InferenceFilter can load them
    sym = symbol.upper()
    xgb_path = tmpdir / f"ml_filter_{sym}.pkl"
    with open(xgb_path, "wb") as f:
        pickle.dump(xgb_m, f)

    lgbm_path = None
    if lgbm_m:
        lgbm_path = tmpdir / f"ml_filter_{sym}.lgbm.pkl"
        with open(lgbm_path, "wb") as f:
            pickle.dump(lgbm_m, f)

    # Build InferenceFilter manually from the in-temp-dir models
    threshold = cfg.get("ml", {}).get("min_probability_threshold", 0.62)
    inf = object.__new__(InferenceFilter)
    inf._enabled = True
    inf._threshold = threshold
    inf._sym = sym
    inf._setup_models: dict = {}   # no per-setup-type models in the combined clean filter
    inf._fallback_model = xgb_m
    inf._lgbm_fallback = lgbm_m    # stash for combined inference

    # Store the exact feature list used during training so inference uses the same columns
    inf._train_features = avail

    # Patch allows_trade to use only training features and the freshly-trained models
    def _patched_allows_trade(self_inner, features: dict) -> tuple[bool, float]:
        train_feats = getattr(self_inner, "_train_features", avail)
        row = pd.DataFrame([{c: features.get(c, 0) for c in train_feats}])
        probs = []
        try:
            probs.append(float(self_inner._fallback_model.predict_proba(row)[0, 1]))
        except Exception:
            pass
        if getattr(self_inner, "_lgbm_fallback", None):
            try:
                probs.append(float(self_inner._lgbm_fallback.predict_proba(row)[0, 1]))
            except Exception:
                pass
        if not probs:
            return True, 0.5
        p = float(np.mean(probs))
        return p >= self_inner._threshold, p

    import types
    inf.allows_trade = types.MethodType(_patched_allows_trade, inf)

    logger.info(
        "Clean InferenceFilter built: threshold={}, LGBM={}",
        threshold, lgbm_m is not None,
    )
    return inf


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_clean_eval(
    symbol: str,
    cutoff_date: str,
    test_start: str | None = None,
    test_end: str | None = None,
    settings_path: str = "config/settings.yaml",
) -> dict:
    labelled_path = ROOT / "data" / "labelled_trades.csv"
    all_trades = _load_labelled_trades(labelled_path)

    cutoff_ts = pd.Timestamp(cutoff_date, tz="UTC")
    test_start_ts = pd.Timestamp(test_start or cutoff_date)
    test_end_ts   = pd.Timestamp(test_end or "2025-12-31")

    if all_trades.empty:
        logger.warning("No labelled trades found — running baseline only")
        train_df = pd.DataFrame()
    else:
        train_df = all_trades[all_trades["entry_ts"] < cutoff_ts].copy()
        test_leak = all_trades[all_trades["entry_ts"] >= cutoff_ts]
        logger.info(
            "Temporal split at {}: {} training samples (pre-cutoff), {} test-period samples IN labelled_trades "
            "(these are in-sample leakage — the clean eval uses NONE of them for training)",
            cutoff_date, len(train_df), len(test_leak),
        )

    # Load bar data once (load_data also calls configure_for_symbol internally)
    _loader = BacktestHarness(settings_path)
    df_1m, df_15m = _loader.load_data(symbol)

    # ── BASELINE: raw strategy, no ML filter ────────────────────────────────
    harness_raw = BacktestHarness(settings_path)
    harness_raw.configure_for_symbol(symbol)
    harness_raw.set_ml_filter(None)   # disable ML AFTER configure_for_symbol

    logger.info("=== BASELINE (no ML filter) — test period {} to {} ===",
                test_start_ts.date(), test_end_ts.date())
    raw_trades = harness_raw.run(
        df_1m, df_15m,
        test_start_ts.strftime("%Y-%m-%d"),
        test_end_ts.strftime("%Y-%m-%d"),
    )

    # ── Train clean ML filter on pre-cutoff data ─────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        clean_filter = _train_clean_filter(
            train_df, symbol, settings_path, Path(tmpdir)
        )

    harness_ml = BacktestHarness(settings_path)
    harness_ml.configure_for_symbol(symbol)
    harness_ml.set_ml_filter(clean_filter)  # set AFTER configure_for_symbol

    if clean_filter is not None:
        logger.info("=== CLEAN ML (trained on pre-{}) — test period {} to {} ===",
                    cutoff_date, test_start_ts.date(), test_end_ts.date())
        ml_trades = harness_ml.run(
            df_1m, df_15m,
            test_start_ts.strftime("%Y-%m-%d"),
            test_end_ts.strftime("%Y-%m-%d"),
        )
    else:
        logger.warning("No clean ML filter (insufficient pre-cutoff samples) — ML column will show N/A")
        ml_trades = pd.DataFrame()

    return {
        "symbol": symbol,
        "cutoff": cutoff_date,
        "test_start": test_start_ts.strftime("%Y-%m-%d"),
        "test_end": test_end_ts.strftime("%Y-%m-%d"),
        "train_samples_pre_cutoff": len(train_df),
        "raw_trades": raw_trades,
        "ml_trades": ml_trades,
    }


def _metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "net": 0.0, "exp": 0.0}
    wins = df[df["pnl_net"] > 0]["pnl_net"].sum()
    losses = abs(df[df["pnl_net"] < 0]["pnl_net"].sum())
    return {
        "n":   len(df),
        "wr":  round((df["pnl_net"] > 0).mean() * 100, 1),
        "pf":  round(wins / losses, 2) if losses else 9999.0,
        "net": round(df["pnl_net"].sum(), 2),
        "exp": round(df["pnl_net"].mean(), 2),
    }


def _print_report(result: dict) -> None:
    sym       = result["symbol"]
    cutoff    = result["cutoff"]
    t_start   = result["test_start"]
    t_end     = result["test_end"]
    n_train   = result["train_samples_pre_cutoff"]
    raw       = _metrics(result["raw_trades"])
    ml        = _metrics(result["ml_trades"])

    W = 72
    print("\n" + "=" * W)
    print(f"  HONEST FORWARD-TEST RESULTS — {sym}")
    print("=" * W)
    print(f"  Train/test split  : trades before {cutoff} used to train ML")
    print(f"  Pre-cutoff samples: {n_train} labelled trades available for training")
    print(f"  Test period       : {t_start} -> {t_end}")
    print("-" * W)
    print(f"  {'Metric':<28} {'Raw (no ML)':>16} {'Clean ML (OOS)':>18}")
    print("-" * W)
    print(f"  {'Trades':28} {raw['n']:>16d} {ml['n'] if ml['n'] else 'N/A':>18}")
    print(f"  {'Win Rate':28} {raw['wr']:>15.1f}% {str(ml['wr'])+'%' if ml['n'] else 'N/A':>18}")
    print(f"  {'Profit Factor':28} {raw['pf']:>16.2f} {ml['pf'] if ml['n'] else 'N/A':>18}")
    _ml_net = f"${ml['net']:,.0f}" if ml['n'] else 'N/A'
    _ml_exp = f"${ml['exp']:.2f}" if ml['n'] else 'N/A'
    print(f"  {'Net P&L':28} ${raw['net']:>14,.0f} {_ml_net:>18}")
    print(f"  {'Expectancy / trade':28} ${raw['exp']:>14.2f} {_ml_exp:>18}")
    print("=" * W)

    if raw["n"] > 0:
        if raw["pf"] < 1.0:
            print("  RAW EDGE: Strategy has NEGATIVE expectancy — review setup rules")
        elif raw["pf"] < 1.5:
            print("  RAW EDGE: Modest (PF < 1.5) — viable but thin margin")
        elif raw["pf"] < 2.5:
            print("  RAW EDGE: Good (PF 1.5-2.5) — solid rules-based alpha")
        else:
            print("  RAW EDGE: Strong (PF > 2.5) — verify simulation assumptions")

    if ml["n"] > 0 and raw["n"] > 0:
        lift = ml["pf"] - raw["pf"]
        if lift > 20:
            print(f"  ML ALPHA:  PF lift = +{lift:.2f} — SUSPICIOUS: re-check cutoff / leakage")
        elif lift > 0:
            print(f"  ML ALPHA:  PF lift = +{lift:.2f} — genuine ML improvement")
        else:
            print(f"  ML ALPHA:  PF lift = {lift:.2f} — ML not helping on this holdout period")

    print()

    # Save clean ML results to a distinct file so they don't overwrite contaminated ones
    if not result["ml_trades"].empty:
        out = ROOT / "data" / f"clean_eval_{sym}_{t_start}_{t_end}.csv"
        result["ml_trades"].to_csv(out, index=False)
        logger.info("Saved clean-eval trades to {}", out)
    if not result["raw_trades"].empty:
        out_raw = ROOT / "data" / f"clean_eval_{sym}_raw_{t_start}_{t_end}.csv"
        result["raw_trades"].to_csv(out_raw, index=False)
        logger.info("Saved raw-strategy trades to {}", out_raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Honest temporal-holdout backtest evaluation")
    parser.add_argument("--symbol", default="NQ")
    parser.add_argument(
        "--cutoff", default="2024-01-01",
        help="Date dividing train (before) from test (after). Default: 2024-01-01",
    )
    parser.add_argument("--start",  default=None, help="Test period start (default=cutoff)")
    parser.add_argument("--end",    default="2025-12-31", help="Test period end")
    parser.add_argument("--settings", default="config/settings.yaml")
    args = parser.parse_args()

    result = run_clean_eval(
        symbol=args.symbol,
        cutoff_date=args.cutoff,
        test_start=args.start,
        test_end=args.end,
        settings_path=args.settings,
    )
    _print_report(result)


if __name__ == "__main__":
    main()
