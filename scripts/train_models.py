"""
scripts/train_models.py

Train XGBoost + LightGBM ensemble models using:
  - Purged + embargoed time-series CV (no label-horizon leakage)
  - Isotonic probability calibration
  - EV-based policy gating (replaces fixed 0.62 threshold)
  - dataset_manifest.json saved alongside each model

Reads from: data/labelled_trades.csv (built by scripts/build_dataset.py)
Writes to:  data/models/pending/  (then user approves via dashboard)

Usage:
    python scripts/train_models.py --symbols ES NQ
    python scripts/train_models.py --symbols NQ --setup-types BREAK_RETEST REJECTION
    python scripts/train_models.py --symbols ES NQ --activate  # skip approval, move to active/
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from loguru import logger

from src.ml_filter import MLFilter, FEATURE_COLS, _schema_hash, _manifest_path_for


SETUP_TYPES = ["BREAK_RETEST", "REJECTION", "BOUNCE", "SWEEP_REVERSE"]
MIN_SAMPLES  = 50    # minimum trades needed to train a specialist model
MIN_COMBINED = 100   # minimum for a combined symbol model


def _load_labelled(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.error(
            "Labelled dataset not found: {}. "
            "Run: python scripts/build_dataset.py first",
            path,
        )
        sys.exit(1)
    df = pd.read_csv(path)
    if "entry_ts" in df.columns:
        df["entry_ts"] = pd.to_datetime(df["entry_ts"], errors="coerce", utc=True)
        df = df.sort_values("entry_ts").reset_index(drop=True)
    logger.info("Loaded {} labelled trades from {}", len(df), path)
    return df


def _split_timestamps(df: pd.DataFrame) -> pd.Series | None:
    """Return entry timestamps for purged CV if available."""
    if "entry_ts" in df.columns and df["entry_ts"].notna().any():
        return df["entry_ts"]
    return None


def _model_size_summary(labelled: pd.DataFrame, symbol: str) -> None:
    """Log sample counts for each setup type."""
    sym_mask = pd.Series([True] * len(labelled))  # all rows if no symbol column
    if "symbol" in labelled.columns:
        sym_mask = labelled["symbol"].str.upper() == symbol.upper()

    sym_df = labelled[sym_mask]
    logger.info(
        "{}: {} total labelled trades  label_method={}",
        symbol, len(sym_df),
        "realized_R_binary" if "realized_R" in labelled.columns else "tp_binary",
    )
    if "setup_type" in sym_df.columns:
        for st, grp in sym_df.groupby("setup_type"):
            pos = (grp.get("label", pd.Series()) == 1).sum()
            logger.info("  {} {}: {} trades ({} pos / {} neg)", symbol, st, len(grp), pos, len(grp) - pos)


def train_symbol(
    labelled: pd.DataFrame,
    symbol: str,
    setup_types: list[str],
    output_dir: Path,
    activate: bool = False,
) -> dict:
    """
    Train combined + per-setup-type models for one symbol.
    Returns a summary dict.
    """
    results: dict[str, dict] = {}
    _model_size_summary(labelled, symbol)

    # Filter to this symbol if the column exists
    if "symbol" in labelled.columns:
        sym_df = labelled[labelled["symbol"].str.upper() == symbol.upper()].copy()
    else:
        sym_df = labelled.copy()

    if len(sym_df) < MIN_COMBINED:
        logger.warning(
            "{}: only {} samples — need at least {} for combined model. Skipping.",
            symbol, len(sym_df), MIN_COMBINED,
        )
        return results

    timestamps = _split_timestamps(sym_df)

    # ── Combined (fallback) model ─────────────────────────────────────────
    logger.info("{}: Training combined model ({} samples)...", symbol, len(sym_df))
    ml_combined = MLFilter(settings_path="config/settings.yaml", symbol=symbol)
    X, y = ml_combined.prepare_dataset(sym_df)
    if len(X) >= MIN_COMBINED:
        m = ml_combined.train(X, y, timestamps=timestamps)
        results["combined"] = m
        auc = m.get("mean_cv_auc", 0)
        cal = m.get("calibration", "none")
        logger.info(
            "{} combined: AUC={:.3f} cal={} schema={} folds={}",
            symbol, auc, cal, m.get("schema_hash", "?"),
            m.get("n_folds_used", "?"),
        )
    else:
        logger.warning("{}: combined dataset too small after filtering ({})", symbol, len(X))

    # ── Per-setup-type specialist models ─────────────────────────────────
    for st in setup_types:
        st_df = sym_df.copy()
        if "setup_type" in st_df.columns:
            st_df = st_df[st_df["setup_type"].str.upper() == st].copy()

        if len(st_df) < MIN_SAMPLES:
            logger.info("{} {}: {} samples < {} min — skipping", symbol, st, len(st_df), MIN_SAMPLES)
            continue

        logger.info("{}: Training {} specialist ({} samples)...", symbol, st, len(st_df))
        ml_st = MLFilter(settings_path="config/settings.yaml", symbol=symbol, setup_type=st)
        ts_st = _split_timestamps(st_df)
        X_st, y_st = ml_st.prepare_dataset(st_df)
        if len(X_st) < MIN_SAMPLES:
            logger.info("{} {}: {} filtered samples — skipping", symbol, st, len(X_st))
            continue

        m_st = ml_st.train(X_st, y_st, timestamps=ts_st)
        results[st] = m_st
        auc_st = m_st.get("mean_cv_auc", 0)
        logger.info(
            "{} {}: AUC={:.3f} cal={} folds={}",
            symbol, st, auc_st,
            m_st.get("calibration", "none"),
            m_st.get("n_folds_used", "?"),
        )

    # Move models to pending/ (or active/ if --activate)
    src_dir = ROOT / "data" / "models" / "active"
    if activate:
        dest_dir = src_dir
        logger.info("{}: Models written directly to active/", symbol)
    else:
        dest_dir = ROOT / "data" / "models" / "pending"
        dest_dir.mkdir(parents=True, exist_ok=True)
        # Copy model files to pending
        for f in src_dir.glob(f"ml_filter_{symbol.upper()}*"):
            shutil.copy2(f, dest_dir / f.name)
            logger.info("Copied {} -> {}", f.name, dest_dir)
        logger.info("{}: Models in pending/ — approve via dashboard to go live", symbol)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML models with purged CV + EV gating")
    parser.add_argument("--symbols", nargs="+", default=["ES", "NQ"],
                        help="Symbols to train (default: ES NQ)")
    parser.add_argument("--setup-types", nargs="*", default=SETUP_TYPES,
                        help="Setup types to train specialist models for")
    parser.add_argument("--labelled", default="data/labelled_trades.csv",
                        help="Path to labelled_trades.csv")
    parser.add_argument("--activate", action="store_true",
                        help="Move models directly to active/ without pending approval")
    args = parser.parse_args()

    labelled = _load_labelled(ROOT / args.labelled)
    output_dir = ROOT / "data" / "models" / "active"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}
    for sym in args.symbols:
        logger.info("\n{'=' * 60}")
        logger.info("Training models for {}", sym)
        r = train_symbol(
            labelled=labelled,
            symbol=sym,
            setup_types=args.setup_types,
            output_dir=output_dir,
            activate=args.activate,
        )
        all_results[sym] = r

    # ── Run report ────────────────────────────────────────────────────────
    report = {
        "run_at": datetime.now().isoformat(),
        "symbols": args.symbols,
        "setup_types": args.setup_types,
        "labelled_file": args.labelled,
        "activated": args.activate,
        "results": {
            sym: {
                model_key: {
                    "auc": m.get("mean_cv_auc"),
                    "schema_hash": m.get("schema_hash"),
                    "calibration": m.get("calibration"),
                    "n_folds": m.get("n_folds_used"),
                    "ev_policy": m.get("ev_policy"),
                    "manifest_path": m.get("manifest_path"),
                }
                for model_key, m in models.items()
            }
            for sym, models in all_results.items()
        },
    }
    rp = ROOT / "data" / "train_models_report.json"
    rp.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Run report: {}", rp)

    print("\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    for sym, models in all_results.items():
        for model_key, m in models.items():
            auc = m.get("mean_cv_auc", 0)
            cal = m.get("calibration", "none")
            status = "PASS" if auc >= 0.58 else "marginal"
            print(f"  {sym} {model_key:<25} AUC={auc:.3f} cal={cal:<8} [{status}]")
    print("=" * 60)
    if not args.activate:
        print("\n  Models saved to data/models/pending/")
        print("  Approve via the dashboard 'Test & Train AI' tab to go live.")
    else:
        print("\n  Models written directly to data/models/active/")


if __name__ == "__main__":
    main()
