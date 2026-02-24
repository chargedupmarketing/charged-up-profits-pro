"""
scripts/build_dataset.py

Build the ML training dataset (labelled_trades.csv) from the latest backtest logs.

What it does:
  1. Finds all backtest_*.csv files (or the specified files)
  2. Applies the configured label_method (realized_R_binary by default)
  3. Expands features_json into individual feature columns
  4. Enforces schema: reports missing features, fills with 0 and warns
  5. Saves labelled_trades.csv with all 57 feature columns + label + realized_R
  6. Writes build_dataset_report.json with schema_hash, label stats, date range

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --symbols ES NQ --label-method realized_R_binary
    python scripts/build_dataset.py --files data/backtest_ES_*.csv data/backtest_NQ_*.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from loguru import logger

from src.ml_filter import FEATURE_COLS, _schema_hash
from src.labeling.labeler import Labeler


def _add_signal_fingerprint(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a `signal_fingerprint` column that uniquely identifies a signal cluster.

    Fingerprint = hash of:
      (date_bucket, setup_type, direction, entry_price_rounded_to_tick)

    Date bucket = entry_ts truncated to 2-minute windows, so two signals
    at the same level generated 1 minute apart are treated as the same signal.

    Also adds a `trade_id` (UUID-style unique row identifier) if not already present.
    """
    import hashlib
    df = df.copy()

    if "entry_ts" not in df.columns:
        df["signal_fingerprint"] = range(len(df))
        return df

    ts = pd.to_datetime(df["entry_ts"], errors="coerce", utc=True)
    # 2-minute bucket (floor to nearest even minute)
    ts_bucket = ts.dt.floor("2min").dt.strftime("%Y%m%d_%H%M").fillna("unknown")
    setup     = df.get("setup_type", pd.Series(["?"] * len(df))).fillna("?").str.upper()
    direction = df.get("direction", pd.Series(["?"] * len(df))).fillna("?").str.upper()
    # Include symbol to prevent cross-instrument fingerprint collisions
    symbol    = df.get("symbol", pd.Series(["?"] * len(df))).fillna("?").str.upper()
    # Round entry_price to nearest 0.25 (ES/NQ tick)
    entry_r = df.get("entry_price", pd.Series([0.0] * len(df))).fillna(0).round(0)

    df["signal_fingerprint"] = [
        hashlib.md5(f"{sym}|{b}|{s}|{d}|{e}".encode()).hexdigest()[:12]
        for sym, b, s, d, e in zip(symbol, ts_bucket, setup, direction, entry_r)
    ]

    # Per-feature missingness report (log at INFO)
    if len(df) > 0:
        from src.ml_filter import FEATURE_COLS
        missing_pcts = {}
        for col in FEATURE_COLS:
            if col in df.columns:
                miss_pct = float(df[col].isna().mean()) * 100
                if miss_pct > 10:
                    missing_pcts[col] = round(miss_pct, 1)
        if missing_pcts:
            logger.warning(
                "Feature missingness > 10%: {}",
                {k: f"{v}%" for k, v in missing_pcts.items()},
            )

    return df


def build_dataset(
    csv_files: list[str],
    label_method: str = "realized_R_binary",
    output_path: str = "data/labelled_trades.csv",
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load backtest CSVs, apply labels, expand features_json, enforce schema.
    Returns the labelled DataFrame.
    """
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            # Always inject the symbol column from the filename if not present.
            # This is required for deduplication to work correctly across files
            # (ES and NQ signals at the same timestamp must NOT be deduplicated).
            if "symbol" not in df.columns:
                fname = Path(f).stem
                parts = fname.split("_")
                inferred_sym = parts[1].upper() if len(parts) >= 2 and parts[0] == "backtest" else "?"
                df["symbol"] = inferred_sym

            # Filter by symbol if requested
            if symbols:
                df = df[df["symbol"].str.upper().isin([s.upper() for s in symbols])]
                if df.empty:
                    continue
            dfs.append(df)
            logger.info("Loaded {} trades from {}", len(df), f)
        except Exception as e:
            logger.warning("Could not load {}: {}", f, e)

    if not dfs:
        logger.error("No valid trade files loaded.")
        sys.exit(1)

    trades_df = pd.concat(dfs, ignore_index=True)
    logger.info("Total: {} trades from {} files", len(trades_df), len(dfs))

    # ── Step 1: Aggregate partial-exit rows into single combined trade rows ─
    # The backtest harness records partial exits as SEPARATE rows for the same
    # entry (same symbol, entry_ts, setup_type, direction, entry_price).
    # A trade that hit partial_tp (+$290) then a stop (-$60) appears as 2 rows.
    # If we dedup by keeping one row we drop the partial win and mislabel
    # net-profitable trades as losses.
    # FIX: group by trade identity and aggregate:
    #   - pnl_net         = sum (total net P&L across all exit legs)
    #   - realized_R      = total_pnl_net / initial_risk_dollars of the trade
    #   - exit_reason     = last non-partial exit reason (tp/sl/eod)
    #   - all other cols  = first row's values (features, entry info unchanged)
    trade_id_cols = ["setup_type", "direction", "entry_ts", "entry_price"]
    if "symbol" in trades_df.columns:
        trade_id_cols = ["symbol"] + trade_id_cols

    before = len(trades_df)
    if all(c in trades_df.columns for c in trade_id_cols):
        # Identify multi-leg trades
        multi_mask = trades_df.duplicated(subset=trade_id_cols, keep=False)
        singles = trades_df[~multi_mask].copy()
        multi   = trades_df[multi_mask].copy()

        if len(multi) > 0:
            combined_rows = []
            for _, grp in multi.groupby(trade_id_cols, sort=False):
                row = grp.iloc[0].copy()
                total_pnl_net = grp["pnl_net"].sum() if "pnl_net" in grp.columns else 0.0
                row["pnl_net"] = total_pnl_net
                # Recompute realized_R from combined PnL
                if "initial_risk_dollars" in grp.columns:
                    irisk = float(grp["initial_risk_dollars"].iloc[0])
                    row["realized_R"] = (total_pnl_net / irisk) if irisk > 0 else 0.0
                else:
                    row["realized_R"] = grp["realized_R"].sum()
                # Use the final/primary exit reason (prefer tp/sl/eod over partial_tp)
                final_reasons = grp["exit_reason"] if "exit_reason" in grp.columns else pd.Series(["?"])
                primary = [r for r in final_reasons if r not in ("partial_tp",)]
                row["exit_reason"] = primary[-1] if primary else final_reasons.iloc[-1]
                row["pnl_gross"]   = grp["pnl_gross"].sum() if "pnl_gross" in grp.columns else row.get("pnl_gross", 0)
                combined_rows.append(row)

            combined_df = pd.DataFrame(combined_rows).reset_index(drop=True)
            trades_df   = pd.concat([singles, combined_df], ignore_index=True)
            logger.info(
                "Partial-exit aggregation: {} raw rows → {} trades "
                "({} multi-leg trades combined)",
                before, len(trades_df), len(combined_rows),
            )

    # ── Step 2: Signal fingerprint + deduplication (Section E) ───────────
    trades_df = _add_signal_fingerprint(trades_df)
    before = len(trades_df)
    # Hard dedup: exact same symbol + timestamp + setup + direction + price.
    trades_df = trades_df.drop_duplicates(
        subset=trade_id_cols, keep="last"
    ).reset_index(drop=True)
    # Soft dedup: same fingerprint (same symbol + level + setup + 2-min bucket).
    if "signal_fingerprint" in trades_df.columns:
        trades_df = trades_df.drop_duplicates(
            subset=["signal_fingerprint"], keep="last"
        ).reset_index(drop=True)
    if len(trades_df) < before:
        logger.info(
            "Deduplication (exact + fingerprint): {} -> {} trades removed {}",
            before, len(trades_df), before - len(trades_df),
        )

    # ── Apply labels ──────────────────────────────────────────────────────
    labeler = Labeler(label_method=label_method)
    labelled = labeler.from_trade_log(trades_df)

    # ── Expand features_json ──────────────────────────────────────────────
    if "features_json" in labelled.columns and labelled["features_json"].notna().any():
        logger.info("Expanding features_json...")
        feat_rows = []
        for _, row in labelled.iterrows():
            feat = {}
            if row.get("features_json"):
                try:
                    feat = json.loads(row["features_json"])
                except (json.JSONDecodeError, TypeError):
                    pass
            feat_rows.append(feat)
        feat_df = pd.DataFrame(feat_rows)

        # Merge feature columns into labelled
        for col in FEATURE_COLS:
            if col in feat_df.columns:
                labelled[col] = feat_df[col].values
            elif col not in labelled.columns:
                labelled[col] = 0.0
    else:
        logger.warning("No features_json found — feature columns will be all zeros")
        for col in FEATURE_COLS:
            if col not in labelled.columns:
                labelled[col] = 0.0

    # ── Schema validation ─────────────────────────────────────────────────
    missing = [c for c in FEATURE_COLS if c not in labelled.columns]
    if missing:
        logger.error(
            "SCHEMA VIOLATION: {} required features missing after expansion: {}",
            len(missing), missing,
        )
        for col in missing:
            labelled[col] = 0.0

    present = [c for c in FEATURE_COLS if c in labelled.columns]
    schema_h = _schema_hash(present)
    feature_coverage = len(present) / len(FEATURE_COLS) * 100
    logger.info(
        "Feature coverage: {}/{} ({:.0f}%)  schema_hash={}",
        len(present), len(FEATURE_COLS), feature_coverage, schema_h,
    )

    # ── Sort by entry_ts ──────────────────────────────────────────────────
    if "entry_ts" in labelled.columns:
        labelled["entry_ts"] = pd.to_datetime(labelled["entry_ts"], errors="coerce", utc=True)
        labelled = labelled.sort_values("entry_ts").reset_index(drop=True)

    # ── Save ──────────────────────────────────────────────────────────────
    out = ROOT / output_path
    out.parent.mkdir(parents=True, exist_ok=True)
    labelled.to_csv(out, index=False)

    pos = (labelled["label"] == 1).sum() if "label" in labelled.columns else 0
    neg = (labelled["label"] == 0).sum() if "label" in labelled.columns else 0
    logger.success(
        "Dataset saved: {}  trades={} pos={} neg={} "
        "win_rate={:.1f}%  mean_R={}",
        out, len(labelled), pos, neg,
        pos / len(labelled) * 100 if len(labelled) else 0,
        round(labelled["realized_R"].mean(), 4) if "realized_R" in labelled.columns else "n/a",
    )

    # ── Run report ────────────────────────────────────────────────────────
    report = {
        "run_at": datetime.now().isoformat(),
        "label_method": label_method,
        "n_trades": len(labelled),
        "n_positive": int(pos),
        "n_negative": int(neg),
        "win_rate_pct": round(pos / len(labelled) * 100, 2) if len(labelled) else 0,
        "mean_realized_R": round(float(labelled["realized_R"].mean()), 4)
            if "realized_R" in labelled.columns else None,
        "schema_hash": schema_h,
        "features_present": present,
        "features_missing": missing,
        "feature_coverage_pct": round(feature_coverage, 1),
        "output": str(out),
        "source_files": csv_files,
    }
    rp = ROOT / "data" / "build_dataset_report.json"
    rp.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Run report: {}", rp)

    return labelled


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ML training dataset")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Filter by symbol (default: all)")
    parser.add_argument("--files", nargs="*", default=None,
                        help="Explicit CSV file paths (default: all data/backtest_*.csv)")
    parser.add_argument("--label-method", default="realized_R_binary",
                        choices=["realized_R_binary", "tp_binary", "realized_R", "r3class"],
                        help="Label method (default: realized_R_binary)")
    parser.add_argument("--output", default="data/labelled_trades.csv",
                        help="Output path (default: data/labelled_trades.csv)")
    parser.add_argument("--all-files", action="store_true",
                        help="Include ALL backtest CSVs, not just canonical (YYYYMMDD) ones")
    args = parser.parse_args()

    if args.files:
        csv_files = args.files
    else:
        all_files = sorted(glob.glob(str(ROOT / "data" / "backtest_*.csv")))

        # ── Canonical-only filter (default: ON) ──────────────────────────
        # Only use backtest files from the current rebuild pipeline.
        # These are named backtest_{SYM}_{YYYYMMDD}_{YYYYMMDD}.csv (no hyphens in dates).
        # Old files with hyphenated dates (backtest_ES_2020-01-01_...) are
        # often from earlier runs with different feature schemas, no realized_R
        # column, and artificially inflated or deflated win rates that corrupt
        # the training label distribution.
        import re
        _canonical_pattern = re.compile(
            r"backtest_[A-Z]+_\d{8}_\d{8}\.csv$", re.IGNORECASE
        )
        canonical_files = [f for f in all_files if _canonical_pattern.search(f)]

        if not args.all_files and canonical_files:
            csv_files = canonical_files
            logger.info(
                "Using {} canonical backtest files (use --all-files to include all)",
                len(canonical_files),
            )
        else:
            csv_files = all_files
            if not canonical_files:
                logger.warning(
                    "No canonical files found — using all {} backtest files. "
                    "Run rebuild_backtests.py to generate canonical files.",
                    len(all_files),
                )
            # WFA files only when explicitly using all files
            csv_files += sorted(glob.glob(str(ROOT / "data" / "wfa_trades_*.csv")))

        if not csv_files:
            logger.error(
                "No backtest CSV files found in data/. "
                "Run: python scripts/rebuild_backtests.py first"
            )
            sys.exit(1)

    build_dataset(
        csv_files=csv_files,
        label_method=args.label_method,
        output_path=args.output,
        symbols=args.symbols,
    )
    print("\nNext step: python scripts/train_models.py")


if __name__ == "__main__":
    main()
