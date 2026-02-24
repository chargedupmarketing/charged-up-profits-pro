"""
scripts/rebuild_backtests.py

Regenerate backtest CSV logs so every row has features_json containing all 57
features (including the 2 new regime features: efficiency_ratio, vol_state).

Run this whenever:
  - feature_builder.py gains new features
  - config/feature_schema.yaml is updated
  - You want to freshen the ML training dataset

Usage:
    python scripts/rebuild_backtests.py --symbols ES NQ --start 2020-01-01 --end 2025-12-31
    python scripts/rebuild_backtests.py --symbols ES NQ  # uses full available history

After this, run:
    python scripts/build_dataset.py
    python scripts/train_models.py --symbols ES NQ
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger
from backtest.harness import BacktestHarness


def rebuild(
    symbols: list[str],
    start: str | None = None,
    end:   str | None = None,
) -> dict[str, Path]:
    """
    Re-run backtests for all symbols and save fresh CSV logs to data/.
    Returns {symbol: csv_path}.
    """
    output_paths: dict[str, Path] = {}

    for sym in symbols:
        logger.info("=" * 60)
        logger.info("Rebuilding backtest for {} ({} to {})", sym, start or "all", end or "all")
        logger.info("=" * 60)

        harness = BacktestHarness()
        harness.set_ml_filter(None)   # no ML filter during rebuild — raw strategy only
        df_1m, df_15m = harness.load_data(sym)

        results = harness.run(df_1m, df_15m, start_date=start, end_date=end)

        if results.empty:
            logger.warning("{}: No trades generated — check data range and config", sym)
            continue

        # Verify feature coverage
        if "features_json" in results.columns:
            n_with_features = (results["features_json"].notna() & (results["features_json"] != "")).sum()
            logger.info(
                "{}: {} / {} trades have features_json ({:.0f}%)",
                sym, n_with_features, len(results), n_with_features / len(results) * 100,
            )

            # Spot-check feature count
            if n_with_features > 0:
                sample_json = results.loc[
                    results["features_json"].notna() & (results["features_json"] != ""),
                    "features_json"
                ].iloc[0]
                try:
                    sample_feat = json.loads(sample_json)
                    logger.info("{}: Sample features_json has {} keys", sym, len(sample_feat))
                except Exception:
                    pass

        # Build output filename
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        start_str = start.replace("-", "") if start else "all"
        end_str   = end.replace("-", "")   if end   else "all"
        out_path  = ROOT / "data" / f"backtest_{sym}_{start_str}_{end_str}.csv"

        results.to_csv(out_path, index=False)
        output_paths[sym] = out_path
        logger.success("{}: Saved {} trades to {}", sym, len(results), out_path)

        # Print quick metrics
        metrics = BacktestHarness.compute_metrics(results)
        logger.info(
            "{}: trades={} win_rate={:.1f}% PF={} mean_R={}",
            sym,
            metrics.get("total_trades"),
            metrics.get("win_rate"),
            metrics.get("profit_factor"),
            metrics.get("mean_realized_R", "n/a"),
        )

    # Write run report
    report = {
        "run_at": datetime.now().isoformat(),
        "symbols": symbols,
        "start": start,
        "end": end,
        "outputs": {k: str(v) for k, v in output_paths.items()},
    }
    report_path = ROOT / "data" / "rebuild_backtests_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Run report: {}", report_path)

    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild backtest CSV logs with all features")
    parser.add_argument("--symbols", nargs="+", default=["ES", "NQ"],
                        help="Symbols to backtest (default: ES NQ)")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD (default: all history)")
    parser.add_argument("--end",   default=None, help="End date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    output_paths = rebuild(args.symbols, args.start, args.end)

    if not output_paths:
        logger.error("No backtests completed — check historical data files.")
        sys.exit(1)

    print("\nRebuilt backtest files:")
    for sym, path in output_paths.items():
        print(f"  {sym}: {path}")
    print("\nNext step: python scripts/build_dataset.py")


if __name__ == "__main__":
    main()
