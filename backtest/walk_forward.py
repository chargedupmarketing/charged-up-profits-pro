"""
backtest/walk_forward.py

Walk-forward evaluation framework.

Splits the full data range into non-overlapping train/test windows,
runs the backtest on each test window independently, then aggregates.
This prevents fitting the strategy parameters to the test period.

Usage:
    python backtest/walk_forward.py --symbol ES --start 2022-01-03 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.harness import BacktestHarness
from backtest.clean_eval import _load_labelled_trades, _train_clean_filter


def _load_settings(path: str = "config/settings.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


@dataclass
class WalkForwardWindow:
    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    metrics: dict
    trade_count: int
    pnl_net: float


class WalkForwardAnalysis:
    """
    Walk-forward test with per-window ML retraining.

      ┌──────────────────┬──────────┐
      │  TRAIN (18 mo)   │ TEST (6m)│  Window 1
      ├──────────────────┬──────────┤
      │     (shift 6m)   │          │
      │  TRAIN (18 mo)   │ TEST (6m)│  Window 2
      └──────────────────┴──────────┘  ...

    For each window the ML model is retrained on ONLY the trades whose
    entry_ts falls within [train_start, train_end].  The test window is
    then evaluated with that window-specific model — giving truly OOS results.

    If fewer than min_total_trades labelled samples exist for a training
    window, the window runs without an ML filter (raw strategy baseline).
    """

    def __init__(self, settings_path: str = "config/settings.yaml") -> None:
        self._cfg = _load_settings(settings_path)
        self._bt_cfg = self._cfg["backtest"]
        self._settings_path = settings_path
        self._labelled_trades: pd.DataFrame | None = None  # lazy-loaded

    def _get_labelled_trades(self) -> pd.DataFrame:
        if self._labelled_trades is None:
            self._labelled_trades = _load_labelled_trades(ROOT / "data" / "labelled_trades.csv")
        return self._labelled_trades

    def _train_window_filter(self, symbol: str, train_start: str, train_end: str, tmpdir: Path):
        """Return a per-window InferenceFilter trained strictly on [train_start, train_end]."""
        all_lt = self._get_labelled_trades()
        if all_lt.empty:
            return None
        ts_start = pd.Timestamp(train_start, tz="UTC")
        ts_end   = pd.Timestamp(train_end,   tz="UTC")
        window_df = all_lt[
            (all_lt["entry_ts"] >= ts_start) & (all_lt["entry_ts"] <= ts_end)
        ].copy()
        if window_df.empty:
            logger.info("Window train={} → {}: no labelled trades, running without ML filter", train_start, train_end)
            return None
        logger.info(
            "Window train={} → {}: {} labelled samples for ML retraining",
            train_start, train_end, len(window_df),
        )
        return _train_clean_filter(window_df, symbol, self._settings_path, tmpdir)

    def run(
        self,
        symbol: str,
        overall_start: str,
        overall_end: str,
    ) -> tuple[list[WalkForwardWindow], pd.DataFrame]:
        """
        Returns (windows, combined_test_trades_df).
        Each test window uses an ML model trained ONLY on that window's training data.
        """
        harness = BacktestHarness()
        df_1m, df_15m = harness.load_data(symbol)

        windows = self._generate_windows(overall_start, overall_end)
        if not windows:
            logger.error("Could not generate any walk-forward windows for the given date range")
            sys.exit(1)

        total_windows = len(windows)
        logger.info("Running walk-forward with {} windows (per-window ML retraining)", total_windows)
        all_test_trades = []
        results: list[WalkForwardWindow] = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info("WF_WINDOW {}/{}", i + 1, total_windows)
            logger.info(
                "Window {}: train={} -> {} | embargo | test={} -> {}",
                i + 1, train_start, train_end, test_start, test_end
            )

            # Train a window-specific ML filter on the training period only
            with tempfile.TemporaryDirectory() as tmpdir:
                window_filter = self._train_window_filter(symbol, train_start, train_end, Path(tmpdir))

                test_harness = BacktestHarness(self._settings_path)
                test_harness.configure_for_symbol(symbol)
                # Inject the window-specific model (or None for raw baseline)
                test_harness.set_ml_filter(window_filter)
                if window_filter is None:
                    logger.info("Window {}: no ML filter (insufficient training data) — raw rules only", i + 1)
                test_trades = test_harness.run(df_1m, df_15m, test_start, test_end)

            metrics = BacktestHarness.compute_metrics(test_trades)
            pnl = test_trades["pnl_net"].sum() if not test_trades.empty else 0.0

            wf_window = WalkForwardWindow(
                window_id=i + 1,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                metrics=metrics,
                trade_count=len(test_trades),
                pnl_net=round(pnl, 2),
            )
            results.append(wf_window)

            if not test_trades.empty:
                test_trades["window_id"] = i + 1
                all_test_trades.append(test_trades)

        combined = (
            pd.concat(all_test_trades, ignore_index=True)
            if all_test_trades
            else pd.DataFrame()
        )
        return results, combined

    def _generate_windows(
        self,
        overall_start: str,
        overall_end: str,
    ) -> list[tuple[str, str, str, str]]:
        """
        Generate (train_start, train_end, test_start, test_end) tuples.

        An embargo gap is inserted between train_end and test_start to prevent
        label leakage from overlapping bars near the boundary.  The embargo
        period defaults to 10 days and is configurable via
        backtest.embargo_days in settings.yaml.
        """
        train_months = self._bt_cfg["train_months"]
        test_months = self._bt_cfg["test_months"]
        rolls = self._bt_cfg.get("walk_forward_rolls", 3)
        embargo_days = self._bt_cfg.get("embargo_days", 10)

        start = pd.Timestamp(overall_start)
        end = pd.Timestamp(overall_end)
        windows = []

        for roll in range(rolls):
            offset_months = roll * test_months
            train_start = start + pd.DateOffset(months=offset_months)
            train_end = train_start + pd.DateOffset(months=train_months) - pd.Timedelta(days=1)
            # Embargo: skip N days after train_end before test starts
            test_start = train_end + pd.Timedelta(days=embargo_days + 1)
            test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)

            if test_end > end:
                test_end = end
            if test_start > end:
                break

            windows.append((
                train_start.strftime("%Y-%m-%d"),
                train_end.strftime("%Y-%m-%d"),
                test_start.strftime("%Y-%m-%d"),
                test_end.strftime("%Y-%m-%d"),
            ))

        return windows

    @staticmethod
    def print_summary(windows: list[WalkForwardWindow], combined: pd.DataFrame) -> None:
        print("\n" + "=" * 70)
        print("  WALK-FORWARD SUMMARY")
        print("=" * 70)
        print(f"  {'Window':<8} {'Test Period':<24} {'Trades':<8} {'P&L Net $':<12} {'Win%':<8} {'PF':<6}")
        print("-" * 70)
        for w in windows:
            win_rate = w.metrics.get("win_rate", 0)
            pf = w.metrics.get("profit_factor", 0)
            period = f"{w.test_start} to {w.test_end}"
            print(f"  {w.window_id:<8} {period:<24} {w.trade_count:<8} {w.pnl_net:<12.2f} {win_rate:<8.1f} {pf:<6.2f}")

        if not combined.empty:
            combined_metrics = BacktestHarness.compute_metrics(combined)
            print("-" * 70)
            print("  COMBINED OUT-OF-SAMPLE:")
            for k in ["total_trades", "win_rate", "profit_factor", "total_pnl_net",
                       "expectancy_per_trade", "max_drawdown", "sharpe_ratio"]:
                print(f"    {k:<30} {combined_metrics.get(k, 'N/A')}")

        print("=" * 70)

        # Stability check
        positive_windows = sum(1 for w in windows if w.pnl_net > 0)
        pct = positive_windows / len(windows) * 100 if windows else 0
        print(f"\n  Profitable windows: {positive_windows}/{len(windows)} ({pct:.0f}%)")
        if pct >= 67:
            print("  ASSESSMENT: Strategy shows stable positive expectancy")
        elif pct >= 50:
            print("  ASSESSMENT: Marginal — review filters before going live")
        else:
            print("  ASSESSMENT: Strategy failed OOS test — do NOT go live")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest")
    parser.add_argument("--symbol", default="ES", help="Instrument symbol (ES, NQ, MNQ)")
    parser.add_argument("--start", default="2022-01-03")
    parser.add_argument("--end", default="2024-12-31")
    args = parser.parse_args()

    wfa = WalkForwardAnalysis()
    windows, combined = wfa.run(args.symbol, args.start, args.end)
    WalkForwardAnalysis.print_summary(windows, combined)

    if not combined.empty:
        out = ROOT / "data" / f"wfa_trades_{args.symbol}.csv"
        combined.to_csv(out, index=False)
        logger.info("Saved combined OOS trades to {}", out)


if __name__ == "__main__":
    main()
