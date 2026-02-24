"""
backtest/analyze_trades.py

Diagnostic analysis of backtest trade logs.

Outputs:
  - P&L and win rate by setup type
  - P&L and win rate by hour-of-day
  - P&L and win rate by day-of-week
  - P&L and win rate by ATR regime (high/low vol)
  - Per-direction breakdown (LONG vs SHORT)
  - Worst trades and best trades
  - Equity curve summary

Usage:
    python backtest/analyze_trades.py
    python backtest/analyze_trades.py --file data/backtest_ES_2022-01-03_2024-12-31.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_latest_csv() -> Path:
    candidates = sorted((ROOT / "data").glob("backtest_*.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        logger.error("No backtest CSV files found in data/. Run backtest first.")
        sys.exit(1)
    return candidates[-1]


def _load_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True)
    df["entry_hour"] = df["entry_ts"].dt.hour
    df["entry_minute"] = df["entry_ts"].dt.minute
    df["entry_dow"] = df["entry_ts"].dt.dayofweek  # 0=Mon, 4=Fri
    df["entry_dow_name"] = df["entry_ts"].dt.day_name()
    df["win"] = (df["pnl_net"] > 0).astype(int)

    # Extract ATR regime from features_json if available
    if "features_json" in df.columns:
        def _extract_atr_regime(row):
            if not isinstance(row, str) or not row:
                return None
            try:
                d = json.loads(row)
                return d.get("atr_regime", None)
            except Exception:
                return None
        df["atr_regime"] = df["features_json"].apply(_extract_atr_regime)
    else:
        df["atr_regime"] = None

    return df


def _section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def _table(df: pd.DataFrame, index_label: str) -> None:
    hdr = f"  {index_label:<20} {'Trades':>8} {'Win%':>8} {'PnL Net$':>12} {'Avg Win$':>10} {'Avg Loss$':>11}"
    print(hdr)
    print("  " + "-" * 63)
    for _, row in df.iterrows():
        print(
            f"  {str(row['label']):<20} {int(row['trades']):>8} "
            f"{row['win_rate']:>7.1f}% {row['pnl_net']:>12.2f} "
            f"{row['avg_win']:>10.2f} {row['avg_loss']:>11.2f}"
        )


def _group_stats(df: pd.DataFrame, group_col: str, label_col: str | None = None) -> pd.DataFrame:
    label_col = label_col or group_col
    rows = []
    for val, grp in df.groupby(group_col):
        wins = grp[grp["win"] == 1]
        losses = grp[grp["win"] == 0]
        rows.append({
            "label": val,
            "trades": len(grp),
            "win_rate": grp["win"].mean() * 100,
            "pnl_net": grp["pnl_net"].sum(),
            "avg_win": wins["pnl_net"].mean() if len(wins) else 0.0,
            "avg_loss": losses["pnl_net"].mean() if len(losses) else 0.0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis sections
# ---------------------------------------------------------------------------

def analyze_by_setup(df: pd.DataFrame) -> None:
    _section("P&L by Setup Type")
    stats = _group_stats(df, "setup_type")
    _table(stats, "Setup Type")


def analyze_by_direction(df: pd.DataFrame) -> None:
    _section("P&L by Direction")
    stats = _group_stats(df, "direction")
    _table(stats, "Direction")


def analyze_by_hour(df: pd.DataFrame) -> None:
    _section("P&L by Entry Hour (ET)")
    stats = _group_stats(df, "entry_hour")
    stats["label"] = stats["label"].apply(lambda h: f"{int(h):02d}:00")
    stats = stats.sort_values("label")
    _table(stats, "Hour (ET)")


def analyze_by_dow(df: pd.DataFrame) -> None:
    _section("P&L by Day of Week")
    order = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4}
    stats = _group_stats(df, "entry_dow_name")
    stats["sort_key"] = stats["label"].map(order)
    stats = stats.sort_values("sort_key")
    _table(stats, "Day of Week")


def analyze_by_regime(df: pd.DataFrame) -> None:
    _section("P&L by ATR Regime")
    if df["atr_regime"].isna().all():
        print("  (No ATR regime data — re-run backtest with updated harness to capture features)")
        return
    # Classify into Low / Normal / High
    def _classify(v):
        if v is None or pd.isna(v):
            return "unknown"
        try:
            v = float(v)
        except (TypeError, ValueError):
            return "unknown"
        if v < 0.75:
            return "Low Vol (<0.75x)"
        elif v > 1.5:
            return "High Vol (>1.5x)"
        else:
            return "Normal Vol"
    df = df.copy()
    df["regime_label"] = df["atr_regime"].apply(_classify)
    stats = _group_stats(df, "regime_label")
    _table(stats, "ATR Regime")


def analyze_exit_reasons(df: pd.DataFrame) -> None:
    _section("Exit Reason Breakdown")
    stats = _group_stats(df, "exit_reason")
    _table(stats, "Exit Reason")


def analyze_equity_curve(df: pd.DataFrame) -> None:
    _section("Equity Curve Summary")
    df_sorted = df.sort_values("entry_ts").copy()
    df_sorted["cumulative_pnl"] = df_sorted["pnl_net"].cumsum()
    peak = df_sorted["cumulative_pnl"].cummax()
    drawdown = df_sorted["cumulative_pnl"] - peak

    print(f"  Starting equity (relative):  $0.00")
    print(f"  Ending equity  (relative):   ${df_sorted['cumulative_pnl'].iloc[-1]:,.2f}")
    print(f"  Peak equity:                 ${peak.max():,.2f}")
    print(f"  Max drawdown:                ${drawdown.min():,.2f}")

    # Monthly P&L
    df_sorted["month"] = df_sorted["entry_ts"].dt.to_period("M")
    monthly = df_sorted.groupby("month")["pnl_net"].sum()
    pos_months = (monthly > 0).sum()
    neg_months = (monthly <= 0).sum()
    print(f"\n  Monthly P&L: {pos_months} profitable months / {neg_months} losing months")
    print(f"  Best month:  ${monthly.max():,.2f}  ({monthly.idxmax()})")
    print(f"  Worst month: ${monthly.min():,.2f}  ({monthly.idxmin()})")


def analyze_top_bottom_trades(df: pd.DataFrame, n: int = 5) -> None:
    _section(f"Top {n} Wins and Bottom {n} Losses")
    top = df.nlargest(n, "pnl_net")[["date", "setup_type", "direction", "entry_price",
                                      "exit_price", "exit_reason", "pnl_net"]]
    bot = df.nsmallest(n, "pnl_net")[["date", "setup_type", "direction", "entry_price",
                                       "exit_price", "exit_reason", "pnl_net"]]
    print(f"\n  TOP {n} WINS:")
    print("  " + top.to_string(index=False).replace("\n", "\n  "))
    print(f"\n  BOTTOM {n} LOSSES:")
    print("  " + bot.to_string(index=False).replace("\n", "\n  "))


def analyze_consecutive_runs(df: pd.DataFrame) -> None:
    _section("Consecutive Win/Loss Streaks")
    df_sorted = df.sort_values("entry_ts")
    wins = df_sorted["win"].tolist()

    max_win_streak = max_loss_streak = 0
    cur_win = cur_loss = 0
    for w in wins:
        if w:
            cur_win += 1
            cur_loss = 0
        else:
            cur_loss += 1
            cur_win = 0
        max_win_streak = max(max_win_streak, cur_win)
        max_loss_streak = max(max_loss_streak, cur_loss)

    print(f"  Max consecutive wins:   {max_win_streak}")
    print(f"  Max consecutive losses: {max_loss_streak}")


def print_summary(df: pd.DataFrame) -> None:
    _section("Overall Summary")
    n = len(df)
    wins = df[df["win"] == 1]
    losses = df[df["win"] == 0]
    gross_profit = wins["pnl_net"].sum()
    gross_loss = abs(losses["pnl_net"].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    print(f"  Total trades:      {n}")
    print(f"  Win rate:          {df['win'].mean()*100:.1f}%")
    print(f"  Profit factor:     {pf:.2f}")
    print(f"  Net P&L:           ${df['pnl_net'].sum():,.2f}")
    print(f"  Gross P&L:         ${df['pnl_gross'].sum():,.2f}")
    print(f"  Total commission:  ${df['commission_dollars'].sum():,.2f}")
    print(f"  Total slippage:    ${df['slippage_dollars'].sum():,.2f}")
    print(f"  Avg win:           ${wins['pnl_net'].mean():,.2f}" if len(wins) else "  Avg win:  N/A")
    print(f"  Avg loss:          ${losses['pnl_net'].mean():,.2f}" if len(losses) else "  Avg loss: N/A")
    print(f"  Expectancy/trade:  ${df['pnl_net'].mean():,.2f}")

    # Setup breakdown in summary
    by_setup = df.groupby("setup_type")["pnl_net"].sum()
    print(f"\n  P&L by setup:")
    for st, pnl in by_setup.items():
        print(f"    {st:<20} ${pnl:,.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze backtest trade logs")
    parser.add_argument("--file", default=None, help="Path to trades CSV file")
    args = parser.parse_args()

    path = Path(args.file) if args.file else _find_latest_csv()
    logger.info("Analyzing trades from: {}", path)

    df = _load_trades(path)
    logger.info("Loaded {} trades", len(df))

    print_summary(df)
    analyze_by_setup(df)
    analyze_by_direction(df)
    analyze_by_hour(df)
    analyze_by_dow(df)
    analyze_by_regime(df)
    analyze_exit_reasons(df)
    analyze_equity_curve(df)
    analyze_consecutive_runs(df)
    analyze_top_bottom_trades(df)
    print()


if __name__ == "__main__":
    main()
