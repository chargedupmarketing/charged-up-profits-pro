"""
src/labeling/labeler.py

Unified labeling module for the ChargedUp Profits Bot ML pipeline.

Fixes two problems with the old TripleBarrierLabeler approach:
  1. "Two different problem" ambiguity: exit_reason="tp" treated the same as a
     full TP even for partial_tp (which may have hit stop on remainder), and
     exit_reason="eod" always = 0 even when EOD exit was actually profitable.
  2. Fixed barriers (+20/-8) in Path 2 didn't match per-signal planned distances.

This module provides:
  ─ Labeler.from_trade_log()     — Path 1: labels from backtest/live trade CSV
  ─ Labeler.from_raw_signals()   — Path 2: labels from raw bar data + signal params
  ─ Labeler.compute_realized_R() — core R-multiple calculation

Label methods available:
  realized_R_binary  (default, recommended)
      label = 1 if realized_R > 0   (any positive net PnL)
      label = 0 if realized_R <= 0
      Fixes: partial_tp, EOD distortion.

  tp_binary  (legacy, backward-compat)
      label = 1 if exit_reason in ("tp", "partial_tp")
      label = 0 otherwise
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent.parent


def _load_settings(path: str = "config/settings.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Core R-multiple helper
# ---------------------------------------------------------------------------

def compute_realized_R(
    entry_price: float,
    exit_price: float,
    stop_price: float,
    direction: str,           # "LONG" or "SHORT"
    point_value: float,
    contracts: int,
    total_costs: float,       # commissions + slippage in dollars
) -> float:
    """
    Compute realized R-multiple for a single trade record.

    realized_R = net_pnl_dollars / initial_risk_dollars

    initial_risk = abs(entry - stop) * point_value * contracts
    net_pnl = (directional price change) * point_value * contracts - costs

    Returns:
      +4.0  = closed at target (approx 4R win)
      +1.0  = closed at 1R profit (e.g. partial TP)
      -1.0  = stopped out for full loss
      -0.1  = small loss after slippage/commission from an otherwise near-BE exit
       0.0  = exact breakeven (rare)
    """
    if direction.upper() == "LONG":
        pnl_pts = float(exit_price) - float(entry_price)
    else:
        pnl_pts = float(entry_price) - float(exit_price)

    initial_risk_pts = abs(float(entry_price) - float(stop_price))
    if initial_risk_pts <= 0:
        return 0.0

    pnl_net_dollars = pnl_pts * float(point_value) * int(contracts) - float(total_costs)
    initial_risk_dollars = initial_risk_pts * float(point_value) * int(contracts)

    return round(pnl_net_dollars / initial_risk_dollars, 4)


# ---------------------------------------------------------------------------
# Labeler
# ---------------------------------------------------------------------------

class Labeler:
    """
    Unified labeling class.  Instantiate with the desired label_method and
    settings, then call from_trade_log() or from_raw_signals().
    """

    def __init__(
        self,
        label_method: str = "realized_R_binary",
        settings_path: str = "config/settings.yaml",
    ) -> None:
        """
        Parameters
        ----------
        label_method : str
            "realized_R_binary"  — label = 1 if realized_R > 0, else 0  (default)
            "tp_binary"          — label = 1 if exit_reason in {tp, partial_tp} (legacy)
            "realized_R"         — regression target: the raw R-multiple value
        settings_path : str
            Path to settings.yaml (used for ML triple-barrier config).
        """
        self._label_method = label_method
        cfg = _load_settings(settings_path)
        ml_cfg = cfg.get("ml", {})
        self._tp_pts = ml_cfg.get("triple_barrier_tp_points", 20)
        self._sl_pts = ml_cfg.get("triple_barrier_sl_points", 8)
        self._time_limit_bars = ml_cfg.get("triple_barrier_time_limit_bars", 60)

    # ------------------------------------------------------------------
    # Path 1 — from backtest/live trade log
    # ------------------------------------------------------------------

    def from_trade_log(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        Label and compute realized_R for each row in a trade log DataFrame.

        Required columns: entry_price, exit_price, stop_price, direction,
                          point_value, contracts, commission_dollars,
                          slippage_dollars, exit_reason.

        Adds columns:
          realized_R   — float: net PnL / initial_risk
          label        — int (0/1) based on label_method
        """
        df = trades_df.copy()

        # Use pre-computed realized_R if the column is already present and valid.
        # This preserves the exact values written by the backtest harness (which
        # includes session-phase slippage and precise commission accounting).
        # Only recompute when the column is absent, has nulls, or contains
        # implausible outliers (|R| > 50 means a computation error occurred).
        _has_precomputed_R = (
            "realized_R" in df.columns
            and df["realized_R"].notna().all()
            and len(df) > 0
            and (df["realized_R"].abs() < 50).all()
        )

        if not _has_precomputed_R:
            def _row_R(row: pd.Series) -> float:
                costs = (
                    float(row.get("commission_dollars", 0))
                    + float(row.get("slippage_dollars", 0))
                )
                try:
                    return compute_realized_R(
                        entry_price=row["entry_price"],
                        exit_price=row["exit_price"],
                        stop_price=row["stop_price"],
                        direction=str(row["direction"]),
                        point_value=float(row.get("point_value", 50) or 50),
                        contracts=int(row.get("contracts", 1) or 1),
                        total_costs=costs,
                    )
                except Exception:
                    return 0.0

            df["realized_R"] = df.apply(_row_R, axis=1)
            logger.debug("realized_R recomputed from raw columns")

        # Apply label method
        if self._label_method == "realized_R_binary":
            df["label"] = (df["realized_R"] > 0).astype(int)
        elif self._label_method == "realized_R":
            df["label"] = df["realized_R"]   # regression target
        elif self._label_method == "tp_binary":
            df["label"] = (df["exit_reason"].isin(["tp", "partial_tp"])).astype(int)
        elif self._label_method == "r3class":
            # 3-class: LOSS=0, SMALL_WIN=1, BIG_WIN=2
            def _class(r: float) -> int:
                if r <= -0.25:
                    return 0
                if r < 1.5:
                    return 1
                return 2
            df["label"] = df["realized_R"].apply(_class)
        else:
            raise ValueError(
                f"Unknown label_method={self._label_method!r}. "
                "Choose from: realized_R_binary, tp_binary, realized_R, r3class"
            )

        if self._label_method in ("realized_R_binary", "tp_binary"):
            pos = (df["label"] == 1).sum()
            neg = (df["label"] == 0).sum()
            imbalance = max(pos, neg) / len(df) if len(df) > 0 else 0.5
            logger.info(
                "Labeling ({}): {} positive / {} negative  imbalance={:.1f}%",
                self._label_method, pos, neg, imbalance * 100,
            )
            if imbalance > 0.85:
                logger.warning(
                    "Class imbalance is {:.1f}% — consider collecting more data "
                    "before training.", imbalance * 100
                )

        return df

    # ------------------------------------------------------------------
    # Path 2 — from raw bar data + signal parameters
    # ------------------------------------------------------------------

    def from_raw_signals(
        self,
        signals_df: pd.DataFrame,
        df_1m: pd.DataFrame,
        direction_col: str = "direction",
        entry_col: str = "entry_price",
        stop_col: str = "stop_price",
        target_col: str = "target_price",
        signal_ts_col: str = "entry_ts",
        point_value_col: str = "point_value",
        contracts_col: str = "contracts",
    ) -> pd.DataFrame:
        """
        Apply triple-barrier labeling to raw signals that may never have been
        executed.  Uses the signal's own TP/SL distances rather than fixed
        global values, fixing the mis-alignment with Path 1.

        For LONG:   win if bar.high >= entry + tp_pts before bar.low <= entry - sl_pts
        For SHORT:  win if bar.low  <= entry - tp_pts before bar.high >= entry + sl_pts
        Tie (same bar):  loss (conservative).
        Timeout:         label based on final price vs entry_price (R sign).

        Adds: realized_R, label, exit_reason_sim
        """
        df = signals_df.copy()
        labels, exit_reasons, realized_rs = [], [], []

        for _, row in df.iterrows():
            entry = float(row[entry_col])
            stop  = float(row[stop_col])
            # Use per-signal distances if target column exists; else use global defaults
            if target_col in row.index and not pd.isna(row.get(target_col)):
                target = float(row[target_col])
                tp_pts = abs(target - entry)
            else:
                tp_pts = self._tp_pts

            sl_pts = abs(entry - stop) if abs(entry - stop) > 0 else self._sl_pts
            direction = str(row[direction_col]).upper()
            ts = pd.Timestamp(row[signal_ts_col])

            pv = float(row[point_value_col]) if point_value_col in row.index else 50.0
            qty = int(row[contracts_col]) if contracts_col in row.index else 1

            # Get forward bars up to time limit
            forward = df_1m.loc[df_1m.index > ts].head(self._time_limit_bars)

            if forward.empty:
                labels.append(0)
                exit_reasons.append("no_data")
                realized_rs.append(-1.0)
                continue

            if direction == "LONG":
                tp_price = entry + tp_pts
                sl_price = entry - sl_pts
            else:
                tp_price = entry - tp_pts
                sl_price = entry + sl_pts

            hit_tp, hit_sl = False, False
            for _, bar in forward.iterrows():
                if direction == "LONG":
                    if bar["high"] >= tp_price:
                        hit_tp = True
                        break
                    if bar["low"] <= sl_price:
                        hit_sl = True
                        break
                else:
                    if bar["low"] <= tp_price:
                        hit_tp = True
                        break
                    if bar["high"] >= sl_price:
                        hit_sl = True
                        break

            # Determine exit
            if hit_tp and not hit_sl:
                reason = "tp_sim"
                sim_exit = tp_price
            elif hit_sl:
                reason = "sl_sim"
                sim_exit = sl_price
            else:
                # Time limit expired — use last bar close for R computation
                reason = "eod_sim"
                sim_exit = float(forward.iloc[-1]["close"])

            r = compute_realized_R(
                entry_price=entry, exit_price=sim_exit, stop_price=stop,
                direction=direction, point_value=pv, contracts=qty,
                total_costs=0.0,   # no cost data for raw signals
            )

            labels.append(1 if r > 0 else 0)
            exit_reasons.append(reason)
            realized_rs.append(r)

        df["realized_R"] = realized_rs
        df["label"] = labels
        df["exit_reason_sim"] = exit_reasons
        return df


# ---------------------------------------------------------------------------
# Convenience wrapper kept for backward-compat with ml_filter.py retrain path
# ---------------------------------------------------------------------------

class TripleBarrierLabeler:
    """
    Backward-compatible shim.  Delegates to Labeler with the configured
    label_method so the retrain pipeline picks up the new logic transparently.
    """

    def __init__(self, settings_path: str = "config/settings.yaml") -> None:
        cfg = _load_settings(settings_path)
        method = cfg.get("ml", {}).get("label_method", "realized_R_binary")
        self._labeler = Labeler(label_method=method, settings_path=settings_path)

    def label_from_backtest_trades(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        return self._labeler.from_trade_log(trades_df)

    def label_from_bars(
        self,
        signals_df: pd.DataFrame,
        df_1m: pd.DataFrame,
        direction_col: str = "direction",
        entry_col: str = "entry_price",
        signal_ts_col: str = "entry_ts",
    ) -> pd.DataFrame:
        return self._labeler.from_raw_signals(
            signals_df=signals_df,
            df_1m=df_1m,
            direction_col=direction_col,
            entry_col=entry_col,
            signal_ts_col=signal_ts_col,
        )
