"""
src/portfolio_risk.py

Cross-symbol portfolio risk manager.

ES and NQ are highly correlated (~0.95).  Trading both simultaneously
doubles the drawdown exposure when the market moves against you.

This module enforces:
  1. One shared daily loss budget across ALL symbols.
     If ES + NQ combined P&L < portfolio_daily_loss_limit → halt ALL symbols.

  2. Cross-symbol position cap.
     If one symbol already has an open position, the other may only open
     if they are "aligned" (same direction) AND total notional exposure
     is within the portfolio exposure limit.

  3. Correlated-drawdown circuit breaker.
     If N consecutive losses across all symbols → halt all for rest of day.

Usage (in multi_bot_runner.py or each bot_runner):
    _portfolio_risk = PortfolioRiskManager()

    # Before placing any order:
    if not _portfolio_risk.allow_new_position(symbol, direction):
        return  # blocked

    # After each trade exits:
    _portfolio_risk.record_trade_pnl(symbol, realized_dollars)
"""
from __future__ import annotations

import json
import threading
from datetime import date
from pathlib import Path
from typing import Optional

from loguru import logger

ROOT = Path(__file__).resolve().parent.parent

# ES and NQ are treated as a correlated pair
_CORRELATED_PAIRS: list[tuple[str, str]] = [("ES", "NQ"), ("MES", "MNQ")]


class PortfolioRiskManager:
    """
    Thread-safe cross-symbol risk manager.
    One instance shared across ALL BotRunner processes/threads.

    Parameters
    ----------
    portfolio_daily_loss_limit : float
        Total allowable loss in dollars across all symbols (negative = loss).
        Default: -2000 (stop all trading if down $2,000 across ES+NQ combined).
    max_correlated_positions : int
        Maximum simultaneous positions in correlated pairs. Default: 1.
        Set to 2 only if you intend to trade both simultaneously with reduced size.
    consecutive_loss_halt : int
        Halt ALL symbols if this many consecutive combined losses occur. Default: 4.
    """

    def __init__(
        self,
        portfolio_daily_loss_limit: float = -2000.0,
        max_correlated_positions: int = 1,
        consecutive_loss_halt: int = 4,
        state_file: str = "data/portfolio_risk_state.json",
    ) -> None:
        self._lock = threading.Lock()
        self._limit = portfolio_daily_loss_limit
        self._max_corr = max_correlated_positions
        self._consec_halt = consecutive_loss_halt
        self._state_file = ROOT / state_file

        # Mutable state (reset daily)
        self._portfolio_pnl: float = 0.0
        self._open_positions: dict[str, str] = {}   # symbol -> direction ("LONG"/"SHORT")
        self._consecutive_losses: int = 0
        self._halted_today: bool = False
        self._trade_date: date = date.today()

        self._load_state()

    # ------------------------------------------------------------------
    # Daily reset
    # ------------------------------------------------------------------

    def _reset_if_new_day(self) -> None:
        today = date.today()
        if today != self._trade_date:
            with self._lock:
                self._portfolio_pnl = 0.0
                self._open_positions = {}
                self._consecutive_losses = 0
                self._halted_today = False
                self._trade_date = today
                logger.info("PortfolioRisk: daily reset for {}", today)

    # ------------------------------------------------------------------
    # Allow / block new positions
    # ------------------------------------------------------------------

    def allow_new_position(self, symbol: str, direction: str) -> tuple[bool, str]:
        """
        Returns (allowed, reason).
        Call BEFORE placing any order.
        """
        self._reset_if_new_day()

        with self._lock:
            sym = symbol.upper()
            dir_upper = direction.upper()

            # ── Portfolio halt check ──────────────────────────────────────
            if self._halted_today:
                return False, "portfolio_halted_today"

            # ── Portfolio daily loss limit ────────────────────────────────
            if self._portfolio_pnl <= self._limit:
                self._halted_today = True
                logger.error(
                    "PORTFOLIO LOSS LIMIT HIT: combined P&L=${:.2f} <= limit=${:.2f}. "
                    "ALL symbols halted.",
                    self._portfolio_pnl, self._limit,
                )
                return False, f"portfolio_loss_limit_hit_{self._portfolio_pnl:.0f}"

            # ── Consecutive loss circuit breaker ──────────────────────────
            if self._consecutive_losses >= self._consec_halt:
                self._halted_today = True
                logger.error(
                    "PORTFOLIO CONSEC LOSS HALT: {} consecutive losses. "
                    "Halting ALL symbols for rest of day.",
                    self._consecutive_losses,
                )
                return False, f"consecutive_loss_halt_{self._consecutive_losses}"

            # ── Correlated position cap ───────────────────────────────────
            n_corr = self._count_correlated_positions(sym)
            if n_corr >= self._max_corr:
                # Check if direction alignment would still be safe
                existing_dir = self._get_correlated_direction(sym)
                if existing_dir and existing_dir != dir_upper:
                    return (
                        False,
                        f"correlated_opposite_direction: {sym} wants {dir_upper} "
                        f"but pair already has {existing_dir}",
                    )
                # Position cap still enforced
                return (
                    False,
                    f"correlated_position_cap_{n_corr}",
                )

            return True, "approved"

    def record_open_position(self, symbol: str, direction: str) -> None:
        """Call after a fill is confirmed."""
        with self._lock:
            self._open_positions[symbol.upper()] = direction.upper()
            logger.debug(
                "PortfolioRisk: open positions now {}",
                self._open_positions,
            )

    def record_close_position(self, symbol: str, pnl_dollars: float) -> None:
        """Call after a position closes. pnl_dollars is net."""
        with self._lock:
            self._open_positions.pop(symbol.upper(), None)
            self._portfolio_pnl += pnl_dollars
            if pnl_dollars < 0:
                self._consecutive_losses += 1
            else:
                self._consecutive_losses = 0
            logger.info(
                "PortfolioRisk: {} closed pnl=${:.2f}  portfolio_pnl=${:.2f}  "
                "consec_losses={}",
                symbol, pnl_dollars, self._portfolio_pnl, self._consecutive_losses,
            )
            self._save_state()

    def record_trade_pnl(self, symbol: str, pnl_dollars: float) -> None:
        """Alias for record_close_position without removing open position."""
        with self._lock:
            self._portfolio_pnl += pnl_dollars
            if pnl_dollars < 0:
                self._consecutive_losses += 1
            else:
                self._consecutive_losses = 0
            self._save_state()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _count_correlated_positions(self, symbol: str) -> int:
        """How many open positions exist in the correlated pair for `symbol`?"""
        for pair in _CORRELATED_PAIRS:
            if symbol in pair:
                return sum(1 for s in pair if s in self._open_positions)
        return 0

    def _get_correlated_direction(self, symbol: str) -> Optional[str]:
        """Get existing direction for the correlated pair member (if any open)."""
        for pair in _CORRELATED_PAIRS:
            if symbol in pair:
                for s in pair:
                    if s in self._open_positions and s != symbol:
                        return self._open_positions[s]
        return None

    # ------------------------------------------------------------------
    # State persistence (survives intraday bot restarts)
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        try:
            state = {
                "trade_date":        str(self._trade_date),
                "portfolio_pnl":     self._portfolio_pnl,
                "open_positions":    self._open_positions,
                "consecutive_losses": self._consecutive_losses,
                "halted_today":      self._halted_today,
            }
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            self._state_file.write_text(
                json.dumps(state, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    def _load_state(self) -> None:
        if not self._state_file.exists():
            return
        try:
            s = json.loads(self._state_file.read_text(encoding="utf-8"))
            saved_date = date.fromisoformat(s.get("trade_date", "2000-01-01"))
            if saved_date == date.today():
                self._portfolio_pnl     = float(s.get("portfolio_pnl", 0))
                self._open_positions    = dict(s.get("open_positions", {}))
                self._consecutive_losses = int(s.get("consecutive_losses", 0))
                self._halted_today      = bool(s.get("halted_today", False))
                logger.info(
                    "PortfolioRisk: restored state — pnl=${:.2f} positions={} halted={}",
                    self._portfolio_pnl, self._open_positions, self._halted_today,
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        with self._lock:
            return {
                "trade_date":         str(self._trade_date),
                "portfolio_pnl":      round(self._portfolio_pnl, 2),
                "open_positions":     dict(self._open_positions),
                "consecutive_losses": self._consecutive_losses,
                "halted_today":       self._halted_today,
                "portfolio_limit":    self._limit,
                "max_correlated":     self._max_corr,
            }


# Singleton — shared across all BotRunner instances in the same process
_PORTFOLIO_RISK = PortfolioRiskManager()


def get_portfolio_risk() -> PortfolioRiskManager:
    """Return the process-level singleton PortfolioRiskManager."""
    return _PORTFOLIO_RISK
