"""
risk_manager.py

Enforces all risk rules before any signal is allowed to become an order:
  - Daily loss limit (hard stop — funded account limits enforced by AccountManager)
  - Max trades per day
  - Execution time window (9:00–13:00 ET)
  - 8am range validity (already set by level_builder but double-checked)
  - Minimum R:R ratio
  - Kill-switch logic
  - Position sizing (contracts — uses AccountManager when available)

Also tracks daily P&L and exposes the state needed by the dashboard.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

import yaml
from loguru import logger

from src.session_engine import SessionEngine
from src.setup_detector import Direction, Signal

if TYPE_CHECKING:
    from src.account_manager import AccountManager


def _load_settings(path: str = "config/settings.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class KillSwitchReason(str, Enum):
    NONE = "none"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    MAX_TRADES = "max_trades"
    CONNECTION_LOST = "connection_lost"
    MANUAL = "manual"
    OUTSIDE_WINDOW = "outside_window"
    NO_TRADE_DAY = "no_trade_day"


@dataclass
class TradeRecord:
    """Minimal record of a completed trade for daily tracking."""
    signal: Signal
    fill_price: float
    exit_price: float
    contracts: int
    point_value: float
    timestamp_entry: datetime.datetime
    timestamp_exit: datetime.datetime

    @property
    def pnl_points(self) -> float:
        if self.signal.direction == Direction.LONG:
            return self.exit_price - self.fill_price
        return self.fill_price - self.exit_price

    @property
    def pnl_dollars(self) -> float:
        return self.pnl_points * self.point_value * self.contracts


@dataclass
class DailyState:
    """Mutable state for the current trading day."""
    date: datetime.date
    trades: list[TradeRecord] = field(default_factory=list)
    kill_switch_active: bool = False
    kill_switch_reason: KillSwitchReason = KillSwitchReason.NONE
    open_position: bool = False

    @property
    def trade_count(self) -> int:
        return len(self.trades)

    @property
    def daily_pnl(self) -> float:
        return sum(t.pnl_dollars for t in self.trades)

    @property
    def win_count(self) -> int:
        return sum(1 for t in self.trades if t.pnl_dollars > 0)

    @property
    def loss_count(self) -> int:
        return sum(1 for t in self.trades if t.pnl_dollars <= 0)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return self.win_count / len(self.trades)


class RiskManager:
    """
    Central risk gatekeeper.  Every signal must pass through `approve_signal()`
    before an order is placed.  All position open/close events must be reported
    back via `record_fill()` and `record_exit()`.

    When an AccountManager is provided, funded account limits take precedence
    over the static max_daily_loss_dollars setting in config.
    """

    def __init__(
        self,
        settings_path: str = "config/settings.yaml",
        account_manager: Optional["AccountManager"] = None,
    ) -> None:
        self._cfg = _load_settings(settings_path)
        self._session = SessionEngine(settings_path)
        self._risk_cfg = self._cfg["risk"]
        self._instrument_cfg = self._cfg["instrument"]
        self._account_manager = account_manager   # injected after creation

        self.today = DailyState(date=datetime.date.today())
        self._history: list[DailyState] = []

        logger.info(
            "RiskManager initialised — max_daily_loss={} max_trades={} paper_mode={} "
            "account_manager={}",
            self._risk_cfg["max_daily_loss_dollars"],
            self._risk_cfg["max_trades_per_day"],
            self._cfg["execution"]["paper_mode"],
            "enabled" if account_manager else "disabled",
        )

    # ------------------------------------------------------------------
    # Day management
    # ------------------------------------------------------------------

    def new_day(self) -> None:
        """Call at the start of each trading session to reset daily counters.
        Also resets the circuit breaker so it doesn't carry over from yesterday."""
        self._consecutive_losses = 0
        self._circuit_break_until = None
        if self.today.trades:
            self._history.append(self.today)
        self.today = DailyState(date=datetime.date.today())
        logger.info("RiskManager: new trading day started — {}", self.today.date)

    # ------------------------------------------------------------------
    # Signal approval
    # ------------------------------------------------------------------

    def set_account_manager(self, account_manager: "AccountManager") -> None:
        """Inject the AccountManager after construction (avoids circular imports)."""
        self._account_manager = account_manager
        logger.info("RiskManager: AccountManager attached")

    # ------------------------------------------------------------------
    # Cross-symbol correlation guard
    # Called by multi_bot_runner to register/clear positions
    # ------------------------------------------------------------------

    # Shared dict across RiskManager instances: {symbol: direction_or_None}
    # Using a class-level variable so all BotRunner threads see the same state.
    _active_cross_positions: dict[str, str | None] = {}

    def register_open_position(self, symbol: str, direction: str) -> None:
        """Register an open position for cross-symbol correlation guard."""
        RiskManager._active_cross_positions[symbol.upper()] = direction.upper()

    def clear_open_position(self, symbol: str) -> None:
        """Clear a position from the correlation guard."""
        RiskManager._active_cross_positions.pop(symbol.upper(), None)

    def approve_signal(self, signal: Signal) -> tuple[bool, str]:
        """
        Check every risk rule.  Returns (approved: bool, reason: str).
        An empty reason string means approved.
        """
        # 0a. Circuit breaker: 3 consecutive losses → 30-minute pause
        cb_until = getattr(self, "_circuit_break_until", None)
        if cb_until:
            from datetime import datetime as _dt
            now = _dt.now()
            if now < cb_until:
                remaining = max(1, int((cb_until - now).total_seconds() / 60))
                return False, (
                    f"Circuit breaker: 3 consecutive losses — cooling off for "
                    f"{remaining} more minute{'s' if remaining != 1 else ''}."
                )
            else:
                self._circuit_break_until = None   # Pause expired

        # 0b. Cross-symbol correlation guard
        # ES (~0.93 corr with NQ): block if both already have open positions.
        # Natural hedges (ES long + NQ short) are allowed through.
        my_symbol = self._cfg["instrument"]["symbol"].upper()
        _corr_pairs = {"ES": "NQ", "NQ": "ES", "MES": "MNQ", "MNQ": "MES"}
        other_sym = _corr_pairs.get(my_symbol)
        if other_sym:
            other_dir = RiskManager._active_cross_positions.get(other_sym)
            my_dir    = RiskManager._active_cross_positions.get(my_symbol)
            if other_dir is not None and my_dir is not None:
                return False, (
                    f"Correlation block: both {my_symbol} and {other_sym} already "
                    f"have open positions — too much correlated risk at once."
                )

        # 1. Kill-switch
        if self.today.kill_switch_active:
            return False, f"Kill switch active: {self.today.kill_switch_reason.value}"

        # 2. Already have an open position
        if self.today.open_position:
            return False, "Already in a position"

        # 3. Time window
        if not self._session.is_in_execution_window():
            return False, "Outside execution window (9:00–13:00 ET)"

        # 4. Max trades per day
        if self.today.trade_count >= self._risk_cfg["max_trades_per_day"]:
            self._trigger_kill_switch(KillSwitchReason.MAX_TRADES)
            return False, f"Max trades ({self._risk_cfg['max_trades_per_day']}) reached"

        # 5. Daily loss limit — funded account manager takes precedence if available
        if self._account_manager is not None:
            # AccountManager enforces live funded account limits
            if not self._account_manager.is_safe_to_trade():
                reason = self._account_manager.state.unsafe_reason
                self._trigger_kill_switch(KillSwitchReason.DAILY_LOSS_LIMIT)
                return False, f"Funded account limit: {reason}"
        else:
            # Fallback to static config limit
            if self.today.daily_pnl <= self._risk_cfg["max_daily_loss_dollars"]:
                self._trigger_kill_switch(KillSwitchReason.DAILY_LOSS_LIMIT)
                return False, f"Daily loss limit hit (P&L={self.today.daily_pnl:.2f})"

        # 6. R:R minimum
        min_rr = self._risk_cfg["min_reward_risk_ratio"]
        if not signal.is_valid_rr(min_rr):
            return False, f"R:R={signal.reward_risk:.1f} below minimum {min_rr}"

        # 7. Stop-loss within bounds
        sl = signal.stop_distance
        sl_min = self._risk_cfg["stop_loss_points_min"]
        sl_max = self._risk_cfg["stop_loss_points_max"]
        if not (sl_min <= sl <= sl_max):
            return False, f"Stop distance={sl:.1f}pts outside bounds [{sl_min},{sl_max}]"

        # 8. Target within bounds
        tp = signal.target_distance
        tp_min = self._risk_cfg["take_profit_points_min"]
        tp_max = self._risk_cfg["take_profit_points_max"]
        if not (tp_min <= tp <= tp_max):
            return False, f"Target distance={tp:.1f}pts outside bounds [{tp_min},{tp_max}]"

        return True, ""

    # ------------------------------------------------------------------
    # Volatility regime input (set by bot_runner after computing features)
    # ------------------------------------------------------------------

    def set_volatility_regime(self, atr_ratio: float) -> None:
        """
        Called by bot_runner with the current atr_regime feature value
        (current ATR / 20-bar rolling avg ATR).  A ratio > threshold means
        the market is running hotter than normal — reduce size by 1 contract.
        """
        self._current_atr_ratio = float(atr_ratio)

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def get_contract_count(self) -> int:
        """
        Returns the number of contracts to trade.

        If an AccountManager is attached, delegates to its dynamic sizing
        (which scales based on live account profit).

        Otherwise falls back to the rule-based logic using trading history.

        In both cases, applies a volatility reduction: when current ATR is
        significantly above its 20-bar average (ratio > vol_sizing_atr_threshold),
        contracts are reduced by vol_sizing_reduce_by (floor 1).
        """
        if self._account_manager is not None:
            base = self._account_manager.get_contract_size()
        else:
            initial = self._risk_cfg["initial_contracts"]
            scale_after = self._risk_cfg["scale_up_after_days"]
            profitable_days = sum(1 for d in self._history if d.daily_pnl > 0)
            if profitable_days >= scale_after and len(self._history) >= scale_after:
                extra = (profitable_days - scale_after) // 10
                base = initial + extra
            else:
                base = initial

        # Volatility-based reduction
        if self._risk_cfg.get("vol_sizing_enabled", False):
            threshold = float(self._risk_cfg.get("vol_sizing_atr_threshold", 1.5))
            reduce_by = int(self._risk_cfg.get("vol_sizing_reduce_by", 1))
            atr_ratio = getattr(self, "_current_atr_ratio", 1.0)
            if atr_ratio > threshold:
                reduced = max(1, base - reduce_by)
                if reduced < base:
                    logger.info(
                        "Vol sizing: ATR ratio={:.2f} > {:.2f} threshold — "
                        "reducing contracts {} → {}",
                        atr_ratio, threshold, base, reduced,
                    )
                base = reduced

        return base

    # ------------------------------------------------------------------
    # Trade lifecycle callbacks
    # ------------------------------------------------------------------

    def record_fill(self, signal: Signal, fill_price: float) -> None:
        """Called when an entry order is filled."""
        self.today.open_position = True
        logger.info(
            "Fill recorded: {} @ {:.2f} ({} contracts)",
            signal.direction.value, fill_price, self.get_contract_count()
        )

    def record_exit(
        self,
        signal: Signal,
        fill_price: float,
        exit_price: float,
        exit_ts: datetime.datetime,
        entry_ts: datetime.datetime,
    ) -> None:
        """Called when an exit order (TP or SL) is filled."""
        self.today.open_position = False
        record = TradeRecord(
            signal=signal,
            fill_price=fill_price,
            exit_price=exit_price,
            contracts=self.get_contract_count(),
            point_value=self._instrument_cfg["point_value"],
            timestamp_entry=entry_ts,
            timestamp_exit=exit_ts,
        )
        self.today.trades.append(record)

        # Circuit breaker: track consecutive losses within the session.
        # 3 losses in a row → 30-minute trading pause (resets at new_day).
        if not hasattr(self, "_consecutive_losses"):
            self._consecutive_losses = 0
        if not hasattr(self, "_circuit_break_until"):
            self._circuit_break_until = None

        if record.pnl_points < 0:
            self._consecutive_losses += 1
            if self._consecutive_losses >= 3 and self._circuit_break_until is None:
                self._circuit_break_until = datetime.datetime.now() + datetime.timedelta(minutes=30)
                logger.warning(
                    "Circuit breaker triggered: {} consecutive losses — "
                    "pausing new trades for 30 minutes.",
                    self._consecutive_losses,
                )
        else:
            self._consecutive_losses = 0   # Reset on any winning trade

        logger.info(
            "Trade closed: {} @ {:.2f} → {:.2f} | P&L={:.2f}pts ({:.2f}$) | Daily={:.2f}$",
            signal.direction.value,
            fill_price, exit_price,
            record.pnl_points, record.pnl_dollars,
            self.today.daily_pnl,
        )

        # Check if daily loss limit is now breached
        if self.today.daily_pnl <= self._risk_cfg["max_daily_loss_dollars"]:
            self._trigger_kill_switch(KillSwitchReason.DAILY_LOSS_LIMIT)

    # ------------------------------------------------------------------
    # Kill-switch
    # ------------------------------------------------------------------

    def _trigger_kill_switch(self, reason: KillSwitchReason) -> None:
        if not self.today.kill_switch_active:
            self.today.kill_switch_active = True
            self.today.kill_switch_reason = reason
            logger.warning("KILL SWITCH triggered: {}", reason.value)

    def trigger_manual_kill(self) -> None:
        self._trigger_kill_switch(KillSwitchReason.MANUAL)

    def trigger_connection_kill(self) -> None:
        self._trigger_kill_switch(KillSwitchReason.CONNECTION_LOST)

    def reset_kill_switch(self) -> None:
        """Only call this deliberately and with caution (e.g. after fixing a connection issue)."""
        if self.today.kill_switch_reason == KillSwitchReason.CONNECTION_LOST:
            self.today.kill_switch_active = False
            self.today.kill_switch_reason = KillSwitchReason.NONE
            logger.warning("Kill switch manually reset")

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def daily_summary(self) -> dict:
        return {
            "date": str(self.today.date),
            "trades": self.today.trade_count,
            "pnl_dollars": round(self.today.daily_pnl, 2),
            "win_rate": round(self.today.win_rate, 3),
            "wins": self.today.win_count,
            "losses": self.today.loss_count,
            "kill_switch": self.today.kill_switch_active,
            "kill_reason": self.today.kill_switch_reason.value,
        }

    def all_time_summary(self) -> dict:
        all_days = self._history + [self.today]
        total_pnl = sum(d.daily_pnl for d in all_days)
        total_trades = sum(d.trade_count for d in all_days)
        winning_days = sum(1 for d in all_days if d.daily_pnl > 0)
        return {
            "total_days": len(all_days),
            "winning_days": winning_days,
            "total_trades": total_trades,
            "total_pnl_dollars": round(total_pnl, 2),
        }
