"""
src/account_manager.py

Funded account safety manager for TopstepX.

Responsibilities:
  - Connect to TopstepX API and poll real-time account balance,
    daily P&L, and trailing drawdown usage
  - Enforce funded account rules (daily loss limit, trailing drawdown)
  - Provide dynamic contract sizing that scales as the account grows
  - Write account state to data/account_state.json so the dashboard
    can display live account health without needing its own API connection

TopstepX funded account limits (approximate — verify on your dashboard):
  50K  account: $2,000 daily loss limit, $3,000 trailing drawdown
  75K  account: $2,750 daily loss limit, $3,750 trailing drawdown
  100K account: $3,500 daily loss limit, $4,500 trailing drawdown

Usage (in bot_runner.py):
    from src.account_manager import AccountManager
    acct = AccountManager(settings_path="config/settings.yaml")
    await acct.start()               # begin polling
    if acct.is_safe_to_trade():
        contracts = acct.get_contract_size()
    await acct.stop()
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = ROOT / "data" / "account_state.json"


# ---------------------------------------------------------------------------
# TopstepX funded account limits by account size
# ---------------------------------------------------------------------------

TOPSTEPX_LIMITS: dict[int, dict[str, float]] = {
    25_000:  {"daily_loss": 1_500, "trailing_drawdown": 1_500},
    50_000:  {"daily_loss": 2_000, "trailing_drawdown": 3_000},
    75_000:  {"daily_loss": 2_750, "trailing_drawdown": 3_750},
    100_000: {"daily_loss": 3_500, "trailing_drawdown": 4_500},
    150_000: {"daily_loss": 5_000, "trailing_drawdown": 6_000},
    200_000: {"daily_loss": 6_000, "trailing_drawdown": 8_000},
}


def _get_limits(account_size: int) -> dict[str, float]:
    """Return the closest matching TopstepX limit tier for the given account size."""
    sizes = sorted(TOPSTEPX_LIMITS.keys())
    for size in sizes:
        if account_size <= size:
            return TOPSTEPX_LIMITS[size]
    return TOPSTEPX_LIMITS[max(sizes)]


# ---------------------------------------------------------------------------
# Account state dataclass
# ---------------------------------------------------------------------------

@dataclass
class AccountState:
    """Snapshot of the funded account at a point in time."""
    account_name: str = ""
    account_size: float = 0.0
    current_balance: float = 0.0
    daily_pnl: float = 0.0
    open_pnl: float = 0.0           # Unrealized P&L on open positions
    daily_loss_limit: float = 0.0
    trailing_drawdown_limit: float = 0.0
    trailing_drawdown_used: float = 0.0
    safety_buffer: float = 500.0
    is_safe_to_trade: bool = True
    unsafe_reason: str = ""
    last_updated: str = ""
    paper_mode: bool = True
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def daily_loss_remaining(self) -> float:
        return self.daily_loss_limit + self.daily_pnl  # daily_pnl is negative when losing

    @property
    def drawdown_remaining(self) -> float:
        return self.trailing_drawdown_limit - self.trailing_drawdown_used

    @property
    def pct_daily_limit_used(self) -> float:
        if self.daily_loss_limit == 0:
            return 0.0
        return min(100.0, abs(min(0, self.daily_pnl)) / self.daily_loss_limit * 100)

    @property
    def pct_drawdown_used(self) -> float:
        if self.trailing_drawdown_limit == 0:
            return 0.0
        return min(100.0, self.trailing_drawdown_used / self.trailing_drawdown_limit * 100)


# ---------------------------------------------------------------------------
# AccountManager
# ---------------------------------------------------------------------------

class AccountManager:
    """
    Polls TopstepX for account health and provides trading safety checks.

    In paper_mode=True (default): reads daily P&L from risk_manager state
    and enforces simulated account limits using the funded_account config.

    In paper_mode=False: connects to TopstepX API and reads live balance/drawdown.
    """

    def __init__(
        self,
        settings_path: str = "config/settings.yaml",
        risk_manager=None,  # Optional[RiskManager] — injected to read paper-mode P&L
    ) -> None:
        self._settings_path = settings_path
        self._cfg = self._load_settings()
        self._risk_manager = risk_manager
        self._paper_mode = self._cfg["execution"]["paper_mode"]
        self._funded_cfg = self._cfg.get("funded_account", {})
        self._account_size = int(self._funded_cfg.get("account_size", 50_000))
        self._safety_buffer = float(self._funded_cfg.get("safety_buffer_dollars", 500))
        self._limits = _get_limits(self._account_size)

        self._state = AccountState(
            account_size=self._account_size,
            daily_loss_limit=self._limits["daily_loss"],
            trailing_drawdown_limit=self._limits["trailing_drawdown"],
            safety_buffer=self._safety_buffer,
            paper_mode=self._paper_mode,
        )

        self._suite = None          # project-x-py TradingSuite (live mode only)
        self._poll_task: Optional[asyncio.Task] = None
        self._running = False
        self._poll_interval = 30    # seconds between API polls

        logger.info(
            "AccountManager init — account_size=${:,} daily_limit=${:,} "
            "drawdown_limit=${:,} paper_mode={}",
            self._account_size,
            self._limits["daily_loss"],
            self._limits["trailing_drawdown"],
            self._paper_mode,
        )

    def _load_settings(self) -> dict:
        with open(self._settings_path, "r") as f:
            return yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start background account polling."""
        self._running = True
        if not self._paper_mode:
            await self._connect_live()
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("AccountManager started (paper_mode={})", self._paper_mode)

    async def stop(self) -> None:
        """Stop polling and disconnect."""
        self._running = False
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        self._write_state_file()
        logger.info("AccountManager stopped")

    # ------------------------------------------------------------------
    # Safety checks (called before every trade)
    # ------------------------------------------------------------------

    def is_safe_to_trade(self) -> bool:
        """Return True if it is safe to place a new trade."""
        self._refresh_state_sync()
        return self._state.is_safe_to_trade

    def get_contract_size(self) -> int:
        """
        Dynamic contract sizing based on account profit accumulation.
        Starts at 1 contract and scales up as the account grows.
        """
        # Base: 1 contract
        contracts = 1

        # Get current profit above starting balance
        profit_above_start = max(0.0, self._state.current_balance - self._account_size)
        if profit_above_start <= 0 and self._paper_mode:
            # In paper mode, use risk_manager history if available
            if self._risk_manager is not None:
                profit_above_start = max(0.0, self._risk_manager.all_time_summary().get("total_pnl_dollars", 0))

        # Apply scale-up tiers from config
        scale_tiers = self._funded_cfg.get("scale_contracts_at_profit", [])
        for tier in sorted(scale_tiers, key=lambda t: t["profit_threshold"], reverse=True):
            if profit_above_start >= tier["profit_threshold"]:
                contracts = tier["contracts"]
                break

        return contracts

    @property
    def state(self) -> AccountState:
        return self._state

    # ------------------------------------------------------------------
    # State refresh
    # ------------------------------------------------------------------

    def _refresh_state_sync(self) -> None:
        """Synchronous refresh — uses risk_manager data in paper mode."""
        if self._paper_mode and self._risk_manager is not None:
            daily_pnl = self._risk_manager.today.daily_pnl
            self._state.daily_pnl = daily_pnl
            self._state.current_balance = self._account_size + daily_pnl
            self._state.last_updated = datetime.datetime.now().isoformat()
            self._evaluate_safety()

    async def _poll_loop(self) -> None:
        """Background coroutine that polls account state every N seconds."""
        while self._running:
            try:
                await self._fetch_account_state()
                self._write_state_file()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("AccountManager poll error: {}", e)
                self._state.error = str(e)
            await asyncio.sleep(self._poll_interval)

    async def _fetch_account_state(self) -> None:
        """Fetch real account state from TopstepX API or simulate it in paper mode."""
        if self._paper_mode:
            # Paper mode: use risk_manager daily P&L
            if self._risk_manager is not None:
                daily_pnl = self._risk_manager.today.daily_pnl
                self._state.daily_pnl = daily_pnl
                self._state.current_balance = self._account_size + daily_pnl
            self._state.last_updated = datetime.datetime.now().isoformat()
            self._state.error = ""
            self._evaluate_safety()
            return

        # Live mode: query TopstepX API via project-x-py
        if self._suite is None:
            logger.warning("AccountManager: no live connection available")
            return

        try:
            # project-x-py TradingSuite account info
            accounts = await self._suite.accounts.get_accounts()
            account_name = os.getenv("PROJECT_X_ACCOUNT_NAME", "")

            target = None
            if account_name:
                for acct in accounts:
                    if acct.name == account_name or str(acct.id) == account_name:
                        target = acct
                        break
            if target is None and accounts:
                target = accounts[0]

            if target is None:
                self._state.error = "No account found"
                return

            self._state.account_name = getattr(target, "name", str(getattr(target, "id", "")))
            self._state.current_balance = float(getattr(target, "balance", self._account_size))

            # Calculate daily P&L relative to start-of-day balance
            # TopstepX exposes daily P&L directly on some account objects
            if hasattr(target, "daily_pnl"):
                self._state.daily_pnl = float(target.daily_pnl)
            else:
                # Approximate from balance change
                self._state.daily_pnl = self._state.current_balance - self._account_size

            # Trailing drawdown: use balance vs high-water mark
            if hasattr(target, "max_loss_threshold"):
                # Some SDKs expose the threshold directly
                threshold = float(target.max_loss_threshold)
                self._state.trailing_drawdown_used = max(
                    0.0, self._limits["trailing_drawdown"] - threshold
                )
            else:
                # Approximate: drawdown used = account_size - current_balance (if negative)
                self._state.trailing_drawdown_used = max(
                    0.0, self._account_size - self._state.current_balance
                )

            self._state.last_updated = datetime.datetime.now().isoformat()
            self._state.error = ""
            self._evaluate_safety()

        except Exception as e:
            self._state.error = f"API error: {e}"
            logger.warning("AccountManager live fetch failed: {}", e)

    def _evaluate_safety(self) -> None:
        """Check all safety rules and update is_safe_to_trade."""
        # Rule 1: Daily loss within safety buffer
        daily_remaining = self._limits["daily_loss"] + self._state.daily_pnl
        if daily_remaining <= self._safety_buffer:
            self._state.is_safe_to_trade = False
            self._state.unsafe_reason = (
                f"Daily loss limit approaching: ${daily_remaining:.0f} remaining "
                f"(buffer=${self._safety_buffer:.0f})"
            )
            logger.warning("AccountManager: UNSAFE — {}", self._state.unsafe_reason)
            return

        # Rule 2: Trailing drawdown within safety buffer
        drawdown_remaining = self._limits["trailing_drawdown"] - self._state.trailing_drawdown_used
        if drawdown_remaining <= self._safety_buffer:
            self._state.is_safe_to_trade = False
            self._state.unsafe_reason = (
                f"Trailing drawdown limit approaching: ${drawdown_remaining:.0f} remaining"
            )
            logger.warning("AccountManager: UNSAFE — {}", self._state.unsafe_reason)
            return

        self._state.is_safe_to_trade = True
        self._state.unsafe_reason = ""

    # ------------------------------------------------------------------
    # Live connection (only used in paper_mode=False)
    # ------------------------------------------------------------------

    async def _connect_live(self) -> None:
        """Connect to TopstepX API for live account polling."""
        try:
            from projectx import TradingSuite
            username = os.getenv("PROJECT_X_USERNAME", "")
            api_key = os.getenv("PROJECT_X_API_KEY", "")
            if not username or not api_key:
                logger.error("AccountManager: PROJECT_X_USERNAME / PROJECT_X_API_KEY not set in .env")
                return
            self._suite = await TradingSuite.create(username=username, api_key=api_key)
            logger.info("AccountManager: connected to TopstepX API")
        except Exception as e:
            logger.error("AccountManager: failed to connect — {}", e)
            self._state.error = f"Connection failed: {e}"

    # ------------------------------------------------------------------
    # State file (for dashboard)
    # ------------------------------------------------------------------

    def _write_state_file(self) -> None:
        """Write account state to data/account_state.json for the dashboard."""
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            state_dict = self._state.to_dict()
            # Add computed fields for dashboard display
            state_dict["daily_loss_remaining"] = round(self._state.daily_loss_remaining, 2)
            state_dict["drawdown_remaining"] = round(self._state.drawdown_remaining, 2)
            state_dict["pct_daily_limit_used"] = round(self._state.pct_daily_limit_used, 1)
            state_dict["pct_drawdown_used"] = round(self._state.pct_drawdown_used, 1)
            state_dict["suggested_contracts"] = self.get_contract_size()
            STATE_FILE.write_text(json.dumps(state_dict, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug("AccountManager: could not write state file: {}", e)

    def write_state_now(self) -> None:
        """Public method for bot_runner to call synchronously."""
        self._refresh_state_sync()
        self._write_state_file()

    # ------------------------------------------------------------------
    # Summary helper
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a compact summary dict for logging."""
        return {
            "account": self._state.account_name,
            "balance": round(self._state.current_balance, 2),
            "daily_pnl": round(self._state.daily_pnl, 2),
            "daily_remaining": round(self._state.daily_loss_remaining, 2),
            "drawdown_remaining": round(self._state.drawdown_remaining, 2),
            "safe": self._state.is_safe_to_trade,
            "contracts": self.get_contract_size(),
        }
