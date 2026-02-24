"""
src/order_state.py

Idempotent order state machine — prevents duplicate orders on reconnect/retry.

Problem: If the bot reconnects mid-session and the execution engine resets,
it may not know about orders already submitted in the current session.
Placing a second bracket order would double the exposure.

Solution: The `OrderStateTracker` maintains a persistent JSON file with all
orders submitted today.  Before submitting any new order:
  1. Check if an order for this signal already exists (fingerprint match).
  2. If the existing order is still OPEN/PENDING — skip and wait for it.
  3. If FILLED — position already on, skip entry.
  4. If CANCELLED/REJECTED — OK to retry.

Signal fingerprint = hash(date, symbol, setup_type, direction, entry_price_rounded)
Same fingerprint used in build_dataset.py dedup logic.

Usage:
    tracker = OrderStateTracker()

    # Before placing order:
    if tracker.is_duplicate(symbol, signal):
        logger.warning("Duplicate order prevented")
        return

    # After placement:
    tracker.record_submitted(symbol, signal, order_id)

    # On fill:
    tracker.record_filled(order_id, fill_price)

    # On cancel:
    tracker.record_cancelled(order_id)
"""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from loguru import logger

ROOT = Path(__file__).resolve().parent.parent


def _signal_fingerprint(
    symbol: str,
    setup_type: str,
    direction: str,
    entry_price: float,
    ts: "datetime | None" = None,
) -> str:
    """Stable fingerprint for a signal — matches build_dataset.py logic."""
    date_str  = (ts or datetime.now()).strftime("%Y%m%d")
    price_r   = round(entry_price, 0)
    key = f"{date_str}|{symbol.upper()}|{setup_type.upper()}|{direction.upper()}|{price_r}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


class OrderStateTracker:
    """
    Thread-safe persistent order state tracker.

    State is persisted to `data/order_state_{date}.json`.
    On each new trading day the state file is automatically rolled over.
    """

    PENDING   = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED    = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED  = "REJECTED"

    def __init__(
        self,
        state_dir: str = "data",
    ) -> None:
        self._lock = threading.Lock()
        self._state_dir = ROOT / state_dir
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._trade_date = date.today()
        self._orders: dict[str, dict] = {}   # fingerprint -> order_record
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_duplicate(
        self,
        symbol: str,
        setup_type: str,
        direction: str,
        entry_price: float,
        ts: Optional[datetime] = None,
    ) -> bool:
        """
        Returns True if an order with this fingerprint is already
        SUBMITTED or FILLED today — preventing a duplicate submission.
        """
        self._roll_if_new_day()
        fp = _signal_fingerprint(symbol, setup_type, direction, entry_price, ts)
        with self._lock:
            rec = self._orders.get(fp)
            if rec is None:
                return False
            status = rec.get("status", "")
            if status in (self.SUBMITTED, self.FILLED):
                logger.warning(
                    "OrderState: duplicate {} {} signal at {} — "
                    "existing order status={} order_id={}",
                    symbol, setup_type, entry_price,
                    status, rec.get("order_id", "?"),
                )
                return True
            return False

    def record_submitted(
        self,
        symbol: str,
        setup_type: str,
        direction: str,
        entry_price: float,
        order_id: str,
        ts: Optional[datetime] = None,
    ) -> str:
        """Record an order as submitted. Returns fingerprint."""
        self._roll_if_new_day()
        fp = _signal_fingerprint(symbol, setup_type, direction, entry_price, ts)
        with self._lock:
            self._orders[fp] = {
                "fingerprint":  fp,
                "symbol":       symbol.upper(),
                "setup_type":   setup_type.upper(),
                "direction":    direction.upper(),
                "entry_price":  entry_price,
                "order_id":     order_id,
                "status":       self.SUBMITTED,
                "submitted_at": datetime.now().isoformat(),
                "fill_price":   None,
            }
            self._save()
        return fp

    def record_filled(self, fingerprint: str, fill_price: float) -> None:
        with self._lock:
            rec = self._orders.get(fingerprint)
            if rec:
                rec["status"]    = self.FILLED
                rec["fill_price"] = fill_price
                rec["filled_at"] = datetime.now().isoformat()
                self._save()

    def record_filled_by_order_id(self, order_id: str, fill_price: float) -> None:
        with self._lock:
            for rec in self._orders.values():
                if rec.get("order_id") == order_id:
                    rec["status"]    = self.FILLED
                    rec["fill_price"] = fill_price
                    rec["filled_at"] = datetime.now().isoformat()
                    break
            self._save()

    def record_cancelled(self, fingerprint: str) -> None:
        with self._lock:
            rec = self._orders.get(fingerprint)
            if rec:
                rec["status"] = self.CANCELLED
                self._save()

    def record_rejected(self, fingerprint: str, reason: str = "") -> None:
        with self._lock:
            rec = self._orders.get(fingerprint)
            if rec:
                rec["status"]          = self.REJECTED
                rec["rejection_reason"] = reason
                self._save()

    def today_orders(self) -> list[dict]:
        """Return all order records for today."""
        with self._lock:
            return list(self._orders.values())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _state_file(self) -> Path:
        return self._state_dir / f"order_state_{self._trade_date}.json"

    def _roll_if_new_day(self) -> None:
        today = date.today()
        if today != self._trade_date:
            with self._lock:
                self._trade_date = today
                self._orders = {}
                logger.info("OrderStateTracker: rolled to new day {}", today)

    def _save(self) -> None:
        try:
            self._state_file().write_text(
                json.dumps(self._orders, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("OrderStateTracker: could not save state: {}", e)

    def _load(self) -> None:
        sf = self._state_file()
        if sf.exists():
            try:
                self._orders = json.loads(sf.read_text(encoding="utf-8"))
                n = len(self._orders)
                if n:
                    logger.info(
                        "OrderStateTracker: restored {} orders from {}",
                        n, sf.name,
                    )
            except Exception:
                self._orders = {}
