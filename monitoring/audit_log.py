"""
monitoring/audit_log.py

Persistent trade and decision logging using SQLAlchemy.

Stores every:
  - Signal detected (even rejected ones) with full feature context
  - Order placement, fill, and exit events
  - Daily summary rows
  - Kill-switch events

Default backend: SQLite (trading_bot/data/audit.db)
Upgrade to PostgreSQL by changing db_url in settings.yaml.
"""

from __future__ import annotations

import datetime
import json
from typing import Optional

import yaml
from loguru import logger
from sqlalchemy import (
    Column, DateTime, Float, Integer, String, Text, create_engine
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.setup_detector import Signal


def _load_settings(path: str = "config/settings.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------

class SignalLog(Base):
    """Every signal detected — approved or rejected."""
    __tablename__ = "signal_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    date = Column(String(10))
    setup_type = Column(String(30))
    direction = Column(String(10))
    entry_price = Column(Float)
    stop_price = Column(Float)
    target_price = Column(Float)
    stop_distance = Column(Float)
    target_distance = Column(Float)
    reward_risk = Column(Float)
    level_price = Column(Float, nullable=True)
    level_origin = Column(String(20), nullable=True)
    approved = Column(Integer)           # 1 = approved, 0 = rejected
    rejection_reason = Column(String(200), nullable=True)
    features_json = Column(Text, nullable=True)  # JSON dump of ML features


class TradeLog(Base):
    """Every completed trade."""
    __tablename__ = "trade_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(10))
    setup_type = Column(String(30))
    direction = Column(String(10))
    entry_price = Column(Float)
    fill_price = Column(Float)
    exit_price = Column(Float)
    stop_price = Column(Float)
    target_price = Column(Float)
    contracts = Column(Integer)
    pnl_points = Column(Float)
    pnl_dollars = Column(Float)
    commission_dollars = Column(Float)
    exit_reason = Column(String(20))    # "tp", "sl", "eod", "manual"
    entry_ts = Column(DateTime)
    exit_ts = Column(DateTime)
    hold_minutes = Column(Float)


class DailyLog(Base):
    """End-of-day summary row."""
    __tablename__ = "daily_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(10), unique=True)
    total_trades = Column(Integer)
    wins = Column(Integer)
    losses = Column(Integer)
    win_rate = Column(Float)
    pnl_dollars = Column(Float)
    kill_switch_triggered = Column(Integer)
    kill_reason = Column(String(50), nullable=True)


class EventLog(Base):
    """Operational events: kill-switch, reconnect, etc."""
    __tablename__ = "event_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    event_type = Column(String(50))
    detail = Column(Text, nullable=True)


class ModelVersion(Base):
    """
    Tracks all ML model training runs, safety gate evaluations, and
    deployment decisions (approve / reject / rollback).

    Status values: "pending", "active", "archived", "rejected"
    """
    __tablename__ = "model_versions"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    timestamp         = Column(DateTime, default=datetime.datetime.utcnow)
    symbol            = Column(String(10))
    setup_type        = Column(String(30))
    auc               = Column(Float, nullable=True)
    win_rate_sim      = Column(Float, nullable=True)
    brier_score       = Column(Float, nullable=True)
    n_trades          = Column(Integer, nullable=True)
    status            = Column(String(20))  # "pending", "active", "archived", "rejected"
    gate_results_json = Column(Text, nullable=True)   # JSON: per-gate pass/fail details
    comparison_json   = Column(Text, nullable=True)   # JSON: old vs new side-by-side metrics
    approved_at       = Column(DateTime, nullable=True)
    rejected_at       = Column(DateTime, nullable=True)


# ---------------------------------------------------------------------------
# AuditLogger
# ---------------------------------------------------------------------------

class AuditLogger:
    """
    Thread-safe audit logger.  All writes go through SQLAlchemy sessions.
    Supports both SQLite (default) and PostgreSQL (production).
    """

    def __init__(self, settings_path: str = "config/settings.yaml") -> None:
        cfg = _load_settings(settings_path)
        db_url = cfg["monitoring"]["db_url"]

        # Ensure directory exists for SQLite
        if db_url.startswith("sqlite:///"):
            from pathlib import Path
            db_path = Path(db_url.replace("sqlite:///", ""))
            db_path.parent.mkdir(parents=True, exist_ok=True)

        self._engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)

        # Alert webhook
        self._webhook_url = cfg["monitoring"].get("alert_webhook_url", "")
        self._alert_cfg = cfg["monitoring"]

        logger.info("AuditLogger initialised — db={}", db_url)

    # ------------------------------------------------------------------
    # Signal events
    # ------------------------------------------------------------------

    def log_signal(
        self,
        signal: Signal,
        approved: bool,
        rejection_reason: str = "",
        features: Optional[dict] = None,
    ) -> None:
        with self._Session() as session:
            row = SignalLog(
                date=str(signal.formed_at.date()),
                setup_type=signal.setup_type.value,
                direction=signal.direction.value,
                entry_price=signal.entry_price,
                stop_price=signal.stop_price,
                target_price=signal.target_price,
                stop_distance=signal.stop_distance,
                target_distance=signal.target_distance,
                reward_risk=signal.reward_risk,
                level_price=signal.level_ref.price if signal.level_ref else None,
                level_origin=signal.level_ref.session_origin if signal.level_ref else None,
                approved=int(approved),
                rejection_reason=rejection_reason or None,
                features_json=json.dumps(features) if features else None,
            )
            session.add(row)
            session.commit()

    # ------------------------------------------------------------------
    # Trade events
    # ------------------------------------------------------------------

    def log_trade(
        self,
        signal: Signal,
        fill_price: float,
        exit_price: float,
        contracts: int,
        pnl_points: float,
        pnl_dollars: float,
        commission_dollars: float,
        exit_reason: str,
        entry_ts: datetime.datetime,
        exit_ts: datetime.datetime,
        point_value: float,
    ) -> None:
        hold_minutes = (exit_ts - entry_ts).total_seconds() / 60

        with self._Session() as session:
            row = TradeLog(
                date=str(entry_ts.date()),
                setup_type=signal.setup_type.value,
                direction=signal.direction.value,
                entry_price=signal.entry_price,
                fill_price=fill_price,
                exit_price=exit_price,
                stop_price=signal.stop_price,
                target_price=signal.target_price,
                contracts=contracts,
                pnl_points=round(pnl_points, 2),
                pnl_dollars=round(pnl_dollars, 2),
                commission_dollars=round(commission_dollars, 2),
                exit_reason=exit_reason,
                entry_ts=entry_ts,
                exit_ts=exit_ts,
                hold_minutes=round(hold_minutes, 1),
            )
            session.add(row)
            session.commit()

        if self._alert_cfg.get("alert_on_trade_fill"):
            self._send_alert(
                f"Trade closed [{exit_reason.upper()}] {signal.direction.value} "
                f"P&L: {pnl_points:+.2f}pts / ${pnl_dollars:+.2f}"
            )

    # ------------------------------------------------------------------
    # Daily summary
    # ------------------------------------------------------------------

    def log_daily_summary(
        self,
        date: str,
        total_trades: int,
        wins: int,
        losses: int,
        win_rate: float,
        pnl_dollars: float,
        kill_switch_triggered: bool,
        kill_reason: str = "",
    ) -> None:
        with self._Session() as session:
            # Upsert: update if exists
            existing = session.query(DailyLog).filter_by(date=date).first()
            if existing:
                existing.total_trades = total_trades
                existing.wins = wins
                existing.losses = losses
                existing.win_rate = win_rate
                existing.pnl_dollars = pnl_dollars
                existing.kill_switch_triggered = int(kill_switch_triggered)
                existing.kill_reason = kill_reason or None
            else:
                row = DailyLog(
                    date=date,
                    total_trades=total_trades,
                    wins=wins,
                    losses=losses,
                    win_rate=win_rate,
                    pnl_dollars=pnl_dollars,
                    kill_switch_triggered=int(kill_switch_triggered),
                    kill_reason=kill_reason or None,
                )
                session.add(row)
            session.commit()

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def log_event(self, event_type: str, detail: str = "") -> None:
        with self._Session() as session:
            row = EventLog(event_type=event_type, detail=detail or None)
            session.add(row)
            session.commit()

        if event_type == "kill_switch" and self._alert_cfg.get("alert_on_kill_switch"):
            self._send_alert(f"KILL SWITCH: {detail}")
        elif event_type == "daily_loss_limit" and self._alert_cfg.get("alert_on_daily_loss_limit"):
            self._send_alert(f"DAILY LOSS LIMIT HIT: {detail}")

    # ------------------------------------------------------------------
    # Read helpers for dashboard
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Model version tracking
    # ------------------------------------------------------------------

    def log_model_version(
        self,
        symbol: str,
        setup_type: str,
        status: str,
        auc: float = 0.0,
        win_rate_sim: float = 0.0,
        brier_score: float = 0.0,
        n_trades: int = 0,
        gate_results: dict | None = None,
        comparison: dict | None = None,
        approved_at: "datetime.datetime | None" = None,
        rejected_at: "datetime.datetime | None" = None,
    ) -> int:
        """Log a model version record. Returns the new row id."""
        import json as _json
        row = ModelVersion(
            symbol=symbol,
            setup_type=setup_type,
            auc=auc,
            win_rate_sim=win_rate_sim,
            brier_score=brier_score,
            n_trades=n_trades,
            status=status,
            gate_results_json=_json.dumps(gate_results or {}),
            comparison_json=_json.dumps(comparison or {}),
            approved_at=approved_at,
            rejected_at=rejected_at,
        )
        with self._Session() as session:
            session.add(row)
            session.commit()
            return row.id

    def update_model_version_status(self, version_id: int, status: str, **kwargs) -> None:
        """Update the status (and optionally approved_at / rejected_at) of a model version."""
        with self._Session() as session:
            row = session.get(ModelVersion, version_id)
            if row:
                row.status = status
                for k, v in kwargs.items():
                    if hasattr(row, k):
                        setattr(row, k, v)
                session.commit()

    def get_model_versions(
        self,
        symbol: str | None = None,
        setup_type: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """
        Return model version history, newest first.
        Optional filters: symbol, setup_type.
        """
        with self._Session() as session:
            q = session.query(ModelVersion).order_by(ModelVersion.timestamp.desc())
            if symbol:
                q = q.filter(ModelVersion.symbol == symbol.upper())
            if setup_type:
                q = q.filter(ModelVersion.setup_type == setup_type.upper())
            rows = q.limit(limit).all()
            return [
                {
                    "id": r.id,
                    "timestamp": str(r.timestamp),
                    "symbol": r.symbol,
                    "setup_type": r.setup_type,
                    "auc": r.auc,
                    "win_rate_sim": r.win_rate_sim,
                    "brier_score": r.brier_score,
                    "n_trades": r.n_trades,
                    "status": r.status,
                    "gate_results": r.gate_results_json,
                    "comparison": r.comparison_json,
                    "approved_at": str(r.approved_at) if r.approved_at else None,
                    "rejected_at": str(r.rejected_at) if r.rejected_at else None,
                }
                for r in rows
            ]

    def get_pending_models(self) -> list[dict]:
        """
        Return all model versions with status='pending', newest first.
        Used by the dashboard Model Approval Panel.
        """
        return self.get_model_versions(limit=100)  # Filter in Python for flexibility
        # Note: ModelLifecycle.get_pending_models() is the primary source;
        # this method is a DB-level backup query.

    def get_recent_trades(self, days: int = 30) -> list[dict]:
        cutoff = (datetime.datetime.utcnow() - datetime.timedelta(days=days)).date()
        with self._Session() as session:
            rows = (
                session.query(TradeLog)
                .filter(TradeLog.date >= str(cutoff))
                .order_by(TradeLog.entry_ts.desc())
                .all()
            )
            return [
                {c.name: getattr(r, c.name) for c in TradeLog.__table__.columns}
                for r in rows
            ]

    def get_daily_history(self, days: int = 90) -> list[dict]:
        cutoff = (datetime.datetime.utcnow() - datetime.timedelta(days=days)).date()
        with self._Session() as session:
            rows = (
                session.query(DailyLog)
                .filter(DailyLog.date >= str(cutoff))
                .order_by(DailyLog.date.asc())
                .all()
            )
            return [
                {c.name: getattr(r, c.name) for c in DailyLog.__table__.columns}
                for r in rows
            ]

    def get_recent_signals(self, limit: int = 50) -> list[dict]:
        with self._Session() as session:
            rows = (
                session.query(SignalLog)
                .order_by(SignalLog.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [
                {c.name: getattr(r, c.name) for c in SignalLog.__table__.columns}
                for r in rows
            ]

    # ------------------------------------------------------------------
    # Alerting
    # ------------------------------------------------------------------

    def _send_alert(self, message: str) -> None:
        """Send a webhook alert (Discord/Telegram/Slack compatible)."""
        if not self._webhook_url:
            return
        try:
            import requests
            payload = {"content": f"[ChargedUp] {message}", "text": f"[ChargedUp] {message}"}
            requests.post(self._webhook_url, json=payload, timeout=5)
        except Exception as e:
            logger.warning("Alert delivery failed: {}", e)
