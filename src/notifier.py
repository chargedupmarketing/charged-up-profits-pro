"""
src/notifier.py

Email notifications — supports two backends:

  1. Gmail SMTP  (default, no domain needed)
     Set NOTIFY_FROM_EMAIL + NOTIFY_GMAIL_APP_PASSWORD in .env.
     Sends to any address instantly.

  2. Resend API  (fallback if gmail credentials absent)
     Set RESEND_API_KEY in .env. Requires a verified domain to send
     to addresses other than the Resend account owner.

Sends clean HTML emails on:
  - Trade ENTRY  : symbol, direction, entry price, stop, target, R:R
  - Trade EXIT   : win/loss, PnL in points and dollars, cumulative day PnL
  - Kill switch  : reason the bot stopped trading for the day
  - Error alert  : critical bot errors (optional)

Configuration in config/settings.yaml under ``notifications:``
and credentials in .env.
"""

from __future__ import annotations

import datetime
import os
import smtplib
import threading
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
ET = ZoneInfo("America/New_York")


def _load_cfg() -> dict:
    p = ROOT / "config" / "settings.yaml"
    with open(p) as f:
        return yaml.safe_load(f).get("notifications", {})


# ── colour palette ────────────────────────────────────────────────────────────
_GREEN  = "#27ae60"
_RED    = "#e74c3c"
_BLUE   = "#2980b9"
_DARK   = "#1a1a2e"
_LIGHT  = "#f5f6fa"
_BORDER = "#dcdde1"


def _base_html(title: str, accent: str, body: str) -> str:
    """Wrap body HTML in a clean branded email shell."""
    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body {{font-family:Arial,sans-serif;background:{_LIGHT};margin:0;padding:20px;}}
    .card {{background:#fff;border-radius:8px;border:1px solid {_BORDER};
            max-width:520px;margin:0 auto;overflow:hidden;}}
    .header {{background:{accent};color:#fff;padding:18px 24px;}}
    .header h2 {{margin:0;font-size:20px;letter-spacing:.5px;}}
    .header p  {{margin:4px 0 0;font-size:13px;opacity:.85;}}
    .body {{padding:22px 24px;}}
    .row {{display:flex;justify-content:space-between;
           padding:9px 0;border-bottom:1px solid {_BORDER};font-size:14px;}}
    .row:last-child {{border-bottom:none;}}
    .label {{color:#666;}}
    .value {{font-weight:bold;color:{_DARK};}}
    .pill  {{display:inline-block;padding:2px 10px;border-radius:12px;
             font-size:12px;font-weight:bold;color:#fff;}}
    .green {{background:{_GREEN};}}
    .red   {{background:{_RED};}}
    .blue  {{background:{_BLUE};}}
    .footer{{text-align:center;font-size:11px;color:#aaa;padding:14px;}}
  </style>
</head>
<body>
  <div class="card">
    <div class="header">
      <h2>⚡ ChargedUp Profits Bot</h2>
      <p>{title}</p>
    </div>
    <div class="body">
      {body}
    </div>
    <div class="footer">ChargedUp Profits Bot &bull; Paper-mode notification &bull;
      {datetime.datetime.now(ET).strftime('%b %d, %Y %I:%M %p ET')}
    </div>
  </div>
</body>
</html>
"""


def _row(label: str, value: str) -> str:
    return f'<div class="row"><span class="label">{label}</span><span class="value">{value}</span></div>'


def _pill(text: str, colour: str) -> str:
    return f'<span class="pill {colour}">{text}</span>'


# ── Notifier class ────────────────────────────────────────────────────────────

class Notifier:
    """
    Thread-safe email notifier.  All sends are fire-and-forget in a daemon
    thread so they never block the trading loop.
    """

    def __init__(self) -> None:
        self._cfg = _load_cfg()
        self._enabled: bool = self._cfg.get("enabled", True)

        # ── Recipients (comma-separated) ──────────────────────────────
        _to_raw: str = self._cfg.get("to_email") or os.getenv("NOTIFY_TO_EMAIL", "")
        self._to_addrs: list[str] = [e.strip() for e in _to_raw.split(",") if e.strip()]

        if self._enabled and not self._to_addrs:
            logger.warning("Notifier: NOTIFY_TO_EMAIL not set — notifications disabled.")
            self._enabled = False
            return

        # ── Backend selection ─────────────────────────────────────────
        # Prefer Gmail SMTP (no domain required, sends to anyone).
        # Fall back to Resend if Gmail credentials are absent.
        self._gmail_user: str = (
            self._cfg.get("gmail_from") or os.getenv("NOTIFY_FROM_EMAIL", "")
        )
        self._gmail_pass: str = (
            self._cfg.get("gmail_app_password") or os.getenv("NOTIFY_GMAIL_APP_PASSWORD", "")
        )
        self._resend_key: str = (
            self._cfg.get("resend_api_key") or os.getenv("RESEND_API_KEY", "")
        )
        self._from_addr: str = self._cfg.get("from_email", "ChargedUp Bot")

        if self._gmail_user and self._gmail_pass:
            self._backend = "gmail"
            logger.info(
                "Notifier ready (Gmail SMTP) — sending from {} to {}",
                self._gmail_user, ", ".join(self._to_addrs),
            )
        elif self._resend_key:
            self._backend = "resend"
            import resend as _resend_mod
            _resend_mod.api_key = self._resend_key
            logger.info(
                "Notifier ready (Resend) — sending to {}",
                ", ".join(self._to_addrs),
            )
        else:
            logger.warning(
                "Notifier: no Gmail or Resend credentials found — notifications disabled."
            )
            self._enabled = False

    # ── public methods ────────────────────────────────────────────────────────

    def notify_entry(
        self,
        symbol: str,
        direction: str,        # "LONG" or "SHORT"
        entry_price: float,
        stop_price: float,
        target_price: float,
        contracts: int,
        setup_type: str,
        paper_mode: bool = True,
    ) -> None:
        """Fire email on trade entry."""
        if not self._enabled:
            return

        rr = abs(target_price - entry_price) / max(abs(entry_price - stop_price), 0.01)
        dir_colour = "green" if direction == "LONG" else "red"
        mode_tag   = "PAPER" if paper_mode else "LIVE"

        body = (
            _row("Mode",       _pill(mode_tag, "blue"))
            + _row("Symbol",   f"{symbol}")
            + _row("Direction",_pill(direction, dir_colour))
            + _row("Setup",    setup_type)
            + _row("Entry",    f"{entry_price:.2f}")
            + _row("Stop",     f"{stop_price:.2f}")
            + _row("Target",   f"{target_price:.2f}")
            + _row("R:R",      f"{rr:.1f}:1")
            + _row("Contracts",str(contracts))
            + _row("Time",     datetime.datetime.now(ET).strftime("%I:%M:%S %p ET"))
        )

        subject = f"[{mode_tag}] {symbol} {direction} ENTRY @ {entry_price:.2f}"
        accent  = _GREEN if direction == "LONG" else _RED
        self._send_async(
            subject,
            _base_html(f"{symbol} Trade Entry", accent, body),
        )

    def notify_exit(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        pnl_points: float,
        pnl_dollars: float,
        exit_reason: str,       # "tp" or "sl"
        day_pnl_dollars: float,
        contracts: int,
        paper_mode: bool = True,
    ) -> None:
        """Fire email on trade exit (TP or SL)."""
        if not self._enabled:
            return

        won        = pnl_dollars >= 0
        result_txt = "✅ WIN" if won else "❌ LOSS"
        reason_txt = "Take Profit" if exit_reason == "tp" else "Stop Loss"
        pnl_colour = "green" if won else "red"
        mode_tag   = "PAPER" if paper_mode else "LIVE"

        body = (
            _row("Mode",        _pill(mode_tag, "blue"))
            + _row("Symbol",    f"{symbol}")
            + _row("Direction", direction)
            + _row("Result",    _pill(result_txt, pnl_colour))
            + _row("Exit via",  reason_txt)
            + _row("Entry",     f"{entry_price:.2f}")
            + _row("Exit",      f"{exit_price:.2f}")
            + _row("P&L pts",   f"{pnl_points:+.2f}")
            + _row("P&L $",     f"${pnl_dollars:+,.2f}")
            + _row("Day P&L",   f"${day_pnl_dollars:+,.2f}")
            + _row("Contracts", str(contracts))
            + _row("Time",      datetime.datetime.now(ET).strftime("%I:%M:%S %p ET"))
        )

        subject = (
            f"[{mode_tag}] {symbol} EXIT {'WIN' if won else 'LOSS'}"
            f" {pnl_points:+.1f}pts (${pnl_dollars:+,.0f})"
        )
        accent  = _GREEN if won else _RED
        self._send_async(
            subject,
            _base_html(f"{symbol} Trade Exit — {result_txt}", accent, body),
        )

    def notify_kill_switch(
        self,
        symbol: str,
        reason: str,
        day_pnl_dollars: float,
        paper_mode: bool = True,
    ) -> None:
        """Fire email when the kill switch halts trading for the day."""
        if not self._enabled:
            return

        mode_tag = "PAPER" if paper_mode else "LIVE"
        body = (
            _row("Mode",      _pill(mode_tag, "blue"))
            + _row("Symbol",  symbol)
            + _row("Reason",  reason)
            + _row("Day P&L", f"${day_pnl_dollars:+,.2f}")
            + _row("Time",    datetime.datetime.now(ET).strftime("%I:%M:%S %p ET"))
        )
        subject = f"[{mode_tag}] {symbol} ⚠️ Kill Switch — {reason}"
        self._send_async(
            subject,
            _base_html(f"{symbol} Kill Switch Triggered", _RED, body),
        )

    def notify_error(
        self,
        symbol: str,
        message: str,
        paper_mode: bool = True,
    ) -> None:
        """Fire email on a critical bot error."""
        if not self._enabled:
            return
        if not self._cfg.get("notify_on_error", True):
            return

        mode_tag = "PAPER" if paper_mode else "LIVE"
        body = (
            _row("Mode",    _pill(mode_tag, "blue"))
            + _row("Symbol", symbol)
            + _row("Error",  message[:300])
            + _row("Time",   datetime.datetime.now(ET).strftime("%I:%M:%S %p ET"))
        )
        subject = f"[{mode_tag}] {symbol} 🚨 Bot Error"
        self._send_async(
            subject,
            _base_html(f"{symbol} Bot Error", _RED, body),
        )

    # ── internal ──────────────────────────────────────────────────────────────

    def _send_async(self, subject: str, html: str) -> None:
        """Send email in a fire-and-forget daemon thread."""
        t = threading.Thread(target=self._send, args=(subject, html), daemon=True)
        t.start()

    def _send(self, subject: str, html: str) -> None:
        try:
            if self._backend == "gmail":
                self._send_gmail(subject, html)
            else:
                self._send_resend(subject, html)
        except Exception as exc:
            logger.warning("Email send failed: {}", exc)

    def _send_gmail(self, subject: str, html: str) -> None:
        """Send via Gmail SMTP using an App Password."""
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"{self._from_addr} <{self._gmail_user}>"
        msg["To"]      = ", ".join(self._to_addrs)
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(self._gmail_user, self._gmail_pass)
            server.sendmail(self._gmail_user, self._to_addrs, msg.as_string())
        logger.debug("Email sent via Gmail to {}", self._to_addrs)

    def _send_resend(self, subject: str, html: str) -> None:
        """Send via Resend API."""
        import resend
        resend.api_key = self._resend_key
        params: resend.Emails.SendParams = {
            "from": f"{self._from_addr} <{self._resend_key and 'onboarding@resend.dev'}>",
            "to":   self._to_addrs,
            "subject": subject,
            "html": html,
        }
        resp = resend.Emails.send(params)
        logger.debug("Email sent via Resend: id={}", resp.get("id", "?"))
