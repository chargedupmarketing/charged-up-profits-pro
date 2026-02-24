"""
monitoring/dashboard.py

ChargedUp Profits Bot — Full-Control Web Panel

A browser-based management interface that replaces the need to use the
terminal or IDE to run and monitor the trading bot.

What you can do from this panel:
  ✅ Start and stop the bot for ES and/or NQ separately
  ✅ Switch between paper (simulation) and live trading modes
  ✅ Monitor real-time P&L, open positions, and account health
  ✅ View the account drawdown meter (color-coded safety indicator)
  ✅ Adjust key settings (regime filter, ML threshold, time window) with sliders
  ✅ Emergency flatten button — closes all positions immediately
  ✅ View the full trade history and signal rejection log
  ✅ See performance charts: equity curve, setup breakdown, win rate by day

Run with:
    streamlit run monitoring/dashboard.py

Then open: http://localhost:8501 in any browser (or phone on the same network).
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as _stc
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Timezone helpers — all display times are US Eastern (ET)
# ---------------------------------------------------------------------------
_ET = ZoneInfo("America/New_York")


def _now_et() -> datetime:
    """Current datetime in US Eastern timezone."""
    return datetime.now(tz=_ET)


def _fmt_time_et(dt: datetime) -> str:
    """Format a datetime as 12-hour Eastern time, e.g. '2:45:07 PM ET'."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_ET)
    dt_et = dt.astimezone(_ET)
    return dt_et.strftime("%I:%M:%S %p ET").lstrip("0")


def _fmt_short_et(dt: datetime) -> str:
    """Format as short 12-hour ET, e.g. '2:45 PM ET'."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_ET)
    dt_et = dt.astimezone(_ET)
    return dt_et.strftime("%I:%M %p ET").lstrip("0")


def _to_et(dt: datetime) -> datetime:
    """Convert a datetime (naive or aware) to US Eastern timezone."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_ET)
    return dt.astimezone(_ET)

from monitoring.audit_log import AuditLogger

# ---------------------------------------------------------------------------
# Background task helpers — run scripts as subprocesses, stream output
# ---------------------------------------------------------------------------

TASK_LOG_DIR = ROOT / "data" / "run_logs"
TASK_LOG_DIR.mkdir(parents=True, exist_ok=True)

TASKS: dict[str, dict] = {
    "backtest":           {"label": "Full Backtest",                    "log": TASK_LOG_DIR / "backtest.log"},
    "backtest_all":       {"label": "Full Backtest (All Symbols)",      "log": TASK_LOG_DIR / "backtest_all.log"},
    "walkforward":        {"label": "Walk-Forward Test",                "log": TASK_LOG_DIR / "walkforward.log"},
    "continuous_learner": {"label": "AI Re-Learning (Background)",      "log": TASK_LOG_DIR / "continuous_learner.log"},
    "walkforward_all": {"label": "Walk-Forward Test (All Symbols)",  "log": TASK_LOG_DIR / "walkforward_all.log"},
    "ml_retrain":      {"label": "Retrain ML Filter",                "log": TASK_LOG_DIR / "ml_retrain.log"},
    "ml_retrain_all":  {"label": "Retrain ML Filter (All Symbols)",  "log": TASK_LOG_DIR / "ml_retrain_all.log"},
    "youtube":         {"label": "YouTube Strategy Scan",            "log": TASK_LOG_DIR / "youtube.log"},
    "automation":      {"label": "Automation Loop",                  "log": TASK_LOG_DIR / "automation.log"},
}
TASK_PIDS_FILE = ROOT / "data" / "task_pids.json"

# Maps each task key → the script filename to look for in process command lines
_TASK_SCRIPT_MAP: dict[str, str] = {
    "backtest":        "harness.py",
    "backtest_all":    "run_all.py",
    "walkforward":     "walk_forward.py",
    "walkforward_all": "wf_all.py",
    "ml_retrain":      "ml_filter.py",
    "ml_retrain_all":  "ml_retrain_all.py",
    "youtube":         "youtube_analyzer.py",
    "automation":      "run_automation.py",
}


def _load_task_pids() -> dict:
    return _read_json(TASK_PIDS_FILE)


def _save_task_pids(pids: dict) -> None:
    _write_json(TASK_PIDS_FILE, pids)


def _reconcile_task_pids() -> None:
    """
    Called once per render to keep task_pids.json accurate.

    Two jobs:
      1. Prune entries whose process has already exited (prevents 'stuck' badges).
      2. Discover tasks that are running but aren't in the file yet —
         e.g. started from the terminal, or after a panel restart.

    Uses psutil to inspect process command lines.  Fails silently if psutil
    is not available (it's optional).
    """
    try:
        import psutil as _psutil
    except ImportError:
        return

    pids = _load_task_pids()
    changed = False

    # 1. Remove entries for dead processes
    for task_key in list(pids.keys()):
        pid = _get_task_pid(task_key)
        if pid > 0 and not _is_process_running(pid):
            pids.pop(task_key, None)
            changed = True

    # 2. Scan for running processes we're not tracking
    for proc in _psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            cmdline_str = " ".join(cmdline)
            for task_key, script_name in _TASK_SCRIPT_MAP.items():
                if script_name in cmdline_str and task_key not in pids:
                    pids[task_key] = {
                        "pid": proc.info["pid"],
                        "started": datetime.fromtimestamp(
                            proc.info["create_time"]
                        ).isoformat(),
                    }
                    changed = True
        except Exception:
            pass

    if changed:
        _save_task_pids(pids)


# -- helpers to read structured task metadata (pid + start time) ---------------

def _get_task_pid(task_key: str) -> int:
    """Return the PID for a task, handling both old (int) and new (dict) formats."""
    entry = _load_task_pids().get(task_key, 0)
    if isinstance(entry, dict):
        return entry.get("pid", 0)
    return int(entry) if entry else 0


def _get_task_started(task_key: str) -> Optional[datetime]:
    """Return the datetime when a task was last started, or None."""
    entry = _load_task_pids().get(task_key)
    if isinstance(entry, dict) and "started" in entry:
        try:
            return datetime.fromisoformat(entry["started"])
        except ValueError:
            pass
    return None


def _format_duration(secs: float) -> str:
    """Turn a number of seconds into a human-readable string, e.g. '3m 27s'."""
    secs = max(0.0, secs)
    h, rem = divmod(int(secs), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _parse_task_progress(task_key: str) -> tuple[int, int, str]:
    """
    Parse log output to determine task progress.

    Returns:
        (done, total, phase_label)
        done=0 / total=0 means 'unknown / starting up'
    """
    log_text = _read_task_log(task_key, last_n_lines=600)
    if not log_text:
        return 0, 0, "Starting up…"

    if task_key == "backtest_all":
        # Top-level: ALL_PROGRESS X/N (SYMBOL)
        all_hits = re.findall(r"ALL_PROGRESS (\d+)/(\d+) \((\w+)\)", log_text)
        if "ALL_DONE" in log_text:
            _t = int(all_hits[-1][1]) if all_hits else 3
            return _t, _t, "All symbols complete ✅"
        if all_hits:
            done_sym, total_sym, current_sym = all_hits[-1]
            done_sym, total_sym = int(done_sym), int(total_sym)
            # Also check inner PROGRESS for the current symbol
            progress_hits = re.findall(r"PROGRESS (\d+)/(\d+)", log_text)
            if progress_hits:
                inner_done, inner_total = int(progress_hits[-1][0]), int(progress_hits[-1][1])
                inner_pct = inner_done / inner_total if inner_total else 0
                # Scale inner progress into the current symbol's 1/3 slice
                overall_pct = ((done_sym - 1) + inner_pct) / total_sym
                overall_done = int(overall_pct * total_sym * 100)
                overall_total = total_sym * 100
                return overall_done, overall_total, f"{current_sym}: day {inner_done}/{inner_total} ({done_sym}/{total_sym} symbols)"
            return done_sym - 1, total_sym, f"Starting {current_sym}… ({done_sym}/{total_sym} symbols)"
        return 0, 3, "Starting up…"

    if task_key == "backtest":
        # Total from "Running backtest over X trading days"
        m_total = re.search(r"Running backtest over (\d+) trading days", log_text)
        total = int(m_total.group(1)) if m_total else 0
        # Done from most-recent "PROGRESS X/Y"
        progress_hits = re.findall(r"PROGRESS (\d+)/(\d+)", log_text)
        if progress_hits:
            done, total = int(progress_hits[-1][0]), int(progress_hits[-1][1])
            pct = done / total * 100 if total else 0
            return done, total, f"Day {done}/{total} ({pct:.0f}%)"
        if total:
            return 0, total, "Loading data…"
        return 0, 0, "Starting up…"

    elif task_key == "walkforward_all":
        all_hits = re.findall(r"WF_ALL_PROGRESS (\d+)/(\d+) \((\w+)\)", log_text)
        if "WF_ALL_DONE" in log_text:
            _t = int(all_hits[-1][1]) if all_hits else 3
            return _t, _t, "All symbols complete ✅"
        if all_hits:
            done_sym, total_sym, current_sym = all_hits[-1]
            done_sym, total_sym = int(done_sym), int(total_sym)
            # Inner window progress for the current symbol
            wf_hits = re.findall(r"WF_WINDOW (\d+)/(\d+)", log_text)
            if wf_hits:
                inner_done, inner_total = int(wf_hits[-1][0]), int(wf_hits[-1][1])
                inner_pct = inner_done / inner_total if inner_total else 0
                overall_done = int(((done_sym - 1) + inner_pct) / total_sym * total_sym * 100)
                overall_total = total_sym * 100
                return overall_done, overall_total, f"{current_sym}: window {inner_done}/{inner_total} ({done_sym}/{total_sym} symbols)"
            return done_sym - 1, total_sym, f"Starting {current_sym}… ({done_sym}/{total_sym} symbols)"
        return 0, 3, "Starting up…"

    elif task_key == "walkforward":
        # Total from "Running walk-forward with X windows"
        m_total = re.search(r"Running walk-forward with (\d+) windows", log_text)
        total = int(m_total.group(1)) if m_total else 0
        # Done from most-recent "WF_WINDOW X/Y"
        wf_hits = re.findall(r"WF_WINDOW (\d+)/(\d+)", log_text)
        if wf_hits:
            done, total = int(wf_hits[-1][0]), int(wf_hits[-1][1])
            return done, total, f"Window {done}/{total}"
        if total:
            return 0, total, "Loading data…"
        return 0, 0, "Starting up…"

    elif task_key == "ml_retrain_all":
        all_hits = re.findall(r"ML_ALL_PROGRESS (\d+)/(\d+) \((\w+)\)", log_text)
        if "ML_ALL_DONE" in log_text:
            _t = int(all_hits[-1][1]) if all_hits else 3
            return _t, _t, "All models retrained ✅"
        if all_hits:
            done_sym, total_sym, current_sym = all_hits[-1]
            done_sym, total_sym = int(done_sym), int(total_sym)
            # Check inner fold progress for the current symbol
            fold_hits = re.findall(r"Fold (\d+)/(\d+)", log_text)
            if fold_hits:
                inner_done, inner_total = int(fold_hits[-1][0]), int(fold_hits[-1][1])
                inner_pct = inner_done / inner_total if inner_total else 0
                overall_done = int(((done_sym - 1) + inner_pct) / total_sym * total_sym * 100)
                overall_total = total_sym * 100
                return overall_done, overall_total, f"{current_sym}: fold {inner_done}/{inner_total} ({done_sym}/{total_sym} models)"
            return done_sym - 1, total_sym, f"Training {current_sym}… ({done_sym}/{total_sym} models)"
        return 0, 3, "Starting up…"

    elif task_key == "ml_retrain":
        # Rough phases: loading → training fold → evaluating → done
        if "SHAP" in log_text or "Evaluation complete" in log_text:
            return 4, 4, "Evaluating & SHAP analysis"
        if "Training XGBoost" in log_text or "XGBClassifier" in log_text:
            fold_hits = re.findall(r"Fold (\d+)/(\d+)", log_text)
            if fold_hits:
                done, total = int(fold_hits[-1][0]), int(fold_hits[-1][1])
                return done, total, f"Cross-val fold {done}/{total}"
            return 2, 4, "Training XGBoost model"
        if "feature" in log_text.lower() or "Loaded" in log_text:
            return 1, 4, "Loading features & building dataset"
        return 0, 4, "Starting up…"

    elif task_key == "youtube":
        # Count transcripts attempted vs total videos list
        total_hits = re.findall(r"Processing (\d+) video", log_text)
        done_hits = re.findall(r"Transcript downloaded|rules extracted|Error downloading", log_text)
        total = int(total_hits[-1]) if total_hits else 0
        done = len(done_hits)
        if total:
            return done, total, f"Video {done}/{total}"
        return 0, 0, "Fetching video list…"

    elif task_key == "automation":
        # AUTO_CYCLE_START N / AUTO_STEP_START step symbols / AUTO_CYCLE_DONE N
        if "AUTO_FINISHED" in log_text:
            cycle_done = re.findall(r"AUTO_CYCLE_DONE (\d+)", log_text)
            total_cycles = int(cycle_done[-1]) if cycle_done else 1
            return total_cycles, total_cycles, f"Completed {total_cycles} cycle(s) ✅"
        step_starts = re.findall(r"AUTO_STEP_START (\w+)", log_text)
        step_dones  = re.findall(r"AUTO_STEP_DONE (\w+)", log_text)
        cycle_starts = re.findall(r"AUTO_CYCLE_START (\d+)", log_text)
        cycle_dones  = re.findall(r"AUTO_CYCLE_DONE (\d+)", log_text)
        current_cycle = int(cycle_starts[-1]) if cycle_starts else 0
        done_cycles   = int(cycle_dones[-1])  if cycle_dones  else 0
        if step_starts:
            current_step = step_starts[-1]
            done_steps   = len(step_dones)
            step_label = {"retrain": "Retraining AI", "backtest": "Running Backtest",
                          "wfa": "Walk-Forward Test"}.get(current_step, current_step)
            return done_cycles * 10 + done_steps, max((current_cycle) * 10, 1), \
                   f"Cycle {current_cycle} — {step_label} ({done_steps} steps done)"
        if current_cycle:
            return done_cycles * 10, current_cycle * 10, f"Starting cycle {current_cycle}…"
        return 0, 0, "Starting automation…"

    return 0, 0, ""


def _task_completed_info(task_key: str) -> Optional[tuple[str, str]]:
    """
    Return (elapsed_str, finish_time_str) if the task's log exists but the
    process is no longer running — i.e. it completed (or was stopped).
    Returns None if still running or no log exists.
    """
    if _is_task_running(task_key):
        return None
    log_path = TASKS[task_key]["log"]
    if not log_path.exists():
        return None
    try:
        mtime = datetime.fromtimestamp(log_path.stat().st_mtime)
        started = _get_task_started(task_key)
        if started:
            elapsed = _format_duration((mtime - started).total_seconds())
        else:
            elapsed = "unknown duration"
        return elapsed, _fmt_time_et(mtime)
    except Exception:
        return None


def _render_task_status_badge(task_key: str) -> bool:
    """
    Render a compact status badge for a task and return True if it is running
    (so the caller can disable the Start button).

    States shown:
      ● RUNNING  — blue, with elapsed time
      ✅ COMPLETED — green, with finish time and elapsed
      ○ IDLE     — grey (no previous run this session)
    """
    running = _is_task_running(task_key)
    label = TASKS[task_key]["label"]
    icon = _TASK_ICON.get(task_key, "⚙️")

    if running:
        started = _get_task_started(task_key)
        elapsed = _format_duration(
            (datetime.now() - started).total_seconds() if started else 0
        )
        pid = _get_task_pid(task_key)
        st.markdown(
            f"<span style='background:#1e3a5f;color:#7ec8e3;padding:4px 10px;"
            f"border-radius:12px;font-size:0.85em;font-weight:bold;'>"
            f"● RUNNING — {elapsed} elapsed (PID {pid})</span>",
            unsafe_allow_html=True,
        )
        return True

    completed = _task_completed_info(task_key)
    if completed:
        elapsed_str, finish_str = completed
        st.markdown(
            f"<span style='background:#1a3a1a;color:#57e389;padding:4px 10px;"
            f"border-radius:12px;font-size:0.85em;font-weight:bold;'>"
            f"✅ COMPLETED — finished at {finish_str} (ran for {elapsed_str})</span>",
            unsafe_allow_html=True,
        )
        return False

    st.markdown(
        f"<span style='background:#2d2d2d;color:#888;padding:4px 10px;"
        f"border-radius:12px;font-size:0.85em;'>○ IDLE — not started</span>",
        unsafe_allow_html=True,
    )
    return False


# Expected duration hints (seconds) used for the indeterminate progress bar
_TASK_DURATION_HINT: dict[str, int] = {
    "backtest":        5 * 60,    # ~5 minutes per symbol
    "backtest_all":    15 * 60,   # ~15 minutes for all 3 symbols
    "walkforward":     15 * 60,   # ~15 minutes per symbol
    "walkforward_all": 45 * 60,   # ~45 minutes for all 3 symbols
    "ml_retrain":      3 * 60,    # ~3 minutes
    "ml_retrain_all":  8 * 60,    # ~8 minutes for all 3 symbols
    "youtube":         2 * 60,    # ~2 minutes
    "automation":      60 * 60,   # varies — runs until stop time
}

# Labels shown inside the status header
_TASK_ICON: dict[str, str] = {
    "backtest":        "📊",
    "backtest_all":    "🔁",
    "walkforward":     "🔄",
    "walkforward_all": "🔁",
    "ml_retrain":      "🤖",
    "ml_retrain_all":  "🤖",
    "youtube":         "🎬",
    "automation":      "🔄",
}


def _render_progress_panel(task_key: str) -> None:
    """
    Render a rich status panel for a background task:
      • ✅ Running indicator with PID
      • ⏱  Elapsed timer
      • ⌛  ETA (calculated from progress or estimated from historical duration)
      • Progress bar (exact % when data available, animated when not)
      • Phase label (what stage the task is at)
      • Last log line
      • ⏹ Stop button
    If the task is NOT running, renders nothing (caller handles that).
    """
    running = _is_task_running(task_key)
    if not running:
        return

    icon = _TASK_ICON.get(task_key, "⚙️")
    label = TASKS[task_key]["label"]
    pid = _get_task_pid(task_key)
    started = _get_task_started(task_key)

    # Elapsed time
    elapsed_secs = (datetime.now() - started).total_seconds() if started else 0.0
    elapsed_str = _format_duration(elapsed_secs)

    # Progress
    done, total, phase_label = _parse_task_progress(task_key)

    # ETA calculation
    if done > 0 and total > 0 and elapsed_secs > 0:
        rate = elapsed_secs / done          # seconds per unit
        remaining = rate * (total - done)
        eta_str = f"~{_format_duration(remaining)} remaining"
        pct = done / total
    else:
        # Fall back to time-based estimate
        hint = _TASK_DURATION_HINT.get(task_key, 300)
        if elapsed_secs > 0 and hint > 0:
            pct = min(elapsed_secs / hint, 0.95)   # cap at 95% until we know it finished
            remaining = max(0, hint - elapsed_secs)
            eta_str = f"~{_format_duration(remaining)} remaining (estimated)"
        else:
            pct = 0.0
            eta_str = "calculating…"

    # ── Render ──────────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='background:#1e3a5f;border-radius:8px;padding:12px 16px;margin-bottom:8px;'>"
        f"<b>{icon} {label}</b> &nbsp; "
        f"<span style='color:#7ec8e3;font-size:0.85em;'>● RUNNING (PID {pid})</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    col_timer, col_eta, col_stop = st.columns([2, 3, 1])
    with col_timer:
        st.metric("⏱ Elapsed", elapsed_str)
    with col_eta:
        st.metric("⌛ ETA", eta_str)
    with col_stop:
        st.markdown("")
        if st.button(f"⏹ Stop", key=f"stop_panel_{task_key}", type="secondary"):
            _stop_task(task_key)
            st.rerun()

    # Progress bar
    bar_text = phase_label if phase_label else "Working…"
    if done > 0 and total > 0:
        bar_text = f"{bar_text}  —  {done:,} / {total:,} units"
    st.progress(min(max(pct, 0.01), 1.0), text=bar_text)

    # Last log line (strip ANSI, skip blank/timestamp-only lines)
    log_text = _read_task_log(task_key, last_n_lines=50)
    if log_text:
        ansi_re = re.compile(r"\x1b\[[0-9;]*[mGKHF]")
        last_lines = [
            ansi_re.sub("", ln).strip()
            for ln in log_text.splitlines()
            if ln.strip() and "|" in ln
        ]
        if last_lines:
            st.caption(f"Last: `{last_lines[-1][:120]}`")


def _is_task_running(task_key: str) -> bool:
    pid = _get_task_pid(task_key)
    return pid > 0 and _is_process_running(pid)


def _start_task(task_key: str, cmd: list[str]) -> tuple[bool, str]:
    """Launch a command as a background subprocess, output → task log file."""
    if _is_task_running(task_key):
        return False, f"{TASKS[task_key]['label']} is already running."

    log_path: Path = TASKS[task_key]["log"]
    try:
        kwargs: dict = {"cwd": str(ROOT)}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        # Open log in binary write mode — avoids text/encoding conflicts on Windows.
        # Do NOT use a `with` block: the file must stay open while the subprocess runs.
        log_f = open(log_path, "wb")

        proc = subprocess.Popen(
            cmd,
            **kwargs,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,   # never inherit Streamlit's stdin
        )
        # log_f stays open; OS reclaims it when the child process exits

        pids = _load_task_pids()
        pids[task_key] = {"pid": proc.pid, "started": datetime.now().isoformat()}
        _save_task_pids(pids)

        return True, f"Started (PID {proc.pid})"
    except Exception as e:
        return False, f"Failed to start: {e}"


def _stop_task(task_key: str) -> tuple[bool, str]:
    pid = _get_task_pid(task_key)
    if pid <= 0:
        return False, "Not running."
    ok = _kill_process(pid)
    pids = _load_task_pids()
    pids.pop(task_key, None)
    _save_task_pids(pids)
    return ok, "Stopped." if ok else f"Could not stop PID {pid}."


def _read_task_log(task_key: str, last_n_lines: int = 80) -> str:
    """Read the last N lines of a task's output log (binary-safe)."""
    log_path: Path = TASKS[task_key]["log"]
    if not log_path.exists():
        return ""
    try:
        raw = log_path.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        return "\n".join(lines[-last_n_lines:])
    except Exception:
        return ""


def _load_backtest_csv(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Load a backtest results CSV if it exists."""
    fname = ROOT / "data" / f"backtest_{symbol}_{start}_{end}.csv"
    if fname.exists():
        try:
            return pd.read_csv(fname)
        except Exception:
            pass
    # Try any recent backtest file for this symbol
    candidates = sorted((ROOT / "data").glob(f"backtest_{symbol}_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        try:
            return pd.read_csv(candidates[0])
        except Exception:
            pass
    return pd.DataFrame()


def _load_wfa_csv(symbol: str) -> pd.DataFrame:
    candidates = sorted((ROOT / "data").glob(f"wfa_trades_{symbol}.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        try:
            return pd.read_csv(candidates[0])
        except Exception:
            pass
    return pd.DataFrame()


def _backtest_summary(df: pd.DataFrame) -> dict:
    """Compute key metrics from a backtest CSV.

    Backtest CSVs use the column ``pnl_net`` (net P&L in dollars after
    commission).  The live-trade audit-log uses ``pnl_dollars`` — these are
    different data sources, so we resolve whichever column is present.
    """
    # Resolve the P&L column name (backtest CSV = pnl_net, audit log = pnl_dollars)
    pnl_col = "pnl_net" if "pnl_net" in df.columns else "pnl_dollars" if "pnl_dollars" in df.columns else None
    if df.empty or pnl_col is None:
        return {}
    wins = (df[pnl_col] > 0).sum()
    losses = (df[pnl_col] <= 0).sum()
    total = len(df)
    net = df[pnl_col].sum()
    win_rate = wins / total * 100 if total > 0 else 0
    avg_win = df[df[pnl_col] > 0][pnl_col].mean() if wins > 0 else 0
    avg_loss = df[df[pnl_col] < 0][pnl_col].mean() if losses > 0 else 0
    pf_wins = df[df[pnl_col] > 0][pnl_col].sum()
    pf_losses = abs(df[df[pnl_col] < 0][pnl_col].sum())
    pf = round(pf_wins / pf_losses, 2) if pf_losses > 0 else 0
    equity = df[pnl_col].cumsum()
    max_dd = (equity - equity.cummax()).min() if len(equity) > 0 else 0
    return {
        "total_trades": total, "wins": wins, "losses": losses,
        "win_rate": win_rate, "net_pnl": net, "profit_factor": pf,
        "avg_win": avg_win, "avg_loss": avg_loss, "max_drawdown": max_dd,
        "_pnl_col": pnl_col,   # pass through so callers can use the right column
    }


def _generate_improvement_plan(user_request: str, stats: dict) -> str:
    """
    Generate a targeted improvement checklist based on user's plain-English request
    and the current backtest stats. No external AI API needed — rule-based logic.
    """
    req = user_request.lower()
    lines = [
        "## Improvement Plan",
        f"*Based on your request: \"{user_request.strip()}\"*",
        "",
    ]

    if stats:
        lines += [
            "### Current Performance Baseline",
            f"- Trades: **{stats.get('total_trades', '?')}** | "
            f"Win Rate: **{stats.get('win_rate', 0):.1f}%** | "
            f"Net P&L: **${stats.get('net_pnl', 0):+,.2f}**",
            f"- Profit Factor: **{stats.get('profit_factor', 0)}** | "
            f"Max Drawdown: **${stats.get('max_drawdown', 0):,.2f}**",
            "",
        ]

    lines.append("### Recommended Actions")
    lines.append("")

    action_num = 1

    # More trades
    if any(w in req for w in ["more trades", "more signals", "too few trades", "trade more", "trade frequency"]):
        lines += [
            f"**{action_num}. Increase Trade Frequency**",
            "   - Go to **Settings** tab → raise *Volatility Threshold* from 2.2 → 2.5",
            "   - This allows trading on more days by relaxing the volatility gate",
            "   - Consider adding NQ in **Settings** (checkbox) to double opportunities",
            "   - Extend execution window to 13:30 if you want even more afternoon setups",
            "   - *After changing: re-run Backtest to verify quality doesn't drop*",
            "",
        ]
        action_num += 1

    # Better win rate
    if any(w in req for w in ["win rate", "win more", "losing", "accuracy", "60%", "70%", "65%"]):
        lines += [
            f"**{action_num}. Improve Win Rate**",
            "   - Go to **Settings** → raise *Minimum Win Probability* from 0.60 → 0.65–0.70",
            "   - This makes the AI filter stricter — fewer trades but more winners",
            "   - Retrain the ML model (section above) after any new backtest",
            "   - Check the Signal Log tab to see which setups have the lowest win rate",
            "   - Consider disabling the weakest setup type under ⚙️ Setup Detector in settings.yaml",
            "",
        ]
        action_num += 1

    # More profit / bigger wins
    if any(w in req for w in ["more profit", "bigger wins", "more money", "maximize", "1000", "profitab"]):
        lines += [
            f"**{action_num}. Maximize Profit Per Trade**",
            "   - The current setup uses partial TP at 1.5R — this is conservative but safe",
            "   - On your funded account, scaling to 2 contracts after +$2,000 profit auto-activates",
            "   - The *Account Health* tab shows when you'll scale up",
            "   - Do NOT increase risk per trade — instead let contract scaling work over time",
            "   - Consider running both ES and NQ simultaneously to compound opportunities",
            "",
        ]
        action_num += 1

    # Specific day losses
    if any(w in req for w in ["monday", "tuesday", "wednesday", "thursday", "friday", "day of week"]):
        for day_name, day_num in [("monday", 0), ("tuesday", 1), ("wednesday", 2), ("thursday", 3), ("friday", 4)]:
            if day_name in req:
                lines += [
                    f"**{action_num}. Fix {day_name.title()} Performance**",
                    f"   - Go to **Settings** → check *Skip {day_name.title()}s* to exclude that day",
                    f"   - Check the **Performance** tab → Win Rate by Day chart to confirm",
                    f"   - Alternatively: run a Backtest and check the breakdown by day in the CSV",
                    "",
                ]
                action_num += 1

    # Drawdown / losing streaks / account safety
    if any(w in req for w in ["drawdown", "losing streak", "account", "safe", "protect", "blow"]):
        lines += [
            f"**{action_num}. Reduce Risk and Protect Account**",
            "   - Go to **Settings** → lower *Max Trades Per Day* to 2",
            "   - Raise *Safety Buffer* to $750 in the funded account section",
            "   - Enable the Regime Filter if not already on",
            "   - The Account Health tab shows a live meter — check it daily",
            "",
        ]
        action_num += 1

    # Research / new setups
    if any(w in req for w in ["research", "new setup", "youtube", "rp profits", "strategy", "improve strategy"]):
        lines += [
            f"**{action_num}. Research New Strategy Rules**",
            "   - Run the *YouTube Strategy Scanner* (section above) for fresh video analysis",
            "   - Review the strategy notes for any rules not yet in the bot",
            "   - *Sweep and reverse* (16x mentions) is the top uncoded setup — highest priority to add",
            "   - News event filter (FOMC, CPI, NFP avoidance) mentioned 9x — worth adding",
            "",
        ]
        action_num += 1

    if action_num == 1:
        # Generic fallback
        lines += [
            "**1. Run a Fresh Backtest**",
            "   - Use the *Full Backtest* section above to see current performance",
            "",
            "**2. Review Setup Breakdown**",
            "   - Go to the *Performance* tab to see which setups make money",
            "   - Disable underperforming setups in the Settings tab",
            "",
            "**3. Re-run YouTube Scanner**",
            "   - Use the *YouTube Strategy Scanner* to check for new rule ideas",
            "",
            "**4. Retrain the AI Filter**",
            "   - After any strategy change, retrain the ML model to update the AI scoring",
            "",
        ]

    lines += [
        "---",
        "### Next Step Checklist",
        "- [ ] Run the Backtest with current settings to get a baseline",
        "- [ ] Make one change at a time (don't change everything at once)",
        "- [ ] Re-run Backtest after each change to see if it helped",
        "- [ ] If trade count drops below 50, loosen the volatility gate (+0.2)",
        "- [ ] If win rate drops below 55%, tighten the ML threshold (+0.05)",
        "- [ ] Retrain the ML model after significant rule changes",
        "",
        "*Always test in paper mode first. Never skip the walk-forward validation.*",
    ]

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Page config — must be FIRST streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ChargedUp Profits Bot — Control Panel",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SETTINGS_PATH = ROOT / "config" / "settings.yaml"
PIDS_FILE = ROOT / "data" / "bot_pids.json"
FLATTEN_FLAG = ROOT / "data" / "flatten_requested.flag"
SYMBOLS = ["ES", "NQ", "MNQ"]
SCHEDULE_FILE = ROOT / "data" / "bot_schedule.json"

# Valid schedule keys and their human labels
_SCHEDULE_OPTIONS: dict[str, dict] = {
    "ES":         {"label": "ES only",       "symbols": ["ES"],           "multi": False},
    "NQ":         {"label": "NQ only",       "symbols": ["NQ"],           "multi": False},
    "MNQ":        {"label": "MNQ only",      "symbols": ["MNQ"],          "multi": False},
    "ES+NQ":      {"label": "ES + NQ",       "symbols": ["ES", "NQ"],     "multi": True},
    "ES+NQ+MNQ":  {"label": "All Three",     "symbols": ["ES", "NQ", "MNQ"], "multi": True},
}


def _load_settings() -> dict:
    with open(SETTINGS_PATH, "r") as f:
        return yaml.safe_load(f)


def _load_schedule() -> dict:
    """Load the 9am auto-start schedule from disk."""
    try:
        if SCHEDULE_FILE.exists():
            with open(SCHEDULE_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {"active": None, "last_autostart_date": None}


def _save_schedule(data: dict) -> None:
    SCHEDULE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SCHEDULE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _save_settings(cfg: dict) -> None:
    """Write config back to settings.yaml with a comment header preserved."""
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _read_json(path: Path) -> dict:
    """Safely read a JSON file; return {} if missing or corrupt."""
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _load_pids() -> dict[str, int]:
    return _read_json(PIDS_FILE)


def _save_pids(pids: dict) -> None:
    _write_json(PIDS_FILE, pids)


def _is_process_running(pid: int) -> bool:
    """Return True if the given PID is a running process."""
    if pid <= 0:
        return False
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        # Fallback: os.kill with signal 0 (doesn't kill, just checks)
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


def _kill_process(pid: int) -> bool:
    """Terminate a process by PID. Returns True if successful."""
    if pid <= 0:
        return False
    try:
        import psutil
        p = psutil.Process(pid)
        p.terminate()
        try:
            p.wait(timeout=5)
        except psutil.TimeoutExpired:
            p.kill()
        return True
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
            return True
        except Exception:
            return False


def _bot_state(symbol: str) -> dict:
    """Read the bot state file for a given symbol."""
    state_file = ROOT / "data" / f"bot_state_{symbol}.json"
    return _read_json(state_file)


# ---------------------------------------------------------------------------
# TradingView Lightweight Charts renderer
# ---------------------------------------------------------------------------

_TV_CDN = "https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"

def _render_tv_chart(
    symbol: str,
    trades_df: "pd.DataFrame",
    timeframe: str = "1m",
    height: int = 440,
) -> None:
    """
    Render a professional TradingView Lightweight Chart for *symbol* using
    bar data written by BotRunner._write_bars_file().

    Shows:
      • Candlestick bars (1m or 15m, last 300 / 100 bars)
      • Today's price levels as dashed horizontal lines
      • Trade entry markers (green arrow-up = long win, red = loss, etc.)
    """
    bars_file = ROOT / "data" / f"bars_{symbol}.json"
    if not bars_file.exists():
        st.info(
            f"No live chart data for **{symbol}** yet. "
            f"Start the bot and the chart will appear here automatically."
        )
        return

    try:
        bar_data = json.loads(bars_file.read_text(encoding="utf-8"))
    except Exception:
        st.warning("Could not read bar data — file may be mid-write. Refresh in a moment.")
        return

    bars = bar_data.get(f"bars_{timeframe}", [])
    levels = bar_data.get("levels", [])
    last_upd = bar_data.get("last_update", "")

    if not bars:
        st.info(f"Waiting for first {timeframe} bars for {symbol}...")
        return

    # Build trade markers from today's audit trades
    markers: list = []
    if not trades_df.empty:
        _sym_col = "symbol" if "symbol" in trades_df.columns else None
        _df = trades_df[trades_df[_sym_col] == symbol] if _sym_col else trades_df
        if not _df.empty and "entry_ts" in _df.columns:
            _today = date.today()
            _td = _df[pd.to_datetime(_df["entry_ts"], utc=True, errors="coerce").dt.date == _today]
            for _, tr in _td.iterrows():
                try:
                    _t = int(pd.Timestamp(tr["entry_ts"]).timestamp())
                    _dir = str(tr.get("direction", "LONG")).upper()
                    _res = str(tr.get("exit_reason", "tp")).lower()
                    _pnl = float(tr.get("pnl_dollars", 0))
                    _win = _res == "tp"
                    markers.append({
                        "time":      _t,
                        "position":  "belowBar" if _dir == "LONG" else "aboveBar",
                        "color":     "#26a69a" if _win else "#ef5350",
                        "shape":     "arrowUp" if _dir == "LONG" else "arrowDown",
                        "text":      f"{'✓' if _win else '✗'} {_dir} ${_pnl:+.0f}",
                    })
                except Exception:
                    pass

    # Sort markers by time (required by Lightweight Charts)
    markers.sort(key=lambda x: x["time"])

    bars_json    = json.dumps(bars)
    levels_json  = json.dumps(levels)
    markers_json = json.dumps(markers)
    tf_label     = "1-min" if timeframe == "1m" else "15-min"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0e0e17;overflow:hidden}}
  #wrap{{position:relative;width:100%;height:{height}px}}
  #tv{{width:100%;height:{height}px}}
  #info{{position:absolute;top:8px;left:10px;z-index:10;pointer-events:none;
         font-family:-apple-system,sans-serif;font-size:12px;color:#aaa}}
  #info b{{color:#e0e0e0;font-size:13px}}
  #upd{{position:absolute;top:8px;right:10px;z-index:10;pointer-events:none;
        font-family:-apple-system,sans-serif;font-size:10px;color:#555}}
</style></head>
<body>
<div id="wrap">
  <div id="info"><b>{symbol} — {tf_label}</b> &nbsp;|&nbsp; {len(bars)} bars</div>
  <div id="upd">Updated {last_upd[11:19] if len(last_upd) > 18 else last_upd}</div>
  <div id="tv"></div>
</div>
<script src="{_TV_CDN}"></script>
<script>
const bars    = {bars_json};
const levels  = {levels_json};
const markers = {markers_json};

const chart = LightweightCharts.createChart(document.getElementById('tv'), {{
  width:  window.innerWidth,
  height: {height},
  layout: {{
    background: {{type:'solid', color:'#0e0e17'}},
    textColor:  '#c0c0c0',
    fontSize:   11,
  }},
  grid: {{
    vertLines: {{color:'#1e1e2e', style:1}},
    horzLines: {{color:'#1e1e2e', style:1}},
  }},
  timeScale: {{
    timeVisible:    true,
    secondsVisible: false,
    borderColor:    '#2a2a3e',
    fixLeftEdge:    false,
    fixRightEdge:   true,
  }},
  rightPriceScale: {{borderColor:'#2a2a3e'}},
  crosshair: {{mode: LightweightCharts.CrosshairMode.Normal}},
}});

const cs = chart.addCandlestickSeries({{
  upColor:        '#26a69a',
  downColor:      '#ef5350',
  borderUpColor:  '#26a69a',
  borderDownColor:'#ef5350',
  wickUpColor:    '#26a69a',
  wickDownColor:  '#ef5350',
}});
cs.setData(bars);

// Price levels
levels.forEach(function(lvl) {{
  cs.createPriceLine({{
    price:            lvl.price,
    color:            lvl.color,
    lineWidth:        1,
    lineStyle:        2,
    axisLabelVisible: true,
    title:            lvl.label,
  }});
}});

// Trade markers
if (markers.length > 0) cs.setMarkers(markers);

chart.timeScale().scrollToRealTime();

window.addEventListener('resize', function() {{
  chart.applyOptions({{width: window.innerWidth}});
}});
</script>
</body></html>"""

    _stc.html(html, height=height, scrolling=False)


def _account_state() -> dict:
    return _read_json(ROOT / "data" / "account_state.json")


def _is_bot_running_individual(symbol: str) -> bool:
    """Check if an INDIVIDUAL (non-multi) bot process for this symbol is alive."""
    pids = _load_pids()
    pid = pids.get(symbol, 0)
    if pid <= 0:
        return False
    # Skip if this PID belongs to the multi-bot (handled separately)
    if pid == pids.get(_MULTI_BOT_KEY, -1):
        return False
    if _is_process_running(pid):
        return True
    # Process is dead — clean up the PID
    pids.pop(symbol, None)
    _save_pids(pids)
    return False


def _is_bot_running(symbol: str) -> bool:
    """Check if the bot for this symbol is running — individual OR multi-bot."""
    # 1. Check individual bot
    if _is_bot_running_individual(symbol):
        return True
    # 2. Check if multi-bot is running and covers this symbol
    pids = _load_pids()
    multi_pid = pids.get(_MULTI_BOT_KEY, 0)
    if multi_pid > 0 and _is_process_running(multi_pid):
        multi_syms = pids.get(_MULTI_SYMBOLS_KEY, [])
        if symbol in multi_syms:
            return True
    return False


# ---------------------------------------------------------------------------
# Process control
# ---------------------------------------------------------------------------

BOT_LOG_DIR = ROOT / "data" / "run_logs"
BOT_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _bot_log_path(symbol: str) -> Path:
    return BOT_LOG_DIR / f"bot_{symbol}.log"


def start_bot(symbol: str, paper_mode: bool) -> tuple[bool, str]:
    """Launch bot_runner.py as a background subprocess, stdout → log file."""
    if _is_bot_running(symbol):
        return False, f"{symbol} bot is already running."

    python_exe = sys.executable
    cmd = [python_exe, "-u", str(ROOT / "src" / "bot_runner.py"), "--symbol", symbol]
    log_path = _bot_log_path(symbol)

    try:
        kwargs: dict = {"cwd": str(ROOT)}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        # Force UTF-8 encoding for the subprocess
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        kwargs["env"] = env

        # Open log file in binary write mode — avoids text/encoding conflicts on Windows.
        log_f = open(log_path, "wb")

        proc = subprocess.Popen(
            cmd,
            **kwargs,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,   # never inherit Streamlit's stdin
        )
        pids = _load_pids()
        pids[symbol] = proc.pid
        _save_pids(pids)
        return True, f"{symbol} bot started (PID {proc.pid})"
    except Exception as e:
        return False, f"Failed to start {symbol} bot: {e}"


def stop_bot(symbol: str) -> tuple[bool, str]:
    """Kill the running bot process for this symbol."""
    pids = _load_pids()
    pid = pids.get(symbol, 0)

    if pid <= 0 or not _is_process_running(pid):
        pids.pop(symbol, None)
        _save_pids(pids)
        return False, f"{symbol} bot is not running."

    success = _kill_process(pid)
    if success:
        pids.pop(symbol, None)
        _save_pids(pids)
        return True, f"{symbol} bot stopped (PID {pid})"
    return False, f"Could not stop {symbol} bot (PID {pid})"


# Multi-symbol key used in bot_pids.json
_MULTI_BOT_KEY = "__multi__"
_MULTI_SYMBOLS_KEY = "__multi_symbols__"


def _is_multi_bot_running() -> bool:
    """Return True if the multi-symbol bot process is alive."""
    pids = _load_pids()
    pid = pids.get(_MULTI_BOT_KEY, 0)
    if pid > 0 and _is_process_running(pid):
        return True
    # Clean up stale multi-bot PID if process is dead
    if pid > 0:
        pids.pop(_MULTI_BOT_KEY, None)
        pids.pop(_MULTI_SYMBOLS_KEY, None)
        _save_pids(pids)
    return False


def _multi_bot_symbols() -> list[str]:
    """Return the list of symbols the multi-bot is currently trading."""
    pids = _load_pids()
    return pids.get(_MULTI_SYMBOLS_KEY, [])


def start_multi_bot(symbols: list[str], paper_mode: bool) -> tuple[bool, str]:
    """Launch multi_bot_runner.py with all requested symbols in one process."""
    if _is_multi_bot_running():
        return False, "Multi-symbol bot is already running."

    # Refuse if any individual bot is still running (would conflict)
    for sym in symbols:
        if _is_bot_running_individual(sym):
            return False, f"Stop the individual {sym} bot first before starting multi-symbol mode."

    python_exe = sys.executable
    cmd = [
        python_exe, "-u",
        str(ROOT / "src" / "multi_bot_runner.py"),
        "--symbols", *symbols,
    ]
    log_path = BOT_LOG_DIR / "bot_MULTI.log"

    try:
        kwargs: dict = {"cwd": str(ROOT)}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        # Force UTF-8 encoding for the subprocess to avoid emoji/unicode errors
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        kwargs["env"] = env

        log_f = open(log_path, "wb")
        proc = subprocess.Popen(
            cmd,
            **kwargs,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,   # never inherit Streamlit's stdin
        )
        pids = _load_pids()
        pids[_MULTI_BOT_KEY] = proc.pid
        pids[_MULTI_SYMBOLS_KEY] = symbols   # remember which symbols are included
        # Also register individual symbol PIDs so _is_bot_running works immediately
        for sym in symbols:
            pids[sym] = proc.pid
        _save_pids(pids)
        sym_str = " + ".join(symbols)
        return True, f"Multi-symbol bot started for {sym_str} (PID {proc.pid})"
    except Exception as e:
        return False, f"Failed to start multi-symbol bot: {e}"


def stop_multi_bot() -> tuple[bool, str]:
    """Kill the running multi-symbol bot process."""
    pids = _load_pids()
    pid = pids.get(_MULTI_BOT_KEY, 0)

    if pid <= 0 or not _is_process_running(pid):
        # Clean up everything related to multi-bot
        multi_syms = pids.get(_MULTI_SYMBOLS_KEY, [])
        pids.pop(_MULTI_BOT_KEY, None)
        pids.pop(_MULTI_SYMBOLS_KEY, None)
        for sym in multi_syms:
            if pids.get(sym) == pid:
                pids.pop(sym, None)
        _save_pids(pids)
        return False, "Multi-symbol bot is not running."

    success = _kill_process(pid)
    if success:
        multi_syms = pids.get(_MULTI_SYMBOLS_KEY, [])
        pids.pop(_MULTI_BOT_KEY, None)
        pids.pop(_MULTI_SYMBOLS_KEY, None)
        # Clean up individual symbol PIDs that belong to this multi-bot
        for sym in multi_syms:
            if pids.get(sym) == pid:
                pids.pop(sym, None)
        _save_pids(pids)
        return True, f"Multi-symbol bot stopped (PID {pid})"
    return False, f"Could not stop multi-symbol bot (PID {pid})"


def emergency_flatten(symbol: str) -> tuple[bool, str]:
    """
    Write a flatten flag file and stop the bot.
    Bot checks for this file on each loop iteration.
    """
    FLATTEN_FLAG.write_text(
        json.dumps({"symbol": symbol, "requested_at": datetime.now().isoformat()}),
        encoding="utf-8",
    )
    ok, msg = stop_bot(symbol)
    return True, f"FLATTEN requested for {symbol}. {msg}"


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=15)
def load_trades(days: int = 90) -> pd.DataFrame:
    try:
        audit = AuditLogger(str(SETTINGS_PATH))
        rows = audit.get_recent_trades(days=days)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        for col in ("entry_ts", "exit_ts"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=15)
def load_daily(days: int = 90) -> pd.DataFrame:
    try:
        audit = AuditLogger(str(SETTINGS_PATH))
        rows = audit.get_daily_history(days=days)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=10)
def load_signals(limit: int = 150) -> pd.DataFrame:
    try:
        audit = AuditLogger(str(SETTINGS_PATH))
        rows = audit.get_recent_signals(limit=limit)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Live console helpers
# ---------------------------------------------------------------------------

import re as _re

# Strip ANSI escape codes (colour sequences loguru emits on terminals)
_ANSI_RE = _re.compile(r"\x1b\[[0-9;]*[mGKHF]")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


# Log level → (CSS colour, bold)
_LEVEL_STYLES: dict[str, tuple[str, bool]] = {
    "CRITICAL": ("#ff5555", True),
    "ERROR":    ("#f44747", True),
    "WARNING":  ("#f5e642", False),
    "SUCCESS":  ("#26a269", False),
    "INFO":     ("#4ec9b0", False),
    "DEBUG":    ("#858585", False),
    "TRACE":    ("#555555", False),
}


def _colorize_log_html(raw_lines: list[str]) -> str:
    """
    Convert raw loguru log lines to an HTML block with colour-coded levels.
    Returns an HTML string safe for st.markdown(unsafe_allow_html=True).
    """
    html_lines: list[str] = []
    for raw in raw_lines:
        line = _strip_ansi(raw).rstrip()
        if not line:
            html_lines.append("<br/>")
            continue

        # Detect log level in the line (loguru format: "... | LEVEL    | ...")
        colour = "#d4d4d4"
        bold = False
        for level, (col, b) in _LEVEL_STYLES.items():
            if f"| {level}" in line or f"|{level}" in line:
                colour = col
                bold = b
                break

        # Escape HTML special chars
        line_esc = (
            line.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
        )
        weight = "bold" if bold else "normal"
        html_lines.append(
            f'<span style="color:{colour};font-weight:{weight};'
            f'font-family:\'Consolas\',\'Courier New\',monospace;'
            f'font-size:12px;white-space:pre-wrap;">{line_esc}</span>'
        )

    return "<br/>".join(html_lines)


def _read_log_lines(path: Path, last_n: int = 200) -> list[str]:
    """Read the last N lines of a log file (binary-safe, handles ANSI)."""
    if not path.exists():
        return []
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        return lines[-last_n:]
    except Exception:
        return []


def _all_log_sources() -> dict[str, Path]:
    """Return all available log files keyed by display name."""
    sources: dict[str, Path] = {}
    # Bot logs
    for sym in SYMBOLS:
        p = _bot_log_path(sym)
        label = f"Bot {sym}"
        sources[label] = p
    # Task logs
    for key, info in TASKS.items():
        label = info["label"]
        sources[label] = info["log"]
    return sources


def _merge_log_lines(paths: list[Path], last_n: int = 300) -> list[str]:
    """
    Merge multiple log files sorted by timestamp (best-effort).
    Falls back to file order when timestamps are absent.
    """
    all_lines: list[str] = []
    for path in paths:
        lines = _read_log_lines(path, last_n=last_n)
        all_lines.extend(lines)
    # Simple sort by timestamp prefix (loguru: "2026-02-22 10:30:00.123")
    def _ts_key(line: str) -> str:
        m = _re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", _strip_ansi(line))
        return m.group(1) if m else ""
    all_lines.sort(key=_ts_key)
    return all_lines[-last_n:]


# ---------------------------------------------------------------------------
# Password gate — IP-based 24-hour remember-me + session state fallback
# ---------------------------------------------------------------------------

_PANEL_PASSWORD    = "7310512131105"
_AUTH_IPS_FILE     = ROOT / "data" / "auth_ips.json"
_AUTH_EXPIRY_HOURS = 24


def _get_client_ip() -> str:
    """
    Return the connecting client's IP address.

    Streamlit 1.33+ exposes st.context.headers; older builds fall back to a
    best-effort approach.  Returns "unknown" if nothing can be determined.
    """
    try:
        headers = st.context.headers  # type: ignore[attr-defined]
        for hdr in ("X-Forwarded-For", "X-Real-Ip", "CF-Connecting-IP"):
            val = headers.get(hdr, "")
            if val:
                return val.split(",")[0].strip()
        # No proxy header → direct connection (local or same-machine)
        return headers.get("Host", "local").split(":")[0] or "local"
    except Exception:
        return "unknown"


def _load_auth_ips() -> dict:
    if _AUTH_IPS_FILE.exists():
        try:
            return json.loads(_AUTH_IPS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_auth_ips(data: dict) -> None:
    try:
        _AUTH_IPS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def _is_ip_trusted(ip: str) -> bool:
    """Return True if the IP authenticated within the last 24 hours."""
    if ip in ("unknown",):
        return False
    ips = _load_auth_ips()
    entry = ips.get(ip)
    if not entry:
        return False
    try:
        ts = datetime.fromisoformat(entry["ts"])
        return (datetime.now() - ts).total_seconds() < _AUTH_EXPIRY_HOURS * 3600
    except Exception:
        return False


def _trust_ip(ip: str) -> None:
    """Record this IP as authenticated with the current timestamp."""
    if ip in ("unknown",):
        return
    ips = _load_auth_ips()
    # Purge expired entries while we're here
    now = datetime.now()
    ips = {
        k: v for k, v in ips.items()
        if (now - datetime.fromisoformat(v["ts"])).total_seconds() < _AUTH_EXPIRY_HOURS * 3600
    }
    ips[ip] = {"ts": now.isoformat()}
    _save_auth_ips(ips)


def _revoke_ip(ip: str) -> None:
    """Remove this IP from the trusted list (called on Lock Panel)."""
    ips = _load_auth_ips()
    ips.pop(ip, None)
    _save_auth_ips(ips)


# ── Determine auth state ────────────────────────────────────────────────────
_client_ip = _get_client_ip()

# Auto-authenticate if this IP is already trusted (remembered for 24 h)
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = _is_ip_trusted(_client_ip)

if not st.session_state["authenticated"]:
    st.markdown(
        "<div style='display:flex;flex-direction:column;align-items:center;"
        "justify-content:center;min-height:60vh;'>"
        "<div style='background:#161b22;border:1px solid #30363d;border-radius:14px;"
        "padding:48px 56px;max-width:400px;width:100%;text-align:center;"
        "box-shadow:0 8px 32px #0008;'>"
        "<div style='font-size:48px;margin-bottom:12px;'>🔒</div>"
        "<h2 style='color:#e6edf3;margin:0 0 6px;'>ChargedUp Profits Bot</h2>"
        "<p style='color:#8b949e;font-size:0.9em;margin-bottom:28px;'>"
        "Enter your password to continue</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    _pw_col = st.columns([1, 2, 1])[1]
    with _pw_col:
        _pw_input = st.text_input(
            "Password", type="password", key="_pw_field",
            placeholder="Enter password…",
            label_visibility="collapsed",
        )
        _pw_btn = st.button("Unlock →", type="primary", use_container_width=True)

    if _pw_btn or (_pw_input and st.session_state.get("_pw_field") == _PANEL_PASSWORD):
        if _pw_input == _PANEL_PASSWORD:
            st.session_state["authenticated"] = True
            _trust_ip(_client_ip)   # remember this IP for 24 hours
            st.rerun()
        else:
            st.error("❌ Incorrect password. Please try again.")
    st.stop()   # Do not render anything else until authenticated

# ---------------------------------------------------------------------------
# Auto-refresh timer — 5s when research tasks are running, else 15s
# ---------------------------------------------------------------------------

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

# Keep task_pids.json in sync: prune dead entries, discover new processes.
_reconcile_task_pids()

_any_task_running = any(_is_task_running(k) for k in TASKS)
_refresh_interval = 5 if _any_task_running else 15

# NOTE: auto-refresh is now at the BOTTOM of the script (after all
# button actions) so that it never steals a user's button click.

# ---------------------------------------------------------------------------
# 7:45am auto-start scheduler — fires once per trading day at pre-market open
# ---------------------------------------------------------------------------
_sched      = _load_schedule()
_et_now_sched = _now_et()
_sched_active = _sched.get("active")  # e.g. "ES+NQ" or None
_sched_today  = _et_now_sched.strftime("%Y-%m-%d")
_sched_triggered_key = f"_745am_triggered_{_sched_today}"

if (
    _sched_active
    and _sched_active in _SCHEDULE_OPTIONS
    and _et_now_sched.weekday() < 5              # Mon-Fri only
    and _et_now_sched.hour == 7
    and 44 <= _et_now_sched.minute < 50          # 7:44–7:49 window (targets 7:45)
    and _sched.get("last_autostart_date") != _sched_today
    and not st.session_state.get(_sched_triggered_key)   # once per render cycle
):
    _sched_cfg  = _SCHEDULE_OPTIONS[_sched_active]
    _sched_syms = _sched_cfg["symbols"]
    _sched_paper = cfg["execution"].get("paper_mode", True)
    if _sched_cfg["multi"]:
        _s_ok, _s_msg = start_multi_bot(_sched_syms, _sched_paper)
    else:
        _s_ok, _s_msg = start_bot(_sched_syms[0], _sched_paper)
    if _s_ok:
        _sched["last_autostart_date"] = _sched_today
        _save_schedule(_sched)
        st.session_state[_sched_triggered_key] = True
        st.toast(
            f"⏰ 7:45am auto-start fired: {_sched_cfg['label']} bot(s) launched for pre-market!",
            icon="🚀",
        )
        time.sleep(0.4)
        st.rerun()

# ---------------------------------------------------------------------------
# Load data once per render
# ---------------------------------------------------------------------------

cfg = _load_settings()
trades_df = load_trades()
daily_df = load_daily()
signals_df = load_signals()

# ---------------------------------------------------------------------------
# SIDEBAR — Control Panel
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/candle-sticks.png", width=60)
    st.title("ChargedUp Profits Bot")

    # --- Lock button ---
    if st.button("🔒 Lock Panel", key="lock_panel", use_container_width=True):
        st.session_state["authenticated"] = False
        _revoke_ip(_client_ip)   # remove IP trust so the lock screen shows
        st.rerun()

    st.markdown("")

    # --- Mode badge ---
    paper_mode = cfg["execution"].get("paper_mode", True)
    mode_label = "PAPER MODE 🟡" if paper_mode else "⚠️ LIVE MODE 🔴"
    mode_color = "#b5891a" if paper_mode else "#c01c28"
    st.markdown(
        f"<div style='background:{mode_color};padding:8px;border-radius:6px;"
        f"text-align:center;font-weight:bold;color:white'>{mode_label}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # --- Paper / Live toggle ---
    with st.expander("Switch Trading Mode", expanded=False):
        st.warning(
            "⚠️ Switching to LIVE mode will trade on your TopstepX account. "
            "During evaluation or funded stages, breaking loss limits will cost "
            "you the account. Only switch to LIVE after you've verified backtest "
            "and walk-forward results, and you've passed your evaluation."
        )
        new_paper = st.radio(
            "Mode", ["Paper (Simulation)", "Live (Real Money)"],
            index=0 if paper_mode else 1,
            key="mode_radio",
        )
        if st.button("Apply Mode Change", key="apply_mode"):
            cfg["execution"]["paper_mode"] = (new_paper == "Paper (Simulation)")
            _save_settings(cfg)
            st.success("Mode updated. Restart bots to apply.")
            st.cache_data.clear()
            st.rerun()

    st.markdown("---")

    # --- Bot start/stop per symbol ---
    _sidebar_multi = _is_multi_bot_running()
    _sidebar_multi_syms = _multi_bot_symbols() if _sidebar_multi else []

    st.subheader("Bot Control")
    for sym in SYMBOLS:
        running = _is_bot_running(sym)
        in_multi = _sidebar_multi and sym in _sidebar_multi_syms
        state = _bot_state(sym)
        sym_enabled = cfg.get("symbols", {}).get(sym, {}).get("enabled", True)

        col_status, col_btn = st.columns([2, 1])
        with col_status:
            if in_multi:
                status_icon = "🟢 Running (Multi)"
            elif running:
                status_icon = "🟢 Running"
            else:
                status_icon = "⚫ Stopped"
            st.markdown(f"**{sym}** — {status_icon}")
            if state:
                pnl = state.get("today_pnl", 0)
                pnl_str = f"${pnl:+.2f}" if pnl != 0 else "—"
                st.caption(f"P&L: {pnl_str} | Trades: {state.get('today_trades', 0)}")

        with col_btn:
            if in_multi:
                st.caption("via Multi")
            elif running:
                if st.button("Stop", key=f"stop_{sym}", type="secondary"):
                    ok, msg = stop_bot(sym)
                    st.toast(msg, icon="🛑" if ok else "⚠️")
                    time.sleep(0.5)
                    st.rerun()
            else:
                if not sym_enabled:
                    st.caption("Disabled")
                elif _sidebar_multi:
                    st.caption("Multi active")
                elif st.button("Start", key=f"start_{sym}", type="primary"):
                    ok, msg = start_bot(sym, paper_mode)
                    st.toast(msg, icon="🚀" if ok else "⚠️")
                    time.sleep(0.5)
                    st.rerun()

    st.markdown("---")

    # --- Multi-Symbol (single connection) ---
    st.subheader("🔗 Multi-Symbol Mode (One Login)")
    st.caption(
        "Exchange rules allow only **one active session** at a time. "
        "Use this to run two or more symbols on a single connection."
    )

    multi_running = _sidebar_multi
    enabled_syms = [s for s in SYMBOLS if cfg.get("symbols", {}).get(s, {}).get("enabled", True)]
    any_individual_running = any(_is_bot_running_individual(s) for s in SYMBOLS)

    multi_col_sel, multi_col_btn = st.columns([3, 1])
    with multi_col_sel:
        if multi_running:
            _ms = " + ".join(_sidebar_multi_syms) if _sidebar_multi_syms else "?"
            st.markdown(f"**🟢 Multi-symbol bot is RUNNING ({_ms})**")
        else:
            multi_sym_sel = st.multiselect(
                "Symbols to trade together",
                options=SYMBOLS,
                default=enabled_syms[:2],
                key="multi_sym_select",
            )
    with multi_col_btn:
        if multi_running:
            if st.button("Stop All", key="stop_multi", type="secondary"):
                ok, msg = stop_multi_bot()
                st.toast(msg, icon="🛑" if ok else "⚠️")
                time.sleep(0.5)
                st.rerun()
        else:
            # Use the widget return value directly — avoids any session-state
            # timing edge-case where the key hasn't been written yet.
            sel = list(multi_sym_sel)
            if any_individual_running:
                st.warning("Stop individual bots first")
            elif sel and st.button("Start Together", key="start_multi", type="primary"):
                ok, msg = start_multi_bot(sel, paper_mode)
                st.toast(msg, icon="🚀" if ok else "⚠️")
                time.sleep(0.5)
                st.rerun()

    st.markdown("---")

    # --- Emergency Flatten ---
    st.subheader("Emergency Controls")
    for sym in SYMBOLS:
        if _is_bot_running(sym):
            if st.button(
                f"⚡ FLATTEN ALL {sym} POSITIONS",
                key=f"flatten_{sym}",
                type="secondary",
            ):
                ok, msg = emergency_flatten(sym)
                st.error(msg)
                time.sleep(1)
                st.rerun()

    st.markdown("---")

    # --- Refresh controls ---
    _ri = 5 if any(_is_task_running(k) for k in TASKS) else 15
    st.caption(f"Auto-refreshes every {_ri}s {'⚡ (task running)' if _ri == 5 else ''}")
    if st.button("Force Refresh Now"):
        st.cache_data.clear()
        st.session_state.last_refresh = time.time()
        st.rerun()

    next_refresh = max(0, int(_ri - (time.time() - st.session_state.last_refresh)))
    st.caption(f"Next auto-refresh in {next_refresh}s")


# ---------------------------------------------------------------------------
# MAIN AREA — Tabs
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .big-metric { font-size: 2rem; font-weight: bold; }
    .green-text { color: #00c853; }
    .red-text { color: #f44336; }
    .meter-container { background: #1a1a1a; border-radius: 8px; padding: 12px; }

    /* ── Global dark trading theme ────────────────────────────────────── */
    .stApp, .stApp > div { background-color: #0d0d0d !important; color: #e8e8e8 !important; }
    [data-testid="stSidebar"] { background-color: #111111 !important; border-right: 1px solid #262626; }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    [data-testid="stMetric"] { background: #1a1a1a !important; border: 1px solid #262626 !important;
                                border-radius: 8px !important; padding: 14px !important; }
    [data-testid="stMetricLabel"] { color: #9a9a9a !important; font-size: 0.78rem !important; }
    [data-testid="stMetricValue"] { color: #e8e8e8 !important; }
    .stTabs [data-baseweb="tab"] { background: #141414; border-bottom: 2px solid transparent;
                                    color: #6b6b6b !important; font-size: 0.85rem; }
    .stTabs [aria-selected="true"] { border-bottom: 2px solid #1e88e5 !important; color: #e8e8e8 !important; }
    .stButton > button { background: #1a1a1a !important; border: 1px solid #262626 !important;
                         color: #e8e8e8 !important; border-radius: 6px !important; }
    .stButton > button:hover { background: #252525 !important; border-color: #3a3a3a !important; }
    button[kind="primary"], .stButton > button[data-testid="baseButton-primary"] {
        background: #1e88e5 !important; border: none !important; color: white !important; }
    button[kind="secondary"] { background: #c62828 !important; border: none !important; color: white !important; }
    .stTextInput input, .stSelectbox select, .stNumberInput input {
        background: #1a1a1a !important; color: #e8e8e8 !important; border: 1px solid #262626 !important; }
    .stSlider [data-baseweb="slider"] { background: transparent; }
    .stDataFrame, [data-testid="stDataFrame"] { border: 1px solid #262626 !important; }
    .stDataFrame td, .stDataFrame th { background: #111111 !important; color: #d0d0d0 !important;
                                        border-color: #262626 !important; }
    hr { border-color: #262626 !important; }
    .streamlit-expanderHeader { background: #1a1a1a !important; border: 1px solid #262626 !important;
                                  color: #e0e0e0 !important; }
    .streamlit-expanderContent { background: #111111 !important; border: 1px solid #262626 !important; }
    h1, h2, h3, h4, h5, h6 { color: #e8e8e8 !important; }
    p, li, span, label { color: #d0d0d0; }
    .profit-value { color: #00c853 !important; font-weight: 700; }
    .loss-value   { color: #f44336 !important; font-weight: 700; }
    /* Status bar */
    .status-bar { background: #111111; border: 1px solid #262626; border-radius: 8px;
                  padding: 12px 16px; margin-bottom: 16px; display: flex; align-items: center; gap: 24px; }
    .status-running { color: #00c853; font-weight: 700; font-size: 1.1rem; }
    .status-stopped { color: #6b6b6b; font-weight: 700; font-size: 1.1rem; }
    .status-paper   { color: #1e88e5; font-weight: 700; font-size: 1.1rem; }
    /* Approval panel cards */
    .approval-card { background: #1a1a1a; border: 1px solid #1e88e5; border-radius: 10px;
                     padding: 16px; margin-bottom: 16px; }
    .gate-pass { color: #00c853; }
    .gate-fail { color: #f44336; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Tab navigation — persisted across refreshes via st.query_params
_TAB_OPTIONS = [
    "📊 Status",
    "🏦 Account",
    "📈 How Am I Doing?",
    "📋 My Trades",
    "🔔 Today's Trading",
    "⚙️ Settings",
    "🔬 Test & Train AI",
    "🔄 Auto-Run",
    "🖥️ Technical Log",
]

# Mapping from old tab names to new (for any remaining references)
_TAB_ALIAS = {
    "📊 Live":         "📊 Status",
    "📈 Performance":  "📈 How Am I Doing?",
    "📋 Trades":       "📋 My Trades",
    "🔔 Signals":      "🔔 Today's Trading",
    "🔬 Research":     "🔬 Test & Train AI",
    "🔄 Automation":   "🔄 Auto-Run",
    "🖥️ Console":     "🖥️ Technical Log",
}

# Read saved tab from URL, fall back to first tab
_qp = st.query_params
_saved_tab = _qp.get("tab", _TAB_OPTIONS[0])
if _saved_tab not in _TAB_OPTIONS:
    _saved_tab = _TAB_OPTIONS[0]

# Initialise session state once so the default sticks on first load
if "_page_nav" not in st.session_state:
    st.session_state["_page_nav"] = _saved_tab

_active_tab = st.pills(
    "Navigate",
    _TAB_OPTIONS,
    key="_page_nav",
    label_visibility="collapsed",
)

# Persist current selection to URL so a browser refresh returns here
if _active_tab and _active_tab != _qp.get("tab"):
    st.query_params["tab"] = _active_tab

# ---------------------------------------------------------------------------
# Persistent status bar — shown on every page in plain English
# ---------------------------------------------------------------------------
_sb_running_syms = [s for s in SYMBOLS if _is_bot_running(s)]
_sb_multi = _is_multi_bot_running()
_sb_paper = cfg["execution"].get("paper_mode", True)

# Aggregate today's P&L and trades across all running bots
_sb_pnl = 0.0
_sb_trades = 0
_sb_max_trades_per_sym = cfg.get("risk", {}).get("max_trades_per_day", 7)
_sb_daily_loss_limit = cfg.get("risk", {}).get("max_daily_loss_dollars", -2000)
_sb_kill = False
_sb_active_syms = 0
for _s in SYMBOLS:
    _st = _bot_state(_s)
    if _st:
        _sb_active_syms += 1
        _sb_pnl += _st.get("today_pnl", 0.0)
        _sb_trades += _st.get("today_trades", 0)
        if _st.get("kill_switch_active"):
            _sb_kill = True
# Total allowed = per-symbol limit × number of active symbols (min 1 so display is always sane)
_sb_max_trades = _sb_max_trades_per_sym * max(1, _sb_active_syms)

if _sb_running_syms or _sb_multi:
    _status_label = "● RUNNING"
    _status_class = "status-running"
    _status_syms  = " + ".join(_sb_running_syms) if _sb_running_syms else "Multi"
elif _sb_paper:
    _status_label = "◯ PRACTICE MODE"
    _status_class = "status-paper"
    _status_syms  = "Simulation Only"
else:
    _status_label = "◯ STOPPED"
    _status_class = "status-stopped"
    _status_syms  = ""

_pnl_class = "profit-value" if _sb_pnl >= 0 else "loss-value"
_pnl_sign  = "+" if _sb_pnl > 0 else ""

_remaining_loss = _sb_daily_loss_limit - _sb_pnl
_safety_msg = (
    f"<span style='color:#f44336'>⚠️ Daily loss limit reached</span>"
    if _sb_kill else
    f"Safe — you can lose <b>${abs(_remaining_loss):,.0f}</b> more today"
)

_sep = "<span style='color:#9a9a9a; margin:0 8px;'>|</span>"
_sb_syms_html = (
    f"{_sep}<span style='color:#6b6b6b'>{_status_syms}</span>"
    if _status_syms else ""
)
_sb_paper_html = (
    f"{_sep}<span style='color:#1e88e5'>PRACTICE — no real money</span>"
    if _sb_paper else ""
)

# Check for pending AI models — show a badge in the status bar so it's visible
# from any tab, not just "Test & Train AI".
try:
    from src.model_lifecycle import ModelLifecycle as _MLsb
    _sb_pending_count = len(_MLsb().get_pending_models())
except Exception:
    _sb_pending_count = 0
_sb_ai_html = (
    f"{_sep}<a href='#' style='color:#ff9800;font-weight:700;text-decoration:none;' "
    f"title='Go to Test & Train AI tab to review'>🤖 {_sb_pending_count} AI model{'s' if _sb_pending_count != 1 else ''} ready for your approval</a>"
    if _sb_pending_count > 0 else ""
)

_status_bar_html = (
    f"<div class='status-bar'>"
    f"<span class='{_status_class}'>{_status_label}</span>"
    f"{_sb_syms_html}"
    f"{_sep}<span>Today's P&L: <b class='{_pnl_class}'>{_pnl_sign}${_sb_pnl:,.2f}</b></span>"
    f"{_sep}<span>Account safety: {_safety_msg}</span>"
    f"{_sep}<span>Trades today: <b>{_sb_trades}</b> of <b>{_sb_max_trades}</b> allowed "
    f"(<b>{_sb_max_trades_per_sym}</b> per symbol)</span>"
    f"{_sb_paper_html}"
    f"{_sb_ai_html}"
    f"</div>"
)
st.markdown(_status_bar_html, unsafe_allow_html=True)

# ===========================================================================
# TAB 1 — Status (was: Live)
# ===========================================================================

if _active_tab == "📊 Status":
    st.subheader("Today's Live Status")

    today_str = str(date.today())

    # Show a prominent banner if multi-bot mode is active
    _multi_live = _is_multi_bot_running()
    _multi_syms = _multi_bot_symbols() if _multi_live else []
    if _multi_live:
        st.success(
            f"🔗 **Multi-Symbol Mode ACTIVE** — Trading "
            f"**{' + '.join(_multi_syms)}** on a single connection"
        )

    # Pre-market assessment cards (shown when bot has run pre-market analysis)
    _pm_data_any = False
    for sym in SYMBOLS:
        _pm_file = ROOT / "data" / f"premarket_{sym}.json"
        if _pm_file.exists():
            try:
                _pm = json.loads(_pm_file.read_text(encoding="utf-8"))
                if _pm.get("date") == today_str:
                    _pm_data_any = True
                    break
            except Exception:
                pass

    if _pm_data_any:
        st.markdown("#### 🌅 Pre-Market Assessment")
        _pm_cols = st.columns(len(SYMBOLS))
        for _pci, sym in enumerate(SYMBOLS):
            _pm_file = ROOT / "data" / f"premarket_{sym}.json"
            with _pm_cols[_pci]:
                if not _pm_file.exists():
                    continue
                try:
                    _pm = json.loads(_pm_file.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if _pm.get("date") != today_str:
                    continue

                _outlook = _pm.get("session_outlook", "UNKNOWN")
                _outlook_color = {"TRADE": "#26a69a", "CAUTION": "#ff9800", "NO_TRADE": "#ef5350"}.get(_outlook, "#9a9a9a")
                _outlook_icon = {"TRADE": "🟢", "CAUTION": "🟡", "NO_TRADE": "🔴"}.get(_outlook, "⚪")

                st.markdown(
                    f"<div style='border:1px solid {_outlook_color};border-radius:8px;"
                    f"padding:10px;background:#1a1a2e'>"
                    f"<b style='color:{_outlook_color}'>{_outlook_icon} {sym} — {_outlook}</b><br/>"
                    f"<span style='color:#9a9a9a;font-size:0.8em'>"
                    f"Vol: {_pm.get('vol_regime','?')} | "
                    f"Gap: {_pm.get('gap_direction','?')} ({_pm.get('gap_size_atr',0):.1f}× ATR) | "
                    f"Levels: {_pm.get('level_count',0)} ({_pm.get('level_quality','?')})"
                    f"</span></div>",
                    unsafe_allow_html=True,
                )
                _notes = _pm.get("session_notes", [])
                if _notes:
                    with st.expander("Details", expanded=False):
                        for _n in _notes:
                            st.caption(_n)
        st.markdown("---")

    # Status cards per symbol
    for sym in SYMBOLS:
        state = _bot_state(sym)
        running = _is_bot_running(sym)

        with st.container():
            st.markdown(f"#### {sym}")
            if not state:
                if running:
                    st.info(
                        f"🟢 {sym} bot is running but hasn't produced data yet. "
                        f"It may still be connecting..."
                    )
                else:
                    st.info(f"No {sym} state yet. Start the bot to see live data.")
                continue

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            pnl = state.get("today_pnl", 0)
            pnl_delta = f"${pnl:+.2f}" if pnl != 0 else None
            pnl_color = "normal" if pnl >= 0 else "inverse"

            c1.metric("Today P&L", f"${pnl:,.2f}", delta=pnl_delta, delta_color=pnl_color)
            c2.metric("Trades Today", state.get("today_trades", 0))
            c3.metric(
                "Win Rate",
                f"{state.get('today_win_rate', 0) * 100:.0f}%"
                if state.get("today_trades", 0) > 0 else "—"
            )
            c4.metric("W / L", f"{state.get('today_wins', 0)} / {state.get('today_losses', 0)}")
            c5.metric("Position", "OPEN 📍" if state.get("open_position") else "Flat")
            ks_active = state.get("kill_switch_active", False)
            c6.metric("Kill Switch", "ACTIVE ⛔" if ks_active else "OFF ✅")

            if ks_active:
                st.warning(f"Kill switch triggered: {state.get('kill_switch_reason', 'unknown')}")

            mode_txt = "PAPER" if state.get("paper_mode", True) else "LIVE"
            last_upd = state.get("last_update", "")
            if last_upd:
                try:
                    dt = datetime.fromisoformat(last_upd)
                    last_upd = _fmt_time_et(dt)
                except Exception:
                    pass
            run_label = "🟢 Running" if running else "⚫ Stopped"
            if running and _multi_live and sym in _multi_syms:
                run_label = "🟢 Running (Multi-Symbol)"
            status_txt = f"{run_label} | Mode: {mode_txt}"
            if last_upd:
                status_txt += f" | Last update: {last_upd}"
            st.caption(status_txt)
            st.markdown("---")

    # ── TradingView Live Chart ───────────────────────────────────────────────
    st.markdown("#### 📊 Live Chart")
    _chart_any = any(
        (ROOT / "data" / f"bars_{s}.json").exists() for s in SYMBOLS
    )
    if not _chart_any:
        st.info(
            "Start the bot and the live TradingView chart will appear here automatically. "
            "Chart updates every ~3 minutes during market hours."
        )
    else:
        _chart_c1, _chart_c2, _chart_c3 = st.columns([2, 2, 2])
        with _chart_c1:
            _available_syms = [s for s in SYMBOLS if (ROOT / "data" / f"bars_{s}.json").exists()]
            _chart_sym = st.selectbox(
                "Symbol", _available_syms,
                key="tv_sym",
                label_visibility="collapsed",
            )
        with _chart_c2:
            _chart_tf = st.radio(
                "Timeframe", ["1m", "15m"],
                horizontal=True,
                key="tv_tf",
                label_visibility="collapsed",
            )
        with _chart_c3:
            _chart_h = st.select_slider(
                "Height", [340, 440, 540, 640],
                value=440,
                key="tv_height",
                label_visibility="collapsed",
            )
        _render_tv_chart(_chart_sym, trades_df, timeframe=_chart_tf, height=_chart_h)

    st.markdown("---")

    # Combined today's stats from daily log
    if not daily_df.empty:
        today_row = daily_df[daily_df["date"].dt.strftime("%Y-%m-%d") == today_str]
        if not today_row.empty:
            r = today_row.iloc[0]
            st.markdown("#### All Symbols — Combined Today")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Combined P&L", f"${r['pnl_dollars']:+,.2f}")
            m2.metric("Total Trades", int(r["total_trades"]))
            m3.metric("Win Rate", f"{r['win_rate'] * 100:.1f}%")
            m4.metric("Kill Switch", "ACTIVE ⛔" if r["kill_switch_triggered"] else "OFF ✅")


# ===========================================================================
# TAB 2 — Account Health
# ===========================================================================

if _active_tab == "🏦 Account":
    st.subheader("Funded Account Health Monitor")
    st.caption(
        "This panel shows your TopstepX funded account's safety status. "
        "The bot automatically stops trading if you get within $500 of a limit."
    )

    acct = _account_state()

    if not acct:
        # Show config-based limits when no live data yet
        funded_cfg = cfg.get("funded_account", {})
        acct_size = funded_cfg.get("account_size", 50_000)
        st.info(
            f"Account state will appear here once the bot is running. "
            f"Configured account size: ${acct_size:,}"
        )
        # Show the limits from config
        limits_map = {
            25_000:  (1_500, 1_500),
            50_000:  (2_000, 3_000),
            75_000:  (2_750, 3_750),
            100_000: (3_500, 4_500),
        }
        dl, dd = limits_map.get(acct_size, (2_000, 3_000))
        st.markdown(f"""
        **Your Account Limits (${acct_size:,} account):**
        | Limit | Amount |
        |---|---|
        | Daily Loss Limit | ${dl:,} |
        | Trailing Drawdown | ${dd:,} |
        | Safety Buffer | $500 |
        """)
    else:
        # --- Balance row ---
        c1, c2, c3, c4 = st.columns(4)
        balance = acct.get("current_balance", acct.get("account_size", 0))
        daily_pnl = acct.get("daily_pnl", 0)
        c1.metric("Account Balance", f"${balance:,.2f}")
        c2.metric(
            "Today's P&L",
            f"${daily_pnl:+,.2f}",
            delta_color="normal" if daily_pnl >= 0 else "inverse",
        )
        c3.metric("Daily Loss Remaining", f"${acct.get('daily_loss_remaining', 0):,.2f}")
        c4.metric("Drawdown Remaining", f"${acct.get('drawdown_remaining', 0):,.2f}")

        st.markdown("")

        # --- Daily loss meter ---
        pct_daily = acct.get("pct_daily_limit_used", 0)
        dl_color = "#26a269" if pct_daily < 50 else ("#e5a50a" if pct_daily < 80 else "#c01c28")
        st.markdown("**Daily Loss Limit Usage**")
        st.progress(min(pct_daily / 100, 1.0))
        st.caption(
            f"{pct_daily:.1f}% used — "
            f"${acct.get('daily_loss_remaining', 0):,.0f} remaining of "
            f"${acct.get('daily_loss_limit', 0):,.0f} limit"
        )

        st.markdown("")

        # --- Trailing drawdown meter ---
        pct_dd = acct.get("pct_drawdown_used", 0)
        st.markdown("**Trailing Drawdown Usage**")
        st.progress(min(pct_dd / 100, 1.0))
        st.caption(
            f"{pct_dd:.1f}% used — "
            f"${acct.get('drawdown_remaining', 0):,.0f} remaining of "
            f"${acct.get('trailing_drawdown_limit', 0):,.0f} limit"
        )

        st.markdown("")

        # --- Safety status ---
        safe = acct.get("is_safe_to_trade", True)
        unsafe_reason = acct.get("unsafe_reason", "")
        if safe:
            st.success("✅ Account is safe to trade. All limits have adequate buffer.")
        else:
            st.error(f"⛔ TRADING HALTED — {unsafe_reason}")

        # --- Contract sizing ---
        contracts = acct.get("suggested_contracts", 1)
        st.info(
            f"💼 Recommended contract size: **{contracts} contract(s)**\n\n"
            f"This scales automatically as your account grows profit above the starting balance."
        )

        last_upd = acct.get("last_updated", "")
        if last_upd:
            try:
                dt = datetime.fromisoformat(last_upd)
                last_upd = f"{_fmt_time_et(dt)}  {dt.strftime('%Y-%m-%d')}"
            except Exception:
                pass
        st.caption(
            f"Account: {acct.get('account_name', 'N/A')} | "
            f"Mode: {'Paper' if acct.get('paper_mode', True) else 'LIVE'} | "
            f"Last updated: {last_upd}"
        )

        if acct.get("error"):
            st.warning(f"API error: {acct['error']}")

    # --- TopstepX / ProjectX Funded Account Path Explainer ---
    with st.expander("📖 How TopstepX (ProjectX) Works — The Full Path to Real-Money Payouts"):
        st.markdown("""
**The TopstepX path has three stages before you can withdraw real money:**

---

**Stage 1 — Evaluation (Trading Combine)**
- You buy an evaluation account (e.g. $50k, $100k).
- You must hit a **profit target** while staying within loss limits.
- There is **no real money** at this stage — it's a test.
- If you break the rules, you lose the account and must buy a new one.

---

**Stage 2 — Funded Account (Express Funded / Pro Account)**
- Once you pass the evaluation, TopstepX gives you a funded account.
- You're now trading with **TopstepX's capital**, not your own.
- You must follow the **Daily Loss Limit** and **Trailing Drawdown** rules (shown above).
- Breaking either rule ends your funded account.
- **You still cannot withdraw money yet** — you must first build a profit buffer.

---

**Stage 3 — Payouts**
- After meeting the payout requirements (varies by account size, usually a minimum profit + a certain number of trading days), you can request a **withdrawal**.
- TopstepX takes a **profit split** (typically 80/20 or 90/10 depending on your plan — you keep the larger share).
- Payouts are processed via bank transfer or other methods.

---

**Bot Safety Rules (what this panel manages):**

| Rule | What It Does |
|---|---|
| **Daily Loss Limit** | Max you can lose in one day. Bot stops before you hit it. |
| **Trailing Drawdown** | Max total loss from your account's highest point. Breaching this ends the account. |
| **Safety Buffer ($500)** | Bot stops $500 *before* hitting either limit — a cushion for fast-moving markets. |
| **Contract Scaling** | Bot starts with 1 contract and adds more as profits grow. Conservative and safe. |

**Bottom line:** You don't earn real money until you pass the evaluation AND meet payout
requirements on the funded account. This bot is designed to protect the account at every stage
so you never break the rules and lose access.
        """)


# ---------------------------------------------------------------------------
# Backtest / WFA CSV helpers (needed by Performance, Trade Log, Research tabs)
# ---------------------------------------------------------------------------

def _latest_bt_csv(sym: str) -> "pd.DataFrame":
    """Most recently modified backtest_{SYM}_*.csv for the analytics panel."""
    files = sorted(
        (ROOT / "data").glob(f"backtest_{sym}_*.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    try:
        df = pd.read_csv(files[-1]) if files else pd.DataFrame()
        if not df.empty and "symbol" not in df.columns:
            df["symbol"] = sym
        return df
    except Exception:
        return pd.DataFrame()


def _latest_wfa_csv(sym: str) -> "pd.DataFrame":
    p = ROOT / "data" / f"wfa_trades_{sym}.csv"
    try:
        df = pd.read_csv(p) if p.exists() else pd.DataFrame()
        if not df.empty and "symbol" not in df.columns:
            df["symbol"] = sym
        return df
    except Exception:
        return pd.DataFrame()


# ===========================================================================
# TAB 3 — Performance Charts
# ===========================================================================

if _active_tab == "📈 How Am I Doing?":
    st.subheader("Performance Analysis")

    # --- Selectors row -------------------------------------------------------
    _perf_sel1, _perf_sel2 = st.columns([2, 1])
    with _perf_sel1:
        _perf_source_options = ["Backtest Results", "Walk-Forward Results", "Live Trades (Audit DB)"]
        _perf_source = st.radio(
            "Data Source",
            _perf_source_options,
            index=0,
            key="perf_data_source",
            horizontal=True,
        )
    with _perf_sel2:
        _perf_sym_filter = st.selectbox(
            "Symbol",
            ["All"] + list(SYMBOLS),
            key="perf_sym_filter",
        )

    if _perf_source == "Live Trades (Audit DB)":
        _perf_df = trades_df.copy() if not trades_df.empty else pd.DataFrame()
        _perf_pnl_col = "pnl_dollars"
        _perf_ts_col = "entry_ts"
    elif _perf_source == "Walk-Forward Results":
        _perf_parts = [_latest_wfa_csv(s) for s in SYMBOLS]
        _perf_df = pd.concat([p for p in _perf_parts if not p.empty], ignore_index=True) if any(not p.empty for p in _perf_parts) else pd.DataFrame()
        _perf_pnl_col = "pnl_net" if "pnl_net" in _perf_df.columns else "pnl_dollars" if "pnl_dollars" in _perf_df.columns else None
        _perf_ts_col = "entry_ts" if "entry_ts" in _perf_df.columns else "date"
    else:
        _perf_parts = [_latest_bt_csv(s) for s in SYMBOLS]
        _perf_df = pd.concat([p for p in _perf_parts if not p.empty], ignore_index=True) if any(not p.empty for p in _perf_parts) else pd.DataFrame()
        _perf_pnl_col = "pnl_net" if "pnl_net" in _perf_df.columns else "pnl_dollars" if "pnl_dollars" in _perf_df.columns else None
        _perf_ts_col = "entry_ts" if "entry_ts" in _perf_df.columns else "date"

    if not _perf_df.empty and _perf_ts_col in _perf_df.columns:
        _perf_df[_perf_ts_col] = pd.to_datetime(_perf_df[_perf_ts_col], errors="coerce", utc=True)

    # Apply symbol filter
    if _perf_sym_filter != "All" and "symbol" in _perf_df.columns:
        _perf_df = _perf_df[_perf_df["symbol"] == _perf_sym_filter]

    if _perf_df.empty or _perf_pnl_col is None or _perf_pnl_col not in _perf_df.columns:
        st.info(
            "No data for this source yet. "
            "Run a backtest from the **Research** tab, or start the live bot."
        )
    else:
        _pc = _perf_pnl_col  # shorthand
        _syms_in_data = sorted(_perf_df["symbol"].unique().tolist()) if "symbol" in _perf_df.columns else []

        # --- Per-symbol summary table (shown when viewing All symbols) -------
        if _perf_sym_filter == "All" and len(_syms_in_data) > 1:
            st.markdown("#### Results by Symbol")
            _sym_rows = []
            for _s in _syms_in_data:
                _sd = _perf_df[_perf_df["symbol"] == _s]
                _s_wins = (_sd[_pc] > 0).sum()
                _s_losses = (_sd[_pc] <= 0).sum()
                _s_pf_w = _sd[_sd[_pc] > 0][_pc].sum()
                _s_pf_l = abs(_sd[_sd[_pc] < 0][_pc].sum())
                _sym_rows.append({
                    "Symbol": _s,
                    "Trades": len(_sd),
                    "Wins": int(_s_wins),
                    "Losses": int(_s_losses),
                    "Win Rate": f"{(_s_wins / len(_sd) * 100):.1f}%" if len(_sd) else "—",
                    "Net P&L": f"${_sd[_pc].sum():+,.2f}",
                    "Avg P&L": f"${_sd[_pc].mean():+.2f}" if len(_sd) else "—",
                    "Profit Factor": f"{round(_s_pf_w / _s_pf_l, 2):.2f}" if _s_pf_l > 0 else "—",
                })
            st.dataframe(pd.DataFrame(_sym_rows), height=min(150 + len(_sym_rows) * 35, 280))
            st.markdown("---")

        # All-time summary metrics
        total_net = _perf_df[_pc].sum()
        win_rate = (_perf_df[_pc] > 0).mean() * 100
        pf_wins = _perf_df[_perf_df[_pc] > 0][_pc].sum()
        pf_losses = abs(_perf_df[_perf_df[_pc] < 0][_pc].sum())
        profit_factor = round(pf_wins / pf_losses, 2) if pf_losses > 0 else 0.0
        avg_win = _perf_df[_perf_df[_pc] > 0][_pc].mean()
        avg_loss = _perf_df[_perf_df[_pc] < 0][_pc].mean()
        if _perf_ts_col in _perf_df.columns:
            equity = _perf_df.sort_values(_perf_ts_col)[_pc].cumsum()
        else:
            equity = _perf_df[_pc].cumsum()
        max_dd = (equity - equity.cummax()).min() if not equity.empty else 0

        _perf_label = _perf_sym_filter if _perf_sym_filter != "All" else "All Symbols"
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric(f"Total Net P&L ({_perf_label})", f"${total_net:+,.2f}")
        m2.metric("Win Rate", f"{win_rate:.1f}%")
        m3.metric("Profit Factor", profit_factor)
        m4.metric("Avg Win", f"${avg_win:.2f}" if not pd.isna(avg_win) else "—")
        m5.metric("Avg Loss", f"${avg_loss:.2f}" if not pd.isna(avg_loss) else "—")
        m6.metric("Max Drawdown", f"${max_dd:,.2f}")

        _expectancy_per_trade = (total_net / len(_perf_df)) if len(_perf_df) > 0 else 0.0
        st.caption(
            f"Showing **{len(_perf_df):,} trades** from **{_perf_source}** — "
            f"Symbol: **{_perf_label}** — Expectancy: **${_expectancy_per_trade:+.2f}/trade**"
        )
        st.markdown("---")

        # ── Live vs Backtest Expectancy Gap ───────────────────────────────────
        # Always shown; compares live audit trades against latest backtest data.
        _live_trades_for_gap = trades_df.copy() if not trades_df.empty else pd.DataFrame()
        _bt_gap_parts = [_latest_bt_csv(s) for s in SYMBOLS]
        _bt_gap_df = pd.concat([p for p in _bt_gap_parts if not p.empty], ignore_index=True) if any(not p.empty for p in _bt_gap_parts) else pd.DataFrame()

        if not _live_trades_for_gap.empty and not _bt_gap_df.empty:
            _live_exp = _live_trades_for_gap["pnl_dollars"].mean() if "pnl_dollars" in _live_trades_for_gap.columns else 0.0
            _bt_pnl_col = "pnl_net" if "pnl_net" in _bt_gap_df.columns else "pnl_dollars"
            _bt_exp = _bt_gap_df[_bt_pnl_col].mean() if _bt_pnl_col in _bt_gap_df.columns else 0.0
            _live_wr = (_live_trades_for_gap["pnl_dollars"] > 0).mean() * 100 if "pnl_dollars" in _live_trades_for_gap.columns else 0.0
            _bt_wr = (_bt_gap_df[_bt_pnl_col] > 0).mean() * 100 if _bt_pnl_col in _bt_gap_df.columns else 0.0
            _exp_gap = _live_exp - _bt_exp
            _wr_gap = _live_wr - _bt_wr

            _gap_color = "#26a69a" if _exp_gap >= 0 else "#ef5350"
            _gap_icon = "↑" if _exp_gap >= 0 else "↓"
            _wr_color = "#26a69a" if _wr_gap >= 0 else "#ef5350"

            st.markdown("#### Strategy Health: Live vs Backtest")
            _g1, _g2, _g3, _g4, _g5 = st.columns(5)
            _g1.metric("Backtest Expectancy", f"${_bt_exp:+.2f}/trade", help="Average P&L per trade from latest backtest CSV")
            _g2.metric("Live Expectancy", f"${_live_exp:+.2f}/trade",
                       delta=f"{_gap_icon} ${abs(_exp_gap):.2f} vs backtest",
                       delta_color="normal" if _exp_gap >= 0 else "inverse",
                       help="Average P&L per trade from live audit log")
            _g3.metric("Backtest Win Rate", f"{_bt_wr:.1f}%")
            _g4.metric("Live Win Rate", f"{_live_wr:.1f}%",
                       delta=f"{_wr_gap:+.1f}pp vs backtest",
                       delta_color="normal" if _wr_gap >= 0 else "inverse")
            _g5.metric("Live Trade Count", f"{len(_live_trades_for_gap)}")

            _drift_pct = abs(_exp_gap) / abs(_bt_exp) * 100 if _bt_exp != 0 else 0
            if _drift_pct > 30:
                st.warning(
                    f"⚠️ **Model drift detected:** live expectancy is "
                    f"{_drift_pct:.0f}% below backtest. Consider re-running the AI retrain."
                )
            elif _drift_pct > 15:
                st.info(
                    f"📉 Live expectancy is {_drift_pct:.0f}% off backtest — monitor closely."
                )
            else:
                st.success(f"✅ Live performance is tracking backtest within {_drift_pct:.0f}% — no drift detected.")
            st.markdown("---")

        # Equity curve — one line per symbol when viewing All, single line when filtered
        st.markdown("#### Equity Curve (Cumulative P&L)")
        if _perf_ts_col in _perf_df.columns:
            _SYM_LINE_COLORS = {"ES": "#4a9eff", "NQ": "#f5a623", "MNQ": "#a78bfa"}
            fig_eq = go.Figure()
            if _perf_sym_filter == "All" and len(_syms_in_data) > 1:
                for _es in _syms_in_data:
                    _eq_s = _perf_df[_perf_df["symbol"] == _es].sort_values(_perf_ts_col).copy()
                    _eq_s["cumulative_pnl"] = _eq_s[_pc].cumsum()
                    _line_col = _SYM_LINE_COLORS.get(_es, "#4a9eff")
                    fig_eq.add_trace(go.Scatter(
                        x=_eq_s[_perf_ts_col], y=_eq_s["cumulative_pnl"],
                        mode="lines", name=_es,
                        line=dict(width=2, color=_line_col),
                    ))
            else:
                eq_df = _perf_df.sort_values(_perf_ts_col).copy()
                eq_df["cumulative_pnl"] = eq_df[_pc].cumsum()
                _line_col = _SYM_LINE_COLORS.get(_perf_sym_filter, "#4a9eff")
                fig_eq.add_trace(go.Scatter(
                    x=eq_df[_perf_ts_col], y=eq_df["cumulative_pnl"],
                    mode="lines", name=_perf_sym_filter,
                    line=dict(width=2, color=_line_col),
                    fill="tozeroy", fillcolor=f"{_line_col}26",
                ))
            fig_eq.update_layout(
                xaxis_title="Date", yaxis_title="P&L ($)",
                height=280, margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e",
                font=dict(color="#e0e0e0"),
                legend=dict(bgcolor="#1e1e1e", bordercolor="#333"),
            )
            st.plotly_chart(fig_eq)

        col_left, col_right = st.columns(2)

        # Per-symbol P&L bar chart
        with col_left:
            if _perf_sym_filter == "All" and len(_syms_in_data) > 1:
                st.markdown("#### P&L by Symbol")
                _sym_bar = _perf_df.groupby("symbol")[_pc].sum().reset_index()
                _sym_bar.columns = ["symbol", "pnl"]
                _sym_bar["color"] = _sym_bar["pnl"].apply(
                    lambda x: "#26a269" if x >= 0 else "#c01c28"
                )
                fig_sym = px.bar(
                    _sym_bar, x="symbol", y="pnl",
                    color="color",
                    color_discrete_map={"#26a269": "#26a269", "#c01c28": "#c01c28"},
                    labels={"pnl": "Net P&L ($)", "symbol": "Symbol"},
                    height=250,
                )
                fig_sym.update_layout(
                    showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e",
                    font=dict(color="#e0e0e0"),
                )
                st.plotly_chart(fig_sym)
            else:
                st.markdown("#### Daily P&L")
                _date_col2 = _perf_ts_col if _perf_ts_col in _perf_df.columns else "date"
                if _date_col2 in _perf_df.columns:
                    _daily_bt = _perf_df.copy()
                    _daily_bt["_date"] = pd.to_datetime(_daily_bt[_date_col2], errors="coerce", utc=True).dt.date
                    _daily_agg = _daily_bt.groupby("_date")[_pc].sum().reset_index()
                    _daily_agg.columns = ["date", "pnl"]
                    _daily_agg["color"] = _daily_agg["pnl"].apply(
                        lambda x: "#26a269" if x >= 0 else "#c01c28"
                    )
                    fig_daily = px.bar(
                        _daily_agg, x="date", y="pnl",
                        color="color",
                        color_discrete_map={"#26a269": "#26a269", "#c01c28": "#c01c28"},
                        labels={"pnl": "P&L ($)", "date": "Date"},
                        height=250,
                    )
                    fig_daily.update_layout(
                        showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
                        plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e",
                        font=dict(color="#e0e0e0"),
                    )
                    st.plotly_chart(fig_daily)

        # Setup breakdown
        with col_right:
            st.markdown("#### Setup Performance")
            if "setup_type" in _perf_df.columns:
                setup_grp = _perf_df.groupby("setup_type").agg(
                    Trades=(_pc, "count"),
                    Net_PnL=(_pc, "sum"),
                    Avg_PnL=(_pc, "mean"),
                    Win_Rate=(_pc, lambda x: f"{(x > 0).mean() * 100:.1f}%"),
                ).reset_index()
                setup_grp.columns = ["Setup", "Trades", "Net P&L", "Avg P&L", "Win Rate"]
                setup_grp["Net P&L"] = setup_grp["Net P&L"].apply(lambda x: f"${x:+,.2f}")
                setup_grp["Avg P&L"] = setup_grp["Avg P&L"].apply(lambda x: f"${x:+.2f}")
                st.dataframe(setup_grp, height=250)

        col_day, col_exit = st.columns(2)

        # Win rate by day of week
        with col_day:
            st.markdown("#### Win Rate by Day of Week")
            _date_col = _perf_ts_col if _perf_ts_col in _perf_df.columns else "date"
            if _date_col in _perf_df.columns:
                df_dow = _perf_df.copy()
                df_dow["_dt"] = pd.to_datetime(df_dow[_date_col], errors="coerce")
                df_dow = df_dow.dropna(subset=["_dt"])
                df_dow["dow"] = df_dow["_dt"].dt.day_name()
                dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                dow_grp = df_dow.groupby("dow").agg(
                    win_rate=(_pc, lambda x: (x > 0).mean() * 100),
                    trades=(_pc, "count"),
                ).reindex(dow_order).dropna()
                fig_dow = px.bar(
                    dow_grp.reset_index(), x="dow", y="win_rate",
                    color="win_rate",
                    color_continuous_scale=["#c01c28", "#e5a50a", "#26a269"],
                    range_color=[0, 100],
                    labels={"dow": "Day", "win_rate": "Win Rate (%)"},
                    height=220,
                )
                fig_dow.update_layout(
                    showlegend=False, coloraxis_showscale=False,
                    margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e",
                    font=dict(color="#e0e0e0"),
                )
                st.plotly_chart(fig_dow)

        # Exit reason pie
        with col_exit:
            st.markdown("#### Exit Reasons")
            if "exit_reason" in _perf_df.columns:
                exit_grp = _perf_df["exit_reason"].value_counts().reset_index()
                exit_grp.columns = ["exit_reason", "count"]
                fig_pie = px.pie(
                    exit_grp, values="count", names="exit_reason",
                    height=220,
                    color_discrete_sequence=["#26a269", "#c01c28", "#4a9eff", "#e5a50a"],
                )
                fig_pie.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e",
                    font=dict(color="#e0e0e0"),
                )
                st.plotly_chart(fig_pie)


# ===========================================================================
# TAB 4 — Trade Log
# ===========================================================================

if _active_tab == "📋 My Trades":
    st.subheader("Trade Log")

    # --- Data source selector ------------------------------------------------
    _tl_source = st.radio(
        "Data Source",
        ["Backtest Results", "Walk-Forward Results", "Live Trades (Audit DB)"],
        index=0,
        key="trade_data_source",
        horizontal=True,
    )

    if _tl_source == "Live Trades (Audit DB)":
        _tl_df = trades_df.copy() if not trades_df.empty else pd.DataFrame()
        _tl_pnl = "pnl_dollars"
    elif _tl_source == "Walk-Forward Results":
        _tl_parts = [_latest_wfa_csv(s) for s in SYMBOLS]
        _tl_df = pd.concat([p for p in _tl_parts if not p.empty], ignore_index=True) if any(not p.empty for p in _tl_parts) else pd.DataFrame()
        _tl_pnl = "pnl_net" if "pnl_net" in _tl_df.columns else "pnl_dollars" if "pnl_dollars" in _tl_df.columns else None
    else:
        _tl_parts = [_latest_bt_csv(s) for s in SYMBOLS]
        _tl_df = pd.concat([p for p in _tl_parts if not p.empty], ignore_index=True) if any(not p.empty for p in _tl_parts) else pd.DataFrame()
        _tl_pnl = "pnl_net" if "pnl_net" in _tl_df.columns else "pnl_dollars" if "pnl_dollars" in _tl_df.columns else None

    # Ensure date columns are datetime
    for _dc in ("entry_ts", "exit_ts", "date"):
        if _dc in _tl_df.columns:
            _tl_df[_dc] = pd.to_datetime(_tl_df[_dc], errors="coerce", utc=True)

    if _tl_df.empty or _tl_pnl is None or _tl_pnl not in _tl_df.columns:
        st.info("No trade data for this source yet. Run a backtest or start the live bot.")
    else:
        # Filters
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            _tl_syms_available = sorted(_tl_df["symbol"].unique().tolist()) if "symbol" in _tl_df.columns else []
            sym_filter = st.selectbox(
                "Symbol",
                ["All"] + _tl_syms_available,
                key="trade_sym",
            )
        with fc2:
            _tl_setups = sorted(_tl_df["setup_type"].unique().tolist()) if "setup_type" in _tl_df.columns else []
            setup_filter = st.selectbox(
                "Setup type",
                ["All"] + _tl_setups,
                key="trade_setup",
            )
        with fc3:
            dir_filter = st.selectbox(
                "Direction",
                ["All", "LONG", "SHORT"],
                key="trade_dir",
            )
        with fc4:
            outcome_filter = st.selectbox(
                "Outcome",
                ["All", "Winners", "Losers"],
                key="trade_outcome",
            )

        filtered = _tl_df.copy()
        if sym_filter != "All" and "symbol" in filtered.columns:
            filtered = filtered[filtered["symbol"] == sym_filter]
        if setup_filter != "All" and "setup_type" in filtered.columns:
            filtered = filtered[filtered["setup_type"] == setup_filter]
        if dir_filter != "All" and "direction" in filtered.columns:
            filtered = filtered[filtered["direction"] == dir_filter]
        if outcome_filter == "Winners":
            filtered = filtered[filtered[_tl_pnl] > 0]
        elif outcome_filter == "Losers":
            filtered = filtered[filtered[_tl_pnl] <= 0]

        # Summary row for filtered set
        if not filtered.empty:
            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("Filtered Trades", len(filtered))
            sm2.metric("Net P&L", f"${filtered[_tl_pnl].sum():+,.2f}")
            sm3.metric("Win Rate", f"{(filtered[_tl_pnl] > 0).mean() * 100:.1f}%")
            sm4.metric("Avg P&L", f"${filtered[_tl_pnl].mean():+.2f}")

            # Per-symbol breakdown (only shown when multiple symbols are in the filtered set)
            _tl_syms_in_view = sorted(filtered["symbol"].unique().tolist()) if "symbol" in filtered.columns else []
            if len(_tl_syms_in_view) > 1:
                st.markdown("**By Symbol**")
                _tl_sym_rows = []
                for _ts in _tl_syms_in_view:
                    _tsd = filtered[filtered["symbol"] == _ts]
                    _ts_wins = (_tsd[_tl_pnl] > 0).sum()
                    _ts_pf_w = _tsd[_tsd[_tl_pnl] > 0][_tl_pnl].sum()
                    _ts_pf_l = abs(_tsd[_tsd[_tl_pnl] < 0][_tl_pnl].sum())
                    _tl_sym_rows.append({
                        "Symbol": _ts,
                        "Trades": len(_tsd),
                        "Wins": int(_ts_wins),
                        "Losses": int(len(_tsd) - _ts_wins),
                        "Win Rate": f"{(_ts_wins / len(_tsd) * 100):.1f}%" if len(_tsd) else "—",
                        "Net P&L": f"${_tsd[_tl_pnl].sum():+,.2f}",
                        "Avg P&L": f"${_tsd[_tl_pnl].mean():+.2f}",
                        "Profit Factor": f"{round(_ts_pf_w / _ts_pf_l, 2):.2f}" if _ts_pf_l > 0 else "—",
                    })
                st.dataframe(pd.DataFrame(_tl_sym_rows), height=min(100 + len(_tl_sym_rows) * 35, 220))

        display_cols = [
            "date", "symbol", "setup_type", "direction",
            "fill_price", "exit_price", "pnl_points", _tl_pnl,
            "exit_reason", "contracts",
        ]
        show_cols = [c for c in display_cols if c in filtered.columns]

        def _color_pnl(val):
            try:
                return "color: #26a269" if float(val) > 0 else "color: #c01c28"
            except Exception:
                return ""

        _sort_col = "entry_ts" if "entry_ts" in filtered.columns else (
            "date" if "date" in filtered.columns else filtered.columns[0]
        )
        styled = filtered.sort_values(_sort_col, ascending=False)[show_cols]
        if _tl_pnl in show_cols:
            st.dataframe(
                styled.style.map(_color_pnl, subset=[_tl_pnl]),
                height=450,
            )
        else:
            st.dataframe(styled, height=450)

        # Download button
        csv = filtered[show_cols].to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            file_name=f"trades_{date.today()}.csv",
            mime="text/csv",
        )


# ===========================================================================
# TAB 5 — Signal Log
# ===========================================================================

# Plain-English mapping for rejection reason strings
# Applied to the rejection_reason column before display.
_REJECTION_PLAIN = {
    "R:R":                       "Trade skipped — potential profit wasn't big enough compared to the risk",
    "Stop distance":             "Trade skipped — the stop-loss would have been too wide or too tight",
    "Kill switch":               "Trading paused for today — daily safety limit reached",
    "kill_switch":               "Trading paused for today — daily safety limit reached",
    "Outside execution window":  "Outside trading hours — bot only trades 9am–2:30pm ET",
    "execution window":          "Outside trading hours — bot only trades 9am–2:30pm ET",
    "Already in a position":     "Already in a trade — waiting for it to close first",
    "open_position":             "Already in a trade — waiting for it to close first",
    "Regime":                    "Market too choppy — volatility gate blocked this trade",
    "regime":                    "Market too choppy — volatility gate blocked this trade",
    "Trend filter":              "Trade was against the overall market direction",
    "trend_filter":              "Trade was against the overall market direction",
    "ml_filter":                 "AI wasn't confident enough in this trade",
    "ML filter":                 "AI wasn't confident enough in this trade",
    "News":                      "High-impact news day — bot takes the day off",
    "news":                      "High-impact news day — bot takes the day off",
    "max_trades":                "Maximum trades for today already taken (limit is per symbol)",
    "Max trades":                "Maximum trades for today already taken (limit is per symbol)",
    "daily_loss":                "Daily loss limit reached — bot stopped trading for today",
    "Daily loss":                "Daily loss limit reached — bot stopped trading for today",
    "Circuit breaker":           "3 losses in a row — taking a 30-minute break to reset",
    "circuit_breaker":           "3 losses in a row — taking a 30-minute break to reset",
    "Correlation block":         "Bot already trading a highly correlated symbol — reducing risk",
    "correlation":               "Bot already trading a highly correlated symbol — reducing risk",
}


def _humanize_rejection(raw: str) -> str:
    """Convert a raw rejection reason string to plain English."""
    if not raw:
        return ""
    # Exact match first
    for k, v in _REJECTION_PLAIN.items():
        if k.lower() in str(raw).lower():
            return v
    return str(raw)  # Fall back to raw if no match


if _active_tab == "🔔 Today's Trading":
    st.subheader("Today's Trading Activity")
    st.caption(
        "Every signal the bot detects is logged here — both approved trades and rejected ones. "
        "This helps you understand why the bot is or isn't taking trades."
    )

    if signals_df.empty:
        st.info(
            "**No signals logged yet.**  \n\n"
            "The signal log only populates while the **live or paper bot is running**. "
            "Each time the bot detects a potential trade setup, it records the signal "
            "here — whether it was approved and traded, or rejected by the AI filter, "
            "regime filter, or risk rules.  \n\n"
            "**To start generating signal data:**  \n"
            "1. Go to the sidebar and click **Start** on one or more symbols  \n"
            "2. Signals will appear here once the market is in the trading window  \n\n"
            "In the meantime, check the **Trades** and **Performance** tabs which can "
            "display your backtest and walk-forward results."
        )
    else:
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            show_rejected = st.checkbox("Show rejected signals", value=True, key="sig_show_rej")
        with sc2:
            setup_filter_sig = st.selectbox(
                "Setup type",
                ["All"] + sorted(signals_df["setup_type"].unique().tolist()) if "setup_type" in signals_df.columns else ["All"],
                key="sig_setup",
            )
        with sc3:
            today_only = st.checkbox("Today only", value=False, key="sig_today")

        filtered_sig = signals_df.copy()
        if not show_rejected and "approved" in filtered_sig.columns:
            filtered_sig = filtered_sig[filtered_sig["approved"] == 1]
        if setup_filter_sig != "All" and "setup_type" in filtered_sig.columns:
            filtered_sig = filtered_sig[filtered_sig["setup_type"] == setup_filter_sig]
        if today_only and "timestamp" in filtered_sig.columns:
            today_ts = pd.Timestamp.today().normalize()
            filtered_sig = filtered_sig[filtered_sig["timestamp"] >= today_ts]

        # Summary
        if not filtered_sig.empty and "approved" in filtered_sig.columns:
            total_sigs = len(filtered_sig)
            approved_count = (filtered_sig["approved"] == 1).sum()
            rejected_count = total_sigs - approved_count
            pass_rate = approved_count / total_sigs * 100 if total_sigs > 0 else 0

            ss1, ss2, ss3, ss4 = st.columns(4)
            ss1.metric("Total Signals", total_sigs)
            ss2.metric("Approved (Traded)", approved_count)
            ss3.metric("Rejected (Filtered)", rejected_count)
            ss4.metric("Pass Rate", f"{pass_rate:.1f}%")

            # Top rejection reasons (in plain English)
            if rejected_count > 0:
                rej_df = filtered_sig[filtered_sig["approved"] == 0].copy()
                if "rejection_reason" in rej_df.columns:
                    rej_df["plain_reason"] = rej_df["rejection_reason"].apply(_humanize_rejection)
                    top_reasons = rej_df["plain_reason"].value_counts().head(6)
                    with st.expander("Why were trades rejected? (click to expand)"):
                        for reason, count in top_reasons.items():
                            st.markdown(f"- **{count}×** — {reason}")

        # Humanize rejection reasons for display
        display_df = filtered_sig.copy()
        if "rejection_reason" in display_df.columns:
            display_df["rejection_reason"] = display_df["rejection_reason"].apply(_humanize_rejection)

        # Rename columns to plain English
        _col_rename = {
            "timestamp":        "Time",
            "setup_type":       "Setup",
            "direction":        "Direction",
            "entry_price":      "Entry Price",
            "stop_price":       "Stop",
            "target_price":     "Target",
            "reward_risk":      "R:R",
            "approved":         "Taken?",
            "rejection_reason": "Why it was skipped",
        }
        display_sig_cols = ["timestamp", "setup_type", "direction",
                            "entry_price", "stop_price", "target_price",
                            "reward_risk", "approved", "rejection_reason"]
        show_sig_cols = [c for c in display_sig_cols if c in display_df.columns]
        display_df = display_df[show_sig_cols].rename(columns=_col_rename)
        if "Taken?" in display_df.columns:
            display_df["Taken?"] = display_df["Taken?"].map({1: "Yes", 0: "No"})

        st.dataframe(
            display_df.sort_values("Time", ascending=False).head(100),
            height=450,
        )


# ===========================================================================
# TAB 6 — Settings
# ===========================================================================

if _active_tab == "⚙️ Settings":
    st.subheader("Bot Settings")
    st.caption(
        "Adjust how the bot behaves here. Changes are saved immediately. "
        "You'll need to restart the bot (Stop then Start in the sidebar) for changes to take effect."
    )
    st.warning("Changes take effect only after restarting the bot. Click 'Stop' then 'Start' in the sidebar.")

    cfg_edit = _load_settings()
    changed = False

    # --- Trading Hours ---
    st.markdown("#### Trading Hours")
    st.caption("The window of time each day when the bot is allowed to take trades.")
    col_s, col_e = st.columns(2)
    with col_s:
        exec_start = st.text_input(
            "Start trading at (ET, 24-hour clock)",
            value=cfg_edit.get("session", {}).get("execution_start", "09:00"),
            key="exec_start",
            help="The earliest the bot will enter a new trade each day. "
                 "09:00 = 9am Eastern Time. Most setups form in the first hour after open.",
        )
    with col_e:
        exec_end = st.text_input(
            "Stop taking new trades at (ET, 24-hour clock)",
            value=cfg_edit.get("session", {}).get("execution_end", "14:30"),
            key="exec_end",
            help="The bot won't open any new trades after this time. "
                 "14:30 = 2:30pm Eastern. Existing trades can still close after this time.",
        )

    # --- Volatility Gate ---
    st.markdown("#### Volatility Gate")
    st.caption(
        "This gate stops the bot from trading when the market is too choppy or wild. "
        "A higher number allows trading in more volatile conditions."
    )
    regime_cfg = cfg_edit.get("regime_filter", {})
    regime_enabled = st.toggle(
        "Turn on Volatility Gate",
        value=regime_cfg.get("enabled", True),
        key="regime_enabled",
        help="When ON, the bot won't trade if the market is moving too wildly. Recommended: ON.",
    )
    regime_mult = st.slider(
        "How sensitive is the volatility gate? (higher = allows trading in wilder markets)",
        min_value=1.2, max_value=3.5, step=0.1,
        value=float(regime_cfg.get("atr_high_vol_multiplier", 2.2)),
        key="regime_mult",
        help="2.2 means the bot blocks trading when price movement is 2.2× more wild than normal. "
             "Lower = very strict, fewer trades. Higher = more permissive, more trades in volatile days.",
    )

    ban_fri = st.checkbox(
        "Skip Fridays (recommended — Fridays tend to be choppy and harder to predict)",
        value=4 in regime_cfg.get("banned_days_of_week", [4]),
        key="ban_fri",
        help="Our historical data shows Fridays have about 25% lower win rate. "
             "Check this to have the bot automatically take the day off on Fridays.",
    )
    ban_mon = st.checkbox(
        "Skip Mondays (optional — Mondays can be unpredictable after the weekend)",
        value=0 in regime_cfg.get("banned_days_of_week", []),
        key="ban_mon",
        help="Monday openings can be choppy. Leave unchecked unless you're seeing poor Monday results.",
    )

    # --- AI Confidence Minimum ---
    st.markdown("#### AI Confidence Minimum")
    st.caption(
        "The AI scores every trade signal from 0% to 100% confidence. "
        "Only trades above this confidence level are taken."
    )
    ml_cfg = cfg_edit.get("ml", {})
    ml_enabled = st.toggle(
        "Use AI to filter trades",
        value=ml_cfg.get("enabled", True),
        key="ml_enabled",
        help="When ON, the AI scores every signal and skips low-confidence ones. "
             "Strongly recommended once you have 50+ backtest trades.",
    )
    ml_thresh = st.slider(
        "AI confidence minimum — how confident does the AI need to be before taking a trade?",
        min_value=0.50, max_value=0.90, step=0.01,
        value=float(ml_cfg.get("min_probability_threshold", 0.58)),
        key="ml_thresh",
        format="%.0%%",
        help="0.58 = only take trades when the AI is at least 58% confident it will win. "
             "Higher = fewer but better-quality trades. Lower = more trades but lower win rate. "
             "Recommended starting point: 58–62%.",
    )

    # --- Risk Management ---
    st.markdown("#### Risk Management")
    st.caption("How much the bot is allowed to risk each day.")
    risk_cfg = cfg_edit.get("risk", {})
    max_loss = st.number_input(
        "Stop trading for the day if losses reach this amount ($)",
        min_value=-5000, max_value=0, step=50,
        value=int(risk_cfg.get("max_daily_loss_dollars", -200)),
        key="max_loss",
        help="The bot will stop taking new trades for the day once total losses reach this number. "
             "Example: -500 means stop if you've lost $500 total today. "
             "Keep this well inside your funded account's daily loss limit.",
    )
    max_trades = st.slider(
        "Max trades per symbol per day — each symbol (ES, NQ) gets this many trades",
        min_value=1, max_value=10, step=1,
        value=int(risk_cfg.get("max_trades_per_day", 7)),
        key="max_trades",
        help="Each symbol the bot trades (ES, NQ, etc.) is allowed this many trades per day "
             "independently. Running 2 symbols at 7 each = up to 14 trades total. "
             "7 is a good balance between opportunity and overtrading.",
    )

    # --- Only Trade With the Trend (NQ) ---
    st.markdown("#### NQ — Only Trade With the Trend")
    nq_cfg = cfg_edit.get("symbols", {}).get("NQ", {})
    nq_trend_block = st.toggle(
        "Only take NQ trades that match the overall market direction",
        value=nq_cfg.get("trend_overrides", {}).get("macro_trend_hard_block", True),
        key="nq_trend_block",
        help="When ON, the bot will only take long (buy) trades on NQ when the market is going up, "
             "and only short (sell) trades when it's going down. "
             "Strongly recommended for NQ — our data shows this dramatically improves win rate.",
    )

    # --- Funded Account ---
    st.markdown("#### Funded Account Settings")
    st.caption("Match these to your TopstepX account so the bot knows your limits.")
    funded_cfg = cfg_edit.get("funded_account", {})
    acct_size = st.selectbox(
        "My account size",
        options=[25_000, 50_000, 75_000, 100_000, 150_000, 200_000],
        index=[25_000, 50_000, 75_000, 100_000, 150_000, 200_000].index(
            funded_cfg.get("account_size", 50_000)
        ) if funded_cfg.get("account_size", 50_000) in [25_000, 50_000, 75_000, 100_000, 150_000, 200_000] else 1,
        format_func=lambda x: f"${x:,}",
        key="acct_size",
        help="Set this to match your TopstepX account size. Used to calculate safe position sizing.",
    )
    safety_buf = st.slider(
        "Safety cushion — stop trading when this close to the account's loss limit ($)",
        min_value=100, max_value=2000, step=50,
        value=int(funded_cfg.get("safety_buffer_dollars", 500)),
        key="safety_buf",
        help="The bot will stop trading if it gets within this many dollars of the daily loss limit. "
             "Example: $500 cushion means the bot stops $500 before the limit — keeps your account safe.",
    )

    # --- Symbol enables ---
    st.markdown("#### Symbols")
    symbols_cfg = cfg_edit.get("symbols", {})
    sym_c1, sym_c2, sym_c3 = st.columns(3)
    with sym_c1:
        es_enabled = st.checkbox(
            "Trade ES (S&P 500)",
            value=symbols_cfg.get("ES", {}).get("enabled", True),
            key="es_enabled",
            help="ES = $50/pt  |  Full S&P 500 futures",
        )
    with sym_c2:
        nq_enabled = st.checkbox(
            "Trade NQ (Nasdaq)",
            value=symbols_cfg.get("NQ", {}).get("enabled", True),
            key="nq_enabled",
            help="NQ = $20/pt  |  Full Nasdaq futures",
        )
    with sym_c3:
        mnq_enabled = st.checkbox(
            "Trade MNQ (Micro Nasdaq)",
            value=symbols_cfg.get("MNQ", {}).get("enabled", False),
            key="mnq_enabled",
            help="MNQ = $2/pt  |  Micro Nasdaq (1/10th of NQ)\nUses same chart & levels as NQ — great for smaller accounts or sizing up gradually.",
        )
    if mnq_enabled:
        st.info(
            "ℹ️ **MNQ tip:** MNQ moves the same number of points as NQ but each point "
            "is worth $2 instead of $20.  A 40-point stop on MNQ risks $80, vs $800 on NQ.  "
            "Great for building confidence or running extra contracts on the same setup."
        )

    # --- Save button ---
    st.markdown("")
    if st.button("💾 Save All Settings", type="primary"):
        # Apply all changes
        cfg_edit["session"]["execution_start"] = exec_start
        cfg_edit["session"]["execution_end"] = exec_end

        cfg_edit.setdefault("regime_filter", {})["enabled"] = regime_enabled
        cfg_edit["regime_filter"]["atr_high_vol_multiplier"] = round(regime_mult, 1)
        banned = []
        if ban_fri:
            banned.append(4)
        if ban_mon:
            banned.append(0)
        banned.sort()
        cfg_edit["regime_filter"]["banned_days_of_week"] = banned

        cfg_edit.setdefault("ml", {})["enabled"] = ml_enabled
        cfg_edit["ml"]["min_probability_threshold"] = round(ml_thresh, 2)

        cfg_edit["risk"]["max_daily_loss_dollars"] = int(max_loss)
        cfg_edit["risk"]["max_trades_per_day"] = int(max_trades)

        cfg_edit.setdefault("funded_account", {})["account_size"] = int(acct_size)
        cfg_edit["funded_account"]["safety_buffer_dollars"] = int(safety_buf)

        cfg_edit.setdefault("symbols", {}).setdefault("ES",  {})["enabled"] = es_enabled
        cfg_edit.setdefault("symbols", {}).setdefault("NQ",  {})["enabled"] = nq_enabled
        cfg_edit.setdefault("symbols", {}).setdefault("MNQ", {})["enabled"] = mnq_enabled

        # Apply NQ trend block setting
        cfg_edit.setdefault("symbols", {}).setdefault("NQ", {}).setdefault(
            "trend_overrides", {}
        )["macro_trend_hard_block"] = nq_trend_block

        _save_settings(cfg_edit)
        st.cache_data.clear()
        st.success(
            "✅ Settings saved to config/settings.yaml. "
            "Use the sidebar to restart the bot(s) to apply changes."
        )
        st.rerun()

    with st.expander("📄 View Raw settings.yaml"):
        st.code(SETTINGS_PATH.read_text(encoding="utf-8"), language="yaml")


# ===========================================================================
# TAB 7 — Research & Testing  (rebuilt)
# ===========================================================================

# ── helpers scoped to Research tab ─────────────────────────────────────────

_SYM_COLOR = {"ES": "#4fc3f7", "NQ": "#ffb347", "MNQ": "#69db7c"}
_SYM_BG    = {"ES": "#071b29", "NQ": "#261a06", "MNQ": "#071f10"}
_SYM_ICON  = {"ES": "🔵", "NQ": "🟠", "MNQ": "🟢"}



def _sym_stats(df: "pd.DataFrame") -> dict:
    """Compute a compact performance stats dict from a trade DataFrame."""
    if df is None or df.empty:
        return {}
    pnl_col = "pnl_net" if "pnl_net" in df.columns else (
        "pnl_dollars" if "pnl_dollars" in df.columns else None
    )
    if pnl_col is None:
        return {}
    p = df[pnl_col]
    n = len(df)
    if n == 0:
        return {}
    wins   = int((p > 0).sum())
    losses = int((p < 0).sum())
    gw = float(p[p > 0].sum())
    gl = abs(float(p[p < 0].sum()))
    pf = round(gw / gl, 2) if gl > 0 else 999.0
    eq   = p.cumsum()
    peak = eq.cummax()
    dd   = float((eq - peak).min())
    return {
        "n":      n,
        "wr":     round(wins / n * 100, 1),
        "net":    round(float(p.sum()), 0),
        "pf":     pf,
        "max_dd": round(dd, 0),
        "avg_w":  round(float(p[p > 0].mean()), 0) if wins > 0 else 0,
        "avg_l":  round(float(p[p < 0].mean()), 0) if losses > 0 else 0,
    }


def _ml_info(sym: str) -> dict:
    """Return metadata for this symbol's ML model file."""
    sym_path  = ROOT / "data" / f"ml_filter_model_{sym}.pkl"
    legacy    = ROOT / "data" / "ml_filter_model.pkl"
    found     = sym_path if sym_path.exists() else (
        legacy if sym == "ES" and legacy.exists() else None
    )
    if found is None:
        return {"found": False}
    mt = datetime.fromtimestamp(found.stat().st_mtime)
    sz = found.stat().st_size // 1024
    return {"found": True, "path": found, "trained": mt, "name": found.name, "kb": sz}


def _stat_row_html(label: str, value: str, good: "bool | None" = None) -> str:
    clr = ""
    if good is True:
        clr = "color:#57e389;"
    elif good is False:
        clr = "color:#ff6b6b;"
    return (
        f"<div style='display:flex;justify-content:space-between;"
        f"padding:3px 0;border-bottom:1px solid #ffffff0d;'>"
        f"<span style='color:#8b949e;font-size:0.77em;'>{label}</span>"
        f"<span style='font-weight:600;font-size:0.82em;{clr}'>{value}</span>"
        f"</div>"
    )


def _section_header_html(title: str, color: str) -> str:
    return (
        f"<div style='font-size:0.68em;color:{color};font-weight:800;"
        f"letter-spacing:1.5px;text-transform:uppercase;"
        f"margin:10px 0 5px;border-bottom:1px solid {color}33;padding-bottom:3px;'>"
        f"{title}</div>"
    )


if _active_tab == "🔬 Test & Train AI":

    # =======================================================================
    # LEAKAGE WARNING — shown whenever standard backtest results exist
    # =======================================================================
    import glob as _glob
    _contaminated_files = _glob.glob("data/backtest_*.csv")
    _clean_files        = _glob.glob("data/clean_eval_*.csv")
    _labelled_path      = Path("data/labelled_trades.csv")
    _show_leakage_warn  = bool(_contaminated_files) and _labelled_path.exists()

    if _show_leakage_warn:
        try:
            _lt_df = pd.read_csv(_labelled_path)
            _lt_df["entry_ts"] = pd.to_datetime(_lt_df["entry_ts"], utc=True, errors="coerce")
            _lt_min = _lt_df["entry_ts"].min().date()
            _lt_max = _lt_df["entry_ts"].max().date()
            _leakage_detail = f"ML training data spans **{_lt_min}** → **{_lt_max}**."
        except Exception:
            _leakage_detail = "ML training data dates could not be read."

        st.warning(
            "⚠️ **Backtest results may be inflated by data leakage.**  \n"
            "The standard backtest runs with an ML model that was trained on the **same** date range "
            "it is then tested on. This caused win rates of 92–96% and profit factors of 100–1,500×, "
            "which are not realistic.  \n\n"
            f"{_leakage_detail}  \n\n"
            "**Honest baseline** (pre-ML, no leakage): NQ ~42% WR / PF 1.77 · ES ~39% WR / PF 1.31.  \n"
            "Run the **Clean Eval** below to get a properly separated forward-test result.",
        )
        if _clean_files:
            st.success(
                f"✅ {len(_clean_files)} clean-eval result(s) found — see the table below for honest numbers."
            )
        else:
            st.info("No clean-eval results yet. Use the command below to run the honest evaluation.")
            st.code(
                "python backtest/clean_eval.py --symbol NQ --cutoff 2024-01-01\n"
                "python backtest/clean_eval.py --symbol ES --cutoff 2024-01-01",
                language="bash",
            )

        # Show clean-eval results if they exist
        if _clean_files:
            st.markdown("#### Clean Forward-Test Results (honest, no leakage)")
            _clean_rows = []
            for _cf in sorted(_clean_files):
                try:
                    _cdf = pd.read_csv(_cf)
                    _is_raw = "_raw_" in _cf
                    _sym_part = Path(_cf).stem.replace("clean_eval_", "").split("_")[0]
                    _wr = (_cdf["pnl_net"] > 0).mean() * 100
                    _w  = _cdf[_cdf["pnl_net"] > 0]["pnl_net"].sum()
                    _l  = abs(_cdf[_cdf["pnl_net"] < 0]["pnl_net"].sum())
                    _pf = round(_w / _l, 2) if _l else 9999.0
                    _clean_rows.append({
                        "Symbol": _sym_part,
                        "Type": "Raw (no ML)" if _is_raw else "Clean ML (OOS)",
                        "Trades": len(_cdf),
                        "Win Rate": f"{_wr:.1f}%",
                        "Profit Factor": f"{_pf:.2f}",
                        "Net P&L": f"${_cdf['pnl_net'].sum():,.0f}",
                        "Expectancy": f"${_cdf['pnl_net'].mean():.2f}",
                    })
                except Exception:
                    pass
            if _clean_rows:
                st.dataframe(pd.DataFrame(_clean_rows), use_container_width=True)

    st.markdown("---")

    # =======================================================================
    # SECTION 0 — Safety: Live Mode Warning
    # =======================================================================
    _live_flag = Path("data/live_mode_active.flag")
    _cfg_paper = True
    try:
        import yaml as _yaml
        with open("config/settings.yaml") as _sf:
            _cfg_paper = _yaml.safe_load(_sf).get("execution", {}).get("paper_mode", True)
    except Exception:
        pass

    if not _cfg_paper or _live_flag.exists():
        st.markdown(
            "<div style='background:#4a0000;border:2px solid #ff4444;border-radius:8px;"
            "padding:16px;margin-bottom:16px;'>"
            "<b style='color:#ff6666;font-size:1.1rem;'>LIVE TRADING MODE IS ACTIVE</b><br>"
            "<span style='color:#ffcccc;'>Real orders will be placed with real money. "
            "paper_mode = false in config/settings.yaml. "
            "Confirm this is intentional. To switch back to paper: "
            "edit config/settings.yaml → <code>paper_mode: true</code></span>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='background:#0a2a0a;border:1px solid #00c853;border-radius:6px;"
            "padding:8px 14px;margin-bottom:12px;'>"
            "<span style='color:#00c853;font-size:0.9rem;'>Paper trading mode active (safe)</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    # =======================================================================
    # SECTION 0A — Currently Deployed AI Models
    # Always shown so you can see what's live right now.
    # =======================================================================
    try:
        from src.model_lifecycle import ModelLifecycle as _ML
        _lifecycle = _ML()
        _pending_models = _lifecycle.get_pending_models()
        _active_manifest = _lifecycle._load_manifest().get("active", {})
    except Exception as _e:
        _pending_models = []
        _active_manifest = {}

    _setup_labels_ai = {
        "BREAK_RETEST":  "Break + Retest",
        "REJECTION":     "Rejection",
        "BOUNCE":        "Bounce",
        "SWEEP_REVERSE": "Sweep & Reverse",
    }

    if _active_manifest:
        st.markdown(
            "<h3 style='color:#e6edf3;margin-bottom:4px;'>Currently Deployed AI Models</h3>",
            unsafe_allow_html=True,
        )
        _am_cols = st.columns(min(len(_active_manifest), 4))
        for _i, (_am_key, _am_entry) in enumerate(_active_manifest.items()):
            with _am_cols[_i % len(_am_cols)]:
                try:
                    _am_sym, _am_st = _am_key.split("_", 1)
                except ValueError:
                    _am_sym, _am_st = _am_key, "combined"
                _am_label = _setup_labels_ai.get(_am_st, _am_st)
                _am_auc = _am_entry.get("new_auc") or _am_entry.get("auc", 0)
                _am_wr  = _am_entry.get("new_win_rate_sim") or _am_entry.get("win_rate_sim", 0)
                _am_ts  = str(_am_entry.get("timestamp", ""))[:10]
                _am_approved = str(_am_entry.get("approved_at", ""))[:10]

                # Load manifest for schema metadata if available
                _am_manifest_file = Path("data/models/active") / f"ml_filter_{_am_key}_manifest.json"
                _am_manifest: dict = {}
                if _am_manifest_file.exists():
                    try:
                        _am_manifest = json.loads(_am_manifest_file.read_text())
                    except Exception:
                        pass

                _am_schema_h  = _am_manifest.get("schema_hash", "—")[:14]
                _am_cv_method = _am_manifest.get("cv_method", "—")
                _am_cal       = _am_manifest.get("calibration", "—")
                _am_label_m   = _am_manifest.get("label_method", "—")
                _am_ev        = _am_manifest.get("ev_policy", {})
                _am_ev_str    = (
                    f"EV min={_am_ev.get('ev_min_threshold', '?')} "
                    f"R_win={_am_ev.get('avg_R_win', '?')} "
                    f"R_loss={_am_ev.get('avg_R_loss', '?')}"
                ) if _am_ev else "—"

                st.markdown(
                    f"<div style='background:#1a1a1a;border:1px solid #262626;border-radius:8px;"
                    f"padding:12px;margin-bottom:8px;'>"
                    f"<b style='color:#00c853;'>{_am_sym} — {_am_label}</b><br>"
                    f"<span style='color:#9a9a9a;font-size:0.8rem;'>"
                    f"AUC: <b style='color:#e6edf3;'>{float(_am_auc or 0):.3f}</b>"
                    f" &nbsp;|&nbsp; "
                    f"Win rate: <b style='color:#e6edf3;'>{float(_am_wr or 0)*100:.0f}%</b>"
                    f"</span><br>"
                    f"<span style='color:#555;font-size:0.72rem;'>"
                    f"label={_am_label_m} &nbsp;|&nbsp; "
                    f"cv={_am_cv_method} &nbsp;|&nbsp; "
                    f"cal={_am_cal}<br>"
                    f"ev: {_am_ev_str}<br>"
                    f"schema: {_am_schema_h} &nbsp;|&nbsp; "
                    f"deployed {_am_approved or _am_ts}"
                    f"</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
    else:
        st.info("No AI models deployed yet. Run a backtest, then start the AI Re-Learning process to train the first models.")

    st.markdown("---")

    # =======================================================================
    # SECTION 0B — Model Approval Panel
    # Shows whenever there is a pending AI model awaiting human review.
    # =======================================================================

    if _pending_models:
        st.markdown("---")
        st.markdown(
            "<h3 style='color:#1e88e5;'>🤖 New AI Models Ready for Your Review</h3>",
            unsafe_allow_html=True,
        )
        st.info(
            "The AI has been re-trained during market downtime. "
            "Compare each new model against the one currently running, then decide whether to **Approve** or **Reject** it. "
            "You can also **Rollback** to the previous version if needed."
        )

        for _pm in _pending_models:
            _sym = _pm.get("symbol", "?")
            _st_type = _pm.get("setup_type", "?")
            _ts = _pm.get("timestamp", "?")
            _old = _pm.get("old_metrics", {})
            _new = _pm.get("new_metrics", {})
            _br = _pm.get("battery_result", {})

            # Human-readable setup type labels
            _setup_labels = {
                "BREAK_RETEST": "Break + Retest",
                "REJECTION":    "Rejection",
                "BOUNCE":       "Bounce",
                "SWEEP_REVERSE": "Sweep & Reverse",
            }
            _st_label = _setup_labels.get(_st_type, _st_type)

            st.markdown(
                f"<div class='approval-card'>"
                f"<b style='color:#1e88e5;font-size:1.05rem;'>NEW AI MODEL READY — {_sym} {_st_label}</b>"
                f"<span style='color:#6b6b6b; font-size:0.8rem; margin-left:12px;'>Trained: {_ts[:10] if _ts and len(str(_ts)) >= 10 else _ts}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            _c1, _c2 = st.columns(2)
            with _c1:
                st.markdown("**Current Model (running now)**")
                if _old:
                    st.metric("AUC Score", f"{_old.get('auc', 0) or _old.get('mean_cv_auc', 0):.3f}" if _old.get('auc') or _old.get('mean_cv_auc') else "—")
                    st.metric("Win Rate (simulated)", f"{float(_old.get('win_rate_sim', 0) or 0) * 100:.1f}%" if _old.get('win_rate_sim') else "—")
                    st.metric("Trades trained on", str(_old.get("n_trades", "—")))
                else:
                    st.info("No previous model — this would be the first deployment.")

            with _c2:
                st.markdown("**New Candidate Model**")
                _new_auc = float(_new.get("mean_cv_auc", 0) or 0)
                _old_auc = float(_old.get("auc", 0) or _old.get("mean_cv_auc", 0) or 0)
                _auc_delta = _new_auc - _old_auc if _old_auc else None
                _delta_str = (f" ({'+' if _auc_delta >= 0 else ''}{_auc_delta:.3f})" if _auc_delta is not None else "")
                st.metric(
                    "AUC Score",
                    f"{_new_auc:.3f}{_delta_str}",
                    delta=f"{'+' if _auc_delta and _auc_delta >= 0 else ''}{_auc_delta:.3f}" if _auc_delta is not None else None,
                )
                _new_wr = float(_new.get("win_rate_sim", 0) or 0)
                st.metric("Win Rate (simulated)", f"{_new_wr * 100:.1f}%" if _new_wr else "—")
                st.metric("Trades trained on", str(_new.get("n_trades", len(_new.get("label_distribution", {})) or "—")))

                # ── New metadata from dataset_manifest.json ───────────────
                _pm_mf_path = Path("data/models/pending") / f"ml_filter_{_sym}_{_st_type}_manifest.json"
                _pm_manifest: dict = {}
                if _pm_mf_path.exists():
                    try:
                        _pm_manifest = json.loads(_pm_mf_path.read_text())
                    except Exception:
                        pass
                if _pm_manifest:
                    st.markdown("**Model Quality Metadata**")
                    _pm_ev = _pm_manifest.get("ev_policy", {})
                    _pm_metrics = _pm_manifest.get("metrics", {})
                    meta_rows = {
                        "Label method":   _pm_manifest.get("label_method", "—"),
                        "CV method":      _pm_manifest.get("cv_method", "—"),
                        "Calibration":    _pm_manifest.get("calibration", "—"),
                        "Schema hash":    (_pm_manifest.get("schema_hash") or "—")[:18],
                        "Features":       str(_pm_manifest.get("n_features", "—")),
                        "Folds used":     str(_pm_metrics.get("n_splits_used", "—")),
                        "EV min thresh":  str(_pm_ev.get("ev_min_threshold", "—")),
                        "avg_R_win":      str(_pm_ev.get("avg_R_win", "—")),
                        "avg_R_loss":     str(_pm_ev.get("avg_R_loss", "—")),
                        "Train period":   f"{_pm_manifest.get('train_start','?')} to {_pm_manifest.get('train_end','?')}",
                    }
                    for k, v in meta_rows.items():
                        st.markdown(
                            f"<span style='color:#9a9a9a;font-size:0.8rem;'>{k}:</span> "
                            f"<b style='color:#e6edf3;font-size:0.8rem;'>{v}</b>",
                            unsafe_allow_html=True,
                        )

                # Gate results
                _gates = _br.get("gate_results", [])
                if _gates:
                    _passed = sum(1 for g in _gates if g.get("passed"))
                    _total = len(_gates)
                    if _passed == _total:
                        st.success(f"✅ All {_total} safety checks passed")
                    else:
                        st.warning(f"⚠️ {_passed}/{_total} safety checks passed")

            with st.expander(f"View safety check details & feature importance — {_sym} {_st_label}"):
                _gates = _br.get("gate_results", [])
                if _gates:
                    st.markdown("**Safety Gate Results:**")
                    for _g in _gates:
                        _icon = "✅" if _g.get("passed") else "❌"
                        _gname = _g.get("gate_name", "")
                        _greason = _g.get("reason", "")
                        _gate_plain = {
                            "MinimumDataGate":       "Enough training data",
                            "AUCGate":               "Prediction accuracy (AUC)",
                            "WinRateSimGate":        "Win rate simulation",
                            "FeatureStabilityGate":  "Feature stability (not overfitting to noise)",
                            "CalibrationGate":       "Probability calibration",
                            "BacktestApplicationGate": "Backtest P&L check",
                        }
                        _gname_plain = _gate_plain.get(_gname, _gname)
                        st.markdown(f"{_icon} **{_gname_plain}** — {_greason}")

                _top_features = _new.get("top_features", {})
                if _top_features:
                    st.markdown("**Top 5 most important signals the AI learned:**")
                    for _feat, _imp in list(_top_features.items())[:5]:
                        _feat_plain = {
                            "ema50_aligned": "Market trending in trade direction (EMA50)",
                            "momentum_aligned": "Momentum matches trade direction",
                            "gap_size_norm": "Gap from yesterday's close",
                            "session_phase": "Time of day (open/mid/afternoon)",
                            "or_range_vs_5day_avg": "Today's opening range vs. average",
                            "atr_14_15m": "Market volatility",
                            "break_excursion_pts": "How far price broke the level",
                            "volume_ratio": "Volume vs. average",
                        }.get(_feat, _feat.replace("_", " ").title())
                        bar_width = int(float(_imp) * 300)
                        st.markdown(
                            f"<div style='display:flex;align-items:center;gap:8px;margin:2px 0;'>"
                            f"<span style='width:200px;font-size:0.82rem;color:#d0d0d0;'>{_feat_plain}</span>"
                            f"<div style='background:#1e88e5;height:10px;width:{bar_width}px;border-radius:3px;'></div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            # Approve / Reject / Rollback buttons
            _btn_col1, _btn_col2, _btn_col3 = st.columns([2, 2, 2])
            with _btn_col1:
                if st.button(
                    "✅ Approve & Deploy",
                    key=f"approve_{_sym}_{_st_type}",
                    type="primary",
                    help="Deploy this new model. The old model will be archived and can be restored.",
                ):
                    try:
                        ok = _lifecycle.approve(_sym, _st_type)
                        if ok:
                            st.success(f"✅ New model deployed for {_sym} {_st_label}! Restart the bot to use it.")
                            st.rerun()
                        else:
                            st.error("Could not deploy model. Check the log.")
                    except Exception as _e2:
                        st.error(f"Deployment error: {_e2}")
            with _btn_col2:
                if st.button(
                    "❌ Reject",
                    key=f"reject_{_sym}_{_st_type}",
                    type="secondary",
                    help="Discard this candidate. The current model stays active.",
                ):
                    try:
                        _lifecycle.reject(_sym, _st_type)
                        st.info(f"Rejected. Current model for {_sym} {_st_label} remains active.")
                        st.rerun()
                    except Exception as _e3:
                        st.error(f"Reject error: {_e3}")
            with _btn_col3:
                if st.button(
                    "↩️ Rollback to Previous Version",
                    key=f"rollback_{_sym}_{_st_type}",
                    help="Restore the previous archived version of this model.",
                ):
                    try:
                        ok = _lifecycle.rollback(_sym, _st_type)
                        if ok:
                            st.success(f"↩️ Rolled back to previous model for {_sym} {_st_label}.")
                            st.rerun()
                        else:
                            st.warning(f"No archived version found for {_sym} {_st_label}.")
                    except Exception as _e4:
                        st.error(f"Rollback error: {_e4}")

            st.markdown("---")

    # Continuous Learner control
    st.markdown("#### AI Re-Learning Status")
    _cl_state_file = ROOT / "data" / "learner_state.json"
    if _cl_state_file.exists():
        try:
            _cl_state = json.loads(_cl_state_file.read_text(encoding="utf-8"))
            _cl_running = _cl_state.get("running", False)
            _cl_last_check = _cl_state.get("last_check", "—")
            _cl_last_retrain = _cl_state.get("last_retrain", {})
            if _cl_running:
                st.success(f"🟢 AI Re-Learning is running in the background (last check: {_cl_last_check})")
            else:
                st.info("⚫ AI Re-Learning is not running. Use Auto-Run tab or start it manually.")
            if _cl_last_retrain:
                with st.expander("Last retrain timestamps"):
                    for sym, ts in _cl_last_retrain.items():
                        st.caption(f"{sym}: {ts}")
            _cl_log = _cl_state.get("log", [])[-10:]
            if _cl_log:
                with st.expander("Recent AI learning activity log"):
                    for entry in reversed(_cl_log):
                        _lvl_color = "#f44336" if entry.get("level") == "ERROR" else \
                                     "#ff9800" if entry.get("level") == "WARN" else "#9a9a9a"
                        st.markdown(
                            f"<span style='color:{_lvl_color};font-size:0.78rem;'>"
                            f"[{entry.get('ts', '')[:19]}] {entry.get('msg', '')}</span>",
                            unsafe_allow_html=True,
                        )
        except Exception:
            st.caption("Could not read AI learner state.")
    else:
        st.info("AI Re-Learning hasn't run yet. It starts automatically after the market closes.")

    # =======================================================================
    # SECTION 1 — Analytics Dashboard
    # =======================================================================

    _hdr_col, _refresh_col = st.columns([5, 1])
    with _hdr_col:
        st.markdown(
            "<h2 style='margin-bottom:2px;color:#e6edf3;'>📊 Analytics Dashboard</h2>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Live snapshot of the most recent results for each symbol. "
            "Updates automatically after every test completes."
        )
    with _refresh_col:
        st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Refresh", key="analytics_refresh", use_container_width=True):
            st.rerun()

    st.markdown("")

    _ana_cols = st.columns(3)
    for _ai, _asym in enumerate(["ES", "NQ", "MNQ"]):
        _aclr  = _SYM_COLOR[_asym]
        _abg   = _SYM_BG[_asym]
        _aicon = _SYM_ICON[_asym]
        _abdf  = _latest_bt_csv(_asym)
        _awdf  = _latest_wfa_csv(_asym)
        _abst  = _sym_stats(_abdf)
        _awst  = _sym_stats(_awdf)
        _aml   = _ml_info(_asym)

        with _ana_cols[_ai]:

            # Build inner HTML ────────────────────────────────────────────
            # Backtest block
            if _abst:
                _bt_block = (
                    _section_header_html("Backtest", _aclr)
                    + _stat_row_html("Trades",       str(_abst["n"]))
                    + _stat_row_html("Win Rate",      f"{_abst['wr']}%",
                                     _abst["wr"] >= 70)
                    + _stat_row_html("Net P&L",       f"${_abst['net']:,.0f}",
                                     _abst["net"] > 0)
                    + _stat_row_html("Profit Factor", f"{_abst['pf']}",
                                     _abst["pf"] >= 3.0)
                    + _stat_row_html("Max Drawdown",  f"${_abst['max_dd']:,.0f}")
                    + _stat_row_html("Avg W / L",
                                     f"${_abst['avg_w']:,.0f} / ${_abst['avg_l']:,.0f}")
                )
            else:
                _bt_block = (
                    _section_header_html("Backtest", _aclr)
                    + "<div style='color:#444;font-size:0.78em;padding:8px 0;'>"
                    "No data — run a Backtest below</div>"
                )

            # Walk-Forward block
            if _awst:
                _wf_block = (
                    _section_header_html("Walk-Forward", _aclr)
                    + _stat_row_html("Trades",       str(_awst["n"]))
                    + _stat_row_html("Win Rate",      f"{_awst['wr']}%",
                                     _awst["wr"] >= 65)
                    + _stat_row_html("Net P&L",       f"${_awst['net']:,.0f}",
                                     _awst["net"] > 0)
                    + _stat_row_html("Profit Factor", f"{_awst['pf']}",
                                     _awst["pf"] >= 2.0)
                    + _stat_row_html("Max Drawdown",  f"${_awst['max_dd']:,.0f}")
                )
            else:
                _wf_block = (
                    _section_header_html("Walk-Forward", _aclr)
                    + "<div style='color:#444;font-size:0.78em;padding:8px 0;'>"
                    "No data — run Walk-Forward below</div>"
                )

            # ML model footer
            if _aml["found"]:
                _ml_et = _to_et(_aml["trained"])
                _ml_footer = (
                    f"<div style='color:#57e389;font-size:0.8em;'>✅ AI Model Ready</div>"
                    f"<div style='color:#8b949e;font-size:0.7em;'>"
                    f"Trained: {_ml_et.strftime('%b %d %I:%M %p %Z')} &nbsp;·&nbsp; "
                    f"{_aml['kb']} KB</div>"
                )
            else:
                _ml_footer = (
                    "<div style='color:#ff6b6b;font-size:0.8em;'>⚠️ No AI Model</div>"
                    "<div style='color:#555;font-size:0.7em;'>Retrain using the ML Retrain tool below</div>"
                )

            # Assemble full card
            _full_card = (
                f"<div style='background:{_abg};"
                f"border:1.5px solid {_aclr}44;"
                f"border-top:3px solid {_aclr};"
                f"border-radius:10px;"
                f"padding:14px 16px;"
                f"min-height:320px;'>"
                # Header
                f"<div style='color:{_aclr};font-size:1.25em;font-weight:800;"
                f"margin-bottom:4px;'>{_aicon} {_asym}</div>"
                # Backtest + WFA stats
                + _bt_block
                + _wf_block
                # ML footer
                + f"<div style='border-top:1px solid {_aclr}22;"
                f"margin-top:10px;padding-top:8px;text-align:center;'>"
                + _ml_footer
                + "</div></div>"
            )

            st.markdown(_full_card, unsafe_allow_html=True)

    # =======================================================================
    # SECTION 2 — Testing Tools
    # =======================================================================

    st.markdown("")
    st.markdown("---")
    st.markdown(
        "<h2 style='margin-bottom:4px;color:#e6edf3;'>🔬 Testing Tools</h2>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Select your symbols, set the date range, and run. "
        "Results appear below the tool while it runs, then the Analytics section above updates automatically."
    )
    st.markdown("")

    # Persistent date defaults (shared across Backtest + Walk-Forward)
    if "rt_start" not in st.session_state:
        st.session_state["rt_start"] = "2022-01-03"
    if "rt_end" not in st.session_state:
        st.session_state["rt_end"] = "2024-12-31"

    _rt_bt_tab, _rt_wf_tab, _rt_ml_tab = st.tabs(
        ["📊  Backtest", "🔄  Walk-Forward", "🤖  ML Retrain"]
    )

    # -------------------------------------------------------------------
    # Sub-tab: Backtest
    # -------------------------------------------------------------------
    with _rt_bt_tab:

        # Symbol checkboxes
        st.markdown("**Select Symbols**")
        _btc1, _btc2, _btc3 = st.columns(3)
        _bt_es  = _btc1.checkbox("🔵  ES  (E-mini S&P 500)",  value=True, key="rt_bt_es")
        _bt_nq  = _btc2.checkbox("🟠  NQ  (E-mini Nasdaq)",   value=True, key="rt_bt_nq")
        _bt_mnq = _btc3.checkbox("🟢  MNQ (Micro Nasdaq)",    value=True, key="rt_bt_mnq")
        _bt_syms = [s for s, c in [("ES", _bt_es), ("NQ", _bt_nq), ("MNQ", _bt_mnq)] if c]

        # Date range
        st.markdown("**Date Range**")
        _bd1, _bd2 = st.columns(2)
        _bt_start = _bd1.text_input(
            "Start Date",
            value=st.session_state["rt_start"],
            key="rt_bt_start",
            placeholder="YYYY-MM-DD",
        )
        _bt_end = _bd2.text_input(
            "End Date",
            value=st.session_state["rt_end"],
            key="rt_bt_end",
            placeholder="YYYY-MM-DD",
        )
        if _bt_start:
            st.session_state["rt_start"] = _bt_start
        if _bt_end:
            st.session_state["rt_end"] = _bt_end

        st.markdown("")

        # Status badge + Run button
        _bt_running = _is_task_running("backtest_all")
        _bst_badge, _bst_run = st.columns([3, 1])
        with _bst_badge:
            _render_task_status_badge("backtest_all")
        with _bst_run:
            if st.button(
                "⏳ Running…" if _bt_running else "▶️  Run Backtest",
                type="primary",
                key="rt_bt_run",
                disabled=_bt_running or not _bt_syms,
                use_container_width=True,
            ):
                if not _bt_syms:
                    st.warning("Please select at least one symbol.")
                else:
                    _cmd = [
                        sys.executable, "-u",
                        str(ROOT / "backtest" / "run_all.py"),
                        "--start", _bt_start,
                        "--end",   _bt_end,
                        "--symbols", *_bt_syms,
                    ]
                    ok, msg = _start_task("backtest_all", _cmd)
                    if ok:
                        st.session_state["rt_bt_ran_syms"] = list(_bt_syms)
                        st.rerun()
                    else:
                        st.warning(msg)

        if _bt_running:
            if st.button("⏹  Stop Backtest", key="rt_bt_stop", type="secondary"):
                _stop_task("backtest_all")
                st.rerun()

        _render_progress_panel("backtest_all")

        _bt_log = _read_task_log("backtest_all")
        if _bt_log:
            with st.expander("📄 Console Output", expanded=_bt_running):
                st.code(_bt_log[-10000:], language="text")

        # Temporary results block
        _bt_done_ts = _task_completed_info("backtest_all")
        _bt_ran = st.session_state.get("rt_bt_ran_syms", ["ES", "NQ", "MNQ"])
        if _bt_done_ts and not _bt_running and _bt_log:
            st.markdown("")
            st.markdown(
                "<div style='background:#071f10;border:1px solid #57e38966;"
                "border-left:4px solid #57e389;border-radius:8px;"
                "padding:10px 16px;margin-bottom:10px;'>"
                "<b style='color:#57e389;font-size:1em;'>✅ Last Backtest Results</b>"
                f"<span style='color:#8b949e;font-size:0.82em;margin-left:10px;'>"
                f"completed {_bt_done_ts[1]}"
                "</span></div>",
                unsafe_allow_html=True,
            )
            _bt_res_cols = st.columns(len(_bt_ran))
            for _bri, _brs in enumerate(_bt_ran):
                _brdf = _latest_bt_csv(_brs)
                _brst = _sym_stats(_brdf)
                _brc  = _SYM_COLOR.get(_brs, "#fff")
                _brbg = _SYM_BG.get(_brs, "#111")
                if _brst:
                    _bt_res_cols[_bri].markdown(
                        f"<div style='background:{_brbg};"
                        f"border:1.5px solid {_brc}55;"
                        f"border-top:3px solid {_brc};"
                        f"border-radius:8px;padding:14px;text-align:center;'>"
                        f"<div style='color:{_brc};font-weight:800;font-size:1.1em;"
                        f"margin-bottom:6px;'>{_SYM_ICON.get(_brs,'')} {_brs}</div>"
                        f"<div style='color:#57e389;font-size:1.6em;font-weight:700;'>"
                        f"${_brst['net']:,.0f}</div>"
                        f"<div style='color:#8b949e;font-size:0.8em;margin-top:4px;'>"
                        f"{_brst['n']} trades &nbsp;·&nbsp; {_brst['wr']}% WR</div>"
                        f"<div style='color:#8b949e;font-size:0.78em;'>"
                        f"Profit Factor: {_brst['pf']}</div>"
                        f"<div style='color:#666;font-size:0.72em;'>"
                        f"Max DD: ${_brst['max_dd']:,.0f}</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    _bt_res_cols[_bri].info(f"{_brs}: no data yet")

    # -------------------------------------------------------------------
    # Sub-tab: Walk-Forward
    # -------------------------------------------------------------------
    with _rt_wf_tab:

        st.markdown("**Select Symbols**")
        _wfc1, _wfc2, _wfc3 = st.columns(3)
        _wf_es  = _wfc1.checkbox("🔵  ES  (E-mini S&P 500)",  value=True, key="rt_wf_es")
        _wf_nq  = _wfc2.checkbox("🟠  NQ  (E-mini Nasdaq)",   value=True, key="rt_wf_nq")
        _wf_mnq = _wfc3.checkbox("🟢  MNQ (Micro Nasdaq)",    value=True, key="rt_wf_mnq")
        _wf_syms = [s for s, c in [("ES", _wf_es), ("NQ", _wf_nq), ("MNQ", _wf_mnq)] if c]

        st.markdown("**Date Range**")
        _wd1, _wd2 = st.columns(2)
        _wf_start = _wd1.text_input(
            "Start Date",
            value=st.session_state["rt_start"],
            key="rt_wf_start",
            placeholder="YYYY-MM-DD",
        )
        _wf_end = _wd2.text_input(
            "End Date",
            value=st.session_state["rt_end"],
            key="rt_wf_end",
            placeholder="YYYY-MM-DD",
        )
        if _wf_start:
            st.session_state["rt_start"] = _wf_start
        if _wf_end:
            st.session_state["rt_end"] = _wf_end

        st.markdown("")

        _wf_running = _is_task_running("walkforward_all")
        _wfb_badge, _wfb_run = st.columns([3, 1])
        with _wfb_badge:
            _render_task_status_badge("walkforward_all")
        with _wfb_run:
            if st.button(
                "⏳ Running…" if _wf_running else "▶️  Run Walk-Forward",
                type="primary",
                key="rt_wf_run",
                disabled=_wf_running or not _wf_syms,
                use_container_width=True,
            ):
                if not _wf_syms:
                    st.warning("Please select at least one symbol.")
                else:
                    _cmd = [
                        sys.executable, "-u",
                        str(ROOT / "backtest" / "wf_all.py"),
                        "--start", _wf_start,
                        "--end",   _wf_end,
                        "--symbols", *_wf_syms,
                    ]
                    ok, msg = _start_task("walkforward_all", _cmd)
                    if ok:
                        st.session_state["rt_wf_ran_syms"] = list(_wf_syms)
                        st.rerun()
                    else:
                        st.warning(msg)

        if _wf_running:
            if st.button("⏹  Stop Walk-Forward", key="rt_wf_stop", type="secondary"):
                _stop_task("walkforward_all")
                st.rerun()

        _render_progress_panel("walkforward_all")

        _wf_log = _read_task_log("walkforward_all")
        if _wf_log:
            with st.expander("📄 Console Output", expanded=_wf_running):
                st.code(_wf_log[-10000:], language="text")

        # Temporary results
        _wf_done_ts = _task_completed_info("walkforward_all")
        _wf_ran = st.session_state.get("rt_wf_ran_syms", ["ES", "NQ", "MNQ"])
        if _wf_done_ts and not _wf_running and _wf_log:
            st.markdown("")
            st.markdown(
                "<div style='background:#071f10;border:1px solid #57e38966;"
                "border-left:4px solid #57e389;border-radius:8px;"
                "padding:10px 16px;margin-bottom:10px;'>"
                "<b style='color:#57e389;font-size:1em;'>✅ Last Walk-Forward Results</b>"
                f"<span style='color:#8b949e;font-size:0.82em;margin-left:10px;'>"
                f"completed {_wf_done_ts[1]}"
                "</span></div>",
                unsafe_allow_html=True,
            )
            _wf_res_cols = st.columns(len(_wf_ran))
            for _wri, _wrs in enumerate(_wf_ran):
                _wrdf = _latest_wfa_csv(_wrs)
                _wrst = _sym_stats(_wrdf)
                _wrc  = _SYM_COLOR.get(_wrs, "#fff")
                _wrbg = _SYM_BG.get(_wrs, "#111")
                if _wrst:
                    _wf_res_cols[_wri].markdown(
                        f"<div style='background:{_wrbg};"
                        f"border:1.5px solid {_wrc}55;"
                        f"border-top:3px solid {_wrc};"
                        f"border-radius:8px;padding:14px;text-align:center;'>"
                        f"<div style='color:{_wrc};font-weight:800;font-size:1.1em;"
                        f"margin-bottom:6px;'>{_SYM_ICON.get(_wrs,'')} {_wrs}</div>"
                        f"<div style='color:#57e389;font-size:1.6em;font-weight:700;'>"
                        f"${_wrst['net']:,.0f}</div>"
                        f"<div style='color:#8b949e;font-size:0.8em;margin-top:4px;'>"
                        f"{_wrst['n']} trades &nbsp;·&nbsp; {_wrst['wr']}% WR</div>"
                        f"<div style='color:#8b949e;font-size:0.78em;'>"
                        f"Profit Factor: {_wrst['pf']}</div>"
                        f"<div style='color:#666;font-size:0.72em;'>"
                        f"Max DD: ${_wrst['max_dd']:,.0f}</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    _wf_res_cols[_wri].info(f"{_wrs}: no data yet")

    # -------------------------------------------------------------------
    # Sub-tab: ML Retrain
    # -------------------------------------------------------------------
    with _rt_ml_tab:

        st.markdown("**Select Symbols**")
        _mlc1, _mlc2, _mlc3 = st.columns(3)
        _ml_es  = _mlc1.checkbox("🔵  ES  (E-mini S&P 500)",  value=True, key="rt_ml_es")
        _ml_nq  = _mlc2.checkbox("🟠  NQ  (E-mini Nasdaq)",   value=True, key="rt_ml_nq")
        _ml_mnq = _mlc3.checkbox("🟢  MNQ (Micro Nasdaq)",    value=True, key="rt_ml_mnq")
        _ml_syms = [s for s, c in [("ES", _ml_es), ("NQ", _ml_nq), ("MNQ", _ml_mnq)] if c]

        st.info(
            "ℹ️ **What this does:** Trains a new XGBoost AI model for each selected symbol "
            "using all available backtest and walk-forward trade history. "
            "The AI learns which setups historically won and filters out weak ones going forward. "
            "Always retrain after completing new backtests to keep the models current."
        )

        st.markdown("")

        _ml_running = _is_task_running("ml_retrain_all")
        _mlb_badge, _mlb_run = st.columns([3, 1])
        with _mlb_badge:
            _render_task_status_badge("ml_retrain_all")
        with _mlb_run:
            if st.button(
                "⏳ Training…" if _ml_running else "🤖  Retrain AI Models",
                type="primary",
                key="rt_ml_run",
                disabled=_ml_running or not _ml_syms,
                use_container_width=True,
            ):
                if not _ml_syms:
                    st.warning("Please select at least one symbol.")
                else:
                    _cmd = [
                        sys.executable, "-u",
                        str(ROOT / "src" / "ml_retrain_all.py"),
                        "--symbols", *_ml_syms,
                    ]
                    ok, msg = _start_task("ml_retrain_all", _cmd)
                    if ok:
                        st.session_state["rt_ml_ran_syms"] = list(_ml_syms)
                        st.rerun()
                    else:
                        st.warning(msg)

        if _ml_running:
            if st.button("⏹  Stop Training", key="rt_ml_stop", type="secondary"):
                _stop_task("ml_retrain_all")
                st.rerun()

        _render_progress_panel("ml_retrain_all")

        _ml_log = _read_task_log("ml_retrain_all")
        if _ml_log:
            with st.expander("📄 Console Output", expanded=_ml_running):
                st.code(_ml_log[-10000:], language="text")

        # Temporary model status cards
        _ml_done_ts = _task_completed_info("ml_retrain_all")
        _ml_ran = st.session_state.get("rt_ml_ran_syms", ["ES", "NQ", "MNQ"])
        if _ml_done_ts and not _ml_running:
            st.markdown("")
            st.markdown(
                "<div style='background:#071f10;border:1px solid #57e38966;"
                "border-left:4px solid #57e389;border-radius:8px;"
                "padding:10px 16px;margin-bottom:10px;'>"
                "<b style='color:#57e389;font-size:1em;'>✅ Models Updated</b>"
                f"<span style='color:#8b949e;font-size:0.82em;margin-left:10px;'>"
                f"completed {_ml_done_ts[1]}"
                "</span></div>",
                unsafe_allow_html=True,
            )
            _ml_card_cols = st.columns(len(_ml_ran))
            for _mli, _mls in enumerate(_ml_ran):
                _mlinf = _ml_info(_mls)
                _mlc   = _SYM_COLOR.get(_mls, "#fff")
                _mlbg  = _SYM_BG.get(_mls, "#111")
                if _mlinf["found"]:
                    _ml_card_cols[_mli].markdown(
                        f"<div style='background:{_mlbg};"
                        f"border:1.5px solid {_mlc}55;"
                        f"border-top:3px solid {_mlc};"
                        f"border-radius:8px;padding:14px;text-align:center;'>"
                        f"<div style='color:{_mlc};font-weight:800;font-size:1.1em;"
                        f"margin-bottom:6px;'>{_SYM_ICON.get(_mls,'')} {_mls}</div>"
                        f"<div style='color:#57e389;font-size:1em;'>✅ Model Ready</div>"
                        f"<div style='color:#8b949e;font-size:0.78em;margin-top:4px;'>"
                        f"Trained: {_to_et(_mlinf['trained']).strftime('%b %d %I:%M %p %Z')}"
                        f"</div>"
                        f"<div style='color:#555;font-size:0.7em;'>"
                        f"{_mlinf['name']}  ({_mlinf['kb']} KB)</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    _ml_card_cols[_mli].markdown(
                        f"<div style='background:{_mlbg};"
                        f"border:1.5px solid {_mlc}33;"
                        f"border-radius:8px;padding:14px;text-align:center;opacity:0.7;'>"
                        f"<div style='color:{_mlc};font-weight:800;'>{_mls}</div>"
                        "<div style='color:#ff6b6b;'>⚠️ No Model Found</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )

    # =======================================================================
    # SECTION 3 — YouTube Strategy Scanner
    # =======================================================================

    st.markdown("")
    st.markdown("---")
    st.markdown("### 🎬 YouTube Strategy Scanner")
    st.caption(
        "Automatically downloads transcripts from YouTube videos and extracts "
        "codable trading rules: time windows, point targets, entry conditions, filter rules. "
        "Results are saved to data/strategy_notes.md."
    )

    _yt_running = _render_task_status_badge("youtube")
    yt_col1, yt_col2 = st.columns([3, 1])
    with yt_col1:
        yt_url = st.text_input(
            "Add custom YouTube URL (optional — leave blank to scan all known videos)",
            placeholder="https://www.youtube.com/watch?v=...",
            key="yt_url",
        )
    with yt_col2:
        st.markdown("")
        st.markdown("")
        if st.button(
            "⏳ Scanning…" if _yt_running else "▶️ Scan Videos",
            type="primary",
            key="yt_scan_btn",
            disabled=_yt_running,
        ):
            cmd = [sys.executable, "-u", str(ROOT / "tools" / "youtube_analyzer.py")]
            if yt_url.strip():
                cmd += ["--url", yt_url.strip()]
            ok, msg = _start_task("youtube", cmd)
            if ok:
                st.success(f"YouTube scan started! {msg}")
                time.sleep(0.5)
                st.rerun()
            else:
                st.warning(msg)

    yt_running = _is_task_running("youtube")
    _render_progress_panel("youtube")

    yt_log = _read_task_log("youtube")
    if yt_log:
        with st.expander("📄 Scan Console Output", expanded=yt_running):
            st.code(yt_log, language="text")

    notes_path = ROOT / "data" / "strategy_notes.md"
    if notes_path.exists():
        mod_time = datetime.fromtimestamp(notes_path.stat().st_mtime)
        st.markdown(
            f"#### 📋 Strategy Notes "
            f"*(last updated: {mod_time.strftime('%Y-%m-%d')} {_fmt_short_et(mod_time)})*"
        )
        with st.expander("View Full Strategy Notes", expanded=False):
            notes_content = notes_path.read_text(encoding="utf-8")
            st.markdown(notes_content)

        top_findings = []
        for line in notes_content.splitlines():
            if "Mentioned" in line and ("HIGH" in line or "VERY HIGH" in line):
                top_findings.append(line.strip("- ").strip())
        if top_findings:
            st.markdown("**Top High-Confidence Findings:**")
            for f in top_findings[:8]:
                st.markdown(f"- {f}")

    # =======================================================================
    # SECTION 4 — Improvement Planner
    # =======================================================================

    st.markdown("")
    st.markdown("---")
    st.markdown("### 💡 Improvement Suggestions")
    st.caption(
        "Describe what you want the bot to do better in plain English. "
        "This generates a targeted research checklist you can review and action."
    )

    improvement_input = st.text_area(
        "What would you like to improve?",
        placeholder=(
            "Examples:\n"
            "• I want more trades per week without reducing win rate\n"
            "• The bot is losing too often on Wednesdays\n"
            "• I want the bot to target bigger profits on strong trend days\n"
            "• My win rate is 60% but I want 70%"
        ),
        height=120,
        key="improvement_text",
    )

    if st.button("Generate Improvement Plan", key="gen_plan_btn"):
        if not improvement_input.strip():
            st.warning("Please describe what you want to improve first.")
        else:
            bt_df_ctx = _load_backtest_csv("ES", "2022-01-03", "2024-12-31")
            ctx_stats = _backtest_summary(bt_df_ctx) if not bt_df_ctx.empty else {}
            suggestions = _generate_improvement_plan(improvement_input, ctx_stats)
            st.session_state["improvement_plan"] = suggestions
            st.rerun()

    if "improvement_plan" in st.session_state:
        plan = st.session_state["improvement_plan"]
        st.markdown("#### Your Improvement Plan")
        st.markdown(plan)

        if st.button("Clear Plan", key="clear_plan"):
            del st.session_state["improvement_plan"]
            st.rerun()


# ===========================================================================
# TAB 8 — Automation
# ===========================================================================

if _active_tab == "🔄 Auto-Run":
    st.subheader("🔄 Auto-Run Center")
    st.caption(
        "Set up a continuous improvement loop that automatically runs "
        "**Retrain AI → Backtest → Walk-Forward** over and over until a chosen time. "
        "Also control the background AI re-learning daemon here."
    )

    # ── Scheduled 7:45am Bot Start ──────────────────────────────────────────
    st.markdown("#### ⏰ Scheduled Trading Bot Start")
    st.caption(
        "Pick which bots to launch automatically at **7:45 am ET** every weekday. "
        "The bot connects early, builds today's price levels, runs pre-market analysis, "
        "then begins trading at 9:00 am. "
        "The dashboard must be open and running for the schedule to fire. "
        "Only one schedule can be active at a time."
    )

    _sched_now = _load_schedule()
    _active_sched = _sched_now.get("active")
    _last_fired   = _sched_now.get("last_autostart_date")

    # Status line
    _et_now_disp = _now_et()
    _target_745 = _et_now_disp.replace(hour=7, minute=45, second=0, microsecond=0)
    _mins_to_745 = (_target_745 - _et_now_disp).total_seconds() / 60
    if _mins_to_745 < 0:
        _mins_to_745 += 1440  # roll to tomorrow

    if _active_sched:
        _sched_label = _SCHEDULE_OPTIONS.get(_active_sched, {}).get("label", _active_sched)
        if _last_fired == _et_now_disp.strftime("%Y-%m-%d"):
            st.success(
                f"✅ **{_sched_label}** already launched today "
                f"({_last_fired}). Will fire again tomorrow at 7:45 am ET."
            )
        else:
            st.info(
                f"🕘 **{_sched_label}** scheduled — fires in "
                f"**{int(_mins_to_745 // 60)}h {int(_mins_to_745 % 60)}m** "
                f"(at 7:45 am ET for pre-market)"
            )
    else:
        st.warning("No schedule active — click a button below to set one.")

    # Individual symbol buttons
    st.markdown("**Individual symbols:**")
    _ind_cols = st.columns(3)
    for _ci, _sym in enumerate(["ES", "NQ", "MNQ"]):
        _is_sel = _active_sched == _sym
        with _ind_cols[_ci]:
            _btn_type = "primary" if _is_sel else "secondary"
            _btn_label = f"{'✅ ' if _is_sel else ''}Start {_sym} at 7:45am"
            if st.button(_btn_label, key=f"sched_{_sym}", type=_btn_type, use_container_width=True):
                _new_sched = _sched_now.copy()
                if _is_sel:
                    _new_sched["active"] = None   # toggle off
                    st.toast(f"Schedule cleared.", icon="🗑️")
                else:
                    _new_sched["active"] = _sym
                    st.toast(f"{_sym} scheduled for 7:45am ET (pre-market).", icon="⏰")
                _save_schedule(_new_sched)
                st.rerun()

    # Combo buttons
    st.markdown("**Combinations (start together as one process):**")
    _combo_cols = st.columns(2)
    for _ci, _key in enumerate(["ES+NQ", "ES+NQ+MNQ"]):
        _is_sel = _active_sched == _key
        _lbl = _SCHEDULE_OPTIONS[_key]["label"]
        with _combo_cols[_ci]:
            _btn_type = "primary" if _is_sel else "secondary"
            _btn_label = f"{'✅ ' if _is_sel else ''}Start {_lbl} at 7:45am"
            if st.button(_btn_label, key=f"sched_{_key}", type=_btn_type, use_container_width=True):
                _new_sched = _sched_now.copy()
                if _is_sel:
                    _new_sched["active"] = None
                    st.toast("Schedule cleared.", icon="🗑️")
                else:
                    _new_sched["active"] = _key
                    st.toast(f"{_lbl} scheduled for 7:45am ET (pre-market).", icon="⏰")
                _save_schedule(_new_sched)
                st.rerun()

    # Cancel button
    if _active_sched:
        if st.button("🗑️ Cancel Schedule", key="sched_cancel", type="secondary"):
            _new_sched = _sched_now.copy()
            _new_sched["active"] = None
            _save_schedule(_new_sched)
            st.toast("Schedule cancelled.", icon="🗑️")
            st.rerun()

    st.markdown("---")

    # ── AI Re-Learning Daemon control ───────────────────────────────────────
    st.markdown("#### AI Re-Learning Daemon")
    st.caption(
        "This process runs quietly in the background. After the market closes, "
        "it retrains the AI on today's new trades and saves candidate models for your review."
    )
    _cl_running = _is_task_running("continuous_learner")
    _cl_col1, _cl_col2 = st.columns(2)
    with _cl_col1:
        if _cl_running:
            st.success("🟢 AI Re-Learning is running in the background")
            if st.button("Stop AI Re-Learning", key="stop_cl"):
                ok, msg = _stop_task("continuous_learner")
                st.toast(msg, icon="🛑" if ok else "⚠️")
                st.rerun()
        else:
            st.info("⚫ AI Re-Learning is stopped")
            _cl_symbols = st.multiselect(
                "Symbols to learn from",
                options=["ES", "NQ", "MNQ"],
                default=["ES", "NQ"],
                key="cl_symbols",
            )
            if st.button("Start AI Re-Learning", key="start_cl", type="primary"):
                _sym_str = " ".join(_cl_symbols)
                ok, msg = _start_task(
                    "continuous_learner",
                    [sys.executable, "src/continuous_learner.py", "--symbols"] + _cl_symbols,
                )
                st.toast(msg, icon="🤖" if ok else "⚠️")
                st.rerun()
    with _cl_col2:
        _render_progress_panel("continuous_learner")

    st.markdown("---")

    # ── Status badge ────────────────────────────────────────────────────────
    _auto_running = _render_task_status_badge("automation")
    _render_progress_panel("automation")

    st.markdown("---")

    # ── Configuration ───────────────────────────────────────────────────────
    st.markdown("### ⚙️ Configure Automation Run")

    auto_c1, auto_c2 = st.columns(2)

    with auto_c1:
        st.markdown("**Select Symbols**")
        auto_sym_es  = st.checkbox("ES  (E-mini S&P 500)",  value=True, key="auto_sym_es",  disabled=_auto_running)
        auto_sym_nq  = st.checkbox("NQ  (E-mini Nasdaq)",   value=True, key="auto_sym_nq",  disabled=_auto_running)
        auto_sym_mnq = st.checkbox("MNQ (Micro Nasdaq)",    value=True, key="auto_sym_mnq", disabled=_auto_running)
        _auto_symbols = []
        if auto_sym_es:  _auto_symbols.append("ES")
        if auto_sym_nq:  _auto_symbols.append("NQ")
        if auto_sym_mnq: _auto_symbols.append("MNQ")

        st.markdown("")
        st.markdown("**Steps to Include Each Cycle**")
        auto_step_retrain  = st.checkbox("🤖 Retrain AI Models",    value=True, key="auto_step_retrain",  disabled=_auto_running)
        auto_step_backtest = st.checkbox("📊 Run All Backtests",    value=True, key="auto_step_backtest", disabled=_auto_running)
        auto_step_wfa      = st.checkbox("🔄 Walk-Forward Tests",   value=True, key="auto_step_wfa",      disabled=_auto_running)
        _auto_steps = []
        if auto_step_retrain:  _auto_steps.append("retrain")
        if auto_step_backtest: _auto_steps.append("backtest")
        if auto_step_wfa:      _auto_steps.append("wfa")

    with auto_c2:
        st.markdown("**Backtest Date Range**")
        if "auto_bt_start" not in st.session_state:
            st.session_state["auto_bt_start"] = "2022-01-03"
        if "auto_bt_end" not in st.session_state:
            st.session_state["auto_bt_end"] = "2024-12-31"
        auto_bt_start = st.text_input(
            "Start Date (YYYY-MM-DD)", key="auto_bt_start",
            disabled=_auto_running,
        )
        auto_bt_end = st.text_input(
            "End Date (YYYY-MM-DD)", key="auto_bt_end",
            disabled=_auto_running,
        )

        st.markdown("")
        st.markdown("**Stop Automation At (US Eastern Time)**")
        _now_h = _now_et().strftime("%I").lstrip("0") or "12"
        _now_ampm = _now_et().strftime("%p")

        stop_col1, stop_col2, stop_col3 = st.columns([2, 1, 2])
        with stop_col1:
            auto_stop_hour = st.selectbox(
                "Hour", options=[str(h) for h in range(1, 13)],
                index=4,   # default 5
                key="auto_stop_hour", disabled=_auto_running,
                label_visibility="collapsed",
            )
        with stop_col2:
            auto_stop_min = st.selectbox(
                "Min", options=[f"{m:02d}" for m in range(0, 60, 5)],
                index=0,
                key="auto_stop_min", disabled=_auto_running,
                label_visibility="collapsed",
            )
        with stop_col3:
            auto_stop_ampm = st.selectbox(
                "AM/PM", options=["AM", "PM"],
                index=1,   # default PM
                key="auto_stop_ampm", disabled=_auto_running,
                label_visibility="collapsed",
            )

        # Show current ET time for reference
        st.caption(f"Current time: **{_fmt_time_et(_now_et())}**")

    # Convert 12-hour picker to 24h HH:MM for the script arg
    _stop_h12 = int(auto_stop_hour)
    _stop_m   = int(auto_stop_min)
    if auto_stop_ampm == "AM":
        _stop_h24 = 0 if _stop_h12 == 12 else _stop_h12
    else:
        _stop_h24 = 12 if _stop_h12 == 12 else _stop_h12 + 12
    _stop_at_arg = f"{_stop_h24:02d}:{_stop_m:02d}"

    # Estimate cycles (rough: each cycle ~25 min with all 3 steps all 3 symbols)
    _stop_dt_est = _now_et().replace(hour=_stop_h24, minute=_stop_m, second=0, microsecond=0)
    if _stop_dt_est <= _now_et():
        _stop_dt_est = _stop_dt_est + timedelta(days=1)
    _mins_avail = max(0, int((_stop_dt_est - _now_et()).total_seconds() / 60))
    _cycle_mins = (len(_auto_steps) if _auto_steps else 1) * max(len(_auto_symbols), 1) * 5 + 2
    _est_cycles = max(1, _mins_avail // _cycle_mins) if _cycle_mins > 0 else "?"

    # ── Summary row ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Automation Summary")

    if _auto_symbols and _auto_steps:
        steps_label = " → ".join({
            "retrain": "🤖 Retrain AI",
            "backtest": "📊 Backtest",
            "wfa":      "🔄 Walk-Forward",
        }[s] for s in _auto_steps)
        st.info(
            f"**Symbols:** {', '.join(_auto_symbols)}  \n"
            f"**Each cycle:** {steps_label}  \n"
            f"**Stop at:** {auto_stop_hour}:{auto_stop_min} {auto_stop_ampm} ET  \n"
            f"**Time available:** ~{_mins_avail} min  →  estimated **~{_est_cycles} cycle(s)**"
        )
    else:
        st.warning("Select at least one symbol and one step to start automation.")

    # ── Start / Stop buttons ─────────────────────────────────────────────────
    btn_col1, btn_col2 = st.columns([2, 1])

    with btn_col1:
        _can_start = bool(_auto_symbols) and bool(_auto_steps) and not _auto_running
        if st.button(
            "⏳ Already Running…" if _auto_running else "▶️  Start Automation Loop",
            type="primary",
            key="auto_start_btn",
            disabled=not _can_start,
            use_container_width=True,
        ):
            _auto_cmd = [
                sys.executable, "-u",
                str(ROOT / "backtest" / "run_automation.py"),
                "--symbols", *_auto_symbols,
                "--steps",   *_auto_steps,
                "--start",   auto_bt_start,
                "--end",     auto_bt_end,
                "--stop-at", _stop_at_arg,
            ]
            _ok, _msg = _start_task("automation", _auto_cmd)
            if _ok:
                st.success(f"Automation loop started! Will cycle until {auto_stop_hour}:{auto_stop_min} {auto_stop_ampm} ET.")
                time.sleep(0.5)
                st.rerun()
            else:
                st.warning(_msg)

    with btn_col2:
        if _auto_running:
            if st.button("⏹ Stop Automation", key="auto_stop_btn", type="secondary", use_container_width=True):
                _stop_task("automation")
                st.success("Automation loop stopped.")
                time.sleep(0.3)
                st.rerun()

    st.markdown("---")

    # ── Live output ──────────────────────────────────────────────────────────
    st.markdown("### 📄 Automation Console Output")
    _auto_log = _read_task_log("automation")
    if _auto_log:
        with st.expander("📄 Automation Log", expanded=_auto_running):
            st.code(_auto_log, language="text")
    else:
        st.caption("No automation output yet. Click **Start Automation Loop** above.")

    # ── Tips ─────────────────────────────────────────────────────────────────
    with st.expander("💡 Tips & Recommended Settings"):
        st.markdown("""
**Recommended overnight automation run:**
- ✅ Enable all 3 symbols + all 3 steps
- ⏰ Set stop time to **6:00 AM ET** (before market open)
- 📅 Leave dates as 2022-01-03 → 2024-12-31 for full backtest coverage

**What each step does:**
| Step | What happens | Time |
|---|---|---|
| 🤖 Retrain AI | Retrains the ML filter for each symbol using the latest trade data | ~8 min |
| 📊 Backtest | Full 3-year backtest for all 3 symbols | ~3 min |
| 🔄 Walk-Forward | Out-of-sample validation for all 3 symbols | ~15 min |

**After automation finishes:**
1. Go to the **Research & Testing** tab to review backtest results
2. Compare backtest vs walk-forward numbers — they should be within 20–30% of each other
3. If they diverge, the AI may be overfitting — run another automation cycle

**Pro tip:** Run automation overnight before a live trading day so the AI model is always fresh.
        """)


# ===========================================================================
# TAB 9 — Live Console
# ===========================================================================

if _active_tab == "🖥️ Technical Log":
    # -----------------------------------------------------------------------
    # Header bar — source selector, filters, controls (all inline)
    # -----------------------------------------------------------------------
    st.markdown(
        """
        <style>
        .console-container {
            background: #1e1e1e;
            border: 1px solid #3c3c3c;
            border-radius: 6px;
            padding: 12px 16px;
            max-height: 620px;
            overflow-y: auto;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 12px;
            line-height: 1.55;
        }
        .console-header {
            background: #2d2d2d;
            border-bottom: 1px solid #3c3c3c;
            padding: 6px 12px;
            border-radius: 6px 6px 0 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .console-dot { width:12px; height:12px; border-radius:50%; display:inline-block; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Top control row
    cc1, cc2, cc3, cc4, cc5 = st.columns([3, 2, 2, 1, 1])

    all_sources = _all_log_sources()
    source_options = ["All Sources"] + list(all_sources.keys())
    with cc1:
        selected_source = st.selectbox(
            "Log Source",
            source_options,
            key="console_source",
            label_visibility="collapsed",
        )
    with cc2:
        level_filter = st.selectbox(
            "Min Level",
            ["ALL", "DEBUG", "INFO", "WARNING", "ERROR"],
            index=1,   # default: INFO and above
            key="console_level",
            label_visibility="collapsed",
        )
    with cc3:
        line_count = st.slider(
            "Lines",
            min_value=50, max_value=500, value=200, step=50,
            key="console_lines",
            label_visibility="collapsed",
        )
    with cc4:
        wrap_lines = st.toggle("Wrap", value=True, key="console_wrap")
    with cc5:
        if st.button("Clear", key="console_clear"):
            # Clear all log files for the selected source
            if selected_source == "All Sources":
                for p in all_sources.values():
                    try:
                        if p.exists():
                            p.write_text("", encoding="utf-8")
                    except Exception:
                        pass
            else:
                p = all_sources.get(selected_source)
                if p and p.exists():
                    try:
                        p.write_text("", encoding="utf-8")
                    except Exception:
                        pass
            st.rerun()

    # -----------------------------------------------------------------------
    # Status indicator row
    # -----------------------------------------------------------------------
    status_parts = []
    for sym in SYMBOLS:
        running = _is_bot_running(sym)
        dot = "🟢" if running else "⚫"
        status_parts.append(f"{dot} Bot {sym}")
    for key in TASKS:
        if _is_task_running(key):
            status_parts.append(f"⏳ {TASKS[key]['label']}")

    st.caption("  |  ".join(status_parts) if status_parts else "No processes running")

    # -----------------------------------------------------------------------
    # Gather log lines
    # -----------------------------------------------------------------------
    if selected_source == "All Sources":
        active_paths = [p for p in all_sources.values() if p.exists() and p.stat().st_size > 0]
        raw_lines = _merge_log_lines(active_paths, last_n=line_count)
    else:
        log_path = all_sources.get(selected_source, Path("/dev/null"))
        raw_lines = _read_log_lines(log_path, last_n=line_count)

    # Level filter
    _LEVEL_PRIORITY = {"TRACE": 0, "DEBUG": 1, "INFO": 2, "SUCCESS": 2,
                       "WARNING": 3, "ERROR": 4, "CRITICAL": 5}
    min_priority = _LEVEL_PRIORITY.get(level_filter, 0)

    if level_filter != "ALL":
        filtered_lines: list[str] = []
        for line in raw_lines:
            clean = _strip_ansi(line)
            line_priority = -1
            for lvl, pri in _LEVEL_PRIORITY.items():
                if f"| {lvl}" in clean or f"|{lvl}" in clean:
                    line_priority = pri
                    break
            # Include line if it matches the level OR has no level tag (continuation lines)
            if line_priority == -1 or line_priority >= min_priority:
                filtered_lines.append(line)
        raw_lines = filtered_lines

    # -----------------------------------------------------------------------
    # Render terminal window
    # -----------------------------------------------------------------------
    wrap_style = "white-space:pre-wrap;word-break:break-all;" if wrap_lines else "white-space:pre;overflow-x:auto;"

    # macOS-style title bar
    source_label = selected_source if selected_source != "All Sources" else "All Sources (merged)"
    any_running = any(_is_bot_running(s) for s in SYMBOLS) or any(_is_task_running(k) for k in TASKS)
    pulse = " ● LIVE" if any_running else ""
    title_color = "#26a269" if any_running else "#858585"

    st.markdown(
        f"""
        <div style="background:#2d2d2d;border:1px solid #3c3c3c;
             border-radius:8px 8px 0 0;padding:7px 14px;
             display:flex;align-items:center;gap:8px;">
          <span style="background:#ff5f56;width:12px;height:12px;border-radius:50%;display:inline-block;"></span>
          <span style="background:#ffbd2e;width:12px;height:12px;border-radius:50%;display:inline-block;"></span>
          <span style="background:#27c93f;width:12px;height:12px;border-radius:50%;display:inline-block;"></span>
          <span style="margin-left:8px;color:#858585;font-size:12px;font-family:Consolas,monospace;">
            ChargedUp Profits Bot — {source_label}
          </span>
          <span style="margin-left:auto;color:{title_color};font-size:11px;font-family:Consolas,monospace;font-weight:bold;">
            {pulse}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not raw_lines:
        st.markdown(
            f"""
            <div style="background:#1e1e1e;border:1px solid #3c3c3c;border-top:none;
                 border-radius:0 0 8px 8px;padding:24px 16px;text-align:center;">
              <span style="color:#555;font-family:Consolas,monospace;font-size:13px;">
                No log output yet.<br/>
                Start a bot from the sidebar or run a backtest from the Research & Testing tab.
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        html_content = _colorize_log_html(raw_lines)
        st.markdown(
            f"""
            <div id="console-box" style="background:#1e1e1e;border:1px solid #3c3c3c;
                 border-top:none;border-radius:0 0 8px 8px;
                 padding:12px 16px;max-height:580px;overflow-y:auto;{wrap_style}">
              {html_content}
            </div>
            <script>
              // Auto-scroll to bottom
              var box = document.getElementById("console-box");
              if (box) {{ box.scrollTop = box.scrollHeight; }}
            </script>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            f"{len(raw_lines)} lines shown  |  "
            f"Source: {selected_source}  |  "
            f"Filter: {level_filter}  |  "
            f"Auto-refreshes every 15s"
        )

    # -----------------------------------------------------------------------
    # Quick-access: open log files in a download link
    # -----------------------------------------------------------------------
    with st.expander("📥 Download Log Files"):
        for name, path in all_sources.items():
            if path.exists() and path.stat().st_size > 0:
                content = path.read_text(encoding="utf-8", errors="replace")
                st.download_button(
                    f"Download {name}",
                    content,
                    file_name=path.name,
                    mime="text/plain",
                    key=f"dl_log_{path.stem}",
                )

    # -----------------------------------------------------------------------
    # Console-refresh rate override for this tab
    # -----------------------------------------------------------------------
    # When any process is running, refresh every 5s instead of 15s
    if any_running:
        if "console_fast_refresh" not in st.session_state:
            st.session_state.console_fast_refresh = time.time()
        if time.time() - st.session_state.console_fast_refresh > 5:
            st.session_state.console_fast_refresh = time.time()
            st.rerun()


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.caption("ChargedUp Profits Bot — Control Panel")
with col_f2:
    st.caption(f"Settings: config/settings.yaml | DB: data/audit.db")
with col_f3:
    st.caption(f"Last render: {_fmt_time_et(_now_et())} | Auto-refresh: {_refresh_interval}s")


# ---------------------------------------------------------------------------
# Auto-refresh — placed AFTER all button handlers so user clicks are never
# "stolen" by a premature st.rerun().
# ---------------------------------------------------------------------------
if time.time() - st.session_state.last_refresh > _refresh_interval:
    st.session_state.last_refresh = time.time()
    st.cache_data.clear()
    st.rerun()
