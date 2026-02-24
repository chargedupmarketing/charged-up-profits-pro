"""
backtest/run_automation.py

Continuous improvement automation loop.

Runs the following sequence for each selected symbol, cycling repeatedly
until a specified stop time (US Eastern) is reached:
  1. Retrain AI models
  2. Run full backtest
  3. Run walk-forward test

Usage:
    python backtest/run_automation.py
        --symbols ES NQ MNQ
        --start 2022-01-03
        --end 2024-12-31
        --stop-at "17:00"          # Stop after this ET time (24h format internally)
        --steps retrain backtest wfa  # Which steps to include

Progress markers (parsed by dashboard):
    AUTO_CYCLE_START  <n>
    AUTO_STEP_START   <step> <symbol>
    AUTO_STEP_DONE    <step> <symbol>
    AUTO_CYCLE_DONE   <n>
    AUTO_FINISHED     (stop time reached or max cycles done)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# Force UTF-8 on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore

ROOT = Path(__file__).resolve().parent.parent
ET = ZoneInfo("America/New_York")

SCRIPTS = {
    "retrain": ROOT / "src"      / "ml_retrain_all.py",
    "backtest": ROOT / "backtest" / "run_all.py",
    "wfa":      ROOT / "backtest" / "wf_all.py",
}

STEP_LABELS = {
    "retrain":  "Retrain AI Models",
    "backtest": "Run All Backtests",
    "wfa":      "Walk-Forward Tests",
}


def _now_et() -> datetime:
    return datetime.now(tz=ET)


def _fmt_et(dt: datetime) -> str:
    """Format a datetime as 12-hour Eastern time string."""
    return dt.strftime("%I:%M:%S %p ET").lstrip("0")


def _run_step(step: str, symbols: list[str], start: str, end: str) -> int:
    """Run one automation step. Returns exit code."""
    script = SCRIPTS[step]
    cmd = [sys.executable, "-u", str(script)]

    if step in ("backtest", "wfa"):
        cmd += ["--start", start, "--end", end]
    # retrain_all.py doesn't take extra args; symbol filtering handled internally

    print(f"\n{'─'*60}", flush=True)
    print(f"  [{_fmt_et(_now_et())}]  Starting: {STEP_LABELS[step]}", flush=True)
    print(f"{'─'*60}", flush=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(ROOT),
    )
    assert proc.stdout is not None
    for raw in proc.stdout:
        print(raw.decode("utf-8", errors="replace").rstrip(), flush=True)
    proc.wait()
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous bot improvement automation")
    parser.add_argument(
        "--symbols", nargs="+", default=["ES", "NQ", "MNQ"],
        choices=["ES", "NQ", "MNQ"],
        help="Symbols to include (default: all three)",
    )
    parser.add_argument("--start", default="2022-01-03", help="Backtest start date")
    parser.add_argument("--end",   default="2024-12-31", help="Backtest end date")
    parser.add_argument(
        "--stop-at", default="23:59",
        help="Stop after this Eastern time (HH:MM, 24h). Default: 23:59",
    )
    parser.add_argument(
        "--steps", nargs="+",
        default=["retrain", "backtest", "wfa"],
        choices=["retrain", "backtest", "wfa"],
        help="Which steps to run in each cycle",
    )
    args = parser.parse_args()

    # Parse stop time
    stop_h, stop_m = map(int, args.stop_at.split(":"))
    now_et   = _now_et()
    stop_dt  = now_et.replace(hour=stop_h, minute=stop_m, second=0, microsecond=0)
    if stop_dt <= now_et:
        stop_dt += timedelta(days=1)  # If already past, next day

    symbols_str = ", ".join(args.symbols)
    steps_str   = " -> ".join(STEP_LABELS[s] for s in args.steps)

    print("=" * 60, flush=True)
    print("  ChargedUp Profits Bot — Automation Loop", flush=True)
    print("=" * 60, flush=True)
    print(f"  Symbols   : {symbols_str}", flush=True)
    print(f"  Steps     : {steps_str}", flush=True)
    print(f"  Start     : {_fmt_et(now_et)}", flush=True)
    print(f"  Stop at   : {_fmt_et(stop_dt)}", flush=True)
    print("=" * 60, flush=True)

    cycle = 0
    while True:
        now_et = _now_et()
        if now_et >= stop_dt:
            print(f"\nAUTO_FINISHED  Stop time {_fmt_et(stop_dt)} reached — exiting.", flush=True)
            break

        cycle += 1
        remaining = stop_dt - now_et
        hrs, rem  = divmod(int(remaining.total_seconds()), 3600)
        mins      = rem // 60

        print(f"\nAUTO_CYCLE_START {cycle}", flush=True)
        print(f"\n{'='*60}", flush=True)
        print(f"  CYCLE {cycle}  —  {_fmt_et(now_et)}  "
              f"({hrs}h {mins}m remaining)", flush=True)
        print(f"{'='*60}", flush=True)

        for step in args.steps:
            now_et = _now_et()
            if now_et >= stop_dt:
                print(f"\n  Stop time reached mid-cycle — aborting cycle {cycle}.", flush=True)
                print("AUTO_FINISHED", flush=True)
                sys.exit(0)

            print(f"\nAUTO_STEP_START {step} {','.join(args.symbols)}", flush=True)
            rc = _run_step(step, args.symbols, args.start, args.end)
            status = "OK" if rc == 0 else f"exit {rc}"
            print(f"\nAUTO_STEP_DONE {step} {status}", flush=True)

        print(f"\nAUTO_CYCLE_DONE {cycle}", flush=True)

        # Brief pause between cycles before checking time again
        now_et = _now_et()
        if now_et < stop_dt:
            print(f"\n  Cycle {cycle} complete at {_fmt_et(now_et)}.  "
                  f"Next cycle starting in 10 seconds...", flush=True)
            time.sleep(10)

    print("\nAUTO_FINISHED", flush=True)


if __name__ == "__main__":
    main()
