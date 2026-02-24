"""
backtest/wf_all.py

Run walk-forward (out-of-sample) tests for ALL symbols (ES -> NQ -> MNQ)
in sequence.

Usage:
    python backtest/wf_all.py --start 2022-01-03 --end 2024-12-31

Prints structured progress markers that the web panel can parse:
    WF_ALL_PROGRESS 1/3 (ES)
    WF_ALL_PROGRESS 2/3 (NQ)
    WF_ALL_PROGRESS 3/3 (MNQ)
    WF_ALL_DONE -- all 3 symbols complete
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Force UTF-8 output on Windows so unicode characters don't crash cp1252 consoles.
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

ROOT = Path(__file__).resolve().parent.parent
WF_SCRIPT = ROOT / "backtest" / "walk_forward.py"

SYMBOLS = ["ES", "NQ", "MNQ"]


def run_symbol(symbol: str, start: str, end: str) -> int:
    """Run walk_forward.py for one symbol; stream output live. Returns exit code."""
    cmd = [
        sys.executable, "-u", str(WF_SCRIPT),
        "--symbol", symbol,
        "--start", start,
        "--end", end,
    ]
    print(f"\n{'='*60}", flush=True)
    print(f"  Walk-Forward: {symbol}  ({start} -> {end})", flush=True)
    print(f"{'='*60}\n", flush=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(ROOT),
    )
    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.decode("utf-8", errors="replace").rstrip()
        print(line, flush=True)

    proc.wait()
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run walk-forward tests for selected symbols in sequence."
    )
    parser.add_argument("--start",   default="2022-01-03", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",     default="2024-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS,
                        choices=SYMBOLS, help="Symbols to run (default: all three)")
    args = parser.parse_args()

    run_syms = args.symbols
    results: dict[str, int] = {}
    total = len(run_syms)

    for idx, symbol in enumerate(run_syms, start=1):
        # Marker parsed by the dashboard progress tracker
        print(f"\nWF_ALL_PROGRESS {idx}/{total} ({symbol})", flush=True)
        rc = run_symbol(symbol, args.start, args.end)
        results[symbol] = rc
        status = "OK" if rc == 0 else f"exit {rc}"
        print(f"\n[wf_all] {symbol} finished -- {status}", flush=True)

    # Summary
    print("\n" + "="*60, flush=True)
    print("  ALL WALK-FORWARD TESTS COMPLETE", flush=True)
    print("="*60, flush=True)
    for sym, rc in results.items():
        status = "PASS" if rc == 0 else f"FAIL (exit {rc})"
        print(f"  {sym:>4}: {status}", flush=True)
    print("WF_ALL_DONE", flush=True)

    if any(rc != 0 for rc in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
