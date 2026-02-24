"""
src/ml_retrain_all.py

Retrain the ML (AI) filter for ALL symbols (ES -> NQ -> MNQ) in sequence.

Usage:
    python src/ml_retrain_all.py

Prints structured progress markers that the web panel can parse:
    ML_ALL_PROGRESS 1/3 (ES)
    ML_ALL_PROGRESS 2/3 (NQ)
    ML_ALL_PROGRESS 3/3 (MNQ)
    ML_ALL_DONE -- all 3 symbols complete
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Force UTF-8 output on Windows so unicode characters don't crash cp1252 consoles.
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

ROOT = Path(__file__).resolve().parent.parent
ML_FILTER = ROOT / "src" / "ml_filter.py"

SYMBOLS = ["ES", "NQ", "MNQ"]


def retrain_symbol(symbol: str) -> int:
    """Run ml_filter.py retrain for one symbol; stream output live. Returns exit code."""
    cmd = [
        sys.executable, "-u", str(ML_FILTER),
        "retrain",
        "--symbol", symbol,
        "--no-shap",
    ]
    print(f"\n{'='*60}", flush=True)
    print(f"  Retraining ML model for {symbol}", flush=True)
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
    import argparse
    parser = argparse.ArgumentParser(description="Retrain ML models for selected symbols.")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS,
                        choices=SYMBOLS, help="Symbols to retrain (default: all three)")
    args = parser.parse_args()

    run_syms = args.symbols
    results: dict[str, int] = {}
    total = len(run_syms)

    for idx, symbol in enumerate(run_syms, start=1):
        # Marker parsed by the dashboard progress tracker
        print(f"\nML_ALL_PROGRESS {idx}/{total} ({symbol})", flush=True)
        rc = retrain_symbol(symbol)
        results[symbol] = rc
        status = "OK" if rc == 0 else f"exit {rc}"
        print(f"\n[ml_retrain_all] {symbol} finished -- {status}", flush=True)

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("  ALL ML RETRAINS COMPLETE", flush=True)
    print("=" * 60, flush=True)
    for sym, rc in results.items():
        status = "PASS" if rc == 0 else f"FAIL (exit {rc})"
        print(f"  {sym:>4}: {status}", flush=True)
    print("ML_ALL_DONE", flush=True)

    if any(rc != 0 for rc in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
