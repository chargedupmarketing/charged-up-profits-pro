"""
download_historical.py

Downloads 1-minute and 15-minute OHLCV bars for ES and MES futures
from Databento and saves them as Parquet files in data/historical/.

Usage:
    python data/download_historical.py --start 2022-01-01 --end 2024-12-31

Prerequisites:
    Set environment variable DATABENTO_API_KEY or add to .env
    pip install databento python-dotenv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HIST_DIR = ROOT / "data" / "historical"
HIST_DIR.mkdir(parents=True, exist_ok=True)

# Instruments to download
# Format: [ROOT].[ROLL_RULE].[RANK]  — required by Databento continuous symbology
# .c = calendar roll, .0 = front month
INSTRUMENTS = {
    "ES": "ES.c.0",    # Continuous front-month ES
    "NQ": "NQ.c.0",    # Continuous front-month NQ
    "MES": "MES.c.0",  # Micro ES
    "MNQ": "MNQ.c.0",  # Micro NQ
}

DATASET = "GLBX.MDP3"   # CME Globex


def download_bars(
    symbol_key: str,
    symbol: str,
    start: str,
    end: str,
    schema: str = "ohlcv-1m",
) -> pd.DataFrame:
    """Download OHLCV bars from Databento and return as a DataFrame."""
    try:
        import databento as db
    except ImportError:
        logger.error("databento package not installed. Run: pip install databento")
        sys.exit(1)

    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        logger.error("DATABENTO_API_KEY not set. Add it to .env or set as environment variable.")
        sys.exit(1)

    client = db.Historical(api_key)
    logger.info("Downloading {} ({}) schema={} from {} to {}", symbol_key, symbol, schema, start, end)

    data = client.timeseries.get_range(
        dataset=DATASET,
        symbols=[symbol],
        schema=schema,
        start=start,
        end=end,
        stype_in="continuous",   # Required for root-symbol continuous contract resolution
    )
    df = data.to_df()

    # Standardise columns to lowercase OHLCV
    df.columns = [c.lower() for c in df.columns]
    rename_map = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    df = df[[c for c in rename_map if c in df.columns]]
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"
    return df


def resample_to_15min(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-minute OHLCV to 15-minute bars."""
    return df_1m.resample("15min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()


def save_bars(df: pd.DataFrame, symbol_key: str, schema: str) -> Path:
    fname = HIST_DIR / f"{symbol_key}_{schema}.parquet"
    df.to_parquet(fname)
    logger.info("Saved {} rows to {}", len(df), fname)
    return fname


def main() -> None:
    parser = argparse.ArgumentParser(description="Download historical futures bars from Databento")
    parser.add_argument("--start", default="2022-01-03", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2024-12-31", help="End date YYYY-MM-DD")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["ES", "NQ"],
        choices=list(INSTRUMENTS.keys()),
        help="Which instruments to download",
    )
    args = parser.parse_args()

    for key in args.symbols:
        symbol = INSTRUMENTS[key]

        # Download 1-minute bars
        df_1m = download_bars(key, symbol, args.start, args.end, schema="ohlcv-1m")
        save_bars(df_1m, key, "ohlcv-1m")

        # Derive 15-minute bars
        df_15m = resample_to_15min(df_1m)
        save_bars(df_15m, key, "ohlcv-15m")

    logger.info("Download complete. Files in {}", HIST_DIR)


if __name__ == "__main__":
    main()
