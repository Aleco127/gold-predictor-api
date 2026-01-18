#!/usr/bin/env python
"""
Update Historical Data Script
=============================
Downloads and updates historical price data for backtesting.

Usage:
    python scripts/update_historical.py [--symbol XAUUSD] [--timeframe M5] [--days 365]

Schedule this script to run daily via cron or Windows Task Scheduler.
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from src.config import settings
from src.data.capital_connector import CapitalConnector
from src.storage.historical_data import HistoricalDataStore, download_full_history


def setup_logging():
    """Configure logging."""
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f"historical_update_{datetime.now().strftime('%Y%m%d')}.log"

    logger.add(
        log_file,
        rotation="1 day",
        retention="30 days",
        level="INFO",
    )


def create_data_connector():
    """Create and connect to data source."""
    logger.info("Connecting to Capital.com API...")

    connector = CapitalConnector(
        api_key=settings.capital_api_key,
        password=settings.capital_password,
        identifier=settings.capital_identifier or settings.capital_api_key,
        demo=settings.capital_demo,
        symbol=settings.symbol,
    )
    connector.connect()

    logger.info("Connected to Capital.com")
    return connector


def progress_callback(current: int, total: int, message: str):
    """Progress callback for download."""
    percent = (current / total) * 100
    logger.info(f"[{percent:.1f}%] {message}")


def download_initial_data(
    connector,
    symbol: str,
    timeframe: str,
    days: int,
):
    """Download initial historical data."""
    logger.info(f"Downloading {days} days of {symbol} {timeframe} data...")

    store = HistoricalDataStore(
        data_dir=project_root / "data" / "historical",
        data_connector=connector,
    )

    # Download with progress
    df = store.download_historical(
        symbol=symbol,
        timeframe=timeframe,
        days=days,
        progress_callback=progress_callback,
    )

    # Save to parquet
    file_path = store.save(df, symbol, timeframe)

    # Get info
    info = store.get_data_info(symbol, timeframe)

    logger.info(f"Download complete: {info.total_bars} bars saved to {file_path}")
    logger.info(f"Data range: {info.start_date} to {info.end_date}")

    return info


def update_existing_data(
    connector,
    symbol: str,
    timeframe: str,
    update_days: int = 7,
):
    """Update existing data with recent bars."""
    logger.info(f"Updating {symbol} {timeframe} with last {update_days} days...")

    store = HistoricalDataStore(
        data_dir=project_root / "data" / "historical",
        data_connector=connector,
    )

    # Check if data exists
    info = store.get_data_info(symbol, timeframe)

    if info is None:
        logger.warning(f"No existing data for {symbol} {timeframe}. Running full download...")
        return download_initial_data(connector, symbol, timeframe, 365)

    logger.info(f"Existing data: {info.total_bars} bars, last: {info.end_date}")

    # Update
    new_bars, total_bars = store.update(
        symbol=symbol,
        timeframe=timeframe,
        days=update_days,
    )

    # Get updated info
    updated_info = store.get_data_info(symbol, timeframe)

    logger.info(f"Update complete: +{new_bars} new bars, total: {total_bars}")
    logger.info(f"Data range: {updated_info.start_date} to {updated_info.end_date}")

    return updated_info


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download and update historical price data",
    )
    parser.add_argument(
        "--symbol",
        default="XAUUSD",
        help="Trading symbol (default: XAUUSD)",
    )
    parser.add_argument(
        "--timeframe",
        default="M5",
        choices=["M1", "M5", "M15", "M30", "H1", "H4", "D1"],
        help="Timeframe (default: M5)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Days of history for initial download (default: 365)",
    )
    parser.add_argument(
        "--update-days",
        type=int,
        default=7,
        help="Days to fetch for updates (default: 7)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Force full download (not incremental update)",
    )
    parser.add_argument(
        "--multiple-timeframes",
        action="store_true",
        help="Download M5, M15, H1, H4 (for multi-timeframe analysis)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    logger.info("=" * 60)
    logger.info("Historical Data Update Script")
    logger.info(f"Started at: {datetime.now(timezone.utc).isoformat()}")
    logger.info("=" * 60)

    try:
        # Connect to data source
        connector = create_data_connector()

        # Determine timeframes to process
        if args.multiple_timeframes:
            timeframes = ["M5", "M15", "H1", "H4"]
        else:
            timeframes = [args.timeframe]

        # Process each timeframe
        for tf in timeframes:
            logger.info(f"\nProcessing {args.symbol} {tf}...")

            if args.full:
                # Full download
                download_initial_data(connector, args.symbol, tf, args.days)
            else:
                # Incremental update
                update_existing_data(connector, args.symbol, tf, args.update_days)

        logger.info("\n" + "=" * 60)
        logger.info("Update completed successfully!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error during update: {e}")
        raise
    finally:
        # Disconnect
        if 'connector' in locals():
            connector.disconnect()
            logger.info("Disconnected from data source")


if __name__ == "__main__":
    main()
