"""
Historical Data Storage Module
==============================
Stores and manages historical price data in Parquet format for backtesting.
"""

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Default storage path
DEFAULT_DATA_DIR = Path("data/historical")

# Supported timeframes
SUPPORTED_TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]


@dataclass
class DataRangeInfo:
    """Information about stored data range."""
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    total_bars: int
    file_path: str
    file_size_mb: float
    last_updated: datetime

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "total_bars": self.total_bars,
            "file_path": self.file_path,
            "file_size_mb": round(self.file_size_mb, 2),
            "last_updated": self.last_updated.isoformat(),
            "days_of_data": (self.end_date - self.start_date).days,
        }


class HistoricalDataStore:
    """
    Store and manage historical price data in Parquet format.

    Features:
    - Efficient columnar storage with Parquet
    - Incremental updates (append new data)
    - Fast date range queries
    - Automatic compression
    - Support for multiple symbols and timeframes
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        data_connector: Optional[Any] = None,
    ):
        """
        Initialize historical data store.

        Args:
            data_dir: Directory to store parquet files
            data_connector: Data connector for fetching new data
        """
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.data_connector = data_connector

        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Historical data store initialized at {self.data_dir}")

    def _get_file_path(self, symbol: str, timeframe: str) -> Path:
        """Get parquet file path for symbol/timeframe."""
        return self.data_dir / f"{symbol}_{timeframe}.parquet"

    def download_historical(
        self,
        symbol: str = "XAUUSD",
        timeframe: str = "M5",
        days: int = 365,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> pd.DataFrame:
        """
        Download historical data from data connector.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (M1, M5, M15, M30, H1, H4, D1)
            days: Number of days to download
            progress_callback: Optional callback for progress updates

        Returns:
            Downloaded DataFrame
        """
        if self.data_connector is None:
            raise ValueError("Data connector not configured")

        if timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Use: {SUPPORTED_TIMEFRAMES}")

        logger.info(f"Downloading {days} days of {symbol} {timeframe} data...")

        # Download data in chunks to handle large requests
        chunk_size = 30  # Days per chunk
        all_data = []

        total_chunks = (days + chunk_size - 1) // chunk_size
        for i in range(total_chunks):
            chunk_days = min(chunk_size, days - i * chunk_size)
            offset_days = i * chunk_size

            if progress_callback:
                progress_callback(i + 1, total_chunks, f"Downloading chunk {i + 1}/{total_chunks}")

            try:
                # Calculate date range for this chunk
                end_date = datetime.now(timezone.utc) - timedelta(days=offset_days)
                start_date = end_date - timedelta(days=chunk_days)

                # Fetch data from connector
                df = self.data_connector.get_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    days=chunk_days,
                    use_cache=False,
                )

                if df is not None and not df.empty:
                    all_data.append(df)
                    logger.debug(f"Downloaded {len(df)} bars for chunk {i + 1}")

            except Exception as e:
                logger.warning(f"Error downloading chunk {i + 1}: {e}")
                continue

        if not all_data:
            raise ValueError(f"No data downloaded for {symbol} {timeframe}")

        # Combine all chunks
        combined_df = pd.concat(all_data, ignore_index=True)

        # Remove duplicates and sort by time
        if 'time' in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=['time'])
            combined_df = combined_df.sort_values('time').reset_index(drop=True)
        elif 'datetime' in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=['datetime'])
            combined_df = combined_df.sort_values('datetime').reset_index(drop=True)

        logger.info(f"Downloaded {len(combined_df)} total bars for {symbol} {timeframe}")

        return combined_df

    def save(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        append: bool = False,
    ) -> Path:
        """
        Save DataFrame to Parquet file.

        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            timeframe: Timeframe
            append: If True, append to existing file

        Returns:
            Path to saved file
        """
        file_path = self._get_file_path(symbol, timeframe)

        if append and file_path.exists():
            # Load existing data
            existing_df = self.load(symbol, timeframe)

            # Combine with new data
            combined_df = pd.concat([existing_df, df], ignore_index=True)

            # Remove duplicates
            time_col = 'time' if 'time' in combined_df.columns else 'datetime'
            if time_col in combined_df.columns:
                combined_df = combined_df.drop_duplicates(subset=[time_col])
                combined_df = combined_df.sort_values(time_col).reset_index(drop=True)

            df = combined_df

        # Save to parquet with compression
        df.to_parquet(
            file_path,
            engine='pyarrow',
            compression='snappy',
            index=False,
        )

        logger.info(f"Saved {len(df)} bars to {file_path}")

        return file_path

    def load(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load historical data from Parquet file.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Optional start date filter
            end_date: Optional end date filter
            columns: Optional list of columns to load

        Returns:
            DataFrame with historical data
        """
        file_path = self._get_file_path(symbol, timeframe)

        if not file_path.exists():
            raise FileNotFoundError(f"No data found for {symbol} {timeframe}")

        # Load with optional column selection
        df = pd.read_parquet(file_path, columns=columns)

        # Apply date filters if provided
        time_col = 'time' if 'time' in df.columns else 'datetime'
        if time_col in df.columns:
            # Ensure datetime type
            if df[time_col].dtype == 'object':
                df[time_col] = pd.to_datetime(df[time_col])

            if start_date:
                if start_date.tzinfo is None:
                    start_date = start_date.replace(tzinfo=timezone.utc)
                # Handle both timezone-aware and naive timestamps
                if df[time_col].dt.tz is None:
                    df = df[df[time_col] >= start_date.replace(tzinfo=None)]
                else:
                    df = df[df[time_col] >= start_date]

            if end_date:
                if end_date.tzinfo is None:
                    end_date = end_date.replace(tzinfo=timezone.utc)
                if df[time_col].dt.tz is None:
                    df = df[df[time_col] <= end_date.replace(tzinfo=None)]
                else:
                    df = df[df[time_col] <= end_date]

        return df

    def update(
        self,
        symbol: str = "XAUUSD",
        timeframe: str = "M5",
        days: int = 7,
    ) -> Tuple[int, int]:
        """
        Update existing data with recent bars.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            days: Days of recent data to fetch

        Returns:
            Tuple of (new_bars_added, total_bars)
        """
        if self.data_connector is None:
            raise ValueError("Data connector not configured")

        file_path = self._get_file_path(symbol, timeframe)

        # Get existing bar count
        existing_bars = 0
        if file_path.exists():
            existing_df = self.load(symbol, timeframe)
            existing_bars = len(existing_df)

        # Download recent data
        new_df = self.download_historical(
            symbol=symbol,
            timeframe=timeframe,
            days=days,
        )

        # Save with append mode
        self.save(new_df, symbol, timeframe, append=True)

        # Get new total
        updated_df = self.load(symbol, timeframe)
        total_bars = len(updated_df)
        new_bars = total_bars - existing_bars

        logger.info(f"Updated {symbol} {timeframe}: +{new_bars} bars, total: {total_bars}")

        return new_bars, total_bars

    def get_data_info(
        self,
        symbol: str,
        timeframe: str,
    ) -> Optional[DataRangeInfo]:
        """
        Get information about stored data.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            DataRangeInfo or None if no data exists
        """
        file_path = self._get_file_path(symbol, timeframe)

        if not file_path.exists():
            return None

        # Load data to get range
        df = self.load(symbol, timeframe)

        if df.empty:
            return None

        time_col = 'time' if 'time' in df.columns else 'datetime'

        # Ensure datetime type
        if df[time_col].dtype == 'object':
            df[time_col] = pd.to_datetime(df[time_col])

        start_date = df[time_col].min()
        end_date = df[time_col].max()

        # Convert to datetime if needed
        if hasattr(start_date, 'to_pydatetime'):
            start_date = start_date.to_pydatetime()
        if hasattr(end_date, 'to_pydatetime'):
            end_date = end_date.to_pydatetime()

        # Ensure timezone
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        # Get file stats
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        last_modified = datetime.fromtimestamp(
            file_path.stat().st_mtime,
            tz=timezone.utc,
        )

        return DataRangeInfo(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            total_bars=len(df),
            file_path=str(file_path),
            file_size_mb=file_size_mb,
            last_updated=last_modified,
        )

    def list_available_data(self) -> List[DataRangeInfo]:
        """
        List all available historical data files.

        Returns:
            List of DataRangeInfo for all stored data
        """
        available = []

        for file_path in self.data_dir.glob("*.parquet"):
            # Parse symbol and timeframe from filename
            parts = file_path.stem.split("_")
            if len(parts) >= 2:
                symbol = parts[0]
                timeframe = parts[1]

                info = self.get_data_info(symbol, timeframe)
                if info:
                    available.append(info)

        return available

    def delete(self, symbol: str, timeframe: str) -> bool:
        """
        Delete historical data file.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            True if deleted, False if not found
        """
        file_path = self._get_file_path(symbol, timeframe)

        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted {file_path}")
            return True

        return False

    def get_status(self) -> dict:
        """Get storage status."""
        available = self.list_available_data()

        total_size_mb = sum(info.file_size_mb for info in available)
        total_bars = sum(info.total_bars for info in available)

        return {
            "data_dir": str(self.data_dir),
            "data_dir_exists": self.data_dir.exists(),
            "connector_configured": self.data_connector is not None,
            "files_count": len(available),
            "total_size_mb": round(total_size_mb, 2),
            "total_bars": total_bars,
            "available_data": [info.to_dict() for info in available],
        }


def download_full_history(
    data_connector: Any,
    symbol: str = "XAUUSD",
    timeframe: str = "M5",
    days: int = 365,
    data_dir: Optional[Path] = None,
) -> DataRangeInfo:
    """
    Convenience function to download full history.

    Args:
        data_connector: Data connector instance
        symbol: Trading symbol
        timeframe: Timeframe
        days: Days of history
        data_dir: Optional custom data directory

    Returns:
        DataRangeInfo for the saved data
    """
    store = HistoricalDataStore(
        data_dir=data_dir,
        data_connector=data_connector,
    )

    # Download data
    df = store.download_historical(
        symbol=symbol,
        timeframe=timeframe,
        days=days,
    )

    # Save to parquet
    store.save(df, symbol, timeframe)

    # Return info (guaranteed to exist after save)
    info = store.get_data_info(symbol, timeframe)
    assert info is not None, f"Data info should exist after save for {symbol} {timeframe}"
    return info
