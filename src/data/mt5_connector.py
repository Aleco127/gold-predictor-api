"""
MetaTrader 5 Connector Module
=============================
Handles connection to MT5 and fetches OHLCV data for gold (XAUUSD).
"""

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 package not available. Using mock data.")


class MT5ConnectionError(Exception):
    """Raised when MT5 connection fails."""
    pass


class MT5AuthenticationError(Exception):
    """Raised when MT5 authentication fails."""
    pass


class MT5Connector:
    """
    MetaTrader 5 connector for fetching market data.

    Handles:
    - Connection management
    - OHLCV data retrieval
    - Symbol information
    - Graceful fallback to mock data when MT5 unavailable
    """

    TIMEFRAME_MAP = {
        "M1": mt5.TIMEFRAME_M1 if MT5_AVAILABLE else 1,
        "M5": mt5.TIMEFRAME_M5 if MT5_AVAILABLE else 5,
        "M15": mt5.TIMEFRAME_M15 if MT5_AVAILABLE else 15,
        "M30": mt5.TIMEFRAME_M30 if MT5_AVAILABLE else 30,
        "H1": mt5.TIMEFRAME_H1 if MT5_AVAILABLE else 60,
        "H4": mt5.TIMEFRAME_H4 if MT5_AVAILABLE else 240,
        "D1": mt5.TIMEFRAME_D1 if MT5_AVAILABLE else 1440,
    }

    def __init__(
        self,
        login: int,
        password: str,
        server: str,
        symbol: str = "XAUUSD",
        timeframe: str = "M5",
    ):
        """
        Initialize MT5 connector.

        Args:
            login: MT5 account login number
            password: MT5 account password
            server: MT5 server name
            symbol: Trading symbol (default: XAUUSD)
            timeframe: Chart timeframe (default: M5)
        """
        self.login = login
        self.password = password
        self.server = server
        self.symbol = symbol
        self.timeframe = timeframe
        self._connected = False

    def connect(self) -> bool:
        """
        Establish connection to MT5 terminal.

        Returns:
            bool: True if connection successful

        Raises:
            MT5ConnectionError: If terminal initialization fails
            MT5AuthenticationError: If login fails
        """
        if not MT5_AVAILABLE:
            logger.warning("MT5 not available, using mock mode")
            self._connected = True
            return True

        # Initialize MT5 terminal
        if not mt5.initialize():
            error = mt5.last_error()
            raise MT5ConnectionError(f"MT5 initialization failed: {error}")

        # Login to account
        authorized = mt5.login(
            login=self.login,
            password=self.password,
            server=self.server,
        )

        if not authorized:
            mt5.shutdown()
            error = mt5.last_error()
            raise MT5AuthenticationError(f"MT5 login failed: {error}")

        logger.info(f"Connected to MT5 account {self.login} on {self.server}")
        self._connected = True
        return True

    def disconnect(self) -> None:
        """Disconnect from MT5 terminal."""
        if MT5_AVAILABLE and self._connected:
            mt5.shutdown()
            logger.info("Disconnected from MT5")
        self._connected = False

    def is_connected(self) -> bool:
        """Check if connected to MT5."""
        return self._connected

    def get_ohlcv(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        bars: int = 1000,
        from_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from MT5.

        Args:
            symbol: Trading symbol (uses default if None)
            timeframe: Timeframe string (uses default if None)
            bars: Number of bars to fetch
            from_date: Start date for data (fetches from current if None)

        Returns:
            DataFrame with columns: time, open, high, low, close, tick_volume, spread, real_volume
        """
        symbol = symbol or self.symbol
        timeframe = timeframe or self.timeframe
        tf = self.TIMEFRAME_MAP.get(timeframe, self.TIMEFRAME_MAP["M5"])

        if not MT5_AVAILABLE or not self._connected:
            logger.info(f"Generating mock data for {symbol}")
            return self._generate_mock_data(bars)

        # Fetch data from MT5
        if from_date:
            rates = mt5.copy_rates_from(symbol, tf, from_date, bars)
        else:
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)

        if rates is None or len(rates) == 0:
            error = mt5.last_error()
            logger.warning(f"No data received for {symbol}: {error}")
            return self._generate_mock_data(bars)

        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)

        logger.info(f"Fetched {len(df)} bars for {symbol} ({timeframe})")
        return df

    def get_current_price(self, symbol: Optional[str] = None) -> dict:
        """
        Get current bid/ask prices for symbol.

        Args:
            symbol: Trading symbol (uses default if None)

        Returns:
            Dict with bid, ask, last, spread keys
        """
        symbol = symbol or self.symbol

        if not MT5_AVAILABLE or not self._connected:
            # Return mock price
            base_price = 2650.0
            return {
                "bid": base_price - 0.05,
                "ask": base_price + 0.05,
                "last": base_price,
                "spread": 0.10,
                "time": datetime.now(),
            }

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.warning(f"Could not get tick for {symbol}")
            return {}

        return {
            "bid": tick.bid,
            "ask": tick.ask,
            "last": tick.last,
            "spread": tick.ask - tick.bid,
            "time": datetime.fromtimestamp(tick.time),
        }

    def get_symbol_info(self, symbol: Optional[str] = None) -> dict:
        """
        Get symbol information.

        Args:
            symbol: Trading symbol (uses default if None)

        Returns:
            Dict with symbol specifications
        """
        symbol = symbol or self.symbol

        if not MT5_AVAILABLE or not self._connected:
            return {
                "symbol": symbol,
                "digits": 2,
                "point": 0.01,
                "trade_contract_size": 100,
                "volume_min": 0.01,
                "volume_max": 100.0,
            }

        info = mt5.symbol_info(symbol)
        if info is None:
            logger.warning(f"Could not get info for {symbol}")
            return {}

        return {
            "symbol": info.name,
            "digits": info.digits,
            "point": info.point,
            "trade_contract_size": info.trade_contract_size,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "description": info.description,
        }

    def _generate_mock_data(self, bars: int) -> pd.DataFrame:
        """
        Generate mock OHLCV data for testing.

        Args:
            bars: Number of bars to generate

        Returns:
            DataFrame with mock price data
        """
        np.random.seed(42)

        # Generate time index
        end_time = datetime.now()
        time_index = pd.date_range(end=end_time, periods=bars, freq="5min")

        # Generate random walk prices starting from realistic gold price
        base_price = 2650.0
        returns = np.random.randn(bars) * 0.001  # 0.1% volatility
        prices = base_price * np.exp(np.cumsum(returns))

        # Generate OHLC from prices
        high_offset = np.abs(np.random.randn(bars)) * 0.5
        low_offset = np.abs(np.random.randn(bars)) * 0.5

        df = pd.DataFrame({
            "open": prices,
            "high": prices + high_offset,
            "low": prices - low_offset,
            "close": prices + np.random.randn(bars) * 0.2,
            "tick_volume": np.random.randint(100, 10000, bars),
            "spread": np.random.randint(5, 20, bars),
            "real_volume": np.random.randint(0, 1000, bars),
        }, index=time_index)

        # Ensure high >= max(open, close) and low <= min(open, close)
        df["high"] = df[["open", "close", "high"]].max(axis=1)
        df["low"] = df[["open", "close", "low"]].min(axis=1)

        return df

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False
