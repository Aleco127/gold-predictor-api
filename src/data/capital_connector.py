"""
Capital.com API Connector Module
================================
Handles connection to Capital.com REST API for market data.
Documentation: https://open-api.capital.com/
"""

import base64
import hashlib
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Union

import httpx
import pandas as pd
from loguru import logger

# Try to import cryptography for password encryption
try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.backends import default_backend
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    logger.warning("cryptography package not installed. Password encryption unavailable.")


class CapitalConnectionError(Exception):
    """Raised when Capital.com connection fails."""
    pass


class CapitalAuthenticationError(Exception):
    """Raised when Capital.com authentication fails."""
    pass


class DataCache:
    """
    Simple file-based cache for historical data (US-021).

    Caches OHLCV data to disk to avoid repeated API calls.
    Supports automatic cache invalidation based on age.
    """

    def __init__(
        self,
        cache_dir: Union[str, Path] = "data/cache",
        max_age_hours: int = 24,
    ):
        """
        Initialize data cache.

        Args:
            cache_dir: Directory to store cached data
            max_age_hours: Maximum age of cached data in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_hours = max_age_hours

    def _get_cache_key(
        self,
        symbol: str,
        timeframe: str,
        from_date: datetime,
        to_date: datetime,
    ) -> str:
        """Generate unique cache key."""
        key_str = f"{symbol}_{timeframe}_{from_date.isoformat()}_{to_date.isoformat()}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get path to cached file."""
        return self.cache_dir / f"{cache_key}.parquet"

    def _get_metadata_path(self, cache_key: str) -> Path:
        """Get path to cache metadata."""
        return self.cache_dir / f"{cache_key}.json"

    def is_valid(
        self,
        symbol: str,
        timeframe: str,
        from_date: datetime,
        to_date: datetime,
    ) -> bool:
        """Check if cached data exists and is still valid."""
        cache_key = self._get_cache_key(symbol, timeframe, from_date, to_date)
        cache_path = self._get_cache_path(cache_key)
        metadata_path = self._get_metadata_path(cache_key)

        if not cache_path.exists() or not metadata_path.exists():
            return False

        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            cached_time = datetime.fromisoformat(metadata["cached_at"])
            age_hours = (datetime.utcnow() - cached_time).total_seconds() / 3600

            return age_hours < self.max_age_hours
        except Exception:
            return False

    def get(
        self,
        symbol: str,
        timeframe: str,
        from_date: datetime,
        to_date: datetime,
    ) -> Optional[pd.DataFrame]:
        """Retrieve cached data if valid."""
        if not self.is_valid(symbol, timeframe, from_date, to_date):
            return None

        cache_key = self._get_cache_key(symbol, timeframe, from_date, to_date)
        cache_path = self._get_cache_path(cache_key)

        try:
            df = pd.read_parquet(cache_path)
            logger.debug(f"Loaded {len(df)} bars from cache")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None

    def put(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        from_date: datetime,
        to_date: datetime,
    ) -> None:
        """Store data in cache."""
        cache_key = self._get_cache_key(symbol, timeframe, from_date, to_date)
        cache_path = self._get_cache_path(cache_key)
        metadata_path = self._get_metadata_path(cache_key)

        try:
            df.to_parquet(cache_path)

            metadata = {
                "symbol": symbol,
                "timeframe": timeframe,
                "from_date": from_date.isoformat(),
                "to_date": to_date.isoformat(),
                "cached_at": datetime.utcnow().isoformat(),
                "rows": len(df),
            }
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

            logger.debug(f"Cached {len(df)} bars to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache data: {e}")

    def clear(self, older_than_hours: Optional[int] = None) -> int:
        """
        Clear cache files.

        Args:
            older_than_hours: Only clear files older than this. If None, clear all.

        Returns:
            Number of files cleared
        """
        cleared = 0
        for metadata_path in self.cache_dir.glob("*.json"):
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                if older_than_hours is not None:
                    cached_time = datetime.fromisoformat(metadata["cached_at"])
                    age_hours = (datetime.utcnow() - cached_time).total_seconds() / 3600
                    if age_hours < older_than_hours:
                        continue

                # Delete both parquet and json
                cache_key = metadata_path.stem
                cache_path = self._get_cache_path(cache_key)
                if cache_path.exists():
                    cache_path.unlink()
                metadata_path.unlink()
                cleared += 1
            except Exception as e:
                logger.warning(f"Error clearing cache file {metadata_path}: {e}")

        logger.info(f"Cleared {cleared} cached files")
        return cleared


class CapitalConnector:
    """
    Capital.com API connector for fetching market data.

    Handles:
    - Session management with CST and X-SECURITY-TOKEN
    - Historical OHLC data retrieval
    - Real-time price quotes
    - Rate limiting compliance
    """

    # Base URLs
    DEMO_URL = "https://demo-api-capital.backend-capital.com/api/v1"
    LIVE_URL = "https://api-capital.backend-capital.com/api/v1"

    # Resolution mapping (Capital.com format)
    RESOLUTION_MAP = {
        "M1": "MINUTE",
        "M5": "MINUTE_5",
        "M15": "MINUTE_15",
        "M30": "MINUTE_30",
        "H1": "HOUR",
        "H4": "HOUR_4",
        "D1": "DAY",
        "W1": "WEEK",
    }

    # Symbol mapping (Capital.com uses different symbol names)
    SYMBOL_MAP = {
        "XAUUSD": "GOLD",  # Gold spot
        "GOLD": "GOLD",
    }

    def __init__(
        self,
        api_key: str,
        password: str,
        identifier: str = None,
        demo: bool = True,
        symbol: str = "GOLD",
        cache_dir: Optional[Union[str, Path]] = None,
        cache_max_age_hours: int = 24,
    ):
        """
        Initialize Capital.com connector.

        Args:
            api_key: Capital.com API key
            password: Account password
            identifier: Account identifier/email (optional, uses API key if not provided)
            demo: Use demo environment (default: True)
            symbol: Trading symbol (default: GOLD)
            cache_dir: Directory for data cache (US-021)
            cache_max_age_hours: Maximum cache age in hours
        """
        self.api_key = api_key
        self.password = password
        self.identifier = identifier or api_key
        self.demo = demo
        self.symbol = symbol

        self.base_url = self.DEMO_URL if demo else self.LIVE_URL
        self._cst = None
        self._security_token = None
        self._connected = False
        self._last_request_time = 0
        self._min_request_interval = 0.1  # 10 requests per second max

        # Initialize cache (US-021)
        if cache_dir:
            self._cache = DataCache(cache_dir, cache_max_age_hours)
        else:
            self._cache = None

    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _get_encryption_key(self) -> tuple:
        """
        Get RSA encryption key from Capital.com for password encryption.

        Returns:
            tuple: (encryption_key, timestamp)
        """
        self._rate_limit()

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(
                    f"{self.base_url}/session/encryptionKey",
                    headers={"X-CAP-API-KEY": self.api_key},
                )

                if response.status_code == 200:
                    data = response.json()
                    return data.get("encryptionKey"), data.get("timeStamp")
                else:
                    logger.error(f"Failed to get encryption key: {response.status_code}")
                    return None, None

        except httpx.RequestError as e:
            logger.error(f"Network error getting encryption key: {e}")
            return None, None

    def _encrypt_password(self, password: str, encryption_key: str, timestamp: int) -> str:
        """
        Encrypt password using RSA public key from Capital.com.

        Args:
            password: Plain text password
            encryption_key: Base64 encoded RSA public key
            timestamp: Timestamp from encryption key response

        Returns:
            Base64 encoded encrypted password with timestamp
        """
        if not HAS_CRYPTO:
            raise CapitalAuthenticationError(
                "cryptography package required for password encryption. "
                "Install with: pip install cryptography"
            )

        try:
            # Decode the public key
            key_der = base64.b64decode(encryption_key)
            public_key = serialization.load_der_public_key(key_der, backend=default_backend())

            # Combine password with timestamp pipe-separated
            password_with_timestamp = f"{password}|{timestamp}"

            # Encrypt using RSA PKCS1v15 padding
            encrypted = public_key.encrypt(
                password_with_timestamp.encode('utf-8'),
                padding.PKCS1v15()
            )

            # Return base64 encoded
            return base64.b64encode(encrypted).decode('utf-8')

        except Exception as e:
            logger.error(f"Password encryption failed: {e}")
            raise CapitalAuthenticationError(f"Password encryption failed: {e}")

    def connect(self, use_encryption: bool = False) -> bool:
        """
        Establish session with Capital.com.

        Args:
            use_encryption: Whether to encrypt password (optional, defaults to False)

        Returns:
            bool: True if connection successful

        Raises:
            CapitalConnectionError: If connection fails
            CapitalAuthenticationError: If authentication fails
        """
        password_to_use = self.password

        # Optionally encrypt password (some environments require it)
        if use_encryption and HAS_CRYPTO:
            logger.debug("Getting encryption key from Capital.com...")
            encryption_key, timestamp = self._get_encryption_key()

            if encryption_key:
                logger.debug("Encrypting password...")
                password_to_use = self._encrypt_password(self.password, encryption_key, timestamp)
            else:
                logger.warning("Failed to get encryption key, using plain password")

        self._rate_limit()

        try:
            with httpx.Client(timeout=30.0) as client:
                payload = {
                    "identifier": self.identifier,
                    "password": password_to_use,
                }
                if use_encryption and HAS_CRYPTO:
                    payload["encryptedPassword"] = True

                response = client.post(
                    f"{self.base_url}/session",
                    headers={
                        "X-CAP-API-KEY": self.api_key,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )

                if response.status_code == 200:
                    # Extract tokens from headers
                    self._cst = response.headers.get("CST")
                    self._security_token = response.headers.get("X-SECURITY-TOKEN")

                    if self._cst and self._security_token:
                        self._connected = True
                        data = response.json()
                        logger.info(
                            f"Connected to Capital.com {'demo' if self.demo else 'live'} - "
                            f"Account: {data.get('currentAccountId', 'unknown')}"
                        )
                        return True
                    else:
                        raise CapitalAuthenticationError("No tokens received in response headers")

                elif response.status_code == 401:
                    error_data = response.json()
                    raise CapitalAuthenticationError(
                        f"Authentication failed: {error_data.get('errorCode', 'Unknown error')}"
                    )
                else:
                    raise CapitalConnectionError(
                        f"Connection failed with status {response.status_code}: {response.text}"
                    )

        except httpx.RequestError as e:
            raise CapitalConnectionError(f"Network error: {e}")

    def disconnect(self) -> None:
        """End session with Capital.com."""
        if not self._connected:
            return

        try:
            self._rate_limit()
            with httpx.Client(timeout=10.0) as client:
                client.delete(
                    f"{self.base_url}/session",
                    headers=self._auth_headers(),
                )
        except Exception as e:
            logger.warning(f"Error during disconnect: {e}")
        finally:
            self._cst = None
            self._security_token = None
            self._connected = False
            logger.info("Disconnected from Capital.com")

    def is_connected(self) -> bool:
        """Check if connected to Capital.com."""
        return self._connected

    def _auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for requests."""
        return {
            "X-CAP-API-KEY": self.api_key,
            "CST": self._cst or "",
            "X-SECURITY-TOKEN": self._security_token or "",
            "Content-Type": "application/json",
        }

    def _ensure_connected(self):
        """Ensure we have an active session."""
        if not self._connected:
            self.connect()

    def get_ohlcv(
        self,
        symbol: Optional[str] = None,
        timeframe: str = "M5",
        bars: int = 1000,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Capital.com.

        Args:
            symbol: Trading symbol (uses default if None)
            timeframe: Timeframe string (M1, M5, M15, M30, H1, H4, D1, W1)
            bars: Number of bars to fetch (max ~1000 per request)
            from_date: Start date for data
            to_date: End date for data (defaults to now)

        Returns:
            DataFrame with columns: time, open, high, low, close, volume
        """
        self._ensure_connected()
        self._rate_limit()

        # Map symbol
        symbol = symbol or self.symbol
        capital_symbol = self.SYMBOL_MAP.get(symbol.upper(), symbol.upper())

        # Map resolution
        resolution = self.RESOLUTION_MAP.get(timeframe, "MINUTE_5")

        # Build params - don't use date range by default as it can cause issues
        params = {
            "resolution": resolution,
            "max": min(bars, 1000),
        }

        # Only add date range if explicitly provided
        if from_date is not None:
            params["from"] = from_date.strftime("%Y-%m-%dT%H:%M:%S")
        if to_date is not None:
            params["to"] = to_date.strftime("%Y-%m-%dT%H:%M:%S")

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(
                    f"{self.base_url}/prices/{capital_symbol}",
                    headers=self._auth_headers(),
                    params=params,
                )

                if response.status_code == 200:
                    data = response.json()
                    return self._parse_prices(data)

                elif response.status_code == 401:
                    # Session expired, reconnect and retry
                    logger.warning("Session expired, reconnecting...")
                    self._connected = False
                    self.connect()
                    return self.get_ohlcv(symbol, timeframe, bars, from_date, to_date)

                else:
                    logger.error(f"Failed to fetch prices: {response.status_code} - {response.text}")
                    return pd.DataFrame()

        except httpx.RequestError as e:
            logger.error(f"Network error fetching prices: {e}")
            return pd.DataFrame()

    def get_historical_data(
        self,
        symbol: Optional[str] = None,
        timeframe: str = "M5",
        days: int = 365,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch extended historical data with pagination (US-021).

        This method handles the 1000-bar limit by making multiple requests
        and concatenating the results. Supports caching to reduce API calls.

        Args:
            symbol: Trading symbol (uses default if None)
            timeframe: Timeframe string (M1, M5, M15, M30, H1, H4, D1, W1)
            days: Number of days of historical data (default 365 for 1 year)
            use_cache: Whether to use caching (default True)

        Returns:
            DataFrame with complete historical OHLCV data
        """
        symbol = symbol or self.symbol
        to_date = datetime.utcnow()
        from_date = to_date - timedelta(days=days)

        # Check cache first
        if use_cache and self._cache:
            cached = self._cache.get(symbol, timeframe, from_date, to_date)
            if cached is not None:
                logger.info(f"Using cached data: {len(cached)} bars")
                return cached

        # Calculate bars per timeframe
        bars_per_day = self._get_bars_per_day(timeframe)
        total_bars_needed = int(days * bars_per_day)
        max_bars_per_request = 1000

        logger.info(
            f"Fetching {total_bars_needed} bars ({days} days of {timeframe}) "
            f"in chunks of {max_bars_per_request}"
        )

        all_dfs = []
        current_to = to_date
        bars_fetched = 0
        request_count = 0

        while bars_fetched < total_bars_needed and current_to > from_date:
            # Fetch a chunk
            df_chunk = self.get_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                bars=max_bars_per_request,
                to_date=current_to,
            )

            if df_chunk.empty:
                logger.warning(f"Empty response at request {request_count + 1}")
                break

            all_dfs.append(df_chunk)
            bars_fetched += len(df_chunk)
            request_count += 1

            # Move window back
            if len(df_chunk) > 0:
                current_to = df_chunk.index.min() - timedelta(seconds=1)

            logger.debug(
                f"Request {request_count}: fetched {len(df_chunk)} bars, "
                f"total {bars_fetched}/{total_bars_needed}"
            )

            # Rate limiting is handled by get_ohlcv

        if not all_dfs:
            logger.error("Failed to fetch any historical data")
            return pd.DataFrame()

        # Combine all chunks
        df = pd.concat(all_dfs)
        df = df[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True)

        # Filter to requested date range
        df = df[(df.index >= from_date) & (df.index <= to_date)]

        logger.info(
            f"Fetched {len(df)} total bars in {request_count} requests "
            f"from {df.index.min()} to {df.index.max()}"
        )

        # Cache the result
        if use_cache and self._cache and not df.empty:
            self._cache.put(df, symbol, timeframe, from_date, to_date)

        return df

    def _get_bars_per_day(self, timeframe: str) -> float:
        """Calculate approximate bars per day for a timeframe."""
        bars_mapping = {
            "M1": 1440,      # 24 * 60
            "M5": 288,       # 24 * 12
            "M15": 96,       # 24 * 4
            "M30": 48,       # 24 * 2
            "H1": 24,
            "H4": 6,
            "D1": 1,
            "W1": 1/7,
        }
        return bars_mapping.get(timeframe, 288)  # Default to M5

    def get_multi_timeframe_data(
        self,
        symbol: Optional[str] = None,
        timeframes: List[str] = None,
        days: int = 30,
        use_cache: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple timeframes (US-022).

        Args:
            symbol: Trading symbol
            timeframes: List of timeframes (default: M5, M15, H1, H4, D1)
            days: Number of days to fetch
            use_cache: Whether to use caching

        Returns:
            Dict mapping timeframe to DataFrame
        """
        symbol = symbol or self.symbol
        timeframes = timeframes or ["M5", "M15", "H1", "H4", "D1"]

        result = {}
        for tf in timeframes:
            logger.info(f"Fetching {tf} data...")
            result[tf] = self.get_historical_data(
                symbol=symbol,
                timeframe=tf,
                days=days,
                use_cache=use_cache,
            )

        return result

    def _parse_prices(self, data: Dict) -> pd.DataFrame:
        """Parse price response into DataFrame."""
        prices = data.get("prices", [])

        if not prices:
            logger.warning("No price data received")
            return pd.DataFrame()

        records = []
        for price in prices:
            # Capital.com provides bid and ask, we'll use mid price
            open_price = (
                price.get("openPrice", {}).get("bid", 0) +
                price.get("openPrice", {}).get("ask", 0)
            ) / 2
            high_price = (
                price.get("highPrice", {}).get("bid", 0) +
                price.get("highPrice", {}).get("ask", 0)
            ) / 2
            low_price = (
                price.get("lowPrice", {}).get("bid", 0) +
                price.get("lowPrice", {}).get("ask", 0)
            ) / 2
            close_price = (
                price.get("closePrice", {}).get("bid", 0) +
                price.get("closePrice", {}).get("ask", 0)
            ) / 2

            records.append({
                "time": pd.to_datetime(price.get("snapshotTime") or price.get("snapshotTimeUTC")),
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "tick_volume": price.get("lastTradedVolume", 0),
            })

        df = pd.DataFrame(records)
        df.set_index("time", inplace=True)
        df.sort_index(inplace=True)

        logger.info(f"Fetched {len(df)} bars from Capital.com")
        return df

    def get_current_price(self, symbol: Optional[str] = None) -> Dict:
        """
        Get current bid/ask prices for symbol.

        Args:
            symbol: Trading symbol (uses default if None)

        Returns:
            Dict with bid, ask, spread keys
        """
        self._ensure_connected()
        self._rate_limit()

        symbol = symbol or self.symbol
        capital_symbol = self.SYMBOL_MAP.get(symbol.upper(), symbol.upper())

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(
                    f"{self.base_url}/markets/{capital_symbol}",
                    headers=self._auth_headers(),
                )

                if response.status_code == 200:
                    data = response.json()
                    snapshot = data.get("snapshot", {})

                    bid = snapshot.get("bid", 0)
                    ask = snapshot.get("offer", 0)

                    return {
                        "bid": bid,
                        "ask": ask,
                        "last": (bid + ask) / 2,
                        "spread": ask - bid,
                        "time": datetime.utcnow(),
                        "status": data.get("marketStatus", "UNKNOWN"),
                    }

                else:
                    logger.warning(f"Could not get price for {symbol}: {response.status_code}")
                    return {}

        except httpx.RequestError as e:
            logger.error(f"Network error getting price: {e}")
            return {}

    def get_symbol_info(self, symbol: Optional[str] = None) -> Dict:
        """
        Get symbol information.

        Args:
            symbol: Trading symbol (uses default if None)

        Returns:
            Dict with symbol specifications
        """
        self._ensure_connected()
        self._rate_limit()

        symbol = symbol or self.symbol
        capital_symbol = self.SYMBOL_MAP.get(symbol.upper(), symbol.upper())

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(
                    f"{self.base_url}/markets/{capital_symbol}",
                    headers=self._auth_headers(),
                )

                if response.status_code == 200:
                    data = response.json()
                    instrument = data.get("instrument", {})

                    return {
                        "symbol": capital_symbol,
                        "name": instrument.get("name", ""),
                        "type": instrument.get("type", ""),
                        "currency": instrument.get("currencies", [{}])[0].get("code", "USD"),
                        "min_quantity": instrument.get("minDealSize", {}).get("value", 0.01),
                        "lot_size": instrument.get("lotSize", 1),
                        "margin_factor": instrument.get("marginFactor", 0),
                        "market_status": data.get("marketStatus", "UNKNOWN"),
                    }

                else:
                    logger.warning(f"Could not get info for {symbol}: {response.status_code}")
                    return {}

        except httpx.RequestError as e:
            logger.error(f"Network error getting symbol info: {e}")
            return {}

    def search_markets(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for available markets.

        Args:
            query: Search term
            limit: Maximum results

        Returns:
            List of matching market info dicts
        """
        self._ensure_connected()
        self._rate_limit()

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(
                    f"{self.base_url}/markets",
                    headers=self._auth_headers(),
                    params={
                        "searchTerm": query,
                        "limit": limit,
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    markets = data.get("markets", [])
                    return [
                        {
                            "symbol": m.get("epic", ""),
                            "name": m.get("instrumentName", ""),
                            "type": m.get("instrumentType", ""),
                            "status": m.get("marketStatus", ""),
                        }
                        for m in markets
                    ]

                else:
                    logger.warning(f"Market search failed: {response.status_code}")
                    return []

        except httpx.RequestError as e:
            logger.error(f"Network error searching markets: {e}")
            return []

    # ========================================
    # TRADING METHODS
    # ========================================

    def get_account_info(self) -> Dict:
        """
        Get account information including balance.

        Returns:
            Dict with account details
        """
        self._ensure_connected()
        self._rate_limit()

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(
                    f"{self.base_url}/accounts",
                    headers=self._auth_headers(),
                )

                if response.status_code == 200:
                    data = response.json()
                    accounts = data.get("accounts", [])
                    if accounts:
                        acc = accounts[0]
                        return {
                            "account_id": acc.get("accountId"),
                            "account_name": acc.get("accountName"),
                            "balance": acc.get("balance", {}).get("balance", 0),
                            "available": acc.get("balance", {}).get("available", 0),
                            "profit_loss": acc.get("balance", {}).get("profitLoss", 0),
                            "currency": acc.get("currency", "USD"),
                        }
                    return {}
                else:
                    logger.warning(f"Could not get account info: {response.status_code}")
                    return {}

        except httpx.RequestError as e:
            logger.error(f"Network error getting account info: {e}")
            return {}

    def list_positions(self) -> List[Dict]:
        """
        Get all open positions.

        Returns:
            List of position dicts
        """
        self._ensure_connected()
        self._rate_limit()

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(
                    f"{self.base_url}/positions",
                    headers=self._auth_headers(),
                )

                if response.status_code == 200:
                    data = response.json()
                    positions = data.get("positions", [])
                    return [
                        {
                            "deal_id": p.get("position", {}).get("dealId"),
                            "symbol": p.get("market", {}).get("epic"),
                            "direction": p.get("position", {}).get("direction"),
                            "size": p.get("position", {}).get("size"),
                            "open_level": p.get("position", {}).get("level"),
                            "stop_loss": p.get("position", {}).get("stopLevel"),
                            "take_profit": p.get("position", {}).get("limitLevel"),
                            "profit_loss": p.get("position", {}).get("upl"),
                            "created_at": p.get("position", {}).get("createdDateUTC"),
                        }
                        for p in positions
                    ]
                else:
                    logger.warning(f"Could not get positions: {response.status_code} - {response.text}")
                    return []

        except httpx.RequestError as e:
            logger.error(f"Network error getting positions: {e}")
            return []

    def create_position(
        self,
        symbol: Optional[str] = None,
        direction: str = "BUY",
        size: float = 0.01,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        guaranteed_stop: bool = False,
    ) -> Dict:
        """
        Open a new trading position.

        Args:
            symbol: Trading symbol (uses default if None)
            direction: "BUY" or "SELL"
            size: Position size in lots
            stop_loss: Stop loss price level
            take_profit: Take profit price level
            guaranteed_stop: Use guaranteed stop (higher cost)

        Returns:
            Dict with deal reference and confirmation
        """
        self._ensure_connected()
        self._rate_limit()

        symbol = symbol or self.symbol
        capital_symbol = self.SYMBOL_MAP.get(symbol.upper(), symbol.upper())

        payload = {
            "epic": capital_symbol,
            "direction": direction.upper(),
            "size": size,
            "guaranteedStop": guaranteed_stop,
        }

        if stop_loss is not None:
            payload["stopLevel"] = stop_loss
        if take_profit is not None:
            payload["limitLevel"] = take_profit

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    f"{self.base_url}/positions",
                    headers=self._auth_headers(),
                    json=payload,
                )

                if response.status_code == 200:
                    data = response.json()
                    deal_ref = data.get("dealReference")
                    logger.info(f"Position opened: {direction} {size} {capital_symbol} - Ref: {deal_ref}")

                    # Get deal confirmation
                    confirmation = self._get_deal_confirmation(deal_ref)
                    return {
                        "success": True,
                        "deal_reference": deal_ref,
                        "deal_id": confirmation.get("dealId"),
                        "direction": direction,
                        "size": size,
                        "symbol": capital_symbol,
                        "status": confirmation.get("dealStatus"),
                        "level": confirmation.get("level"),
                        "profit_currency": confirmation.get("profitCurrency"),
                    }

                else:
                    error_data = response.json() if response.content else {}
                    error_msg = error_data.get("errorCode", response.text)
                    logger.error(f"Failed to open position: {response.status_code} - {error_msg}")
                    return {
                        "success": False,
                        "error": error_msg,
                        "status_code": response.status_code,
                    }

        except httpx.RequestError as e:
            logger.error(f"Network error creating position: {e}")
            return {"success": False, "error": str(e)}

    def close_position(
        self,
        deal_id: str,
        size: Optional[float] = None,
    ) -> Dict:
        """
        Close an existing position.

        Args:
            deal_id: The deal ID of the position to close
            size: Size to close (None = close entire position)

        Returns:
            Dict with close confirmation
        """
        self._ensure_connected()
        self._rate_limit()

        try:
            with httpx.Client(timeout=30.0) as client:
                # Capital.com uses DELETE with body for partial closes
                headers = self._auth_headers()
                headers["_method"] = "DELETE"

                payload = {}
                if size is not None:
                    payload["size"] = size

                response = client.delete(
                    f"{self.base_url}/positions/{deal_id}",
                    headers=headers,
                )

                if response.status_code == 200:
                    data = response.json()
                    deal_ref = data.get("dealReference")
                    logger.info(f"Position closed: {deal_id} - Ref: {deal_ref}")

                    # Get deal confirmation
                    confirmation = self._get_deal_confirmation(deal_ref)
                    return {
                        "success": True,
                        "deal_reference": deal_ref,
                        "deal_id": deal_id,
                        "status": confirmation.get("dealStatus"),
                    }

                else:
                    error_data = response.json() if response.content else {}
                    error_msg = error_data.get("errorCode", response.text)
                    logger.error(f"Failed to close position: {response.status_code} - {error_msg}")
                    return {
                        "success": False,
                        "error": error_msg,
                        "status_code": response.status_code,
                    }

        except httpx.RequestError as e:
            logger.error(f"Network error closing position: {e}")
            return {"success": False, "error": str(e)}

    def update_position(
        self,
        deal_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop: Optional[bool] = None,
        trailing_stop_distance: Optional[float] = None,
    ) -> Dict:
        """
        Update an existing position's stop-loss or take-profit.

        Args:
            deal_id: The deal ID of the position to update
            stop_loss: New stop-loss level (None = no change)
            take_profit: New take-profit level (None = no change)
            trailing_stop: Enable/disable trailing stop
            trailing_stop_distance: Distance for trailing stop

        Returns:
            Dict with update confirmation
        """
        self._ensure_connected()
        self._rate_limit()

        if stop_loss is None and take_profit is None and trailing_stop is None:
            return {"success": False, "error": "No update parameters provided"}

        try:
            with httpx.Client(timeout=30.0) as client:
                headers = self._auth_headers()

                payload: Dict = {}
                if stop_loss is not None:
                    payload["stopLevel"] = stop_loss
                if take_profit is not None:
                    payload["profitLevel"] = take_profit
                if trailing_stop is not None:
                    payload["trailingStop"] = trailing_stop
                if trailing_stop_distance is not None:
                    payload["trailingStopDistance"] = trailing_stop_distance

                logger.info(f"Updating position {deal_id}: {payload}")

                response = client.put(
                    f"{self.base_url}/positions/{deal_id}",
                    headers=headers,
                    json=payload,
                )

                if response.status_code == 200:
                    data = response.json()
                    deal_ref = data.get("dealReference")
                    logger.info(f"Position updated: {deal_id} - Ref: {deal_ref}")

                    # Get deal confirmation
                    confirmation = self._get_deal_confirmation(deal_ref)
                    return {
                        "success": True,
                        "deal_reference": deal_ref,
                        "deal_id": deal_id,
                        "status": confirmation.get("dealStatus"),
                        "new_stop_loss": stop_loss,
                        "new_take_profit": take_profit,
                    }

                else:
                    error_data = response.json() if response.content else {}
                    error_msg = error_data.get("errorCode", response.text)
                    logger.error(f"Failed to update position: {response.status_code} - {error_msg}")
                    return {
                        "success": False,
                        "error": error_msg,
                        "status_code": response.status_code,
                    }

        except httpx.RequestError as e:
            logger.error(f"Network error updating position: {e}")
            return {"success": False, "error": str(e)}

    def close_all_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Close all open positions (optionally filtered by symbol).

        Args:
            symbol: Only close positions for this symbol (None = all)

        Returns:
            List of close results
        """
        positions = self.list_positions()

        if symbol:
            capital_symbol = self.SYMBOL_MAP.get(symbol.upper(), symbol.upper())
            positions = [p for p in positions if p["symbol"] == capital_symbol]

        results = []
        for pos in positions:
            result = self.close_position(pos["deal_id"])
            result["original_position"] = pos
            results.append(result)

        logger.info(f"Closed {len(results)} positions")
        return results

    def _get_deal_confirmation(self, deal_reference: str, max_retries: int = 5) -> Dict:
        """
        Get confirmation for a deal.

        Args:
            deal_reference: The deal reference to confirm
            max_retries: Maximum retry attempts

        Returns:
            Dict with deal confirmation details
        """
        for attempt in range(max_retries):
            self._rate_limit()

            try:
                with httpx.Client(timeout=10.0) as client:
                    response = client.get(
                        f"{self.base_url}/confirms/{deal_reference}",
                        headers=self._auth_headers(),
                    )

                    if response.status_code == 200:
                        return response.json()

                    # If still processing, wait and retry
                    if attempt < max_retries - 1:
                        time.sleep(0.5)

            except httpx.RequestError as e:
                logger.warning(f"Error getting confirmation: {e}")

        return {}

    def execute_trade_from_signal(
        self,
        signal: str,
        confidence: float,
        current_price: float,
        size: float = 0.01,
        risk_percent: float = 1.0,
        reward_ratio: float = 2.0,
    ) -> Dict:
        """
        Execute a trade based on prediction signal.

        Args:
            signal: Signal string (e.g., "STRONG_BUY", "BUY", "SELL", "STRONG_SELL")
            confidence: Prediction confidence (0-1)
            current_price: Current market price
            size: Position size in lots
            risk_percent: Risk percentage for stop loss calculation
            reward_ratio: Risk:reward ratio for take profit

        Returns:
            Dict with trade execution result
        """
        # Determine direction from signal
        signal_upper = signal.upper()
        if "BUY" in signal_upper:
            direction = "BUY"
        elif "SELL" in signal_upper:
            direction = "SELL"
        else:
            return {
                "success": False,
                "error": f"Cannot execute trade for signal: {signal}",
                "signal": signal,
            }

        # Calculate stop loss and take profit
        # For gold, typical pip value is 0.01, ATR usually 5-15 for M5
        pip_value = 0.1  # Gold pip
        atr_estimate = 5.0 * pip_value  # Rough estimate

        if direction == "BUY":
            stop_loss = current_price - (atr_estimate * 2)
            take_profit = current_price + (atr_estimate * 2 * reward_ratio)
        else:
            stop_loss = current_price + (atr_estimate * 2)
            take_profit = current_price - (atr_estimate * 2 * reward_ratio)

        # Adjust size based on confidence (stronger signals = larger size)
        if "STRONG" in signal_upper:
            adjusted_size = size * 1.5
        else:
            adjusted_size = size

        # Execute the trade
        result = self.create_position(
            direction=direction,
            size=adjusted_size,
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
        )

        result["signal"] = signal
        result["confidence"] = confidence
        result["calculated_sl"] = round(stop_loss, 2)
        result["calculated_tp"] = round(take_profit, 2)

        return result

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False
