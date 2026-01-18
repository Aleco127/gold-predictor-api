"""
Technical Indicators Module
===========================
Calculates various technical analysis indicators for price prediction.
Uses pandas-ta for efficient calculations.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    logger.warning("pandas-ta not available, using manual calculations")


class TechnicalIndicators:
    """
    Technical indicator calculator for OHLCV data.

    Supports:
    - Trend indicators (EMA, SMA)
    - Momentum indicators (RSI, MACD, Stochastic)
    - Volatility indicators (Bollinger Bands, ATR)
    - Volume indicators (OBV, VWAP)
    - Price action features
    """

    def __init__(
        self,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
        ema_periods: List[int] = None,
        atr_period: int = 14,
    ):
        """
        Initialize indicator calculator.

        Args:
            rsi_period: RSI calculation period
            macd_fast: MACD fast EMA period
            macd_slow: MACD slow EMA period
            macd_signal: MACD signal line period
            bb_period: Bollinger Bands period
            bb_std: Bollinger Bands standard deviation multiplier
            ema_periods: List of EMA periods to calculate
            atr_period: ATR calculation period
        """
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.ema_periods = ema_periods or [9, 21, 50]
        self.atr_period = atr_period

    def calculate_all(
        self,
        df: pd.DataFrame,
        include_temporal: bool = True,
        include_regime: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate all technical indicators.

        Args:
            df: DataFrame with OHLCV data (open, high, low, close, volume columns)
            include_temporal: Add temporal features (hour, day of week, session)
            include_regime: Add volatility regime detection

        Returns:
            DataFrame with original data plus indicator columns
        """
        df = df.copy()

        # Ensure required columns exist
        required = ["open", "high", "low", "close"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Volume column (use tick_volume if volume not present)
        if "volume" not in df.columns:
            df["volume"] = df.get("tick_volume", 0)

        # Calculate indicators
        df = self.add_trend_indicators(df)
        df = self.add_momentum_indicators(df)
        df = self.add_volatility_indicators(df)
        df = self.add_price_features(df)

        if df["volume"].sum() > 0:
            df = self.add_volume_indicators(df)

        # Add temporal features (US-006)
        if include_temporal:
            df = self.add_temporal_features(df)

        # Add volatility regime detection (US-007)
        if include_regime:
            df = self.add_volatility_regime(df)

        logger.info(f"Calculated {len(df.columns)} features")
        return df

    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-following indicators."""
        close = df["close"]

        # Exponential Moving Averages
        for period in self.ema_periods:
            if PANDAS_TA_AVAILABLE:
                df[f"ema_{period}"] = ta.ema(close, length=period)
            else:
                df[f"ema_{period}"] = close.ewm(span=period, adjust=False).mean()

        # Simple Moving Averages
        for period in [20, 50]:
            if PANDAS_TA_AVAILABLE:
                df[f"sma_{period}"] = ta.sma(close, length=period)
            else:
                df[f"sma_{period}"] = close.rolling(window=period).mean()

        # EMA crossover signals
        if "ema_9" in df.columns and "ema_21" in df.columns:
            df["ema_cross"] = (df["ema_9"] > df["ema_21"]).astype(int)
            df["ema_distance"] = (df["ema_9"] - df["ema_21"]) / df["ema_21"] * 100

        return df

    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum oscillators."""
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # RSI
        if PANDAS_TA_AVAILABLE:
            df["rsi"] = ta.rsi(close, length=self.rsi_period)
        else:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            df["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        if PANDAS_TA_AVAILABLE:
            macd = ta.macd(close, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
            if macd is not None and len(macd.columns) >= 3:
                df["macd"] = macd.iloc[:, 0]
                df["macd_histogram"] = macd.iloc[:, 1]
                df["macd_signal"] = macd.iloc[:, 2]
        else:
            exp1 = close.ewm(span=self.macd_fast, adjust=False).mean()
            exp2 = close.ewm(span=self.macd_slow, adjust=False).mean()
            df["macd"] = exp1 - exp2
            df["macd_signal"] = df["macd"].ewm(span=self.macd_signal, adjust=False).mean()
            df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # Stochastic Oscillator
        if PANDAS_TA_AVAILABLE:
            stoch = ta.stoch(high, low, close)
            if stoch is not None:
                df["stoch_k"] = stoch.iloc[:, 0]
                df["stoch_d"] = stoch.iloc[:, 1]
        else:
            low_14 = low.rolling(window=14).min()
            high_14 = high.rolling(window=14).max()
            df["stoch_k"] = 100 * (close - low_14) / (high_14 - low_14)
            df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()

        # ROC (Rate of Change)
        df["roc"] = close.pct_change(periods=10) * 100

        # Williams %R
        if PANDAS_TA_AVAILABLE:
            df["willr"] = ta.willr(high, low, close, length=14)
        else:
            highest_high = high.rolling(window=14).max()
            lowest_low = low.rolling(window=14).min()
            df["willr"] = -100 * (highest_high - close) / (highest_high - lowest_low)

        return df

    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Bollinger Bands
        if PANDAS_TA_AVAILABLE:
            bbands = ta.bbands(close, length=self.bb_period, std=self.bb_std)
            if bbands is not None:
                df["bb_lower"] = bbands.iloc[:, 0]
                df["bb_mid"] = bbands.iloc[:, 1]
                df["bb_upper"] = bbands.iloc[:, 2]
                df["bb_bandwidth"] = bbands.iloc[:, 3] if len(bbands.columns) > 3 else None
                df["bb_percent"] = bbands.iloc[:, 4] if len(bbands.columns) > 4 else None
        else:
            sma = close.rolling(window=self.bb_period).mean()
            std = close.rolling(window=self.bb_period).std()
            df["bb_mid"] = sma
            df["bb_upper"] = sma + (std * self.bb_std)
            df["bb_lower"] = sma - (std * self.bb_std)
            df["bb_bandwidth"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
            df["bb_percent"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # ATR (Average True Range)
        if PANDAS_TA_AVAILABLE:
            df["atr"] = ta.atr(high, low, close, length=self.atr_period)
        else:
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df["atr"] = tr.rolling(window=self.atr_period).mean()

        # ATR as percentage of price
        df["atr_percent"] = df["atr"] / close * 100

        # Historical Volatility
        df["volatility"] = close.pct_change().rolling(window=20).std() * np.sqrt(252) * 100

        return df

    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_ = df["open"]

        # Returns
        df["return_1"] = close.pct_change() * 100
        df["return_5"] = close.pct_change(5) * 100
        df["return_10"] = close.pct_change(10) * 100

        # Candle patterns
        df["body"] = close - open_
        df["body_percent"] = df["body"] / open_ * 100
        df["upper_shadow"] = high - df[["open", "close"]].max(axis=1)
        df["lower_shadow"] = df[["open", "close"]].min(axis=1) - low
        df["range"] = high - low

        # Trend strength
        df["higher_high"] = (high > high.shift(1)).astype(int)
        df["higher_low"] = (low > low.shift(1)).astype(int)
        df["lower_high"] = (high < high.shift(1)).astype(int)
        df["lower_low"] = (low < low.shift(1)).astype(int)

        # Distance from recent high/low
        df["distance_from_high_20"] = (close - high.rolling(20).max()) / close * 100
        df["distance_from_low_20"] = (close - low.rolling(20).min()) / close * 100

        return df

    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        close = df["close"]
        volume = df["volume"]

        # On-Balance Volume
        if PANDAS_TA_AVAILABLE:
            df["obv"] = ta.obv(close, volume)
        else:
            obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
            df["obv"] = obv

        # Volume SMA
        df["volume_sma"] = volume.rolling(window=20).mean()
        df["volume_ratio"] = volume / df["volume_sma"]

        # VWAP (simplified - intraday)
        typical_price = (df["high"] + df["low"] + close) / 3
        df["vwap"] = (typical_price * volume).cumsum() / volume.cumsum()

        return df

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal/time-based features (US-006).

        Features:
        - Hour of day (cyclical encoding: sin/cos)
        - Day of week (cyclical encoding: sin/cos)
        - Trading session indicator (Asia/Europe/US)
        - Time since market open
        """
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("DataFrame index is not DatetimeIndex, temporal features may be inaccurate")
            return df

        # Hour of day - cyclical encoding
        hour = df.index.hour
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

        # Day of week - cyclical encoding (0=Monday, 6=Sunday)
        dayofweek = df.index.dayofweek
        df["dow_sin"] = np.sin(2 * np.pi * dayofweek / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dayofweek / 7)

        # Trading session indicators (based on UTC)
        # Asia: 00:00 - 08:00 UTC (Tokyo, Sydney, Hong Kong)
        # Europe: 07:00 - 16:00 UTC (London, Frankfurt)
        # US: 13:00 - 22:00 UTC (New York)
        # Sessions overlap, so we use separate binary indicators
        df["session_asia"] = ((hour >= 0) & (hour < 8)).astype(int)
        df["session_europe"] = ((hour >= 7) & (hour < 16)).astype(int)
        df["session_us"] = ((hour >= 13) & (hour < 22)).astype(int)

        # Combined session indicator (0=Asia-only, 1=Europe, 2=US, 3=overlap)
        df["session"] = (
            df["session_asia"] * 1 +
            df["session_europe"] * 2 +
            df["session_us"] * 4
        )

        # Time since market week open (Monday 00:00 UTC)
        # Measured in fraction of week (0.0 to 1.0)
        minutes_in_week = 7 * 24 * 60
        minutes_from_monday = dayofweek * 24 * 60 + hour * 60 + df.index.minute
        df["week_progress"] = minutes_from_monday / minutes_in_week

        # Is weekend (typically lower liquidity)
        df["is_weekend"] = (dayofweek >= 5).astype(int)

        logger.debug("Added temporal features")
        return df

    def add_volatility_regime(
        self,
        df: pd.DataFrame,
        window: int = 20,
        percentile_low: float = 33,
        percentile_high: float = 67,
    ) -> pd.DataFrame:
        """
        Add volatility regime features (US-007).

        Classifies market into volatility regimes:
        - Low volatility (regime=0): ATR below 33rd percentile
        - Medium volatility (regime=1): ATR between 33rd and 67th percentile
        - High volatility (regime=2): ATR above 67th percentile

        Args:
            df: DataFrame with ATR already calculated
            window: Rolling window for percentile calculation
            percentile_low: Percentile threshold for low volatility
            percentile_high: Percentile threshold for high volatility
        """
        # Ensure ATR exists
        if "atr" not in df.columns:
            df = self.add_volatility_indicators(df)

        atr = df["atr"]

        # Rolling percentile rank of ATR
        def percentile_rank(x):
            if len(x) < 2:
                return 50
            return (x.values < x.values[-1]).sum() / (len(x) - 1) * 100

        df["atr_percentile"] = atr.rolling(window=window * 5).apply(percentile_rank, raw=False)

        # Volatility regime classification
        df["vol_regime"] = 1  # Default to medium
        df.loc[df["atr_percentile"] <= percentile_low, "vol_regime"] = 0  # Low
        df.loc[df["atr_percentile"] >= percentile_high, "vol_regime"] = 2  # High

        # Regime persistence (how long in current regime)
        df["regime_change"] = (df["vol_regime"] != df["vol_regime"].shift(1)).astype(int)
        df["regime_duration"] = df.groupby(df["regime_change"].cumsum()).cumcount() + 1

        # Volatility trend (is volatility increasing or decreasing)
        atr_sma_fast = atr.rolling(window=5).mean()
        atr_sma_slow = atr.rolling(window=20).mean()
        df["vol_trend"] = np.where(atr_sma_fast > atr_sma_slow, 1, -1)

        # Volatility expansion/contraction
        df["vol_expansion"] = (atr > atr.shift(1)).astype(int)

        # Relative volatility (current vs historical average)
        df["vol_relative"] = atr / atr.rolling(window=100).mean()

        # Clean up intermediate column
        df.drop(columns=["regime_change"], inplace=True, errors="ignore")

        logger.debug("Added volatility regime features")
        return df


def calculate_all_indicators(
    df: pd.DataFrame,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_period: int = 20,
    bb_std: float = 2.0,
    ema_periods: List[int] = None,
    atr_period: int = 14,
) -> pd.DataFrame:
    """
    Convenience function to calculate all technical indicators.

    Args:
        df: OHLCV DataFrame
        **kwargs: Indicator parameters

    Returns:
        DataFrame with indicators added
    """
    calculator = TechnicalIndicators(
        rsi_period=rsi_period,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
        bb_period=bb_period,
        bb_std=bb_std,
        ema_periods=ema_periods,
        atr_period=atr_period,
    )
    return calculator.calculate_all(df)
