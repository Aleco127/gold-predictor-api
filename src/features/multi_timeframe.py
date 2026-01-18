"""
Multi-Timeframe Analysis Module
===============================
Fetches and analyzes price data across multiple timeframes for trend confirmation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import logging

import pandas as pd
import numpy as np

from .technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

# Default timeframes for multi-timeframe analysis
DEFAULT_TIMEFRAMES = ["M5", "M15", "H1", "H4"]

# Bars to fetch per timeframe
DEFAULT_BARS = 200


@dataclass
class TimeframeData:
    """Data container for a single timeframe."""
    timeframe: str
    data: pd.DataFrame
    indicators: pd.DataFrame
    last_updated: datetime
    bars_count: int

    # Key indicator values (most recent)
    current_price: float = 0.0
    ema_9: float = 0.0
    ema_21: float = 0.0
    ema_50: float = 0.0
    rsi: float = 0.0
    macd: float = 0.0
    macd_signal: float = 0.0
    atr: float = 0.0
    bb_upper: float = 0.0
    bb_lower: float = 0.0
    bb_middle: float = 0.0

    # Derived signals
    trend_direction: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    ema_alignment: str = "NEUTRAL"  # BULLISH, BEARISH, MIXED
    rsi_condition: str = "NEUTRAL"  # OVERBOUGHT, OVERSOLD, NEUTRAL
    price_vs_bb: str = "MIDDLE"  # ABOVE_UPPER, BELOW_LOWER, MIDDLE

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timeframe": self.timeframe,
            "last_updated": self.last_updated.isoformat(),
            "bars_count": self.bars_count,
            "current_price": round(self.current_price, 2),
            "indicators": {
                "ema_9": round(self.ema_9, 2),
                "ema_21": round(self.ema_21, 2),
                "ema_50": round(self.ema_50, 2),
                "rsi": round(self.rsi, 2),
                "macd": round(self.macd, 4),
                "macd_signal": round(self.macd_signal, 4),
                "atr": round(self.atr, 2),
                "bb_upper": round(self.bb_upper, 2),
                "bb_middle": round(self.bb_middle, 2),
                "bb_lower": round(self.bb_lower, 2),
            },
            "signals": {
                "trend_direction": self.trend_direction,
                "ema_alignment": self.ema_alignment,
                "rsi_condition": self.rsi_condition,
                "price_vs_bb": self.price_vs_bb,
            },
        }


@dataclass
class MultiTimeframeAnalysis:
    """Complete multi-timeframe analysis result."""
    symbol: str
    timeframes: Dict[str, TimeframeData]
    analysis_time: datetime

    # Confluence metrics (calculated in US-006)
    confluence_score: float = 0.0
    overall_trend: str = "NEUTRAL"
    trend_strength: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "analysis_time": self.analysis_time.isoformat(),
            "timeframes": {tf: data.to_dict() for tf, data in self.timeframes.items()},
            "confluence": {
                "score": round(self.confluence_score, 2),
                "overall_trend": self.overall_trend,
                "trend_strength": round(self.trend_strength, 2),
            },
        }


class MultiTimeframeAnalyzer:
    """
    Analyze price data across multiple timeframes.

    Fetches data for M5, M15, H1, H4 timeframes and calculates
    technical indicators for each to provide trend confirmation.
    """

    def __init__(
        self,
        data_connector: Any,
        indicator_calculator: Optional[TechnicalIndicators] = None,
        timeframes: Optional[List[str]] = None,
        bars_per_timeframe: int = DEFAULT_BARS,
    ):
        """
        Initialize multi-timeframe analyzer.

        Args:
            data_connector: Data connector (CapitalConnector or MT5Connector)
            indicator_calculator: Technical indicator calculator (creates default if None)
            timeframes: List of timeframes to analyze (default: M5, M15, H1, H4)
            bars_per_timeframe: Number of bars to fetch per timeframe
        """
        self.data_connector = data_connector
        self.indicator_calculator = indicator_calculator or TechnicalIndicators()
        self.timeframes = timeframes or DEFAULT_TIMEFRAMES
        self.bars_per_timeframe = bars_per_timeframe

        # Cache for timeframe data
        self._cache: Dict[str, TimeframeData] = {}
        self._last_fetch: Optional[datetime] = None

    def analyze(
        self,
        symbol: Optional[str] = None,
        force_refresh: bool = False,
    ) -> MultiTimeframeAnalysis:
        """
        Perform multi-timeframe analysis.

        Args:
            symbol: Trading symbol (uses connector default if None)
            force_refresh: Force data refresh even if cached

        Returns:
            MultiTimeframeAnalysis with data for all timeframes
        """
        resolved_symbol: str = symbol or getattr(self.data_connector, 'symbol', None) or 'XAUUSD'

        # Fetch and analyze each timeframe
        timeframe_results = {}

        for tf in self.timeframes:
            try:
                tf_data = self._analyze_timeframe(resolved_symbol, tf, force_refresh)
                timeframe_results[tf] = tf_data
                self._cache[tf] = tf_data
            except Exception as e:
                logger.error(f"Error analyzing {tf}: {e}")
                # Use cached data if available
                if tf in self._cache:
                    timeframe_results[tf] = self._cache[tf]

        analysis = MultiTimeframeAnalysis(
            symbol=resolved_symbol,
            timeframes=timeframe_results,
            analysis_time=datetime.now(timezone.utc),
        )

        self._last_fetch = datetime.now(timezone.utc)

        return analysis

    def _analyze_timeframe(
        self,
        symbol: str,
        timeframe: str,
        force_refresh: bool = False,
    ) -> TimeframeData:
        """Analyze a single timeframe."""
        # Check cache if not forcing refresh
        if not force_refresh and timeframe in self._cache:
            cache_age = (datetime.now(timezone.utc) - self._cache[timeframe].last_updated).total_seconds()
            # Cache validity based on timeframe
            cache_validity = self._get_cache_validity(timeframe)
            if cache_age < cache_validity:
                logger.debug(f"Using cached {timeframe} data (age: {cache_age:.0f}s)")
                return self._cache[timeframe]

        # Fetch data
        logger.info(f"Fetching {timeframe} data for {symbol}...")

        # Calculate days needed based on timeframe and bars
        days_needed = self._calculate_days_needed(timeframe)

        df = self.data_connector.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            days=days_needed,
            use_cache=True,
        )

        if df is None or df.empty:
            raise ValueError(f"No data received for {timeframe}")

        # Limit to requested bars
        df = df.tail(self.bars_per_timeframe).copy()

        # Calculate indicators
        df_with_indicators = self.indicator_calculator.calculate_all(
            df,
            include_temporal=False,
            include_regime=True,
        )

        # Extract latest values
        latest = df_with_indicators.iloc[-1]

        # Create TimeframeData
        tf_data = TimeframeData(
            timeframe=timeframe,
            data=df,
            indicators=df_with_indicators,
            last_updated=datetime.now(timezone.utc),
            bars_count=len(df),
            current_price=float(latest.get("close", 0)),
            ema_9=float(latest.get("ema_9", 0)),
            ema_21=float(latest.get("ema_21", 0)),
            ema_50=float(latest.get("ema_50", 0)),
            rsi=float(latest.get("rsi", 50)),
            macd=float(latest.get("macd", 0)),
            macd_signal=float(latest.get("macd_signal", 0)),
            atr=float(latest.get("atr", 0)),
            bb_upper=float(latest.get("bb_upper", 0)),
            bb_lower=float(latest.get("bb_lower", 0)),
            bb_middle=float(latest.get("bb_middle", 0)),
        )

        # Calculate derived signals
        self._calculate_signals(tf_data)

        return tf_data

    def _calculate_signals(self, tf_data: TimeframeData) -> None:
        """Calculate derived signals for a timeframe."""
        # Trend direction based on EMA crossover
        if tf_data.ema_9 > tf_data.ema_21 > tf_data.ema_50:
            tf_data.trend_direction = "BULLISH"
            tf_data.ema_alignment = "BULLISH"
        elif tf_data.ema_9 < tf_data.ema_21 < tf_data.ema_50:
            tf_data.trend_direction = "BEARISH"
            tf_data.ema_alignment = "BEARISH"
        else:
            tf_data.trend_direction = "NEUTRAL"
            tf_data.ema_alignment = "MIXED"

        # RSI condition
        if tf_data.rsi > 70:
            tf_data.rsi_condition = "OVERBOUGHT"
        elif tf_data.rsi < 30:
            tf_data.rsi_condition = "OVERSOLD"
        else:
            tf_data.rsi_condition = "NEUTRAL"

        # Price vs Bollinger Bands
        if tf_data.current_price > tf_data.bb_upper:
            tf_data.price_vs_bb = "ABOVE_UPPER"
        elif tf_data.current_price < tf_data.bb_lower:
            tf_data.price_vs_bb = "BELOW_LOWER"
        else:
            tf_data.price_vs_bb = "MIDDLE"

    def _get_cache_validity(self, timeframe: str) -> int:
        """Get cache validity in seconds based on timeframe."""
        validity_map = {
            "M1": 30,     # 30 seconds
            "M5": 120,    # 2 minutes
            "M15": 300,   # 5 minutes
            "M30": 600,   # 10 minutes
            "H1": 1200,   # 20 minutes
            "H4": 3600,   # 1 hour
            "D1": 7200,   # 2 hours
        }
        return validity_map.get(timeframe, 300)

    def _calculate_days_needed(self, timeframe: str) -> int:
        """Calculate days of data needed for the requested bars."""
        # Bars per day for each timeframe
        bars_per_day = {
            "M1": 1440,
            "M5": 288,
            "M15": 96,
            "M30": 48,
            "H1": 24,
            "H4": 6,
            "D1": 1,
        }

        bpd = bars_per_day.get(timeframe, 288)
        days = max(1, int(np.ceil(self.bars_per_timeframe / bpd)) + 1)

        return days

    def get_timeframe_data(self, timeframe: str) -> Optional[TimeframeData]:
        """Get cached data for a specific timeframe."""
        return self._cache.get(timeframe)

    def get_all_cached(self) -> Dict[str, TimeframeData]:
        """Get all cached timeframe data."""
        return self._cache.copy()

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all timeframe signals."""
        if not self._cache:
            return {"message": "No data cached. Run analyze() first."}

        summary: Dict[str, Any] = {
            "last_fetch": self._last_fetch.isoformat() if self._last_fetch else None,
            "timeframes": {},
            "overall_trend": "NEUTRAL",
            "bullish_timeframes": 0,
            "bearish_timeframes": 0,
            "neutral_timeframes": 0,
        }

        bullish_count = 0
        bearish_count = 0

        for tf, data in self._cache.items():
            summary["timeframes"][tf] = {
                "trend": data.trend_direction,
                "rsi": round(data.rsi, 2),
                "ema_alignment": data.ema_alignment,
                "price": round(data.current_price, 2),
            }

            if data.trend_direction == "BULLISH":
                bullish_count += 1
            elif data.trend_direction == "BEARISH":
                bearish_count += 1

        # Overall trend based on majority
        total = len(self._cache)
        if bullish_count > total / 2:
            summary["overall_trend"] = "BULLISH"
        elif bearish_count > total / 2:
            summary["overall_trend"] = "BEARISH"
        else:
            summary["overall_trend"] = "NEUTRAL"

        summary["bullish_timeframes"] = bullish_count
        summary["bearish_timeframes"] = bearish_count
        summary["neutral_timeframes"] = total - bullish_count - bearish_count

        return summary

    def get_status(self) -> dict:
        """Get analyzer status."""
        return {
            "timeframes": self.timeframes,
            "bars_per_timeframe": self.bars_per_timeframe,
            "cached_timeframes": list(self._cache.keys()),
            "last_fetch": self._last_fetch.isoformat() if self._last_fetch else None,
            "data_connector_type": type(self.data_connector).__name__,
        }

    def calculate_confluence(
        self,
        analysis: MultiTimeframeAnalysis,
    ) -> MultiTimeframeAnalysis:
        """
        Calculate confluence score across all timeframes.

        Confluence measures how well multiple timeframes agree on direction.
        Score ranges from 0-100%:
        - 100% = All timeframes strongly agree
        - 0% = Complete disagreement

        Args:
            analysis: MultiTimeframeAnalysis with timeframe data

        Returns:
            Updated MultiTimeframeAnalysis with confluence metrics
        """
        if not analysis.timeframes:
            return analysis

        # Count trend directions
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0

        # Count RSI conditions
        overbought_count = 0
        oversold_count = 0

        # Count EMA alignments
        ema_bullish = 0
        ema_bearish = 0

        total_timeframes = len(analysis.timeframes)

        for tf, data in analysis.timeframes.items():
            # Trend direction
            if data.trend_direction == "BULLISH":
                bullish_count += 1
            elif data.trend_direction == "BEARISH":
                bearish_count += 1
            else:
                neutral_count += 1

            # RSI condition
            if data.rsi_condition == "OVERBOUGHT":
                overbought_count += 1
            elif data.rsi_condition == "OVERSOLD":
                oversold_count += 1

            # EMA alignment
            if data.ema_alignment == "BULLISH":
                ema_bullish += 1
            elif data.ema_alignment == "BEARISH":
                ema_bearish += 1

        # Calculate trend agreement percentage
        max_trend_agreement = max(bullish_count, bearish_count, neutral_count)
        trend_agreement = max_trend_agreement / total_timeframes

        # Calculate EMA agreement percentage
        max_ema_agreement = max(ema_bullish, ema_bearish, total_timeframes - ema_bullish - ema_bearish)
        ema_agreement = max_ema_agreement / total_timeframes

        # RSI confluence bonus/penalty
        # If RSI agrees with trend, it's a bonus
        rsi_factor = 1.0
        if bullish_count > bearish_count and oversold_count > 0:
            # Bullish trend with oversold RSI = strong buy signal
            rsi_factor = 1.1
        elif bearish_count > bullish_count and overbought_count > 0:
            # Bearish trend with overbought RSI = strong sell signal
            rsi_factor = 1.1
        elif bullish_count > bearish_count and overbought_count > total_timeframes / 2:
            # Bullish trend but overbought = weakening
            rsi_factor = 0.9
        elif bearish_count > bullish_count and oversold_count > total_timeframes / 2:
            # Bearish trend but oversold = weakening
            rsi_factor = 0.9

        # Calculate confluence score (0-100%)
        # Weighted: 60% trend agreement, 30% EMA alignment, 10% RSI factor
        raw_confluence = (
            trend_agreement * 0.6 +
            ema_agreement * 0.3 +
            0.1 * rsi_factor
        )
        confluence_score = min(raw_confluence * 100, 100.0)

        # Determine overall trend
        if bullish_count > bearish_count and bullish_count > neutral_count:
            overall_trend = "BULLISH"
        elif bearish_count > bullish_count and bearish_count > neutral_count:
            overall_trend = "BEARISH"
        else:
            overall_trend = "NEUTRAL"

        # Calculate trend strength (0-100%)
        # Based on how strongly the majority agrees
        if total_timeframes > 0:
            trend_strength = (max_trend_agreement / total_timeframes) * 100
        else:
            trend_strength = 0.0

        # Update analysis object
        analysis.confluence_score = confluence_score
        analysis.overall_trend = overall_trend
        analysis.trend_strength = trend_strength

        logger.info(
            f"Confluence calculated: score={confluence_score:.1f}%, "
            f"trend={overall_trend}, strength={trend_strength:.1f}%"
        )

        return analysis

    def analyze_with_confluence(
        self,
        symbol: Optional[str] = None,
        force_refresh: bool = False,
    ) -> MultiTimeframeAnalysis:
        """
        Perform multi-timeframe analysis with confluence calculation.

        This is a convenience method that calls analyze() followed by
        calculate_confluence().

        Args:
            symbol: Trading symbol (uses connector default if None)
            force_refresh: Force data refresh even if cached

        Returns:
            MultiTimeframeAnalysis with confluence metrics populated
        """
        analysis = self.analyze(symbol, force_refresh)
        return self.calculate_confluence(analysis)
