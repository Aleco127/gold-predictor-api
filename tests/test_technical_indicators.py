"""Tests for Technical Indicators module."""

import pytest
import pandas as pd
import numpy as np

from src.features.technical_indicators import TechnicalIndicators, calculate_all_indicators


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data."""
    np.random.seed(42)
    n = 200

    # Generate random walk prices
    base_price = 2650.0
    returns = np.random.randn(n) * 0.001
    prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        "open": prices,
        "high": prices + np.abs(np.random.randn(n)) * 0.5,
        "low": prices - np.abs(np.random.randn(n)) * 0.5,
        "close": prices + np.random.randn(n) * 0.2,
        "volume": np.random.randint(100, 10000, n),
    })

    # Fix OHLC relationships
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)

    return df


class TestTechnicalIndicators:
    """Test suite for TechnicalIndicators."""

    def test_init(self):
        """Test indicator calculator initialization."""
        calc = TechnicalIndicators(
            rsi_period=14,
            macd_fast=12,
            macd_slow=26,
        )
        assert calc.rsi_period == 14
        assert calc.macd_fast == 12
        assert calc.macd_slow == 26

    def test_calculate_all(self, sample_ohlcv):
        """Test calculation of all indicators."""
        calc = TechnicalIndicators()
        result = calc.calculate_all(sample_ohlcv)

        # Check that indicators were added
        assert "rsi" in result.columns
        assert "macd" in result.columns
        assert "bb_upper" in result.columns
        assert "atr" in result.columns
        assert "ema_9" in result.columns

    def test_rsi_bounds(self, sample_ohlcv):
        """Test that RSI is within valid range."""
        calc = TechnicalIndicators()
        result = calc.calculate_all(sample_ohlcv)

        rsi = result["rsi"].dropna()
        assert (rsi >= 0).all()
        assert (rsi <= 100).all()

    def test_bollinger_bands(self, sample_ohlcv):
        """Test Bollinger Bands relationships."""
        calc = TechnicalIndicators()
        result = calc.calculate_all(sample_ohlcv)

        # Upper should be above middle, middle above lower
        valid_mask = result[["bb_upper", "bb_mid", "bb_lower"]].notna().all(axis=1)
        if valid_mask.any():
            assert (result.loc[valid_mask, "bb_upper"] >= result.loc[valid_mask, "bb_mid"]).all()
            assert (result.loc[valid_mask, "bb_mid"] >= result.loc[valid_mask, "bb_lower"]).all()

    def test_atr_positive(self, sample_ohlcv):
        """Test that ATR is always positive."""
        calc = TechnicalIndicators()
        result = calc.calculate_all(sample_ohlcv)

        atr = result["atr"].dropna()
        assert (atr >= 0).all()

    def test_convenience_function(self, sample_ohlcv):
        """Test calculate_all_indicators convenience function."""
        result = calculate_all_indicators(sample_ohlcv)

        assert "rsi" in result.columns
        assert "macd" in result.columns
        assert len(result) == len(sample_ohlcv)

    def test_missing_volume(self, sample_ohlcv):
        """Test handling of missing volume column."""
        df = sample_ohlcv.drop(columns=["volume"])
        calc = TechnicalIndicators()
        result = calc.calculate_all(df)

        # Should still work, just skip volume indicators
        assert "rsi" in result.columns
