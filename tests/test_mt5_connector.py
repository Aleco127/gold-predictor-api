"""Tests for MT5 Connector module."""

import pytest
import pandas as pd
import numpy as np

from src.data.mt5_connector import MT5Connector


class TestMT5Connector:
    """Test suite for MT5Connector."""

    def test_init(self):
        """Test connector initialization."""
        connector = MT5Connector(
            login=12345,
            password="test",
            server="TestServer",
        )
        assert connector.login == 12345
        assert connector.server == "TestServer"
        assert connector.symbol == "XAUUSD"

    def test_mock_data_generation(self):
        """Test mock data generation."""
        connector = MT5Connector(
            login=12345,
            password="test",
            server="TestServer",
        )
        connector._connected = True  # Simulate connection

        df = connector._generate_mock_data(100)

        assert len(df) == 100
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "tick_volume" in df.columns

        # Verify OHLC relationships
        assert (df["high"] >= df["open"]).all()
        assert (df["high"] >= df["close"]).all()
        assert (df["low"] <= df["open"]).all()
        assert (df["low"] <= df["close"]).all()

    def test_get_ohlcv_mock(self):
        """Test OHLCV data retrieval with mock data."""
        connector = MT5Connector(
            login=12345,
            password="test",
            server="TestServer",
        )
        connector._connected = True

        df = connector.get_ohlcv(bars=500)

        assert len(df) == 500
        assert isinstance(df, pd.DataFrame)

    def test_get_current_price_mock(self):
        """Test current price retrieval with mock data."""
        connector = MT5Connector(
            login=12345,
            password="test",
            server="TestServer",
        )
        connector._connected = True

        price = connector.get_current_price()

        assert "bid" in price
        assert "ask" in price
        assert "spread" in price
        assert price["ask"] > price["bid"]

    def test_context_manager(self):
        """Test context manager usage."""
        # Note: This will use mock mode since MT5 isn't available
        with MT5Connector(
            login=12345,
            password="test",
            server="TestServer",
        ) as connector:
            assert connector.is_connected()

        # After exit, should be disconnected
        assert not connector.is_connected()
