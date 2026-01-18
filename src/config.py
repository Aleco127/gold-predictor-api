"""
Configuration module for Gold Predictor.
Uses pydantic-settings for environment variable management.
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # MetaTrader 5 Configuration
    mt5_login: int = 0
    mt5_password: str = ""
    mt5_server: str = "ICMarketsSC-Demo"

    # Capital.com API Configuration
    capital_api_key: str = ""
    capital_password: str = ""
    capital_identifier: str = ""  # Email or API key
    capital_demo: bool = True  # Use demo environment

    # Data source selection: "capital" or "mt5"
    data_source: str = "capital"

    # API Security
    api_key: str = "dev-key-change-in-production"

    # Model Configuration
    model_lookback: int = 60  # Number of candles to look back
    prediction_horizon: int = 1  # Predict next N candles
    confidence_threshold: float = 0.7  # Minimum confidence for signals

    # Trading Symbol
    symbol: str = "XAUUSD"
    timeframe: str = "M5"  # 5-minute timeframe

    # Scheduler Configuration
    prediction_interval_minutes: int = 5

    # Logging
    log_level: str = "INFO"

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000

    # Database
    database_url: str = "sqlite:///./predictions.db"

    # Model Paths
    lstm_model_path: str = "models/lstm_model.pt"
    xgb_model_path: str = "models/xgb_model.json"
    scaler_path: str = "models/scaler.joblib"

    # Ensemble Weights
    lstm_weight: float = 0.6
    xgb_weight: float = 0.4

    # Risk Management
    daily_loss_limit_pct: float = 3.0  # Max daily loss as % of account
    max_daily_trades: int = 50  # Max trades per day
    default_account_balance: float = 10000.0  # Default for risk calculations

    # Position Sizing
    base_position_size: float = 0.01  # Base position size in lots
    max_position_size: float = 0.1  # Maximum position size in lots
    min_position_size: float = 0.01  # Minimum position size in lots
    volatility_high_threshold: float = 1.5  # ATR multiplier for high volatility (reduce size)
    volatility_low_threshold: float = 0.7  # ATR multiplier for low volatility (increase size)
    volatility_lookback: int = 20  # Periods for average ATR calculation

    # Feature Configuration
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    ema_periods: list[int] = [9, 21, 50]
    atr_period: int = 14

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience accessor
settings = get_settings()
