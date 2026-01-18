"""
FastAPI Server Module
=====================
REST API for gold price predictions with authentication and monitoring.
"""

import os
import secrets
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from loguru import logger
from pydantic import BaseModel, Field

from ..config import settings
from ..data.mt5_connector import MT5Connector
from ..data.capital_connector import CapitalConnector
from ..features.technical_indicators import TechnicalIndicators
from ..models.ensemble import EnsemblePredictor, Signal
from ..preprocessing.data_processor import DataProcessor
from ..trading import (
    RiskManager,
    DailyStats,
    PositionSizeResult,
    PositionManager,
    TrailingStopUpdate,
)
from ..data.economic_calendar import EconomicCalendar, ImpactLevel, NewsFilter, NewsFilterResult
from ..data.news_fetcher import NewsFetcher, NewsArticle
from ..features.sentiment_analyzer import SentimentAnalyzer, AggregateSentiment
from ..storage.trade_database import TradeDatabase, TradeFilter, TradeSummary
from ..monitoring.performance_tracker import TradingPerformanceTracker, PerformanceMetrics, DrawdownInfo

# Global instances
predictor: Optional[EnsemblePredictor] = None
data_processor: Optional[DataProcessor] = None
data_connector = None  # Either MT5Connector or CapitalConnector
indicator_calculator: Optional[TechnicalIndicators] = None
risk_manager: Optional[RiskManager] = None
position_manager: Optional[PositionManager] = None
economic_calendar: Optional[EconomicCalendar] = None
news_filter: Optional[NewsFilter] = None
news_fetcher: Optional[NewsFetcher] = None
sentiment_analyzer: Optional[SentimentAnalyzer] = None
trade_db: Optional[TradeDatabase] = None
trading_performance_tracker: Optional[TradingPerformanceTracker] = None


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    symbol: str = Field(default="XAUUSD", description="Trading symbol")
    timeframe: str = Field(default="M5", description="Timeframe")


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    signal: str
    confidence: float
    current_price: float
    predicted_price: float
    predicted_change_percent: float
    direction_probabilities: dict
    should_alert: bool
    timestamp: str
    message: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    data_source: str
    data_connected: bool
    timestamp: str
    version: str = "1.0.0"


class TrainRequest(BaseModel):
    """Request model for training."""
    days: int = Field(default=30, ge=1, le=365, description="Days of historical data")
    epochs: int = Field(default=100, ge=10, le=500, description="Training epochs")


class TrainResponse(BaseModel):
    """Response model for training."""
    status: str
    message: str
    metrics: Optional[dict] = None


class TradeRequest(BaseModel):
    """Request model for trade execution."""
    signal: str = Field(..., description="Trading signal (BUY, SELL, STRONG_BUY, STRONG_SELL)")
    confidence: float = Field(default=0.5, ge=0, le=1, description="Signal confidence 0-1")
    size: float = Field(default=0.01, ge=0.01, le=10.0, description="Position size in lots")
    auto_sl_tp: bool = Field(default=True, description="Auto-calculate SL/TP")
    stop_loss: Optional[float] = Field(default=None, description="Manual stop loss level")
    take_profit: Optional[float] = Field(default=None, description="Manual take profit level")


class TradeResponse(BaseModel):
    """Response model for trade execution."""
    success: bool
    deal_reference: Optional[str] = None
    deal_id: Optional[str] = None
    direction: Optional[str] = None
    size: Optional[float] = None
    symbol: Optional[str] = None
    status: Optional[str] = None
    level: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    error: Optional[str] = None
    timestamp: str


class PositionsResponse(BaseModel):
    """Response model for positions list."""
    positions: list
    count: int
    timestamp: str


class AccountResponse(BaseModel):
    """Response model for account info."""
    account_id: Optional[str] = None
    balance: Optional[float] = None
    available: Optional[float] = None
    profit_loss: Optional[float] = None
    currency: Optional[str] = None
    timestamp: str


class DailyStatsResponse(BaseModel):
    """Response model for daily risk stats."""
    date: str
    total_pnl: float
    trade_count: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    largest_win: float
    largest_loss: float
    trading_allowed: bool
    limit_reached_at: Optional[str] = None


class RiskStatusResponse(BaseModel):
    """Response model for full risk status."""
    date: str
    account_balance: float
    daily_loss_limit_pct: float
    daily_loss_limit_amount: float
    current_daily_pnl: float
    remaining_risk: float
    trades_today: int
    max_daily_trades: int
    trading_allowed: bool
    blocked_reason: Optional[str] = None
    blocked_at: Optional[str] = None


class PositionSizeResponse(BaseModel):
    """Response model for position size calculation."""
    adjusted_size: float
    base_size: float
    adjustment_factor: float
    volatility_regime: str
    current_atr: float
    average_atr: float
    atr_ratio: float
    reason: str


class PositionSizingConfigResponse(BaseModel):
    """Response model for position sizing configuration."""
    base_position_size: float
    max_position_size: float
    min_position_size: float
    volatility_high_threshold: float
    volatility_low_threshold: float
    volatility_lookback: int


class TrackedPositionResponse(BaseModel):
    """Response model for tracked position with trailing stop info."""
    deal_id: str
    direction: str
    symbol: str
    size: float
    entry_price: float
    current_price: float
    current_stop_loss: float
    original_stop_loss: float
    take_profit: Optional[float]
    pnl_pips: float
    trailing_activated: bool
    highest_price: Optional[float] = None
    lowest_price: Optional[float] = None
    tp_levels: list = []
    tp_levels_hit: list = []
    opened_at: str
    last_stop_update: Optional[str] = None


class TrailingStopUpdateResponse(BaseModel):
    """Response model for trailing stop calculation."""
    should_update: bool
    new_stop_loss: float
    current_price: float
    profit_pips: float
    reason: str
    api_update_result: Optional[dict] = None


class TrailingStopConfigResponse(BaseModel):
    """Response model for trailing stop configuration."""
    trailing_atr_multiplier: float
    activation_pips: float
    step_pips: float
    pip_value: float
    tracked_positions_count: int


class AddPositionRequest(BaseModel):
    """Request model for adding a position to track."""
    deal_id: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: Optional[float] = None
    size: float
    symbol: str = "XAUUSD"
    entry_atr: float = 0.0


class UpdateTrailingStopRequest(BaseModel):
    """Request model for updating trailing stop."""
    deal_id: str
    current_price: float
    current_atr: Optional[float] = None
    apply_to_broker: bool = True  # Whether to update stop-loss via API


class EconomicEventResponse(BaseModel):
    """Response model for a single economic event."""
    datetime_utc: str
    currency: str
    impact: str
    event_name: str
    actual: Optional[str] = None
    forecast: Optional[str] = None
    previous: Optional[str] = None
    is_high_impact: bool


class CalendarStatusResponse(BaseModel):
    """Response model for economic calendar status."""
    events_cached: int
    last_fetch: Optional[str] = None
    cache_age_minutes: Optional[float] = None
    refresh_interval_minutes: float
    is_near_high_impact: bool
    near_event: Optional[dict] = None
    relevant_currencies: list


# API Key security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Verify API key from header."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key missing",
        )

    if api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    return api_key


def get_predictor() -> EnsemblePredictor:
    """Get predictor instance."""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor not initialized",
        )
    return predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global predictor, data_processor, data_connector, indicator_calculator, risk_manager, position_manager, economic_calendar, news_filter, news_fetcher, sentiment_analyzer, trade_db, trading_performance_tracker

    logger.info("Starting Gold Predictor API...")

    # Initialize components
    try:
        # Initialize data connector based on configuration
        if settings.data_source.lower() == "capital":
            logger.info("Using Capital.com API as data source...")
            data_connector = CapitalConnector(
                api_key=settings.capital_api_key,
                password=settings.capital_password,
                identifier=settings.capital_identifier or settings.capital_api_key,
                demo=settings.capital_demo,
                symbol=settings.symbol,
            )
            data_connector.connect()
        else:
            logger.info("Using MetaTrader 5 as data source...")
            data_connector = MT5Connector(
                login=settings.mt5_login,
                password=settings.mt5_password,
                server=settings.mt5_server,
                symbol=settings.symbol,
            )
            data_connector.connect()

        # Initialize indicator calculator
        indicator_calculator = TechnicalIndicators(
            rsi_period=settings.rsi_period,
            macd_fast=settings.macd_fast,
            macd_slow=settings.macd_slow,
            macd_signal=settings.macd_signal,
            bb_period=settings.bb_period,
            bb_std=settings.bb_std,
            ema_periods=settings.ema_periods,
            atr_period=settings.atr_period,
        )

        # Load pre-trained models if available
        lstm_path = Path(settings.lstm_model_path)
        xgb_path = Path(settings.xgb_model_path)
        scaler_path = Path(settings.scaler_path)

        if lstm_path.exists() and xgb_path.exists():
            predictor = EnsemblePredictor(
                lstm_weight=settings.lstm_weight,
                xgb_weight=settings.xgb_weight,
                confidence_threshold=settings.confidence_threshold,
            )
            predictor.load_models(
                lstm_path=str(lstm_path),
                xgb_path=str(xgb_path),
            )

            if scaler_path.exists():
                data_processor = DataProcessor.load(str(scaler_path))

            logger.info("Models loaded successfully")
        else:
            logger.warning("Pre-trained models not found. Train models first.")
            predictor = EnsemblePredictor(
                lstm_weight=settings.lstm_weight,
                xgb_weight=settings.xgb_weight,
                confidence_threshold=settings.confidence_threshold,
            )

        # Initialize risk manager with position sizing
        risk_manager = RiskManager(
            daily_loss_limit_pct=settings.daily_loss_limit_pct,
            account_balance=settings.default_account_balance,
            max_daily_trades=settings.max_daily_trades,
            base_position_size=settings.base_position_size,
            max_position_size=settings.max_position_size,
            min_position_size=settings.min_position_size,
            volatility_high_threshold=settings.volatility_high_threshold,
            volatility_low_threshold=settings.volatility_low_threshold,
            volatility_lookback=settings.volatility_lookback,
        )
        logger.info("Risk manager initialized with position sizing")

        # Initialize position manager for trailing stops
        position_manager = PositionManager(
            trailing_atr_multiplier=settings.trailing_stop_atr_multiplier,
            activation_pips=settings.trailing_activation_pips,
            step_pips=settings.trailing_step_pips,
            pip_value=0.01,  # Gold pip value
        )
        logger.info("Position manager initialized for trailing stops")

        # Initialize economic calendar
        economic_calendar = EconomicCalendar(
            refresh_interval_minutes=60,
            relevant_currencies=["USD", "EUR", "GBP", "JPY", "CHF", "CNY"],
        )
        # Fetch initial events
        await economic_calendar.fetch_events()
        logger.info("Economic calendar initialized")

        # Initialize news filter for trading pause
        news_filter = NewsFilter(
            calendar=economic_calendar,
            pause_minutes_before=30,
            pause_minutes_after=30,
        )
        logger.info("News filter initialized")

        # Initialize news fetcher for sentiment analysis
        if settings.news_api_key:
            news_fetcher = NewsFetcher(
                api_key=settings.news_api_key,
                cache_duration_minutes=settings.news_cache_duration_minutes,
            )
            await news_fetcher.fetch_news()
            logger.info("News fetcher initialized with API key")
        else:
            logger.warning("NEWS_API_KEY not configured, news fetcher disabled")

        # Initialize sentiment analyzer (lazy load to avoid slow startup)
        sentiment_analyzer = SentimentAnalyzer(
            cache_duration_minutes=60,
            lazy_load=True,  # Model loaded on first use
        )
        logger.info("Sentiment analyzer initialized (lazy load)")

        # Initialize trade database
        trade_db = TradeDatabase(database_url="sqlite:///data/trades.db")
        logger.info("Trade database initialized")

        # Initialize trading performance tracker
        trading_performance_tracker = TradingPerformanceTracker(
            trade_db=trade_db,
            initial_balance=settings.default_account_balance,
            risk_free_rate=0.05,  # 5% annualized risk-free rate
        )
        logger.info("Trading performance tracker initialized")

    except Exception as e:
        logger.error(f"Error during startup: {e}")

    yield  # Application runs here

    # Cleanup
    logger.info("Shutting down Gold Predictor API...")
    if data_connector:
        data_connector.disconnect()


# Create FastAPI app
app = FastAPI(
    title="Gold Price Predictor API",
    description="ML-based gold price prediction with LSTM + XGBoost ensemble",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None and predictor.lstm_model is not None,
        data_source=settings.data_source,
        data_connected=data_connector is not None and data_connector.is_connected(),
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest = None,
    api_key: str = Depends(verify_api_key),
):
    """
    Generate price prediction.

    Requires API key authentication.
    Returns prediction with signal, confidence, and price information.
    """
    global predictor, data_processor, data_connector, indicator_calculator

    if predictor is None or predictor.lstm_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not trained. Please train models first.",
        )

    if data_processor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Data processor not initialized. Please train models first.",
        )

    try:
        # Get latest data from data connector (Capital.com or MT5)
        symbol = request.symbol if request else settings.symbol
        df = data_connector.get_ohlcv(
            symbol=symbol,
            bars=settings.model_lookback + 100,  # Extra for indicator calculation
        )

        # Calculate indicators
        df = indicator_calculator.calculate_all(df)

        # Clean and prepare data
        df = data_processor.clean_data(df)

        # Get current price
        current_price = float(df["close"].iloc[-1])

        # Prepare features for prediction
        X = data_processor.prepare_latest(df)

        # Get prediction
        prediction = predictor.predict(
            X=X,
            current_price=current_price,
            target_scaler=data_processor.target_scaler,
        )

        # Format response
        return PredictionResponse(
            signal=prediction.signal.value,
            confidence=prediction.confidence,
            current_price=prediction.current_price,
            predicted_price=prediction.predicted_price,
            predicted_change_percent=prediction.predicted_change_percent,
            direction_probabilities=prediction.direction_probabilities,
            should_alert=predictor.should_alert(prediction),
            timestamp=prediction.timestamp.isoformat(),
            message=predictor.format_alert_message(prediction) if predictor.should_alert(prediction) else None,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.get("/price")
async def get_current_price(
    symbol: str = "GOLD",
    api_key: str = Depends(verify_api_key),
):
    """Get current price for symbol."""
    if data_connector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Data connector not initialized",
        )

    price = data_connector.get_current_price(symbol)
    if not price:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Could not get price for {symbol}",
        )

    return {
        "symbol": symbol,
        "bid": price["bid"],
        "ask": price["ask"],
        "spread": price["spread"],
        "data_source": settings.data_source,
        "timestamp": price["time"].isoformat() if price.get("time") else datetime.now().isoformat(),
    }


@app.post("/train", response_model=TrainResponse)
async def train_models(
    request: TrainRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    Train prediction models on historical data.

    This is a long-running operation. Consider running async or in background.
    """
    global predictor, data_processor, data_connector, indicator_calculator

    try:
        from ..models.lstm_model import GoldLSTM, LSTMTrainer
        from ..models.xgboost_model import GoldXGBoost, calculate_direction_labels

        logger.info(f"Starting training with {request.days} days of data from {settings.data_source}...")

        # Fetch historical data
        bars_needed = request.days * 24 * 12  # 5-min bars
        df = data_connector.get_ohlcv(bars=min(bars_needed, 50000))

        # Calculate indicators
        df = indicator_calculator.calculate_all(df)

        # Initialize data processor
        data_processor = DataProcessor(
            lookback=settings.model_lookback,
            horizon=settings.prediction_horizon,
        )

        # Clean data
        df = data_processor.clean_data(df)

        # Prepare sequences
        X, y = data_processor.fit_transform(df)

        # Calculate direction labels for XGBoost
        y_direction = calculate_direction_labels(
            df["close"].values,
            horizon=settings.prediction_horizon,
        )
        # Align with sequences
        y_direction = y_direction[settings.model_lookback:]

        # Split data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_processor.split_data(X, y)

        # Also split direction labels
        train_end = len(X_train)
        val_end = train_end + len(X_val)
        y_dir_train = y_direction[:train_end]
        y_dir_val = y_direction[train_end:val_end]

        # Train LSTM
        logger.info("Training LSTM model...")
        lstm_model = GoldLSTM(
            input_size=X_train.shape[-1],
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
        )
        trainer = LSTMTrainer(lstm_model, learning_rate=0.001)
        lstm_results = trainer.train(
            X_train, y_train,
            X_val, y_val,
            epochs=request.epochs,
            batch_size=32,
            early_stopping=10,
            save_path=settings.lstm_model_path,
        )

        # Train XGBoost
        logger.info("Training XGBoost model...")
        xgb_model = GoldXGBoost()
        xgb_results = xgb_model.fit(
            X_train, y_dir_train,
            X_val, y_dir_val,
        )
        xgb_model.save(settings.xgb_model_path)

        # Save data processor
        data_processor.save(settings.scaler_path)

        # Update global predictor
        predictor = EnsemblePredictor(
            lstm_model=lstm_model,
            xgb_model=xgb_model,
            lstm_weight=settings.lstm_weight,
            xgb_weight=settings.xgb_weight,
            confidence_threshold=settings.confidence_threshold,
        )

        metrics = {
            "lstm_best_val_loss": lstm_results["best_val_loss"],
            "lstm_epochs_trained": lstm_results["epochs_trained"],
            "xgb_train_accuracy": xgb_results["train_accuracy"],
            "xgb_val_accuracy": xgb_results.get("val_accuracy"),
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "test_samples": len(X_test),
        }

        logger.info(f"Training completed: {metrics}")

        return TrainResponse(
            status="success",
            message="Models trained successfully",
            metrics=metrics,
        )

    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {str(e)}",
        )


@app.get("/metrics")
async def get_metrics(api_key: str = Depends(verify_api_key)):
    """Get model performance metrics."""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor not initialized",
        )

    return {
        "models_loaded": predictor.lstm_model is not None and predictor.xgb_model is not None,
        "lstm_weight": predictor.lstm_weight,
        "xgb_weight": predictor.xgb_weight,
        "confidence_threshold": predictor.confidence_threshold,
        "signal_thresholds": predictor.SIGNAL_THRESHOLDS,
    }


# ========================================
# TRADING ENDPOINTS
# ========================================

@app.post("/execute-trade", response_model=TradeResponse)
async def execute_trade(
    request: TradeRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    Execute a trade based on signal.

    Can be called directly with a signal or automatically after prediction.
    Supports Capital.com trading via their REST API.
    """
    global data_connector

    if data_connector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Data connector not initialized",
        )

    # Check if we have Capital.com connector (has trading methods)
    if not hasattr(data_connector, 'execute_trade_from_signal'):
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Trading not supported with current data source. Use Capital.com.",
        )

    try:
        # Get current price for SL/TP calculation
        price_info = data_connector.get_current_price()
        current_price = price_info.get("bid", 0) if "SELL" in request.signal.upper() else price_info.get("ask", 0)

        if current_price == 0:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Could not get current price",
            )

        # Execute trade
        if request.auto_sl_tp:
            result = data_connector.execute_trade_from_signal(
                signal=request.signal,
                confidence=request.confidence,
                current_price=current_price,
                size=request.size,
            )
        else:
            # Manual SL/TP
            direction = "BUY" if "BUY" in request.signal.upper() else "SELL"
            result = data_connector.create_position(
                direction=direction,
                size=request.size,
                stop_loss=request.stop_loss,
                take_profit=request.take_profit,
            )
            result["signal"] = request.signal
            result["confidence"] = request.confidence

        return TradeResponse(
            success=result.get("success", False),
            deal_reference=result.get("deal_reference"),
            deal_id=result.get("deal_id"),
            direction=result.get("direction"),
            size=result.get("size"),
            symbol=result.get("symbol"),
            status=result.get("status"),
            level=result.get("level"),
            stop_loss=result.get("calculated_sl") or request.stop_loss,
            take_profit=result.get("calculated_tp") or request.take_profit,
            error=result.get("error"),
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Trade execution error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Trade execution failed: {str(e)}",
        )


@app.post("/predict-and-trade", response_model=dict)
async def predict_and_trade(
    size: float = 0.01,
    min_confidence: float = 0.6,
    use_volatility_sizing: bool = True,
    api_key: str = Depends(verify_api_key),
):
    """
    Combined endpoint: Get prediction and execute trade if conditions met.

    Only executes trade if:
    - Signal is BUY/SELL (not HOLD)
    - Confidence >= min_confidence
    - Risk manager allows trading (daily loss limit not reached)

    When use_volatility_sizing is True, position size is adjusted based on
    current ATR vs average ATR (higher volatility = smaller position).

    Returns both prediction and trade result with position sizing info.
    """
    global predictor, data_processor, data_connector, indicator_calculator, risk_manager, news_filter

    if predictor is None or predictor.lstm_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not trained",
        )

    if not hasattr(data_connector, 'execute_trade_from_signal'):
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Trading not supported with current data source",
        )

    try:
        # Check if trading is allowed by risk manager
        can_trade, blocked_reason = risk_manager.can_trade() if risk_manager else (True, None)

        # Check if trading is paused due to high-impact news
        news_check = news_filter.check_trading_allowed() if news_filter else NewsFilterResult(trading_allowed=True, is_paused=False)
        news_paused = news_check.is_paused

        # Get prediction first
        df = data_connector.get_ohlcv(
            bars=settings.model_lookback + 100,
        )
        df = indicator_calculator.calculate_all(df)
        df = data_processor.clean_data(df)
        current_price = float(df["close"].iloc[-1])
        X = data_processor.prepare_latest(df)

        prediction = predictor.predict(
            X=X,
            current_price=current_price,
            target_scaler=data_processor.target_scaler,
        )

        # Calculate volatility-adjusted position size if enabled
        position_sizing_info = None
        trade_size = size

        if use_volatility_sizing and risk_manager and "atr" in df.columns:
            atr_values = df["atr"].dropna().tolist()
            position_result = risk_manager.calculate_position_size(
                atr_values=atr_values,
                base_size=size,
            )
            trade_size = position_result.adjusted_size
            position_sizing_info = position_result.to_dict()

        result = {
            "prediction": {
                "signal": prediction.signal.value,
                "confidence": prediction.confidence,
                "current_price": prediction.current_price,
                "predicted_price": prediction.predicted_price,
                "timestamp": prediction.timestamp.isoformat(),
            },
            "trade": None,
            "trade_executed": False,
            "risk_check": {
                "trading_allowed": can_trade,
                "blocked_reason": blocked_reason,
            },
            "news_paused": news_paused,
            "news_filter": news_check.to_dict() if news_paused else None,
            "position_sizing": position_sizing_info,
        }

        # Check if we should trade
        signal = prediction.signal.value
        signal_allows_trade = (
            "BUY" in signal.upper() or "SELL" in signal.upper()
        ) and prediction.confidence >= min_confidence

        # Only trade if signal, risk manager, and news filter all allow
        should_trade = signal_allows_trade and can_trade and not news_paused

        if should_trade:
            trade_result = data_connector.execute_trade_from_signal(
                signal=signal,
                confidence=prediction.confidence,
                current_price=prediction.current_price,
                size=trade_size,  # Use volatility-adjusted size
            )

            result["trade"] = trade_result
            result["trade_executed"] = trade_result.get("success", False)

            # Record P&L from trade if available (for closed positions from P&L tracking)
            # Note: For open positions, P&L will be recorded when position is closed
            if trade_result.get("success") and risk_manager:
                logger.info(f"Trade executed with size {trade_size}, will track P&L on close")

        elif signal_allows_trade and not can_trade:
            result["trade"] = {"error": f"Trading blocked: {blocked_reason}"}
            logger.warning(f"Trade blocked by risk manager: {blocked_reason}")

        elif signal_allows_trade and news_paused:
            result["trade"] = {"error": f"Trading paused: {news_check.reason}"}
            logger.warning(f"Trade blocked by news filter: {news_check.reason}")

        return result

    except Exception as e:
        logger.error(f"Predict and trade error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Operation failed: {str(e)}",
        )


@app.get("/positions", response_model=PositionsResponse)
async def get_positions(api_key: str = Depends(verify_api_key)):
    """Get all open positions."""
    global data_connector

    if data_connector is None or not hasattr(data_connector, 'list_positions'):
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Positions not supported with current data source",
        )

    try:
        positions = data_connector.list_positions()
        return PositionsResponse(
            positions=positions,
            count=len(positions),
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get positions: {str(e)}",
        )


@app.delete("/positions/{deal_id}")
async def close_position(
    deal_id: str,
    api_key: str = Depends(verify_api_key),
):
    """Close a specific position by deal ID."""
    global data_connector

    if data_connector is None or not hasattr(data_connector, 'close_position'):
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Position closing not supported with current data source",
        )

    try:
        result = data_connector.close_position(deal_id)
        return {
            **result,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error closing position: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to close position: {str(e)}",
        )


@app.delete("/positions")
async def close_all_positions(
    symbol: Optional[str] = None,
    api_key: str = Depends(verify_api_key),
):
    """Close all open positions (optionally filtered by symbol)."""
    global data_connector

    if data_connector is None or not hasattr(data_connector, 'close_all_positions'):
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Position closing not supported with current data source",
        )

    try:
        results = data_connector.close_all_positions(symbol)
        return {
            "closed": len([r for r in results if r.get("success")]),
            "failed": len([r for r in results if not r.get("success")]),
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error closing positions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to close positions: {str(e)}",
        )


@app.get("/account", response_model=AccountResponse)
async def get_account(api_key: str = Depends(verify_api_key)):
    """Get account information and balance."""
    global data_connector

    if data_connector is None or not hasattr(data_connector, 'get_account_info'):
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Account info not supported with current data source",
        )

    try:
        info = data_connector.get_account_info()
        return AccountResponse(
            account_id=info.get("account_id"),
            balance=info.get("balance"),
            available=info.get("available"),
            profit_loss=info.get("profit_loss"),
            currency=info.get("currency"),
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Error getting account info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get account info: {str(e)}",
        )


# ========================================
# RISK MANAGEMENT ENDPOINTS
# ========================================

@app.get("/api/daily-stats", response_model=DailyStatsResponse)
async def get_daily_stats(api_key: str = Depends(verify_api_key)):
    """
    Get daily trading statistics.

    Returns P&L, trade counts, win rate, and whether trading is still allowed.
    """
    global risk_manager

    if risk_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Risk manager not initialized",
        )

    stats = risk_manager.get_daily_stats()
    return DailyStatsResponse(**stats.to_dict())


@app.get("/api/risk-status", response_model=RiskStatusResponse)
async def get_risk_status(api_key: str = Depends(verify_api_key)):
    """
    Get full risk manager status.

    Returns account balance, daily limits, current P&L, remaining risk budget,
    and whether trading is currently allowed.
    """
    global risk_manager

    if risk_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Risk manager not initialized",
        )

    status_data = risk_manager.get_status()
    return RiskStatusResponse(**status_data)


@app.post("/api/record-pnl")
async def record_pnl(
    pnl: float,
    direction: str = "UNKNOWN",
    size: float = 0.01,
    entry_price: float = 0.0,
    exit_price: float = 0.0,
    deal_id: Optional[str] = None,
    api_key: str = Depends(verify_api_key),
):
    """
    Record P&L for a closed trade.

    Call this endpoint when a trade closes to track P&L against daily limits.
    """
    global risk_manager

    if risk_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Risk manager not initialized",
        )

    risk_manager.record_pnl(
        pnl=pnl,
        direction=direction,
        size=size,
        entry_price=entry_price,
        exit_price=exit_price,
        deal_id=deal_id,
    )

    can_trade, reason = risk_manager.can_trade()

    return {
        "recorded": True,
        "pnl": pnl,
        "daily_total_pnl": risk_manager.get_daily_stats().total_pnl,
        "trading_allowed": can_trade,
        "blocked_reason": reason,
        "timestamp": datetime.now().isoformat(),
    }


@app.put("/api/update-balance")
async def update_account_balance(
    balance: float,
    api_key: str = Depends(verify_api_key),
):
    """
    Update account balance for risk calculations.

    Call this to sync the risk manager with actual account balance.
    """
    global risk_manager

    if risk_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Risk manager not initialized",
        )

    risk_manager.update_account_balance(balance)

    return {
        "updated": True,
        "new_balance": balance,
        "new_daily_limit": risk_manager.daily_loss_limit,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/position-sizing-config", response_model=PositionSizingConfigResponse)
async def get_position_sizing_config(api_key: str = Depends(verify_api_key)):
    """
    Get current position sizing configuration.

    Returns volatility thresholds and size limits used for position calculations.
    """
    global risk_manager

    if risk_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Risk manager not initialized",
        )

    config = risk_manager.get_position_sizing_config()
    return PositionSizingConfigResponse(**config)


@app.post("/api/calculate-position-size", response_model=PositionSizeResponse)
async def calculate_position_size(
    base_size: float = 0.01,
    api_key: str = Depends(verify_api_key),
):
    """
    Calculate volatility-adjusted position size based on current market conditions.

    Fetches live ATR data and calculates the optimal position size.
    Returns adjusted size and volatility regime information.
    """
    global risk_manager, data_connector, indicator_calculator

    if risk_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Risk manager not initialized",
        )

    if data_connector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Data connector not initialized",
        )

    try:
        # Get recent data with ATR
        df = data_connector.get_ohlcv(bars=100)
        df = indicator_calculator.calculate_all(df)

        if "atr" not in df.columns:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="ATR not calculated in data",
            )

        atr_values = df["atr"].dropna().tolist()
        result = risk_manager.calculate_position_size(
            atr_values=atr_values,
            base_size=base_size,
        )

        return PositionSizeResponse(**result.to_dict())

    except Exception as e:
        logger.error(f"Position size calculation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Calculation failed: {str(e)}",
        )


# ============================================================================
# Trailing Stop Management Endpoints
# ============================================================================


@app.get("/api/trailing-stop-config", response_model=TrailingStopConfigResponse)
async def get_trailing_stop_config(api_key: str = Depends(verify_api_key)):
    """
    Get current trailing stop configuration.

    Returns trailing stop settings including ATR multiplier, activation threshold, and step size.
    """
    global position_manager

    if position_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Position manager not initialized",
        )

    config = position_manager.get_config()
    return TrailingStopConfigResponse(**config)


@app.post("/api/track-position")
async def track_position(
    request: AddPositionRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    Add a position to the trailing stop tracker.

    Call this after opening a position to enable trailing stop management.
    """
    global position_manager

    if position_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Position manager not initialized",
        )

    position = position_manager.add_position(
        deal_id=request.deal_id,
        direction=request.direction,
        entry_price=request.entry_price,
        stop_loss=request.stop_loss,
        take_profit=request.take_profit,
        size=request.size,
        symbol=request.symbol,
        entry_atr=request.entry_atr,
    )

    return {
        "success": True,
        "deal_id": position.deal_id,
        "message": f"Position {position.deal_id} now being tracked",
        "trailing_activated": position.trailing_activated,
        "current_stop_loss": position.current_stop_loss,
    }


@app.delete("/api/track-position/{deal_id}")
async def untrack_position(
    deal_id: str,
    api_key: str = Depends(verify_api_key),
):
    """
    Remove a position from the trailing stop tracker.

    Call this when a position is closed to clean up tracking.
    """
    global position_manager

    if position_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Position manager not initialized",
        )

    position = position_manager.remove_position(deal_id)

    if position is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Position {deal_id} not found in tracker",
        )

    return {
        "success": True,
        "deal_id": deal_id,
        "message": f"Position {deal_id} removed from tracking",
    }


@app.get("/api/tracked-positions")
async def list_tracked_positions(
    api_key: str = Depends(verify_api_key),
):
    """
    List all positions being tracked for trailing stops.

    Returns summary of all tracked positions with their current trailing stop status.
    """
    global position_manager, data_connector

    if position_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Position manager not initialized",
        )

    positions = position_manager.get_all_positions()

    # Try to get current price for status calculation
    current_price = None
    if data_connector is not None:
        try:
            df = data_connector.get_ohlcv(bars=1)
            if not df.empty:
                current_price = float(df["close"].iloc[-1])
        except Exception:
            pass

    results = []
    for pos in positions:
        if current_price:
            status = position_manager.get_position_status(pos.deal_id, current_price)
            results.append(status)
        else:
            results.append({
                "deal_id": pos.deal_id,
                "direction": pos.direction.value,
                "symbol": pos.symbol,
                "entry_price": pos.entry_price,
                "current_stop_loss": pos.current_stop_loss,
                "trailing_activated": pos.trailing_activated,
            })

    return {
        "tracked_positions": results,
        "count": len(results),
        "current_price": current_price,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/tracked-position/{deal_id}", response_model=TrackedPositionResponse)
async def get_tracked_position(
    deal_id: str,
    api_key: str = Depends(verify_api_key),
):
    """
    Get detailed status of a tracked position.

    Returns comprehensive position info including trailing stop status and P&L.
    """
    global position_manager, data_connector

    if position_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Position manager not initialized",
        )

    position = position_manager.get_position(deal_id)
    if position is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Position {deal_id} not found",
        )

    # Get current price
    current_price = position.entry_price  # Default to entry if can't get live
    if data_connector is not None:
        try:
            df = data_connector.get_ohlcv(bars=1)
            if not df.empty:
                current_price = float(df["close"].iloc[-1])
        except Exception:
            pass

    status = position_manager.get_position_status(deal_id, current_price)
    return TrackedPositionResponse(**status)


@app.post("/api/update-trailing-stop", response_model=TrailingStopUpdateResponse)
async def update_trailing_stop(
    request: UpdateTrailingStopRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    Calculate and optionally apply trailing stop update for a position.

    Checks if stop-loss should be moved based on current price.
    If apply_to_broker is True and update is needed, calls Capital.com API to update.
    """
    global position_manager, data_connector

    if position_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Position manager not initialized",
        )

    # Calculate trailing stop update
    update = position_manager.calculate_trailing_stop(
        deal_id=request.deal_id,
        current_price=request.current_price,
        current_atr=request.current_atr,
    )

    api_result = None

    # Apply to broker if requested and update is needed
    if request.apply_to_broker and update.should_update:
        if data_connector is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Data connector not initialized for broker updates",
            )

        # Check if it's Capital.com connector with update method
        if hasattr(data_connector, "update_position"):
            api_result = data_connector.update_position(
                deal_id=request.deal_id,
                stop_loss=update.new_stop_loss,
            )

            if api_result.get("success"):
                # Update local tracking
                position_manager.apply_stop_update(request.deal_id, update.new_stop_loss)
        else:
            api_result = {"success": False, "error": "Broker update not supported"}

    return TrailingStopUpdateResponse(
        should_update=update.should_update,
        new_stop_loss=update.new_stop_loss,
        current_price=update.current_price,
        profit_pips=update.profit_pips,
        reason=update.reason,
        api_update_result=api_result,
    )


@app.post("/api/update-all-trailing-stops")
async def update_all_trailing_stops(
    apply_to_broker: bool = True,
    api_key: str = Depends(verify_api_key),
):
    """
    Check and update trailing stops for all tracked positions.

    Fetches current price and ATR, then processes all tracked positions.
    Returns list of positions with update results.
    """
    global position_manager, data_connector, indicator_calculator

    if position_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Position manager not initialized",
        )

    if data_connector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Data connector not initialized",
        )

    positions = position_manager.get_all_positions()
    if not positions:
        return {
            "message": "No positions being tracked",
            "updates": [],
            "timestamp": datetime.now().isoformat(),
        }

    # Get current price and ATR
    try:
        df = data_connector.get_ohlcv(bars=50)
        df = indicator_calculator.calculate_all(df)
        current_price = float(df["close"].iloc[-1])
        current_atr = float(df["atr"].iloc[-1]) if "atr" in df.columns else None
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get market data: {str(e)}",
        )

    results = []
    for pos in positions:
        # Calculate trailing stop update
        update = position_manager.calculate_trailing_stop(
            deal_id=pos.deal_id,
            current_price=current_price,
            current_atr=current_atr,
        )

        result = {
            "deal_id": pos.deal_id,
            "direction": pos.direction.value,
            "should_update": update.should_update,
            "new_stop_loss": update.new_stop_loss,
            "profit_pips": update.profit_pips,
            "reason": update.reason,
            "api_result": None,
        }

        # Apply to broker if needed
        if apply_to_broker and update.should_update:
            if hasattr(data_connector, "update_position"):
                api_result = data_connector.update_position(
                    deal_id=pos.deal_id,
                    stop_loss=update.new_stop_loss,
                )
                result["api_result"] = api_result

                if api_result.get("success"):
                    position_manager.apply_stop_update(pos.deal_id, update.new_stop_loss)

        results.append(result)

    updates_applied = sum(1 for r in results if r.get("api_result", {}).get("success"))

    return {
        "positions_checked": len(results),
        "updates_applied": updates_applied,
        "current_price": current_price,
        "current_atr": current_atr,
        "updates": results,
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================================
# Partial Take-Profit Endpoints
# ============================================================================


@app.get("/api/partial-tp-config")
async def get_partial_tp_config(api_key: str = Depends(verify_api_key)):
    """
    Get partial take-profit configuration.

    Returns default TP levels (1x, 2x, 3x ATR) and close percentages (50%, 30%, 20%).
    """
    global position_manager

    if position_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Position manager not initialized",
        )

    return position_manager.get_partial_tp_config()


@app.post("/api/calculate-tp-levels")
async def calculate_tp_levels(
    entry_price: float,
    direction: str,
    atr: float,
    api_key: str = Depends(verify_api_key),
):
    """
    Calculate take-profit levels for a position.

    Returns 3 TP levels based on ATR with default close percentages.
    """
    global position_manager

    if position_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Position manager not initialized",
        )

    tp_levels = position_manager.calculate_tp_levels(
        entry_price=entry_price,
        direction=direction,
        atr=atr,
    )

    return {
        "entry_price": entry_price,
        "direction": direction,
        "atr": atr,
        "tp_levels": tp_levels,
    }


@app.post("/api/setup-position-with-tp")
async def setup_position_with_tp(
    request: AddPositionRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    Add a position to tracker with automatically calculated TP levels.

    Creates position with 3 TP levels based on entry ATR.
    """
    global position_manager

    if position_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Position manager not initialized",
        )

    if request.entry_atr <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="entry_atr must be positive for TP level calculation",
        )

    # Calculate TP levels
    tp_levels = position_manager.calculate_tp_levels(
        entry_price=request.entry_price,
        direction=request.direction,
        atr=request.entry_atr,
    )

    # Add position with TP levels
    position = position_manager.add_position(
        deal_id=request.deal_id,
        direction=request.direction,
        entry_price=request.entry_price,
        stop_loss=request.stop_loss,
        take_profit=request.take_profit,
        size=request.size,
        symbol=request.symbol,
        entry_atr=request.entry_atr,
        tp_levels=tp_levels,
    )

    return {
        "success": True,
        "deal_id": position.deal_id,
        "entry_price": position.entry_price,
        "direction": position.direction.value,
        "size": position.size,
        "stop_loss": position.current_stop_loss,
        "tp_levels": position.tp_levels,
        "message": f"Position {position.deal_id} tracked with {len(tp_levels)} TP levels",
    }


@app.post("/api/check-tp-levels/{deal_id}")
async def check_tp_levels(
    deal_id: str,
    execute_closes: bool = False,
    api_key: str = Depends(verify_api_key),
):
    """
    Check if any TP levels have been hit for a position.

    If execute_closes=True, will also execute partial closes via broker API.
    """
    global position_manager, data_connector

    if position_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Position manager not initialized",
        )

    position = position_manager.get_position(deal_id)
    if position is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Position {deal_id} not found",
        )

    # Get current price
    if data_connector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Data connector not initialized",
        )

    try:
        df = data_connector.get_ohlcv(bars=1)
        current_price = float(df["close"].iloc[-1])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get current price: {str(e)}",
        )

    # Process TP hits
    actions = position_manager.process_tp_hit(deal_id, current_price)

    execution_results = []
    if execute_closes and actions:
        for action in actions:
            if action["action"] == "partial_close":
                # Execute partial close via broker
                if hasattr(data_connector, "close_position"):
                    close_result = data_connector.close_position(
                        deal_id=action["deal_id"],
                        size=action["close_size"],
                    )

                    if close_result.get("success"):
                        # Record the partial close
                        position_manager.record_partial_close(
                            deal_id=action["deal_id"],
                            tp_level=action["tp_level"],
                            close_size=action["close_size"],
                            close_price=current_price,
                        )

                    execution_results.append({
                        "action": "partial_close",
                        "tp_level": action["tp_level"],
                        "close_size": action["close_size"],
                        "broker_result": close_result,
                    })

            elif action["action"] == "move_to_breakeven":
                # Move stop to breakeven
                be_result = position_manager.move_stop_to_breakeven(action["deal_id"])

                # Update broker stop-loss
                if be_result and be_result.get("moved") and hasattr(data_connector, "update_position"):
                    broker_update = data_connector.update_position(
                        deal_id=action["deal_id"],
                        stop_loss=be_result["new_stop_loss"],
                    )
                    execution_results.append({
                        "action": "move_to_breakeven",
                        "new_stop_loss": be_result["new_stop_loss"],
                        "broker_result": broker_update,
                    })

    return {
        "deal_id": deal_id,
        "current_price": current_price,
        "actions_identified": actions,
        "executions": execution_results if execute_closes else [],
        "tp_levels": position.tp_levels if position else [],
        "tp_levels_hit": position.tp_levels_hit if position else [],
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/check-all-tp-levels")
async def check_all_tp_levels(
    execute_closes: bool = False,
    api_key: str = Depends(verify_api_key),
):
    """
    Check TP levels for all tracked positions.

    Optionally execute partial closes for any TP levels that are hit.
    """
    global position_manager, data_connector

    if position_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Position manager not initialized",
        )

    if data_connector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Data connector not initialized",
        )

    positions = position_manager.get_all_positions()
    if not positions:
        return {
            "message": "No positions being tracked",
            "results": [],
            "timestamp": datetime.now().isoformat(),
        }

    # Get current price
    try:
        df = data_connector.get_ohlcv(bars=1)
        current_price = float(df["close"].iloc[-1])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get current price: {str(e)}",
        )

    results = []
    total_closes = 0
    total_breakevens = 0

    for pos in positions:
        actions = position_manager.process_tp_hit(pos.deal_id, current_price)

        pos_result = {
            "deal_id": pos.deal_id,
            "direction": pos.direction.value,
            "actions": actions,
            "executions": [],
        }

        if execute_closes and actions:
            for action in actions:
                if action["action"] == "partial_close":
                    if hasattr(data_connector, "close_position"):
                        close_result = data_connector.close_position(
                            deal_id=action["deal_id"],
                            size=action["close_size"],
                        )

                        if close_result.get("success"):
                            position_manager.record_partial_close(
                                deal_id=action["deal_id"],
                                tp_level=action["tp_level"],
                                close_size=action["close_size"],
                                close_price=current_price,
                            )
                            total_closes += 1

                        pos_result["executions"].append({
                            "action": "partial_close",
                            "tp_level": action["tp_level"],
                            "success": close_result.get("success", False),
                        })

                elif action["action"] == "move_to_breakeven":
                    be_result = position_manager.move_stop_to_breakeven(action["deal_id"])

                    if be_result and be_result.get("moved") and hasattr(data_connector, "update_position"):
                        broker_update = data_connector.update_position(
                            deal_id=action["deal_id"],
                            stop_loss=be_result["new_stop_loss"],
                        )
                        if broker_update.get("success"):
                            total_breakevens += 1

                        pos_result["executions"].append({
                            "action": "move_to_breakeven",
                            "success": broker_update.get("success", False),
                        })

        results.append(pos_result)

    return {
        "positions_checked": len(results),
        "current_price": current_price,
        "partial_closes_executed": total_closes,
        "breakeven_moves_executed": total_breakevens,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/move-to-breakeven/{deal_id}")
async def move_stop_to_breakeven(
    deal_id: str,
    update_broker: bool = True,
    api_key: str = Depends(verify_api_key),
):
    """
    Move stop-loss to breakeven (entry price) for a position.

    Typically called after TP1 is hit to lock in zero loss.
    """
    global position_manager, data_connector

    if position_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Position manager not initialized",
        )

    position = position_manager.get_position(deal_id)
    if position is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Position {deal_id} not found",
        )

    result = position_manager.move_stop_to_breakeven(deal_id)

    broker_result = None
    if result and result.get("moved") and update_broker:
        if data_connector and hasattr(data_connector, "update_position"):
            broker_result = data_connector.update_position(
                deal_id=deal_id,
                stop_loss=result["new_stop_loss"],
            )

    return {
        **result,
        "broker_update": broker_result,
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================================
# Economic Calendar Endpoints
# ============================================================================


@app.get("/api/calendar")
async def get_economic_calendar(
    hours_ahead: int = 24,
    min_impact: str = "low",
    force_refresh: bool = False,
    api_key: str = Depends(verify_api_key),
):
    """
    Get upcoming economic calendar events.

    Returns events filtered by time window and impact level.
    Events are sorted by datetime.

    Args:
        hours_ahead: How many hours ahead to look (default: 24)
        min_impact: Minimum impact level (low, medium, high)
        force_refresh: Force fetch fresh events
    """
    global economic_calendar

    if economic_calendar is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Economic calendar not initialized",
        )

    # Refresh events if requested
    if force_refresh:
        await economic_calendar.fetch_events(force=True)

    # Map string to ImpactLevel
    impact_map = {
        "low": ImpactLevel.LOW,
        "medium": ImpactLevel.MEDIUM,
        "high": ImpactLevel.HIGH,
    }
    impact_level = impact_map.get(min_impact.lower(), ImpactLevel.LOW)

    # Get filtered events
    events = economic_calendar.get_upcoming_events(
        hours_ahead=hours_ahead,
        min_impact=impact_level,
    )

    # Convert to response format
    event_list = [event.to_dict() for event in events]

    return {
        "events": event_list,
        "count": len(event_list),
        "hours_ahead": hours_ahead,
        "min_impact": min_impact,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/calendar/high-impact")
async def get_high_impact_events(
    hours_ahead: int = 24,
    api_key: str = Depends(verify_api_key),
):
    """
    Get only high-impact economic events.

    Convenience endpoint for quickly checking major events.
    """
    global economic_calendar

    if economic_calendar is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Economic calendar not initialized",
        )

    events = economic_calendar.get_high_impact_events(hours_ahead=hours_ahead)
    event_list = [event.to_dict() for event in events]

    return {
        "events": event_list,
        "count": len(event_list),
        "hours_ahead": hours_ahead,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/calendar/status", response_model=CalendarStatusResponse)
async def get_calendar_status(api_key: str = Depends(verify_api_key)):
    """
    Get economic calendar status.

    Returns cache status, whether near high-impact event, and relevant currencies.
    """
    global economic_calendar

    if economic_calendar is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Economic calendar not initialized",
        )

    status_data = economic_calendar.get_status()
    return CalendarStatusResponse(**status_data)


@app.post("/api/calendar/refresh")
async def refresh_calendar(api_key: str = Depends(verify_api_key)):
    """
    Force refresh the economic calendar.

    Fetches fresh events from the data source regardless of cache state.
    """
    global economic_calendar

    if economic_calendar is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Economic calendar not initialized",
        )

    events = await economic_calendar.fetch_events(force=True)

    return {
        "refreshed": True,
        "events_fetched": len(events),
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================================
# News Filter Endpoints
# ============================================================================


@app.get("/api/news-filter/status")
async def get_news_filter_status(api_key: str = Depends(verify_api_key)):
    """
    Get news filter status.

    Returns whether trading is currently paused due to high-impact events,
    the reason for pause, and the next upcoming high-impact event.
    """
    global news_filter

    if news_filter is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="News filter not initialized",
        )

    status_data = news_filter.get_status()
    status_data["timestamp"] = datetime.now().isoformat()
    return status_data


@app.get("/api/news-filter/check")
async def check_news_filter(api_key: str = Depends(verify_api_key)):
    """
    Check if trading is allowed by news filter.

    Quick endpoint to check if trading should be paused.
    Returns trading_allowed boolean and details if paused.
    """
    global news_filter

    if news_filter is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="News filter not initialized",
        )

    result = news_filter.check_trading_allowed()
    return {
        **result.to_dict(),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/news-filter/config")
async def get_news_filter_config(api_key: str = Depends(verify_api_key)):
    """
    Get news filter configuration.

    Returns pause windows (minutes before and after high-impact events).
    """
    global news_filter

    if news_filter is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="News filter not initialized",
        )

    return {
        **news_filter.get_config(),
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================================
# News API Endpoints (for Sentiment Analysis)
# ============================================================================


@app.get("/api/news")
async def get_news(
    hours: int = 24,
    max_count: int = 50,
    keyword: Optional[str] = None,
    force_refresh: bool = False,
    api_key: str = Depends(verify_api_key),
):
    """
    Get gold-related financial news.

    Returns recent news articles for gold, Federal Reserve, inflation, and related topics.
    Used as input for sentiment analysis.

    Args:
        hours: How many hours back to look (default: 24)
        max_count: Maximum articles to return (default: 50)
        keyword: Optional keyword to filter articles
        force_refresh: Force fetch fresh news from API
    """
    global news_fetcher

    if news_fetcher is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="News fetcher not initialized. Configure NEWS_API_KEY in environment.",
        )

    # Fetch fresh news if needed
    if force_refresh:
        await news_fetcher.fetch_news(force_refresh=True)

    # Get articles
    if keyword:
        articles = news_fetcher.get_articles_by_keyword(keyword, max_count=max_count)
    else:
        articles = news_fetcher.get_recent_headlines(hours=hours, max_count=max_count)

    return {
        "articles": [a.to_dict() for a in articles],
        "count": len(articles),
        "hours": hours,
        "keyword": keyword,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/news/headlines")
async def get_headlines(
    hours: int = 4,
    max_count: int = 20,
    api_key: str = Depends(verify_api_key),
):
    """
    Get headlines formatted for sentiment analysis.

    Returns concise list of recent headlines with title, source, and timestamp.
    Optimized for FinBERT sentiment scoring.
    """
    global news_fetcher

    if news_fetcher is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="News fetcher not initialized. Configure NEWS_API_KEY in environment.",
        )

    headlines = await news_fetcher.get_headlines_for_sentiment(
        hours=hours,
        max_count=max_count,
    )

    return {
        "headlines": headlines,
        "count": len(headlines),
        "hours": hours,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/news/status")
async def get_news_status(api_key: str = Depends(verify_api_key)):
    """
    Get news fetcher status.

    Returns configuration, cache status, and article count.
    """
    global news_fetcher

    if news_fetcher is None:
        return {
            "configured": False,
            "message": "News fetcher not initialized. Set NEWS_API_KEY in environment.",
            "timestamp": datetime.now().isoformat(),
        }

    status_data = news_fetcher.get_status()
    status_data["timestamp"] = datetime.now().isoformat()
    return status_data


@app.post("/api/news/refresh")
async def refresh_news(api_key: str = Depends(verify_api_key)):
    """
    Force refresh news from API.

    Fetches fresh articles regardless of cache state.
    """
    global news_fetcher

    if news_fetcher is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="News fetcher not initialized. Configure NEWS_API_KEY in environment.",
        )

    articles = await news_fetcher.fetch_news(force_refresh=True)

    return {
        "refreshed": True,
        "articles_fetched": len(articles),
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================================
# Sentiment Analysis Endpoints
# ============================================================================


@app.get("/api/sentiment")
async def get_current_sentiment(
    hours: int = 4,
    force_refresh: bool = False,
    api_key: str = Depends(verify_api_key),
):
    """
    Get current market sentiment from news analysis.

    Analyzes recent news headlines using FinBERT and returns aggregate sentiment.
    Score ranges from -1.0 (bearish) to +1.0 (bullish).

    Args:
        hours: Time period to analyze (default: 4 hours)
        force_refresh: Force recompute even if cached
    """
    global sentiment_analyzer, news_fetcher

    if sentiment_analyzer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sentiment analyzer not initialized",
        )

    if news_fetcher is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="News fetcher not initialized. Configure NEWS_API_KEY for sentiment analysis.",
        )

    try:
        aggregate = await sentiment_analyzer.get_current_sentiment(
            news_fetcher=news_fetcher,
            period_hours=hours,
            force_refresh=force_refresh,
        )

        return {
            **aggregate.to_dict(),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sentiment analysis failed: {str(e)}",
        )


@app.post("/api/sentiment/analyze")
async def analyze_headline(
    headline: str,
    api_key: str = Depends(verify_api_key),
):
    """
    Analyze sentiment of a single headline.

    Returns sentiment label, confidence, and score for the given text.
    """
    global sentiment_analyzer

    if sentiment_analyzer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sentiment analyzer not initialized",
        )

    try:
        score = sentiment_analyzer.analyze_headline(headline)
        return {
            **score.to_dict(),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Headline analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}",
        )


@app.post("/api/sentiment/analyze-batch")
async def analyze_batch(
    headlines: list[str],
    api_key: str = Depends(verify_api_key),
):
    """
    Analyze sentiment of multiple headlines.

    Returns individual scores and aggregate sentiment for the batch.
    """
    global sentiment_analyzer

    if sentiment_analyzer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sentiment analyzer not initialized",
        )

    if not headlines:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No headlines provided",
        )

    try:
        # Convert to expected format
        headline_dicts = [{"title": h} for h in headlines]
        scores = sentiment_analyzer.analyze_headlines(headline_dicts)
        aggregate = sentiment_analyzer.calculate_aggregate(scores, period_hours=24)

        return {
            "individual_scores": [s.to_dict() for s in scores],
            "aggregate": aggregate.to_dict(),
            "count": len(scores),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch analysis failed: {str(e)}",
        )


@app.get("/api/sentiment/status")
async def get_sentiment_status(api_key: str = Depends(verify_api_key)):
    """
    Get sentiment analyzer status.

    Returns model status, cache info, and configuration.
    """
    global sentiment_analyzer

    if sentiment_analyzer is None:
        return {
            "configured": False,
            "message": "Sentiment analyzer not initialized",
            "timestamp": datetime.now().isoformat(),
        }

    status_data = sentiment_analyzer.get_status()
    status_data["timestamp"] = datetime.now().isoformat()
    return status_data


@app.post("/api/sentiment/clear-cache")
async def clear_sentiment_cache(api_key: str = Depends(verify_api_key)):
    """
    Clear sentiment analysis cache.

    Forces recomputation on next request.
    """
    global sentiment_analyzer

    if sentiment_analyzer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sentiment analyzer not initialized",
        )

    result = sentiment_analyzer.clear_cache()
    result["timestamp"] = datetime.now().isoformat()
    return result


# ============================================================================
# Trade History Database Endpoints
# ============================================================================


@app.get("/api/trades")
async def get_trades(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    direction: Optional[str] = None,
    outcome: Optional[str] = None,
    symbol: Optional[str] = None,
    min_pnl: Optional[float] = None,
    max_pnl: Optional[float] = None,
    signal: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    api_key: str = Depends(verify_api_key),
):
    """
    Get trade history with optional filtering.

    Returns list of trades sorted by opened_at (newest first).

    Args:
        start_date: Filter trades opened after this date (ISO format)
        end_date: Filter trades opened before this date (ISO format)
        direction: Filter by direction (BUY or SELL)
        outcome: Filter by outcome (WIN, LOSS, BREAKEVEN, OPEN)
        symbol: Filter by trading symbol
        min_pnl: Filter trades with P&L >= min_pnl
        max_pnl: Filter trades with P&L <= max_pnl
        signal: Filter by signal type
        limit: Maximum trades to return (default: 100)
        offset: Offset for pagination
    """
    global trade_db

    if trade_db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Trade database not initialized",
        )

    # Parse dates if provided
    parsed_start = None
    parsed_end = None
    if start_date:
        try:
            parsed_start = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid start_date format. Use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)",
            )

    if end_date:
        try:
            parsed_end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid end_date format. Use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)",
            )

    # Create filter
    trade_filter = TradeFilter(
        start_date=parsed_start,
        end_date=parsed_end,
        direction=direction,
        outcome=outcome,
        symbol=symbol,
        min_pnl=min_pnl,
        max_pnl=max_pnl,
        signal=signal,
        limit=limit,
        offset=offset,
    )

    trades = trade_db.get_trades(trade_filter)

    return {
        "trades": trades,
        "count": len(trades),
        "limit": limit,
        "offset": offset,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/trades/summary")
async def get_trade_summary(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    symbol: Optional[str] = None,
    api_key: str = Depends(verify_api_key),
):
    """
    Get trade summary statistics.

    Returns aggregate statistics including win rate, profit factor, and P&L metrics.

    Args:
        start_date: Start of period (ISO format)
        end_date: End of period (ISO format)
        symbol: Filter by trading symbol
    """
    global trade_db

    if trade_db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Trade database not initialized",
        )

    # Parse dates if provided
    parsed_start = None
    parsed_end = None
    if start_date:
        try:
            parsed_start = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid start_date format",
            )

    if end_date:
        try:
            parsed_end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid end_date format",
            )

    summary = trade_db.get_summary(
        start_date=parsed_start,
        end_date=parsed_end,
        symbol=symbol,
    )

    return {
        **summary.to_dict(),
        "period": {
            "start_date": start_date,
            "end_date": end_date,
            "symbol": symbol,
        },
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/trades/accuracy")
async def get_prediction_accuracy(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    api_key: str = Depends(verify_api_key),
):
    """
    Get prediction accuracy statistics.

    Returns accuracy metrics for model predictions vs actual outcomes.
    Includes breakdown by signal type.
    """
    global trade_db

    if trade_db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Trade database not initialized",
        )

    # Parse dates if provided
    parsed_start = None
    parsed_end = None
    if start_date:
        try:
            parsed_start = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid start_date format",
            )

    if end_date:
        try:
            parsed_end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid end_date format",
            )

    accuracy = trade_db.get_prediction_accuracy(
        start_date=parsed_start,
        end_date=parsed_end,
    )

    return {
        **accuracy,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/trades/{deal_id}")
async def get_trade(
    deal_id: str,
    api_key: str = Depends(verify_api_key),
):
    """
    Get a single trade by deal ID.

    Returns complete trade record or 404 if not found.
    """
    global trade_db

    if trade_db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Trade database not initialized",
        )

    trade = trade_db.get_trade(deal_id)

    if trade is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trade with deal_id '{deal_id}' not found",
        )

    return {
        "trade": trade,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/trades/log-open")
async def log_trade_open(
    deal_id: str,
    direction: str,
    size: float,
    entry_price: float,
    signal: str,
    confidence: float,
    deal_reference: Optional[str] = None,
    symbol: str = "XAUUSD",
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    predicted_price: Optional[float] = None,
    sentiment_score: Optional[float] = None,
    volatility_regime: Optional[str] = None,
    news_paused: bool = False,
    notes: Optional[str] = None,
    api_key: str = Depends(verify_api_key),
):
    """
    Log a new trade opening.

    Call this when a position is opened to start tracking the trade.
    Records entry details, signal, and prediction for later analysis.
    """
    global trade_db

    if trade_db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Trade database not initialized",
        )

    try:
        record_id = trade_db.log_trade_open(
            deal_id=deal_id,
            direction=direction,
            size=size,
            entry_price=entry_price,
            signal=signal,
            confidence=confidence,
            deal_reference=deal_reference,
            symbol=symbol,
            stop_loss=stop_loss,
            take_profit=take_profit,
            predicted_price=predicted_price,
            sentiment_score=sentiment_score,
            volatility_regime=volatility_regime,
            news_paused=news_paused,
            notes=notes,
        )

        return {
            "success": True,
            "record_id": record_id,
            "deal_id": deal_id,
            "message": f"Trade {deal_id} logged successfully",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error logging trade open: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to log trade: {str(e)}",
        )


@app.post("/api/trades/log-close")
async def log_trade_close(
    deal_id: str,
    exit_price: float,
    pnl: float,
    pnl_pips: Optional[float] = None,
    notes: Optional[str] = None,
    api_key: str = Depends(verify_api_key),
):
    """
    Log trade closure.

    Call this when a position is closed to record exit details and P&L.
    Automatically calculates outcome (WIN/LOSS/BREAKEVEN) and prediction accuracy.
    """
    global trade_db

    if trade_db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Trade database not initialized",
        )

    try:
        trade = trade_db.log_trade_close(
            deal_id=deal_id,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pips=pnl_pips,
            notes=notes,
        )

        if trade is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Trade with deal_id '{deal_id}' not found",
            )

        return {
            "success": True,
            "trade": trade,
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error logging trade close: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to log trade close: {str(e)}",
        )


@app.get("/api/trades/open")
async def get_open_trades(api_key: str = Depends(verify_api_key)):
    """
    Get all open (unclosed) trades.

    Returns list of trades that have not been closed yet.
    """
    global trade_db

    if trade_db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Trade database not initialized",
        )

    trades = trade_db.get_open_trades()

    return {
        "trades": trades,
        "count": len(trades),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/trades/status")
async def get_trade_db_status(api_key: str = Depends(verify_api_key)):
    """
    Get trade database status.

    Returns database info including total trades, open trades, and latest trade.
    """
    global trade_db

    if trade_db is None:
        return {
            "configured": False,
            "message": "Trade database not initialized",
            "timestamp": datetime.now().isoformat(),
        }

    status_data = trade_db.get_status()
    status_data["timestamp"] = datetime.now().isoformat()
    return status_data


# ==================== Performance Metrics Endpoints ====================


@app.get("/api/performance")
async def get_performance_metrics(
    period: str = "all",
    api_key: str = Depends(verify_api_key),
):
    """
    Get trading performance metrics.

    Returns comprehensive performance metrics including:
    - Win rate, profit factor, expectancy
    - Maximum drawdown
    - Sharpe ratio (annualized)
    - Sortino ratio
    - Risk-reward ratio
    - Consecutive win/loss streaks

    Args:
        period: Time period - "7d", "30d", "90d", "365d", "all"
    """
    global trading_performance_tracker

    if trading_performance_tracker is None:
        return {
            "configured": False,
            "message": "Performance tracker not initialized",
            "timestamp": datetime.now().isoformat(),
        }

    try:
        metrics = trading_performance_tracker.get_performance(period=period)
        result = metrics.to_dict()
        result["period"] = period
        result["timestamp"] = datetime.now().isoformat()
        return result
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance metrics: {str(e)}",
        )


@app.get("/api/performance/drawdown")
async def get_drawdown_metrics(
    period: str = "all",
    api_key: str = Depends(verify_api_key),
):
    """
    Get detailed drawdown analysis.

    Returns drawdown metrics including:
    - Maximum drawdown (percentage and amount)
    - Current drawdown
    - Peak and trough equity levels
    - Drawdown period dates
    """
    global trading_performance_tracker

    if trading_performance_tracker is None:
        return {
            "configured": False,
            "message": "Performance tracker not initialized",
            "timestamp": datetime.now().isoformat(),
        }

    try:
        # Get date range based on period
        from datetime import timedelta

        period_map = {
            "7d": timedelta(days=7),
            "30d": timedelta(days=30),
            "90d": timedelta(days=90),
            "365d": timedelta(days=365),
            "all": None,
        }

        delta = period_map.get(period.lower())
        if delta:
            from datetime import timezone
            start_date = datetime.now(timezone.utc) - delta
        else:
            start_date = None

        drawdown = trading_performance_tracker.get_drawdown(start_date=start_date)
        result = drawdown.to_dict()
        result["period"] = period
        result["timestamp"] = datetime.now().isoformat()
        return result
    except Exception as e:
        logger.error(f"Error getting drawdown metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get drawdown metrics: {str(e)}",
        )


@app.get("/api/performance/equity-curve")
async def get_equity_curve(
    period: str = "all",
    api_key: str = Depends(verify_api_key),
):
    """
    Get equity curve data for charting.

    Returns a list of data points with timestamp, equity, and cumulative P&L.
    Useful for plotting equity charts.
    """
    global trading_performance_tracker

    if trading_performance_tracker is None:
        return {
            "configured": False,
            "message": "Performance tracker not initialized",
            "timestamp": datetime.now().isoformat(),
        }

    try:
        # Get date range based on period
        from datetime import timedelta, timezone

        period_map = {
            "7d": timedelta(days=7),
            "30d": timedelta(days=30),
            "90d": timedelta(days=90),
            "365d": timedelta(days=365),
            "all": None,
        }

        delta = period_map.get(period.lower())
        if delta:
            start_date = datetime.now(timezone.utc) - delta
        else:
            start_date = None

        curve = trading_performance_tracker.get_equity_curve(start_date=start_date)
        return {
            "equity_curve": curve,
            "data_points": len(curve),
            "period": period,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting equity curve: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get equity curve: {str(e)}",
        )


@app.get("/api/performance/summary")
async def get_performance_summary(api_key: str = Depends(verify_api_key)):
    """
    Get a quick summary of performance across multiple periods.

    Returns key metrics for 7d, 30d, 90d, and all-time periods.
    """
    global trading_performance_tracker

    if trading_performance_tracker is None:
        return {
            "configured": False,
            "message": "Performance tracker not initialized",
            "timestamp": datetime.now().isoformat(),
        }

    try:
        periods = ["7d", "30d", "90d", "all"]
        summary = {}

        for period in periods:
            metrics = trading_performance_tracker.get_performance(period=period)
            summary[period] = {
                "total_trades": metrics.total_trades,
                "win_rate": round(metrics.win_rate * 100, 2),
                "total_pnl": round(metrics.total_pnl, 2),
                "profit_factor": round(metrics.profit_factor, 4),
                "sharpe_ratio": round(metrics.sharpe_ratio, 4),
                "max_drawdown": round(metrics.max_drawdown * 100, 2),
                "expectancy": round(metrics.expectancy, 2),
            }

        return {
            "periods": summary,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting performance summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance summary: {str(e)}",
        )


@app.get("/api/performance/status")
async def get_performance_tracker_status(api_key: str = Depends(verify_api_key)):
    """
    Get performance tracker status and configuration.
    """
    global trading_performance_tracker

    if trading_performance_tracker is None:
        return {
            "configured": False,
            "message": "Performance tracker not initialized",
            "timestamp": datetime.now().isoformat(),
        }

    status_data = trading_performance_tracker.get_status()
    status_data["configured"] = True
    status_data["timestamp"] = datetime.now().isoformat()
    return status_data


# ==================== Daily Report Endpoints ====================


@app.get("/api/daily-report")
async def get_daily_report(
    format: str = "json",
    api_key: str = Depends(verify_api_key),
):
    """
    Generate daily trading performance report.

    Returns a comprehensive report with:
    - Today's trades summary
    - Win rate and P&L
    - Best and worst trade
    - Equity change and drawdown
    - Comparison with previous periods

    Args:
        format: "json" for raw data, "telegram" for pre-formatted message
    """
    global trading_performance_tracker, trade_db, risk_manager

    if trading_performance_tracker is None or trade_db is None:
        return {
            "configured": False,
            "message": "Performance tracker or trade database not initialized",
            "timestamp": datetime.now().isoformat(),
        }

    try:
        from datetime import timezone, timedelta

        # Get today's date range (UTC)
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = now

        # Get today's trades
        today_filter = TradeFilter(
            start_date=today_start,
            end_date=today_end,
            limit=1000,
        )
        today_trades = trade_db.get_trades(today_filter)
        closed_today = [t for t in today_trades if t.get("pnl") is not None]

        # Calculate today's stats
        total_trades_today = len(closed_today)
        winning_trades = [t for t in closed_today if t.get("outcome") == "WIN"]
        losing_trades = [t for t in closed_today if t.get("outcome") == "LOSS"]

        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate_today = (win_count / total_trades_today * 100) if total_trades_today > 0 else 0

        # P&L calculations
        pnl_list = [t.get("pnl", 0) for t in closed_today]
        total_pnl_today = sum(pnl_list)

        # Best and worst trades
        best_trade = max(closed_today, key=lambda t: t.get("pnl", 0)) if closed_today else None
        worst_trade = min(closed_today, key=lambda t: t.get("pnl", 0)) if closed_today else None

        best_pnl = best_trade.get("pnl", 0) if best_trade else 0
        worst_pnl = worst_trade.get("pnl", 0) if worst_trade else 0

        # Get performance metrics for different periods
        today_metrics = trading_performance_tracker.get_performance(period="all")  # All time for context

        # Get drawdown
        drawdown_info = trading_performance_tracker.get_drawdown()

        # Get daily stats from risk manager if available
        daily_stats = None
        if risk_manager:
            daily_stats = risk_manager.get_daily_stats()

        # Calculate equity change
        initial_balance = trading_performance_tracker.initial_balance
        current_equity = initial_balance + today_metrics.total_pnl

        # Build report data
        report = {
            "report_date": now.strftime("%Y-%m-%d"),
            "report_time_utc": now.strftime("%H:%M:%S"),
            "generated_at": now.isoformat(),

            # Today's summary
            "today": {
                "trades_count": total_trades_today,
                "winning_trades": win_count,
                "losing_trades": loss_count,
                "win_rate": round(win_rate_today, 2),
                "total_pnl": round(total_pnl_today, 2),
                "best_trade_pnl": round(best_pnl, 2),
                "worst_trade_pnl": round(worst_pnl, 2),
                "best_trade_id": best_trade.get("deal_id") if best_trade else None,
                "worst_trade_id": worst_trade.get("deal_id") if worst_trade else None,
            },

            # All-time metrics
            "all_time": {
                "total_trades": today_metrics.total_trades,
                "win_rate": round(today_metrics.win_rate * 100, 2),
                "total_pnl": round(today_metrics.total_pnl, 2),
                "profit_factor": round(today_metrics.profit_factor, 4),
                "sharpe_ratio": round(today_metrics.sharpe_ratio, 4),
                "expectancy": round(today_metrics.expectancy, 2),
            },

            # Equity and drawdown
            "equity": {
                "initial_balance": round(initial_balance, 2),
                "current_equity": round(current_equity, 2),
                "equity_change": round(today_metrics.total_pnl, 2),
                "equity_change_pct": round((today_metrics.total_pnl / initial_balance) * 100, 2) if initial_balance > 0 else 0,
            },

            "drawdown": {
                "max_drawdown_pct": round(drawdown_info.max_drawdown * 100, 2),
                "max_drawdown_amount": round(drawdown_info.max_drawdown_amount, 2),
                "current_drawdown_pct": round(drawdown_info.current_drawdown * 100, 2),
            },

            # Risk status
            "risk_status": {
                "trading_allowed": daily_stats.trading_allowed if daily_stats else True,
                "daily_pnl": round(daily_stats.total_pnl, 2) if daily_stats else round(total_pnl_today, 2),
                "trades_remaining": (
                    (risk_manager.max_daily_trades - daily_stats.trade_count)
                    if daily_stats and risk_manager else None
                ),
            },
        }

        # If telegram format requested, return pre-formatted message
        if format.lower() == "telegram":
            report["telegram_message"] = _format_telegram_daily_report(report)

        return report

    except Exception as e:
        logger.error(f"Error generating daily report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate daily report: {str(e)}",
        )


def _format_telegram_daily_report(report: dict) -> str:
    """Format daily report for Telegram with emojis."""
    today = report["today"]
    all_time = report["all_time"]
    equity = report["equity"]
    drawdown = report["drawdown"]
    risk = report["risk_status"]

    # Determine overall sentiment
    if today["total_pnl"] > 0:
        header_emoji = "📈"
        pnl_emoji = "💰"
        status_text = "PROFITABLE DAY"
    elif today["total_pnl"] < 0:
        header_emoji = "📉"
        pnl_emoji = "💸"
        status_text = "LOSING DAY"
    else:
        header_emoji = "➖"
        pnl_emoji = "💵"
        status_text = "BREAK EVEN"

    # Format the message
    message = f"""{header_emoji} **GOLD DAILY REPORT** {header_emoji}
📅 {report['report_date']}

━━━━━━━━━━━━━━━━━━━━
📊 **TODAY'S SUMMARY**
━━━━━━━━━━━━━━━━━━━━
{pnl_emoji} Status: {status_text}
📈 Trades: {today['trades_count']} ({today['winning_trades']}W / {today['losing_trades']}L)
🎯 Win Rate: {today['win_rate']}%
💵 P&L: ${today['total_pnl']:+.2f}

🏆 Best Trade: ${today['best_trade_pnl']:+.2f}
😔 Worst Trade: ${today['worst_trade_pnl']:+.2f}

━━━━━━━━━━━━━━━━━━━━
💼 **ACCOUNT STATUS**
━━━━━━━━━━━━━━━━━━━━
💰 Equity: ${equity['current_equity']:,.2f}
📊 Change: ${equity['equity_change']:+.2f} ({equity['equity_change_pct']:+.2f}%)
📉 Max Drawdown: {drawdown['max_drawdown_pct']:.2f}%
📉 Current DD: {drawdown['current_drawdown_pct']:.2f}%

━━━━━━━━━━━━━━━━━━━━
📈 **ALL-TIME STATS**
━━━━━━━━━━━━━━━━━━━━
🔢 Total Trades: {all_time['total_trades']}
🎯 Win Rate: {all_time['win_rate']}%
📊 Profit Factor: {all_time['profit_factor']:.2f}
📏 Sharpe Ratio: {all_time['sharpe_ratio']:.2f}
💎 Expectancy: ${all_time['expectancy']:.2f}/trade

━━━━━━━━━━━━━━━━━━━━
⚠️ **RISK STATUS**
━━━━━━━━━━━━━━━━━━━━
{'✅ Trading Allowed' if risk['trading_allowed'] else '🛑 Trading Blocked'}

🤖 _Gold Predictor Bot_
⏰ {report['report_time_utc']} UTC"""

    return message


# Main entry point
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.server:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
