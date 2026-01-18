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
from ..trading import RiskManager, DailyStats, PositionSizeResult

# Global instances
predictor: Optional[EnsemblePredictor] = None
data_processor: Optional[DataProcessor] = None
data_connector = None  # Either MT5Connector or CapitalConnector
indicator_calculator: Optional[TechnicalIndicators] = None
risk_manager: Optional[RiskManager] = None


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
    global predictor, data_processor, data_connector, indicator_calculator, risk_manager

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
    global predictor, data_processor, data_connector, indicator_calculator, risk_manager

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
            "position_sizing": position_sizing_info,
        }

        # Check if we should trade
        signal = prediction.signal.value
        signal_allows_trade = (
            "BUY" in signal.upper() or "SELL" in signal.upper()
        ) and prediction.confidence >= min_confidence

        # Only trade if both signal and risk manager allow
        should_trade = signal_allows_trade and can_trade

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


# Main entry point
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.server:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
