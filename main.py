"""
Gold Price Predictor - Main Entry Point
========================================
ML-based gold price prediction with n8n integration.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)
logger.add(
    "logs/gold_predictor_{time}.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
)


def serve():
    """Run the FastAPI server."""
    import uvicorn
    from src.config import settings

    logger.info("Starting Gold Predictor API server...")
    uvicorn.run(
        "src.api.server:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info",
    )


def get_data_connector():
    """Get the appropriate data connector based on settings."""
    from src.config import settings
    from src.data.mt5_connector import MT5Connector
    from src.data.capital_connector import CapitalConnector

    if settings.data_source.lower() == "capital":
        logger.info("Using Capital.com API as data source...")
        return CapitalConnector(
            api_key=settings.capital_api_key,
            password=settings.capital_password,
            identifier=settings.capital_identifier or settings.capital_api_key,
            demo=settings.capital_demo,
            symbol=settings.symbol,
        )
    else:
        logger.info("Using MetaTrader 5 as data source...")
        return MT5Connector(
            login=settings.mt5_login,
            password=settings.mt5_password,
            server=settings.mt5_server,
        )


def train(days: int = 30, epochs: int = 100):
    """Train the prediction models."""
    from src.config import settings
    from src.features.technical_indicators import TechnicalIndicators
    from src.models.lstm_model import GoldLSTM, LSTMTrainer
    from src.models.xgboost_model import GoldXGBoost, calculate_direction_labels
    from src.preprocessing.data_processor import DataProcessor

    logger.info(f"Training models with {days} days of data, {epochs} epochs...")

    # Initialize data connector based on settings
    connector = get_data_connector()

    try:
        connector.connect()

        # Fetch data
        bars_needed = days * 24 * 12  # 5-min bars
        df = connector.get_ohlcv(bars=min(bars_needed, 50000))
        logger.info(f"Fetched {len(df)} bars")

        # Calculate indicators
        indicator_calc = TechnicalIndicators()
        df = indicator_calc.calculate_all(df)

        # Prepare data
        processor = DataProcessor(
            lookback=settings.model_lookback,
            horizon=settings.prediction_horizon,
        )
        df = processor.clean_data(df)
        X, y = processor.fit_transform(df)

        # Direction labels
        y_direction = calculate_direction_labels(
            df["close"].values,
            horizon=settings.prediction_horizon,
        )
        y_direction = y_direction[settings.model_lookback:]

        # Split
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = processor.split_data(X, y)
        train_end = len(X_train)
        val_end = train_end + len(X_val)
        y_dir_train = y_direction[:train_end]
        y_dir_val = y_direction[train_end:val_end]

        # Train LSTM
        logger.info("Training LSTM model...")
        lstm = GoldLSTM(input_size=X_train.shape[-1], hidden_size=128, num_layers=2)
        trainer = LSTMTrainer(lstm, learning_rate=0.001)
        lstm_results = trainer.train(
            X_train, y_train, X_val, y_val,
            epochs=epochs,
            save_path=settings.lstm_model_path,
        )
        logger.info(f"LSTM training complete: {lstm_results}")

        # Train XGBoost
        logger.info("Training XGBoost model...")
        xgb = GoldXGBoost()
        xgb_results = xgb.fit(X_train, y_dir_train, X_val, y_dir_val)
        xgb.save(settings.xgb_model_path)
        logger.info(f"XGBoost training complete: {xgb_results}")

        # Save processor
        processor.save(settings.scaler_path)

        logger.info("Training complete!")

    finally:
        connector.disconnect()


def backtest(days: int = 7):
    """Run backtesting on historical data."""
    from src.config import settings
    from src.features.technical_indicators import TechnicalIndicators
    from src.models.ensemble import EnsemblePredictor
    from src.preprocessing.data_processor import DataProcessor

    logger.info(f"Running backtest for last {days} days...")

    # Load models
    processor = DataProcessor.load(settings.scaler_path)
    predictor = EnsemblePredictor(
        lstm_weight=settings.lstm_weight,
        xgb_weight=settings.xgb_weight,
    )
    predictor.load_models(settings.lstm_model_path, settings.xgb_model_path)

    # Get data connector based on settings
    connector = get_data_connector()

    try:
        connector.connect()
        bars = days * 24 * 12
        df = connector.get_ohlcv(bars=bars)

        indicator_calc = TechnicalIndicators()
        df = indicator_calc.calculate_all(df)
        df = processor.clean_data(df)

        # Run predictions
        correct = 0
        total = 0
        lookback = settings.model_lookback

        for i in range(lookback, len(df) - 1):
            window = df.iloc[i - lookback:i]
            current_price = df.iloc[i]["close"]
            actual_next = df.iloc[i + 1]["close"]

            X = processor.prepare_latest(window)
            pred = predictor.predict(X, current_price, processor.target_scaler)

            predicted_up = pred.predicted_price > current_price
            actual_up = actual_next > current_price

            if predicted_up == actual_up:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0
        logger.info(f"Backtest complete: {correct}/{total} correct ({accuracy:.1%})")

        return {"accuracy": accuracy, "correct": correct, "total": total}

    finally:
        connector.disconnect()


def schedule(interval: int = 5, webhook_url: str = None):
    """Run the prediction scheduler."""
    from src.scheduler.prediction_scheduler import run_scheduler

    logger.info(f"Starting scheduler with {interval} minute intervals...")
    asyncio.run(run_scheduler(
        interval_minutes=interval,
        webhook_url=webhook_url,
    ))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Gold Price Predictor")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Run the API server")

    # train command
    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument("--days", type=int, default=30, help="Days of data")
    train_parser.add_argument("--epochs", type=int, default=100, help="Training epochs")

    # backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtesting")
    backtest_parser.add_argument("--days", type=int, default=7, help="Days to backtest")

    # schedule command
    schedule_parser = subparsers.add_parser("schedule", help="Run scheduler")
    schedule_parser.add_argument("--interval", type=int, default=5, help="Interval minutes")
    schedule_parser.add_argument("--webhook", help="Webhook URL for alerts")

    args = parser.parse_args()

    if args.command == "serve":
        serve()
    elif args.command == "train":
        train(days=args.days, epochs=args.epochs)
    elif args.command == "backtest":
        backtest(days=args.days)
    elif args.command == "schedule":
        schedule(interval=args.interval, webhook_url=args.webhook)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
