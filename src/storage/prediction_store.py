"""
Prediction Storage Module
=========================
SQLite-based storage for prediction history and analytics.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from loguru import logger
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
    func,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

Base = declarative_base()


class PredictionRecord(Base):
    """SQLAlchemy model for prediction records."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    symbol = Column(String(20), default="XAUUSD")
    signal = Column(String(20))
    confidence = Column(Float)
    current_price = Column(Float)
    predicted_price = Column(Float)
    predicted_change_percent = Column(Float)
    actual_price = Column(Float, nullable=True)  # Filled later for accuracy tracking
    actual_change_percent = Column(Float, nullable=True)
    was_correct = Column(Integer, nullable=True)  # 0=wrong, 1=correct


class PredictionStore:
    """
    Stores and retrieves prediction history.

    Used for:
    - Prediction logging
    - Accuracy tracking
    - Historical analysis
    - Performance metrics
    """

    def __init__(self, database_url: str = "sqlite:///predictions.db"):
        """
        Initialize prediction store.

        Args:
            database_url: SQLAlchemy database URL
        """
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        logger.info(f"Prediction store initialized: {database_url}")

    def save_prediction(
        self,
        signal: str,
        confidence: float,
        current_price: float,
        predicted_price: float,
        predicted_change_percent: float,
        symbol: str = "XAUUSD",
    ) -> int:
        """
        Save a prediction record.

        Args:
            signal: Signal type
            confidence: Confidence score
            current_price: Current price
            predicted_price: Predicted price
            predicted_change_percent: Predicted change %
            symbol: Trading symbol

        Returns:
            Record ID
        """
        with self.SessionLocal() as session:
            record = PredictionRecord(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                signal=signal,
                confidence=confidence,
                current_price=current_price,
                predicted_price=predicted_price,
                predicted_change_percent=predicted_change_percent,
            )
            session.add(record)
            session.commit()
            logger.debug(f"Saved prediction #{record.id}: {signal}")
            return record.id

    def update_actual(
        self,
        record_id: int,
        actual_price: float,
    ) -> None:
        """
        Update record with actual price for accuracy tracking.

        Args:
            record_id: Prediction record ID
            actual_price: Actual price at prediction horizon
        """
        with self.SessionLocal() as session:
            record = session.query(PredictionRecord).filter_by(id=record_id).first()
            if record:
                record.actual_price = actual_price
                record.actual_change_percent = (
                    (actual_price - record.current_price) / record.current_price * 100
                )

                # Determine if prediction was correct
                predicted_direction = 1 if record.predicted_change_percent > 0 else -1
                actual_direction = 1 if record.actual_change_percent > 0 else -1
                record.was_correct = 1 if predicted_direction == actual_direction else 0

                session.commit()
                logger.debug(f"Updated prediction #{record_id}: correct={record.was_correct}")

    def get_recent_predictions(
        self,
        hours: int = 24,
        symbol: str = "XAUUSD",
    ) -> List[PredictionRecord]:
        """
        Get predictions from the last N hours.

        Args:
            hours: Number of hours to look back
            symbol: Trading symbol

        Returns:
            List of prediction records
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        with self.SessionLocal() as session:
            records = (
                session.query(PredictionRecord)
                .filter(
                    PredictionRecord.timestamp >= cutoff,
                    PredictionRecord.symbol == symbol,
                )
                .order_by(PredictionRecord.timestamp.desc())
                .all()
            )
            return records

    def get_accuracy_stats(
        self,
        days: int = 7,
        symbol: str = "XAUUSD",
    ) -> dict:
        """
        Calculate accuracy statistics.

        Args:
            days: Number of days to analyze
            symbol: Trading symbol

        Returns:
            Dict with accuracy metrics
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        with self.SessionLocal() as session:
            # Total predictions
            total = (
                session.query(func.count(PredictionRecord.id))
                .filter(
                    PredictionRecord.timestamp >= cutoff,
                    PredictionRecord.symbol == symbol,
                    PredictionRecord.was_correct.isnot(None),
                )
                .scalar()
            )

            if total == 0:
                return {"total": 0, "accuracy": None, "by_signal": {}}

            # Correct predictions
            correct = (
                session.query(func.count(PredictionRecord.id))
                .filter(
                    PredictionRecord.timestamp >= cutoff,
                    PredictionRecord.symbol == symbol,
                    PredictionRecord.was_correct == 1,
                )
                .scalar()
            )

            # By signal type
            by_signal = {}
            for signal in ["STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"]:
                signal_total = (
                    session.query(func.count(PredictionRecord.id))
                    .filter(
                        PredictionRecord.timestamp >= cutoff,
                        PredictionRecord.symbol == symbol,
                        PredictionRecord.signal == signal,
                        PredictionRecord.was_correct.isnot(None),
                    )
                    .scalar()
                )

                signal_correct = (
                    session.query(func.count(PredictionRecord.id))
                    .filter(
                        PredictionRecord.timestamp >= cutoff,
                        PredictionRecord.symbol == symbol,
                        PredictionRecord.signal == signal,
                        PredictionRecord.was_correct == 1,
                    )
                    .scalar()
                )

                if signal_total > 0:
                    by_signal[signal] = {
                        "total": signal_total,
                        "correct": signal_correct,
                        "accuracy": signal_correct / signal_total,
                    }

            return {
                "total": total,
                "correct": correct,
                "accuracy": correct / total if total > 0 else None,
                "by_signal": by_signal,
                "period_days": days,
            }

    def get_signal_distribution(
        self,
        days: int = 7,
        symbol: str = "XAUUSD",
    ) -> dict:
        """
        Get distribution of signals over time.

        Args:
            days: Number of days to analyze
            symbol: Trading symbol

        Returns:
            Dict with signal counts
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        with self.SessionLocal() as session:
            results = (
                session.query(
                    PredictionRecord.signal,
                    func.count(PredictionRecord.id).label("count"),
                )
                .filter(
                    PredictionRecord.timestamp >= cutoff,
                    PredictionRecord.symbol == symbol,
                )
                .group_by(PredictionRecord.signal)
                .all()
            )

            return {r.signal: r.count for r in results}

    def cleanup_old_records(self, days: int = 90) -> int:
        """
        Delete records older than N days.

        Args:
            days: Records older than this will be deleted

        Returns:
            Number of records deleted
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        with self.SessionLocal() as session:
            deleted = (
                session.query(PredictionRecord)
                .filter(PredictionRecord.timestamp < cutoff)
                .delete()
            )
            session.commit()
            logger.info(f"Cleaned up {deleted} old prediction records")
            return deleted
