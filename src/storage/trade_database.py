"""
Trade History Database Module
=============================
SQLite-based storage for trade history and performance analysis.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional
import logging

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Boolean,
    create_engine,
    func,
    and_,
    or_,
)
from sqlalchemy.orm import Session, sessionmaker, declarative_base

logger = logging.getLogger(__name__)

Base = declarative_base()


class TradeOutcome(str, Enum):
    """Trade outcome classification."""
    WIN = "WIN"
    LOSS = "LOSS"
    BREAKEVEN = "BREAKEVEN"
    OPEN = "OPEN"


class TradeRecord(Base):
    """SQLAlchemy model for trade records."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Trade identification
    deal_id = Column(String(100), unique=True, index=True)
    deal_reference = Column(String(100), nullable=True)
    symbol = Column(String(20), default="XAUUSD", index=True)

    # Trade details
    direction = Column(String(10), index=True)  # BUY or SELL
    size = Column(Float)
    entry_price = Column(Float)
    exit_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)

    # P&L
    pnl = Column(Float, nullable=True)
    pnl_pips = Column(Float, nullable=True)
    pnl_percent = Column(Float, nullable=True)

    # Signal information
    signal = Column(String(20))  # STRONG_BUY, BUY, SELL, STRONG_SELL
    confidence = Column(Float)

    # Prediction vs actual
    predicted_price = Column(Float, nullable=True)
    predicted_direction = Column(String(10), nullable=True)
    actual_direction = Column(String(10), nullable=True)  # UP or DOWN
    prediction_correct = Column(Boolean, nullable=True)

    # Outcome
    outcome = Column(String(20), default="OPEN", index=True)  # WIN, LOSS, BREAKEVEN, OPEN

    # Timestamps
    opened_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    closed_at = Column(DateTime, nullable=True)

    # Additional context
    sentiment_score = Column(Float, nullable=True)
    volatility_regime = Column(String(20), nullable=True)
    news_paused = Column(Boolean, default=False)
    notes = Column(String(500), nullable=True)


@dataclass
class TradeFilter:
    """Filter criteria for querying trades."""

    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    direction: Optional[str] = None
    outcome: Optional[str] = None
    symbol: Optional[str] = None
    min_pnl: Optional[float] = None
    max_pnl: Optional[float] = None
    signal: Optional[str] = None
    limit: int = 100
    offset: int = 0


@dataclass
class TradeSummary:
    """Summary statistics for trades."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    open_trades: int
    win_rate: float
    total_pnl: float
    average_pnl: float
    largest_win: float
    largest_loss: float
    average_win: float
    average_loss: float
    profit_factor: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "breakeven_trades": self.breakeven_trades,
            "open_trades": self.open_trades,
            "win_rate": round(self.win_rate, 4),
            "total_pnl": round(self.total_pnl, 2),
            "average_pnl": round(self.average_pnl, 2),
            "largest_win": round(self.largest_win, 2),
            "largest_loss": round(self.largest_loss, 2),
            "average_win": round(self.average_win, 2),
            "average_loss": round(self.average_loss, 2),
            "profit_factor": round(self.profit_factor, 4),
        }


class TradeDatabase:
    """
    Trade history database for logging and analysis.

    Stores complete trade records including:
    - Entry/exit prices and P&L
    - Signal and confidence at entry
    - Prediction vs actual outcome
    - Trade context (sentiment, volatility)
    """

    def __init__(self, database_url: str = "sqlite:///trades.db"):
        """
        Initialize trade database.

        Args:
            database_url: SQLAlchemy database URL
        """
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        logger.info(f"Trade database initialized: {database_url}")

    def log_trade_open(
        self,
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
    ) -> int:
        """
        Log a new trade opening.

        Args:
            deal_id: Broker deal ID
            direction: BUY or SELL
            size: Position size
            entry_price: Entry price
            signal: Trading signal
            confidence: Signal confidence
            deal_reference: Broker reference
            symbol: Trading symbol
            stop_loss: Stop loss level
            take_profit: Take profit level
            predicted_price: Model's predicted price
            sentiment_score: News sentiment at entry
            volatility_regime: HIGH, NORMAL, or LOW
            news_paused: Whether news filter was active
            notes: Additional notes

        Returns:
            Record ID
        """
        # Determine predicted direction from signal
        predicted_direction = None
        if "BUY" in signal.upper():
            predicted_direction = "UP"
        elif "SELL" in signal.upper():
            predicted_direction = "DOWN"

        with self.SessionLocal() as session:
            record = TradeRecord(
                deal_id=deal_id,
                deal_reference=deal_reference,
                symbol=symbol,
                direction=direction.upper(),
                size=size,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                signal=signal,
                confidence=confidence,
                predicted_price=predicted_price,
                predicted_direction=predicted_direction,
                outcome=TradeOutcome.OPEN.value,
                opened_at=datetime.now(timezone.utc),
                sentiment_score=sentiment_score,
                volatility_regime=volatility_regime,
                news_paused=news_paused,
                notes=notes,
            )
            session.add(record)
            session.commit()
            record_id = record.id
            logger.info(f"Trade opened logged: {deal_id} {direction} {size} @ {entry_price}")
            return record_id

    def log_trade_close(
        self,
        deal_id: str,
        exit_price: float,
        pnl: float,
        pnl_pips: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Log trade closure and calculate outcome.

        Args:
            deal_id: Broker deal ID
            exit_price: Exit price
            pnl: Profit/loss amount
            pnl_pips: P&L in pips
            notes: Additional notes

        Returns:
            Updated trade record as dict, or None if not found
        """
        with self.SessionLocal() as session:
            record = session.query(TradeRecord).filter(
                TradeRecord.deal_id == deal_id
            ).first()

            if not record:
                logger.warning(f"Trade not found for closing: {deal_id}")
                return None

            # Update exit details
            record.exit_price = exit_price
            record.pnl = pnl
            record.pnl_pips = pnl_pips
            record.closed_at = datetime.now(timezone.utc)

            # Calculate P&L percent
            if record.entry_price and record.entry_price > 0:
                record.pnl_percent = (pnl / (record.entry_price * record.size)) * 100

            # Determine outcome
            if pnl > 0.01:  # Small threshold for rounding
                record.outcome = TradeOutcome.WIN.value
            elif pnl < -0.01:
                record.outcome = TradeOutcome.LOSS.value
            else:
                record.outcome = TradeOutcome.BREAKEVEN.value

            # Determine actual direction
            if exit_price > record.entry_price:
                record.actual_direction = "UP"
            elif exit_price < record.entry_price:
                record.actual_direction = "DOWN"
            else:
                record.actual_direction = "FLAT"

            # Check if prediction was correct
            if record.predicted_direction and record.actual_direction:
                record.prediction_correct = (
                    record.predicted_direction == record.actual_direction
                )

            # Append notes
            if notes:
                if record.notes:
                    record.notes = f"{record.notes} | {notes}"
                else:
                    record.notes = notes

            session.commit()

            result = self._record_to_dict(record)
            logger.info(f"Trade closed: {deal_id} @ {exit_price}, P&L: {pnl}, Outcome: {record.outcome}")
            return result

    def get_trade(self, deal_id: str) -> Optional[dict]:
        """
        Get a single trade by deal ID.

        Args:
            deal_id: Broker deal ID

        Returns:
            Trade record as dict, or None
        """
        with self.SessionLocal() as session:
            record = session.query(TradeRecord).filter(
                TradeRecord.deal_id == deal_id
            ).first()

            if record:
                return self._record_to_dict(record)
            return None

    def get_trades(self, filter: Optional[TradeFilter] = None) -> list[dict]:
        """
        Get trades with optional filtering.

        Args:
            filter: TradeFilter with criteria

        Returns:
            List of trade records as dicts
        """
        if filter is None:
            filter = TradeFilter()

        with self.SessionLocal() as session:
            query = session.query(TradeRecord)

            # Apply filters
            conditions = []

            if filter.start_date:
                conditions.append(TradeRecord.opened_at >= filter.start_date)

            if filter.end_date:
                conditions.append(TradeRecord.opened_at <= filter.end_date)

            if filter.direction:
                conditions.append(TradeRecord.direction == filter.direction.upper())

            if filter.outcome:
                conditions.append(TradeRecord.outcome == filter.outcome.upper())

            if filter.symbol:
                conditions.append(TradeRecord.symbol == filter.symbol)

            if filter.min_pnl is not None:
                conditions.append(TradeRecord.pnl >= filter.min_pnl)

            if filter.max_pnl is not None:
                conditions.append(TradeRecord.pnl <= filter.max_pnl)

            if filter.signal:
                conditions.append(TradeRecord.signal == filter.signal)

            if conditions:
                query = query.filter(and_(*conditions))

            # Order by opened_at descending (newest first)
            query = query.order_by(TradeRecord.opened_at.desc())

            # Apply pagination
            query = query.offset(filter.offset).limit(filter.limit)

            records = query.all()
            return [self._record_to_dict(r) for r in records]

    def get_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        symbol: Optional[str] = None,
    ) -> TradeSummary:
        """
        Get trade summary statistics.

        Args:
            start_date: Start of period
            end_date: End of period
            symbol: Filter by symbol

        Returns:
            TradeSummary with statistics
        """
        with self.SessionLocal() as session:
            query = session.query(TradeRecord)

            conditions = []
            if start_date:
                conditions.append(TradeRecord.opened_at >= start_date)
            if end_date:
                conditions.append(TradeRecord.opened_at <= end_date)
            if symbol:
                conditions.append(TradeRecord.symbol == symbol)

            if conditions:
                query = query.filter(and_(*conditions))

            records = query.all()

            if not records:
                return TradeSummary(
                    total_trades=0,
                    winning_trades=0,
                    losing_trades=0,
                    breakeven_trades=0,
                    open_trades=0,
                    win_rate=0.0,
                    total_pnl=0.0,
                    average_pnl=0.0,
                    largest_win=0.0,
                    largest_loss=0.0,
                    average_win=0.0,
                    average_loss=0.0,
                    profit_factor=0.0,
                )

            # Count by outcome
            winning = [r for r in records if r.outcome == TradeOutcome.WIN.value]
            losing = [r for r in records if r.outcome == TradeOutcome.LOSS.value]
            breakeven = [r for r in records if r.outcome == TradeOutcome.BREAKEVEN.value]
            open_trades = [r for r in records if r.outcome == TradeOutcome.OPEN.value]

            # Calculate P&L stats
            closed_records = [r for r in records if r.pnl is not None]
            total_pnl = sum(r.pnl for r in closed_records) if closed_records else 0.0

            wins_pnl = [r.pnl for r in winning if r.pnl is not None]
            losses_pnl = [r.pnl for r in losing if r.pnl is not None]

            largest_win = max(wins_pnl) if wins_pnl else 0.0
            largest_loss = min(losses_pnl) if losses_pnl else 0.0
            average_win = sum(wins_pnl) / len(wins_pnl) if wins_pnl else 0.0
            average_loss = sum(losses_pnl) / len(losses_pnl) if losses_pnl else 0.0

            # Win rate (exclude open trades)
            closed_count = len(winning) + len(losing) + len(breakeven)
            win_rate = len(winning) / closed_count if closed_count > 0 else 0.0

            # Profit factor (gross profit / gross loss)
            gross_profit = sum(wins_pnl) if wins_pnl else 0.0
            gross_loss = abs(sum(losses_pnl)) if losses_pnl else 0.0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

            # Average P&L
            average_pnl = total_pnl / len(closed_records) if closed_records else 0.0

            return TradeSummary(
                total_trades=len(records),
                winning_trades=len(winning),
                losing_trades=len(losing),
                breakeven_trades=len(breakeven),
                open_trades=len(open_trades),
                win_rate=win_rate,
                total_pnl=total_pnl,
                average_pnl=average_pnl,
                largest_win=largest_win,
                largest_loss=largest_loss,
                average_win=average_win,
                average_loss=average_loss,
                profit_factor=profit_factor if profit_factor != float('inf') else 999.99,
            )

    def get_prediction_accuracy(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict:
        """
        Get prediction accuracy statistics.

        Args:
            start_date: Start of period
            end_date: End of period

        Returns:
            Dict with accuracy metrics
        """
        with self.SessionLocal() as session:
            query = session.query(TradeRecord).filter(
                TradeRecord.prediction_correct.isnot(None)
            )

            if start_date:
                query = query.filter(TradeRecord.opened_at >= start_date)
            if end_date:
                query = query.filter(TradeRecord.opened_at <= end_date)

            records = query.all()

            if not records:
                return {
                    "total_predictions": 0,
                    "correct_predictions": 0,
                    "accuracy": 0.0,
                    "by_signal": {},
                }

            correct = [r for r in records if r.prediction_correct]
            accuracy = len(correct) / len(records)

            # Accuracy by signal type
            by_signal: dict[str, dict] = {}
            for record in records:
                signal = record.signal or "UNKNOWN"
                if signal not in by_signal:
                    by_signal[signal] = {"total": 0, "correct": 0}
                by_signal[signal]["total"] += 1
                if record.prediction_correct:
                    by_signal[signal]["correct"] += 1

            for signal in by_signal:
                total = by_signal[signal]["total"]
                correct_count = by_signal[signal]["correct"]
                by_signal[signal]["accuracy"] = correct_count / total if total > 0 else 0.0

            return {
                "total_predictions": len(records),
                "correct_predictions": len(correct),
                "accuracy": round(accuracy, 4),
                "by_signal": by_signal,
            }

    def get_open_trades(self) -> list[dict]:
        """Get all open trades."""
        return self.get_trades(TradeFilter(outcome="OPEN", limit=1000))

    def count_trades(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> int:
        """Count total trades in period."""
        with self.SessionLocal() as session:
            query = session.query(func.count(TradeRecord.id))

            if start_date:
                query = query.filter(TradeRecord.opened_at >= start_date)
            if end_date:
                query = query.filter(TradeRecord.opened_at <= end_date)

            return query.scalar() or 0

    def _record_to_dict(self, record: TradeRecord) -> dict:
        """Convert TradeRecord to dictionary."""
        return {
            "id": record.id,
            "deal_id": record.deal_id,
            "deal_reference": record.deal_reference,
            "symbol": record.symbol,
            "direction": record.direction,
            "size": record.size,
            "entry_price": record.entry_price,
            "exit_price": record.exit_price,
            "stop_loss": record.stop_loss,
            "take_profit": record.take_profit,
            "pnl": record.pnl,
            "pnl_pips": record.pnl_pips,
            "pnl_percent": record.pnl_percent,
            "signal": record.signal,
            "confidence": record.confidence,
            "predicted_price": record.predicted_price,
            "predicted_direction": record.predicted_direction,
            "actual_direction": record.actual_direction,
            "prediction_correct": record.prediction_correct,
            "outcome": record.outcome,
            "opened_at": record.opened_at.isoformat() if record.opened_at else None,
            "closed_at": record.closed_at.isoformat() if record.closed_at else None,
            "sentiment_score": record.sentiment_score,
            "volatility_regime": record.volatility_regime,
            "news_paused": record.news_paused,
            "notes": record.notes,
        }

    def get_status(self) -> dict:
        """Get database status."""
        with self.SessionLocal() as session:
            total = session.query(func.count(TradeRecord.id)).scalar() or 0
            open_count = session.query(func.count(TradeRecord.id)).filter(
                TradeRecord.outcome == TradeOutcome.OPEN.value
            ).scalar() or 0

            latest = session.query(TradeRecord).order_by(
                TradeRecord.opened_at.desc()
            ).first()

        return {
            "database_url": self.database_url,
            "total_trades": total,
            "open_trades": open_count,
            "closed_trades": total - open_count,
            "latest_trade": latest.deal_id if latest else None,
            "latest_trade_at": latest.opened_at.isoformat() if latest else None,
        }
