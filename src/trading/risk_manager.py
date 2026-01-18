"""
Risk Manager Module
===================
Handles daily loss limits, position sizing, and risk controls.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from loguru import logger


@dataclass
class Trade:
    """Represents a closed trade for P&L tracking."""
    timestamp: datetime
    direction: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    symbol: str = "XAUUSD"
    deal_id: Optional[str] = None


@dataclass
class DailyStats:
    """Daily trading statistics."""
    date: str
    total_pnl: float
    trade_count: int
    winning_trades: int
    losing_trades: int
    largest_win: float
    largest_loss: float
    trading_allowed: bool
    limit_reached_at: Optional[str] = None

    @property
    def win_rate(self) -> float:
        """Calculate win rate as percentage."""
        if self.trade_count == 0:
            return 0.0
        return (self.winning_trades / self.trade_count) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "date": self.date,
            "total_pnl": round(self.total_pnl, 2),
            "trade_count": self.trade_count,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 2),
            "largest_win": round(self.largest_win, 2),
            "largest_loss": round(self.largest_loss, 2),
            "trading_allowed": self.trading_allowed,
            "limit_reached_at": self.limit_reached_at,
        }


@dataclass
class PositionSizeResult:
    """Result of position size calculation."""
    adjusted_size: float
    base_size: float
    adjustment_factor: float
    volatility_regime: str  # "low", "normal", "high"
    current_atr: float
    average_atr: float
    atr_ratio: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "adjusted_size": round(self.adjusted_size, 4),
            "base_size": round(self.base_size, 4),
            "adjustment_factor": round(self.adjustment_factor, 4),
            "volatility_regime": self.volatility_regime,
            "current_atr": round(self.current_atr, 4),
            "average_atr": round(self.average_atr, 4),
            "atr_ratio": round(self.atr_ratio, 4),
            "reason": self.reason,
        }


class RiskManager:
    """
    Manages trading risk including daily loss limits and position sizing.

    Features:
    - Daily P&L tracking
    - Daily loss limit protection
    - Volatility-adjusted position sizing
    - Trade logging
    - Risk statistics

    Usage:
        risk_manager = RiskManager(
            daily_loss_limit_pct=3.0,
            account_balance=10000.0,
        )

        # Check before trading
        if risk_manager.can_trade():
            # Execute trade
            pass

        # Calculate position size based on volatility
        position = risk_manager.calculate_position_size(atr_values)

        # Record closed trade
        risk_manager.record_trade(trade)
    """

    def __init__(
        self,
        daily_loss_limit_pct: float = 3.0,
        account_balance: float = 10000.0,
        max_daily_trades: int = 50,
        base_position_size: float = 0.01,
        max_position_size: float = 0.1,
        min_position_size: float = 0.01,
        volatility_high_threshold: float = 1.5,
        volatility_low_threshold: float = 0.7,
        volatility_lookback: int = 20,
    ):
        """
        Initialize risk manager.

        Args:
            daily_loss_limit_pct: Maximum daily loss as percentage of account (default 3%)
            account_balance: Account balance for calculating loss limit
            max_daily_trades: Maximum number of trades per day
            base_position_size: Base position size in lots (default 0.01)
            max_position_size: Maximum position size in lots (default 0.1)
            min_position_size: Minimum position size in lots (default 0.01)
            volatility_high_threshold: ATR multiplier for high volatility (default 1.5)
            volatility_low_threshold: ATR multiplier for low volatility (default 0.7)
            volatility_lookback: Periods for average ATR calculation (default 20)
        """
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.account_balance = account_balance
        self.max_daily_trades = max_daily_trades

        # Position sizing parameters
        self.base_position_size = base_position_size
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.volatility_high_threshold = volatility_high_threshold
        self.volatility_low_threshold = volatility_low_threshold
        self.volatility_lookback = volatility_lookback

        # Daily tracking
        self._current_date: str = self._get_utc_date()
        self._daily_pnl: float = 0.0
        self._daily_trades: List[Trade] = []
        self._trading_blocked: bool = False
        self._blocked_at: Optional[str] = None

        logger.info(
            f"RiskManager initialized: "
            f"daily_loss_limit={daily_loss_limit_pct}%, "
            f"balance=${account_balance:,.2f}, "
            f"base_size={base_position_size} lots"
        )

    @staticmethod
    def _get_utc_date() -> str:
        """Get current UTC date as string."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    @staticmethod
    def _get_utc_timestamp() -> str:
        """Get current UTC timestamp as ISO string."""
        return datetime.now(timezone.utc).isoformat()

    @property
    def daily_loss_limit(self) -> float:
        """Calculate absolute daily loss limit in account currency."""
        return self.account_balance * (self.daily_loss_limit_pct / 100)

    def _reset_if_new_day(self) -> None:
        """Reset daily counters if it's a new trading day."""
        current_date = self._get_utc_date()

        if current_date != self._current_date:
            logger.info(
                f"New trading day: {current_date}. "
                f"Previous day P&L: ${self._daily_pnl:,.2f}"
            )

            # Reset counters
            self._current_date = current_date
            self._daily_pnl = 0.0
            self._daily_trades = []
            self._trading_blocked = False
            self._blocked_at = None

    def update_account_balance(self, new_balance: float) -> None:
        """
        Update account balance for loss limit calculation.

        Args:
            new_balance: New account balance
        """
        old_balance = self.account_balance
        self.account_balance = new_balance
        logger.info(
            f"Account balance updated: ${old_balance:,.2f} -> ${new_balance:,.2f}"
        )

    def can_trade(self) -> tuple[bool, Optional[str]]:
        """
        Check if trading is allowed based on daily limits.

        Returns:
            Tuple of (allowed: bool, reason: str or None)
        """
        self._reset_if_new_day()

        # Check if already blocked
        if self._trading_blocked:
            return False, f"Daily loss limit reached at {self._blocked_at}"

        # Check daily loss limit
        if abs(self._daily_pnl) >= self.daily_loss_limit and self._daily_pnl < 0:
            self._trading_blocked = True
            self._blocked_at = self._get_utc_timestamp()
            logger.warning(
                f"Daily loss limit reached! "
                f"P&L: ${self._daily_pnl:,.2f}, "
                f"Limit: ${self.daily_loss_limit:,.2f}"
            )
            return False, f"Daily loss limit of {self.daily_loss_limit_pct}% reached"

        # Check max daily trades
        if len(self._daily_trades) >= self.max_daily_trades:
            return False, f"Maximum daily trades ({self.max_daily_trades}) reached"

        return True, None

    def record_trade(self, trade: Trade) -> None:
        """
        Record a closed trade and update daily P&L.

        Args:
            trade: Closed trade to record
        """
        self._reset_if_new_day()

        self._daily_trades.append(trade)
        self._daily_pnl += trade.pnl

        logger.info(
            f"Trade recorded: {trade.direction} {trade.size} lots, "
            f"P&L: ${trade.pnl:,.2f}, "
            f"Daily total: ${self._daily_pnl:,.2f}"
        )

        # Check if this trade pushed us over the limit
        self.can_trade()

    def record_pnl(
        self,
        pnl: float,
        direction: str = "UNKNOWN",
        size: float = 0.01,
        entry_price: float = 0.0,
        exit_price: float = 0.0,
        deal_id: Optional[str] = None,
    ) -> None:
        """
        Convenience method to record P&L without creating Trade object.

        Args:
            pnl: Profit/loss amount
            direction: Trade direction (BUY/SELL)
            size: Position size
            entry_price: Entry price
            exit_price: Exit price
            deal_id: Deal identifier
        """
        trade = Trade(
            timestamp=datetime.now(timezone.utc),
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            size=size,
            pnl=pnl,
            deal_id=deal_id,
        )
        self.record_trade(trade)

    def get_daily_stats(self) -> DailyStats:
        """
        Get current daily trading statistics.

        Returns:
            DailyStats object with current day's performance
        """
        self._reset_if_new_day()

        winning_trades = [t for t in self._daily_trades if t.pnl > 0]
        losing_trades = [t for t in self._daily_trades if t.pnl < 0]

        largest_win = max((t.pnl for t in winning_trades), default=0.0)
        largest_loss = min((t.pnl for t in losing_trades), default=0.0)

        can_trade, _ = self.can_trade()

        return DailyStats(
            date=self._current_date,
            total_pnl=self._daily_pnl,
            trade_count=len(self._daily_trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            largest_win=largest_win,
            largest_loss=largest_loss,
            trading_allowed=can_trade,
            limit_reached_at=self._blocked_at,
        )

    def get_remaining_risk(self) -> float:
        """
        Get remaining risk budget for the day.

        Returns:
            Amount that can still be lost before hitting limit
        """
        self._reset_if_new_day()

        if self._daily_pnl >= 0:
            # If we're in profit, full limit is available
            return self.daily_loss_limit
        else:
            # If we're in loss, remaining is limit minus current loss
            return max(0, self.daily_loss_limit - abs(self._daily_pnl))

    def get_status(self) -> Dict[str, Any]:
        """
        Get full risk manager status.

        Returns:
            Dictionary with current risk status
        """
        self._reset_if_new_day()

        can_trade, reason = self.can_trade()

        return {
            "date": self._current_date,
            "account_balance": self.account_balance,
            "daily_loss_limit_pct": self.daily_loss_limit_pct,
            "daily_loss_limit_amount": round(self.daily_loss_limit, 2),
            "current_daily_pnl": round(self._daily_pnl, 2),
            "remaining_risk": round(self.get_remaining_risk(), 2),
            "trades_today": len(self._daily_trades),
            "max_daily_trades": self.max_daily_trades,
            "trading_allowed": can_trade,
            "blocked_reason": reason,
            "blocked_at": self._blocked_at,
        }

    def calculate_position_size(
        self,
        atr_values: List[float],
        base_size: Optional[float] = None,
    ) -> PositionSizeResult:
        """
        Calculate volatility-adjusted position size.

        Uses the relationship between current ATR and average ATR to adjust
        position size. Higher volatility = smaller position, lower volatility
        = larger position.

        Args:
            atr_values: List of ATR values (at least volatility_lookback + 1 values)
            base_size: Override base position size (optional)

        Returns:
            PositionSizeResult with adjusted size and details
        """
        base = base_size if base_size is not None else self.base_position_size

        # Need at least lookback + 1 values
        if len(atr_values) < self.volatility_lookback + 1:
            logger.warning(
                f"Not enough ATR values ({len(atr_values)}) for volatility calculation. "
                f"Need at least {self.volatility_lookback + 1}. Using base size."
            )
            return PositionSizeResult(
                adjusted_size=base,
                base_size=base,
                adjustment_factor=1.0,
                volatility_regime="unknown",
                current_atr=atr_values[-1] if atr_values else 0.0,
                average_atr=0.0,
                atr_ratio=1.0,
                reason="Insufficient ATR data for volatility calculation",
            )

        # Get current ATR and average ATR
        current_atr = atr_values[-1]
        avg_atr = np.mean(atr_values[-self.volatility_lookback - 1:-1])

        # Calculate ATR ratio
        if avg_atr == 0:
            atr_ratio = 1.0
        else:
            atr_ratio = current_atr / avg_atr

        # Determine volatility regime and adjustment factor
        if atr_ratio > self.volatility_high_threshold:
            # High volatility - reduce position size
            volatility_regime = "high"
            # Scale down: at 1.5x use 0.67, at 2x use 0.5, etc.
            adjustment_factor = 1.0 / atr_ratio
            reason = f"High volatility (ATR {atr_ratio:.2f}x avg) - reducing position"
        elif atr_ratio < self.volatility_low_threshold:
            # Low volatility - increase position size
            volatility_regime = "low"
            # Scale up: at 0.7x use 1.43, at 0.5x use 2.0, etc.
            # But cap the increase to prevent oversizing
            adjustment_factor = min(1.0 / atr_ratio, 2.0)
            reason = f"Low volatility (ATR {atr_ratio:.2f}x avg) - increasing position"
        else:
            # Normal volatility - use base size
            volatility_regime = "normal"
            adjustment_factor = 1.0
            reason = f"Normal volatility (ATR {atr_ratio:.2f}x avg) - using base size"

        # Calculate adjusted size
        adjusted_size = base * adjustment_factor

        # Apply limits
        if adjusted_size > self.max_position_size:
            adjusted_size = self.max_position_size
            reason += f" (capped at max {self.max_position_size})"
        elif adjusted_size < self.min_position_size:
            adjusted_size = self.min_position_size
            reason += f" (floored at min {self.min_position_size})"

        # Round to 2 decimal places (standard lot sizing)
        adjusted_size = round(adjusted_size, 2)

        logger.info(
            f"Position size calculated: {base} -> {adjusted_size} lots "
            f"({volatility_regime} volatility, ATR ratio: {atr_ratio:.2f})"
        )

        return PositionSizeResult(
            adjusted_size=adjusted_size,
            base_size=base,
            adjustment_factor=adjustment_factor,
            volatility_regime=volatility_regime,
            current_atr=current_atr,
            average_atr=avg_atr,
            atr_ratio=atr_ratio,
            reason=reason,
        )

    def get_position_sizing_config(self) -> Dict[str, Any]:
        """
        Get current position sizing configuration.

        Returns:
            Dictionary with position sizing parameters
        """
        return {
            "base_position_size": self.base_position_size,
            "max_position_size": self.max_position_size,
            "min_position_size": self.min_position_size,
            "volatility_high_threshold": self.volatility_high_threshold,
            "volatility_low_threshold": self.volatility_low_threshold,
            "volatility_lookback": self.volatility_lookback,
        }
