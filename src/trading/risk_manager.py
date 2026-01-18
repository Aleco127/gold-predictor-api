"""
Risk Manager Module
===================
Handles daily loss limits, position sizing, and risk controls.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
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


class RiskManager:
    """
    Manages trading risk including daily loss limits.

    Features:
    - Daily P&L tracking
    - Daily loss limit protection
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

        # Record closed trade
        risk_manager.record_trade(trade)
    """

    def __init__(
        self,
        daily_loss_limit_pct: float = 3.0,
        account_balance: float = 10000.0,
        max_daily_trades: int = 50,
    ):
        """
        Initialize risk manager.

        Args:
            daily_loss_limit_pct: Maximum daily loss as percentage of account (default 3%)
            account_balance: Account balance for calculating loss limit
            max_daily_trades: Maximum number of trades per day
        """
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.account_balance = account_balance
        self.max_daily_trades = max_daily_trades

        # Daily tracking
        self._current_date: str = self._get_utc_date()
        self._daily_pnl: float = 0.0
        self._daily_trades: List[Trade] = []
        self._trading_blocked: bool = False
        self._blocked_at: Optional[str] = None

        logger.info(
            f"RiskManager initialized: "
            f"daily_loss_limit={daily_loss_limit_pct}%, "
            f"balance=${account_balance:,.2f}"
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
