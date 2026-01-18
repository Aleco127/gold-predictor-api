"""
Performance Tracker Module
==========================
Advanced trading performance metrics including Sharpe ratio, max drawdown, and expectancy.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
import math
import logging

from ..storage.trade_database import TradeDatabase, TradeFilter

logger = logging.getLogger(__name__)

# Trading days per year for annualization
TRADING_DAYS_PER_YEAR = 252

# Risk-free rate (annualized) - typically US Treasury rate
DEFAULT_RISK_FREE_RATE = 0.05  # 5%


@dataclass
class DrawdownInfo:
    """Drawdown information."""
    max_drawdown: float  # Maximum drawdown as decimal (0.10 = 10%)
    max_drawdown_amount: float  # Maximum drawdown in currency
    current_drawdown: float  # Current drawdown as decimal
    peak_equity: float  # Highest equity reached
    trough_equity: float  # Lowest equity during max drawdown
    drawdown_start: Optional[str] = None
    drawdown_end: Optional[str] = None
    recovery_date: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "max_drawdown": round(self.max_drawdown, 4),
            "max_drawdown_percent": round(self.max_drawdown * 100, 2),
            "max_drawdown_amount": round(self.max_drawdown_amount, 2),
            "current_drawdown": round(self.current_drawdown, 4),
            "current_drawdown_percent": round(self.current_drawdown * 100, 2),
            "peak_equity": round(self.peak_equity, 2),
            "trough_equity": round(self.trough_equity, 2),
            "drawdown_start": self.drawdown_start,
            "drawdown_end": self.drawdown_end,
            "recovery_date": self.recovery_date,
        }


@dataclass
class PerformanceMetrics:
    """Complete trading performance metrics."""
    # Basic metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    win_rate: float

    # P&L metrics
    total_pnl: float
    average_pnl: float
    largest_win: float
    largest_loss: float
    average_win: float
    average_loss: float

    # Risk metrics
    profit_factor: float
    expectancy: float  # Expected value per trade
    risk_reward_ratio: float

    # Drawdown
    max_drawdown: float
    max_drawdown_amount: float
    current_drawdown: float

    # Sharpe and Sortino
    sharpe_ratio: float
    sortino_ratio: float

    # Time metrics
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    trading_days: int = 0

    # Additional context
    avg_trades_per_day: float = 0.0
    best_day_pnl: float = 0.0
    worst_day_pnl: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            # Basic metrics
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "breakeven_trades": self.breakeven_trades,
            "win_rate": round(self.win_rate, 4),
            "win_rate_percent": round(self.win_rate * 100, 2),

            # P&L metrics
            "total_pnl": round(self.total_pnl, 2),
            "average_pnl": round(self.average_pnl, 2),
            "largest_win": round(self.largest_win, 2),
            "largest_loss": round(self.largest_loss, 2),
            "average_win": round(self.average_win, 2),
            "average_loss": round(self.average_loss, 2),

            # Risk metrics
            "profit_factor": round(self.profit_factor, 4),
            "expectancy": round(self.expectancy, 2),
            "risk_reward_ratio": round(self.risk_reward_ratio, 4),

            # Drawdown
            "max_drawdown": round(self.max_drawdown, 4),
            "max_drawdown_percent": round(self.max_drawdown * 100, 2),
            "max_drawdown_amount": round(self.max_drawdown_amount, 2),
            "current_drawdown": round(self.current_drawdown, 4),
            "current_drawdown_percent": round(self.current_drawdown * 100, 2),

            # Sharpe and Sortino
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),

            # Time metrics
            "period_start": self.period_start,
            "period_end": self.period_end,
            "trading_days": self.trading_days,
            "avg_trades_per_day": round(self.avg_trades_per_day, 2),

            # Streak metrics
            "best_day_pnl": round(self.best_day_pnl, 2),
            "worst_day_pnl": round(self.worst_day_pnl, 2),
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
        }


class TradingPerformanceTracker:
    """
    Calculate advanced trading performance metrics.

    Uses TradeDatabase to fetch trade history and calculates:
    - Win rate, profit factor, expectancy
    - Maximum drawdown (peak-to-trough)
    - Sharpe ratio (annualized)
    - Sortino ratio (downside deviation)
    - Risk-reward ratio
    - Consecutive win/loss streaks

    Note: This class is different from PerformanceTracker in drift_detector.py,
    which tracks MODEL performance. This class tracks TRADING performance.
    """

    def __init__(
        self,
        trade_db: TradeDatabase,
        initial_balance: float = 10000.0,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    ):
        """
        Initialize performance tracker.

        Args:
            trade_db: TradeDatabase instance for fetching trades
            initial_balance: Starting account balance for equity calculations
            risk_free_rate: Annualized risk-free rate for Sharpe calculation
        """
        self.trade_db = trade_db
        self.initial_balance = initial_balance
        self.risk_free_rate = risk_free_rate

    def get_performance(
        self,
        period: str = "all",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> PerformanceMetrics:
        """
        Calculate performance metrics for a period.

        Args:
            period: Period shorthand - "7d", "30d", "90d", "365d", "all"
            start_date: Custom start date (overrides period)
            end_date: Custom end date (defaults to now)

        Returns:
            PerformanceMetrics with all calculated metrics
        """
        # Calculate date range
        if start_date is None:
            start_date = self._period_to_date(period)

        if end_date is None:
            end_date = datetime.now(timezone.utc)

        # Fetch trades for period
        trade_filter = TradeFilter(
            start_date=start_date,
            end_date=end_date,
            limit=10000,  # Get all trades
        )
        trades = self.trade_db.get_trades(trade_filter)

        # If no trades, return empty metrics
        if not trades:
            return self._empty_metrics(start_date, end_date)

        # Filter to closed trades only (have P&L)
        closed_trades = [t for t in trades if t.get("pnl") is not None]

        if not closed_trades:
            return self._empty_metrics(start_date, end_date)

        # Sort by opened_at for sequential analysis
        closed_trades.sort(key=lambda t: t.get("opened_at", ""))

        # Calculate basic metrics
        total_trades = len(closed_trades)
        winning_trades = [t for t in closed_trades if t.get("outcome") == "WIN"]
        losing_trades = [t for t in closed_trades if t.get("outcome") == "LOSS"]
        breakeven_trades = [t for t in closed_trades if t.get("outcome") == "BREAKEVEN"]

        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        breakeven_count = len(breakeven_trades)

        # Win rate
        win_rate = win_count / total_trades if total_trades > 0 else 0.0

        # P&L calculations
        pnl_list = [t.get("pnl", 0) for t in closed_trades]
        total_pnl = sum(pnl_list)
        average_pnl = total_pnl / total_trades if total_trades > 0 else 0.0

        wins_pnl = [t.get("pnl", 0) for t in winning_trades]
        losses_pnl = [t.get("pnl", 0) for t in losing_trades]

        largest_win = max(wins_pnl) if wins_pnl else 0.0
        largest_loss = min(losses_pnl) if losses_pnl else 0.0
        average_win = sum(wins_pnl) / len(wins_pnl) if wins_pnl else 0.0
        average_loss = sum(losses_pnl) / len(losses_pnl) if losses_pnl else 0.0

        # Profit factor
        gross_profit = sum(wins_pnl) if wins_pnl else 0.0
        gross_loss = abs(sum(losses_pnl)) if losses_pnl else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
            999.99 if gross_profit > 0 else 0.0
        )

        # Expectancy (expected value per trade)
        # E = (Win% * Avg Win) + (Loss% * Avg Loss)
        loss_rate = loss_count / total_trades if total_trades > 0 else 0.0
        expectancy = (win_rate * average_win) + (loss_rate * average_loss)

        # Risk-reward ratio
        risk_reward = abs(average_win / average_loss) if average_loss != 0 else (
            999.99 if average_win > 0 else 0.0
        )

        # Calculate drawdown
        drawdown_info = self._calculate_drawdown(closed_trades)

        # Calculate Sharpe and Sortino ratios
        sharpe = self._calculate_sharpe_ratio(pnl_list, start_date, end_date)
        sortino = self._calculate_sortino_ratio(pnl_list, start_date, end_date)

        # Calculate streaks
        streaks = self._calculate_streaks(closed_trades)

        # Daily metrics
        daily_metrics = self._calculate_daily_metrics(closed_trades, start_date, end_date)

        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=win_count,
            losing_trades=loss_count,
            breakeven_trades=breakeven_count,
            win_rate=win_rate,
            total_pnl=total_pnl,
            average_pnl=average_pnl,
            largest_win=largest_win,
            largest_loss=largest_loss,
            average_win=average_win,
            average_loss=average_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            risk_reward_ratio=risk_reward,
            max_drawdown=drawdown_info.max_drawdown,
            max_drawdown_amount=drawdown_info.max_drawdown_amount,
            current_drawdown=drawdown_info.current_drawdown,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            period_start=start_date.isoformat() if start_date else None,
            period_end=end_date.isoformat() if end_date else None,
            trading_days=daily_metrics["trading_days"],
            avg_trades_per_day=daily_metrics["avg_trades_per_day"],
            best_day_pnl=daily_metrics["best_day_pnl"],
            worst_day_pnl=daily_metrics["worst_day_pnl"],
            max_consecutive_wins=streaks["max_consecutive_wins"],
            max_consecutive_losses=streaks["max_consecutive_losses"],
        )

    def get_drawdown(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> DrawdownInfo:
        """
        Calculate drawdown metrics.

        Args:
            start_date: Start of period
            end_date: End of period

        Returns:
            DrawdownInfo with max and current drawdown
        """
        # Fetch trades
        trade_filter = TradeFilter(
            start_date=start_date,
            end_date=end_date,
            limit=10000,
        )
        trades = self.trade_db.get_trades(trade_filter)

        # Filter to closed trades
        closed_trades = [t for t in trades if t.get("pnl") is not None]

        if not closed_trades:
            return DrawdownInfo(
                max_drawdown=0.0,
                max_drawdown_amount=0.0,
                current_drawdown=0.0,
                peak_equity=self.initial_balance,
                trough_equity=self.initial_balance,
            )

        # Sort by opened_at
        closed_trades.sort(key=lambda t: t.get("opened_at", ""))

        return self._calculate_drawdown(closed_trades)

    def _calculate_drawdown(self, trades: list[dict]) -> DrawdownInfo:
        """Calculate drawdown from trade list."""
        if not trades:
            return DrawdownInfo(
                max_drawdown=0.0,
                max_drawdown_amount=0.0,
                current_drawdown=0.0,
                peak_equity=self.initial_balance,
                trough_equity=self.initial_balance,
            )

        # Build equity curve
        equity = self.initial_balance
        peak_equity = self.initial_balance
        trough_equity = self.initial_balance

        max_drawdown = 0.0
        max_drawdown_amount = 0.0
        max_dd_start = None
        max_dd_end = None

        current_dd_start = None

        for trade in trades:
            pnl = trade.get("pnl", 0)
            equity += pnl

            # Track peak
            if equity > peak_equity:
                peak_equity = equity
                current_dd_start = None  # Reset drawdown start

            # Calculate current drawdown
            if peak_equity > 0:
                current_drawdown = (peak_equity - equity) / peak_equity
                current_drawdown_amount = peak_equity - equity
            else:
                current_drawdown = 0.0
                current_drawdown_amount = 0.0

            # Track max drawdown
            if current_drawdown > max_drawdown:
                max_drawdown = current_drawdown
                max_drawdown_amount = current_drawdown_amount
                trough_equity = equity
                max_dd_end = trade.get("closed_at")
                if current_dd_start is None:
                    current_dd_start = trade.get("opened_at")
                max_dd_start = current_dd_start

        # Calculate current drawdown (from most recent peak)
        final_drawdown = 0.0
        if peak_equity > 0:
            final_drawdown = (peak_equity - equity) / peak_equity

        return DrawdownInfo(
            max_drawdown=max_drawdown,
            max_drawdown_amount=max_drawdown_amount,
            current_drawdown=final_drawdown,
            peak_equity=peak_equity,
            trough_equity=trough_equity,
            drawdown_start=max_dd_start,
            drawdown_end=max_dd_end,
        )

    def _calculate_sharpe_ratio(
        self,
        pnl_list: list[float],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
    ) -> float:
        """
        Calculate annualized Sharpe ratio.

        Sharpe = (Mean Return - Risk Free Rate) / Std Dev of Returns
        Annualized by multiplying by sqrt(periods per year)
        """
        if len(pnl_list) < 2:
            return 0.0

        # Calculate mean return
        mean_return = sum(pnl_list) / len(pnl_list)

        # Calculate standard deviation
        variance = sum((x - mean_return) ** 2 for x in pnl_list) / len(pnl_list)
        std_dev = math.sqrt(variance) if variance > 0 else 0.0

        if std_dev == 0:
            return 0.0

        # Estimate trades per year for annualization
        if start_date and end_date:
            days = (end_date - start_date).days
            if days > 0:
                trades_per_day = len(pnl_list) / days
                trades_per_year = trades_per_day * TRADING_DAYS_PER_YEAR
            else:
                trades_per_year = len(pnl_list)
        else:
            # Assume trades span trading days proportionally
            trades_per_year = TRADING_DAYS_PER_YEAR

        # Risk-free rate per trade
        rf_per_trade = self.risk_free_rate / trades_per_year if trades_per_year > 0 else 0

        # Sharpe ratio (per trade)
        sharpe_per_trade = (mean_return - rf_per_trade) / std_dev

        # Annualize
        annualization_factor = math.sqrt(trades_per_year)
        sharpe_annualized = sharpe_per_trade * annualization_factor

        return sharpe_annualized

    def _calculate_sortino_ratio(
        self,
        pnl_list: list[float],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
    ) -> float:
        """
        Calculate annualized Sortino ratio.

        Sortino = (Mean Return - Risk Free Rate) / Downside Deviation
        Only considers negative returns for volatility
        """
        if len(pnl_list) < 2:
            return 0.0

        # Calculate mean return
        mean_return = sum(pnl_list) / len(pnl_list)

        # Calculate downside deviation (only negative returns)
        negative_returns = [x for x in pnl_list if x < 0]

        if not negative_returns:
            # No losing trades - infinite Sortino (cap at 999)
            return 999.99 if mean_return > 0 else 0.0

        # Downside variance
        downside_variance = sum(x ** 2 for x in negative_returns) / len(pnl_list)
        downside_dev = math.sqrt(downside_variance) if downside_variance > 0 else 0.0

        if downside_dev == 0:
            return 0.0

        # Estimate trades per year
        if start_date and end_date:
            days = (end_date - start_date).days
            if days > 0:
                trades_per_day = len(pnl_list) / days
                trades_per_year = trades_per_day * TRADING_DAYS_PER_YEAR
            else:
                trades_per_year = len(pnl_list)
        else:
            trades_per_year = TRADING_DAYS_PER_YEAR

        # Risk-free rate per trade
        rf_per_trade = self.risk_free_rate / trades_per_year if trades_per_year > 0 else 0

        # Sortino ratio (per trade)
        sortino_per_trade = (mean_return - rf_per_trade) / downside_dev

        # Annualize
        annualization_factor = math.sqrt(trades_per_year)
        sortino_annualized = sortino_per_trade * annualization_factor

        return sortino_annualized

    def _calculate_streaks(self, trades: list[dict]) -> dict:
        """Calculate consecutive win/loss streaks."""
        if not trades:
            return {
                "max_consecutive_wins": 0,
                "max_consecutive_losses": 0,
                "current_streak": 0,
                "current_streak_type": None,
            }

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for trade in trades:
            outcome = trade.get("outcome")

            if outcome == "WIN":
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif outcome == "LOSS":
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
            else:
                # Breakeven doesn't break streak
                pass

        return {
            "max_consecutive_wins": max_wins,
            "max_consecutive_losses": max_losses,
            "current_streak": current_wins if current_wins > 0 else -current_losses,
            "current_streak_type": "WIN" if current_wins > 0 else ("LOSS" if current_losses > 0 else None),
        }

    def _calculate_daily_metrics(
        self,
        trades: list[dict],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
    ) -> dict:
        """Calculate daily trading metrics."""
        if not trades:
            return {
                "trading_days": 0,
                "avg_trades_per_day": 0.0,
                "best_day_pnl": 0.0,
                "worst_day_pnl": 0.0,
            }

        # Group trades by day
        daily_pnl: dict[str, float] = {}
        daily_count: dict[str, int] = {}

        for trade in trades:
            opened_at = trade.get("opened_at", "")
            if opened_at:
                # Extract date part
                day = opened_at[:10]  # YYYY-MM-DD
                pnl = trade.get("pnl", 0)

                daily_pnl[day] = daily_pnl.get(day, 0) + pnl
                daily_count[day] = daily_count.get(day, 0) + 1

        trading_days = len(daily_pnl)

        # Calculate date range
        if start_date and end_date:
            total_days = max(1, (end_date - start_date).days)
        else:
            total_days = trading_days

        avg_trades_per_day = len(trades) / trading_days if trading_days > 0 else 0.0
        best_day = max(daily_pnl.values()) if daily_pnl else 0.0
        worst_day = min(daily_pnl.values()) if daily_pnl else 0.0

        return {
            "trading_days": trading_days,
            "avg_trades_per_day": avg_trades_per_day,
            "best_day_pnl": best_day,
            "worst_day_pnl": worst_day,
        }

    def _period_to_date(self, period: str) -> Optional[datetime]:
        """Convert period string to start date."""
        now = datetime.now(timezone.utc)

        period_map = {
            "7d": timedelta(days=7),
            "30d": timedelta(days=30),
            "90d": timedelta(days=90),
            "365d": timedelta(days=365),
            "1y": timedelta(days=365),
            "all": None,
        }

        delta = period_map.get(period.lower())
        if delta is None:
            return None  # All time

        return now - delta

    def _empty_metrics(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
    ) -> PerformanceMetrics:
        """Return empty metrics structure."""
        return PerformanceMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            breakeven_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            average_pnl=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            average_win=0.0,
            average_loss=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            risk_reward_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_amount=0.0,
            current_drawdown=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            period_start=start_date.isoformat() if start_date else None,
            period_end=end_date.isoformat() if end_date else None,
        )

    def get_equity_curve(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> list[dict]:
        """
        Get equity curve data points.

        Returns list of {timestamp, equity, pnl, cumulative_pnl}
        """
        trade_filter = TradeFilter(
            start_date=start_date,
            end_date=end_date,
            limit=10000,
        )
        trades = self.trade_db.get_trades(trade_filter)

        # Filter to closed trades
        closed_trades = [t for t in trades if t.get("pnl") is not None]

        if not closed_trades:
            return []

        # Sort by closed_at
        closed_trades.sort(key=lambda t: t.get("closed_at", "") or t.get("opened_at", ""))

        equity = self.initial_balance
        cumulative_pnl = 0.0
        curve = []

        # Add starting point
        curve.append({
            "timestamp": closed_trades[0].get("opened_at"),
            "equity": self.initial_balance,
            "pnl": 0,
            "cumulative_pnl": 0,
        })

        for trade in closed_trades:
            pnl = trade.get("pnl", 0)
            equity += pnl
            cumulative_pnl += pnl

            curve.append({
                "timestamp": trade.get("closed_at") or trade.get("opened_at"),
                "equity": round(equity, 2),
                "pnl": round(pnl, 2),
                "cumulative_pnl": round(cumulative_pnl, 2),
                "deal_id": trade.get("deal_id"),
            })

        return curve

    def get_status(self) -> dict:
        """Get performance tracker status."""
        return {
            "initial_balance": self.initial_balance,
            "risk_free_rate": self.risk_free_rate,
            "trading_days_per_year": TRADING_DAYS_PER_YEAR,
            "trade_db_connected": self.trade_db is not None,
        }
