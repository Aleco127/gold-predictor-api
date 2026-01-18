"""
Backtesting Engine
==================
Simulate trading strategies on historical data.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd

from src.storage.historical_data import HistoricalDataStore
from src.features.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Represents a simulated trade during backtesting."""
    entry_time: datetime
    exit_time: datetime
    direction: str  # "BUY" or "SELL"
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_percent: float
    signal_confidence: float
    exit_reason: str  # "take_profit", "stop_loss", "signal_reversal", "end_of_data"
    bars_held: int
    max_favorable: float  # Maximum favorable excursion
    max_adverse: float  # Maximum adverse excursion

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "direction": self.direction,
            "entry_price": round(self.entry_price, 2),
            "exit_price": round(self.exit_price, 2),
            "size": round(self.size, 4),
            "pnl": round(self.pnl, 2),
            "pnl_percent": round(self.pnl_percent, 4),
            "signal_confidence": round(self.signal_confidence, 4),
            "exit_reason": self.exit_reason,
            "bars_held": self.bars_held,
            "max_favorable": round(self.max_favorable, 2),
            "max_adverse": round(self.max_adverse, 2),
        }


@dataclass
class BacktestMetrics:
    """Performance metrics from backtesting."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    total_pnl: float
    total_pnl_percent: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    max_drawdown: float
    max_drawdown_percent: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    expectancy: float
    avg_bars_held: float
    avg_winning_bars: float
    avg_losing_bars: float
    consecutive_wins: int
    consecutive_losses: int
    recovery_factor: float
    risk_reward_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 2),
            "profit_factor": round(self.profit_factor, 2),
            "total_pnl": round(self.total_pnl, 2),
            "total_pnl_percent": round(self.total_pnl_percent, 2),
            "average_win": round(self.average_win, 2),
            "average_loss": round(self.average_loss, 2),
            "largest_win": round(self.largest_win, 2),
            "largest_loss": round(self.largest_loss, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "max_drawdown_percent": round(self.max_drawdown_percent, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "calmar_ratio": round(self.calmar_ratio, 4),
            "expectancy": round(self.expectancy, 2),
            "avg_bars_held": round(self.avg_bars_held, 2),
            "avg_winning_bars": round(self.avg_winning_bars, 2),
            "avg_losing_bars": round(self.avg_losing_bars, 2),
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
            "recovery_factor": round(self.recovery_factor, 2),
            "risk_reward_ratio": round(self.risk_reward_ratio, 2),
        }


@dataclass
class BacktestResult:
    """Complete backtest result."""
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    initial_balance: float
    final_balance: float
    metrics: BacktestMetrics
    trades: List[BacktestTrade]
    equity_curve: List[Dict[str, Any]]
    parameters: Dict[str, Any]
    run_time_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_balance": round(self.initial_balance, 2),
            "final_balance": round(self.final_balance, 2),
            "total_return_percent": round(
                (self.final_balance - self.initial_balance) / self.initial_balance * 100, 2
            ),
            "metrics": self.metrics.to_dict(),
            "trades_count": len(self.trades),
            "trades": [t.to_dict() for t in self.trades[-100:]],  # Last 100 trades
            "equity_curve": self.equity_curve[-500:],  # Last 500 points
            "parameters": self.parameters,
            "run_time_seconds": round(self.run_time_seconds, 2),
        }


class BacktestEngine:
    """
    Backtesting engine for simulating trading strategies.

    Features:
    - Load historical data from HistoricalDataStore
    - Simulate trades using technical indicator signals
    - Apply configurable stop-loss and take-profit
    - Track equity curve and drawdown
    - Calculate comprehensive performance metrics
    """

    def __init__(
        self,
        historical_store: HistoricalDataStore,
        initial_balance: float = 10000.0,
        position_size_percent: float = 2.0,
        stop_loss_atr_multiplier: float = 2.0,
        take_profit_atr_multiplier: float = 3.0,
        max_positions: int = 1,
        commission_percent: float = 0.0,
        slippage_percent: float = 0.01,
    ):
        """
        Initialize backtesting engine.

        Args:
            historical_store: Historical data store instance
            initial_balance: Starting account balance
            position_size_percent: Position size as % of balance
            stop_loss_atr_multiplier: Stop loss distance as ATR multiple
            take_profit_atr_multiplier: Take profit distance as ATR multiple
            max_positions: Maximum concurrent positions
            commission_percent: Commission per trade as %
            slippage_percent: Slippage per trade as %
        """
        self.historical_store = historical_store
        self.initial_balance = initial_balance
        self.position_size_percent = position_size_percent
        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier
        self.take_profit_atr_multiplier = take_profit_atr_multiplier
        self.max_positions = max_positions
        self.commission_percent = commission_percent
        self.slippage_percent = slippage_percent

        # Technical indicator calculator
        self.indicators = TechnicalIndicators()

        logger.info(
            f"BacktestEngine initialized: balance={initial_balance}, "
            f"position_size={position_size_percent}%, "
            f"SL={stop_loss_atr_multiplier}x ATR, TP={take_profit_atr_multiplier}x ATR"
        )

    def run_backtest(
        self,
        symbol: str = "XAUUSD",
        timeframe: str = "M5",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        signal_generator: Optional[Callable[[pd.DataFrame, int], Tuple[str, float]]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe for data
            start_date: Backtest start date (None = all data)
            end_date: Backtest end date (None = all data)
            signal_generator: Custom signal function (df, idx) -> (signal, confidence)
            progress_callback: Progress callback (current, total, message)

        Returns:
            BacktestResult with full results
        """
        import time
        start_time = time.time()

        logger.info(f"Starting backtest for {symbol} {timeframe}")

        # Load historical data
        df = self.historical_store.load(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )

        if df.empty:
            raise ValueError(f"No data found for {symbol} {timeframe}")

        logger.info(f"Loaded {len(df)} bars for backtesting")

        # Calculate indicators
        df = self.indicators.calculate_all(df)

        # Determine time column
        time_col = 'time' if 'time' in df.columns else 'datetime'

        # Get actual date range
        actual_start = pd.to_datetime(df[time_col].iloc[0])
        actual_end = pd.to_datetime(df[time_col].iloc[-1])

        # Use default signal generator if not provided
        if signal_generator is None:
            signal_generator = self._default_signal_generator

        # Run simulation
        trades, equity_curve = self._simulate_trades(
            df=df,
            time_col=time_col,
            signal_generator=signal_generator,
            progress_callback=progress_callback,
        )

        # Calculate final balance
        final_balance = equity_curve[-1]["balance"] if equity_curve else self.initial_balance

        # Calculate metrics
        metrics = self._calculate_metrics(trades, equity_curve)

        run_time = time.time() - start_time

        result = BacktestResult(
            symbol=symbol,
            timeframe=timeframe,
            start_date=actual_start.to_pydatetime(),
            end_date=actual_end.to_pydatetime(),
            initial_balance=self.initial_balance,
            final_balance=final_balance,
            metrics=metrics,
            trades=trades,
            equity_curve=equity_curve,
            parameters={
                "position_size_percent": self.position_size_percent,
                "stop_loss_atr_multiplier": self.stop_loss_atr_multiplier,
                "take_profit_atr_multiplier": self.take_profit_atr_multiplier,
                "max_positions": self.max_positions,
                "commission_percent": self.commission_percent,
                "slippage_percent": self.slippage_percent,
                "bars_tested": len(df),
            },
            run_time_seconds=run_time,
        )

        logger.info(
            f"Backtest complete: {metrics.total_trades} trades, "
            f"win rate: {metrics.win_rate:.1f}%, "
            f"P&L: ${metrics.total_pnl:.2f} ({metrics.total_pnl_percent:.2f}%)"
        )

        return result

    def _default_signal_generator(
        self,
        df: pd.DataFrame,
        idx: int,
    ) -> Tuple[str, float]:
        """
        Default signal generator using EMA crossover and RSI.

        Args:
            df: DataFrame with indicators
            idx: Current bar index

        Returns:
            Tuple of (signal, confidence) where signal is "BUY", "SELL", or "HOLD"
        """
        if idx < 50:  # Need enough data for indicators
            return ("HOLD", 0.0)

        row = df.iloc[idx]

        # Get indicator values with safe defaults
        ema_9 = row.get('ema_9', 0)
        ema_21 = row.get('ema_21', 0)
        ema_50 = row.get('ema_50', 0)
        rsi = row.get('rsi', 50)
        macd = row.get('macd', 0)
        macd_signal = row.get('macd_signal', 0)
        close = row.get('close', 0)
        bb_upper = row.get('bb_upper', close)
        bb_lower = row.get('bb_lower', close)

        # Check for valid data
        if pd.isna(ema_9) or pd.isna(rsi) or close == 0:
            return ("HOLD", 0.0)

        # Calculate signals
        signal_points = 0.0
        max_points = 5.0

        # EMA alignment (2 points)
        if ema_9 > ema_21 > ema_50:
            signal_points += 2.0  # Bullish
        elif ema_9 < ema_21 < ema_50:
            signal_points -= 2.0  # Bearish

        # RSI condition (1 point)
        if rsi < 30:
            signal_points += 1.0  # Oversold - potential buy
        elif rsi > 70:
            signal_points -= 1.0  # Overbought - potential sell
        elif rsi < 45:
            signal_points += 0.5
        elif rsi > 55:
            signal_points -= 0.5

        # MACD crossover (1 point)
        if macd > macd_signal and macd > 0:
            signal_points += 1.0
        elif macd < macd_signal and macd < 0:
            signal_points -= 1.0

        # Bollinger Band position (1 point)
        if close < bb_lower:
            signal_points += 1.0  # Below lower band - oversold
        elif close > bb_upper:
            signal_points -= 1.0  # Above upper band - overbought

        # Determine signal and confidence
        normalized_score = signal_points / max_points
        confidence = abs(normalized_score)

        if normalized_score > 0.4:
            return ("BUY", confidence)
        elif normalized_score < -0.4:
            return ("SELL", confidence)
        else:
            return ("HOLD", 0.0)

    def _simulate_trades(
        self,
        df: pd.DataFrame,
        time_col: str,
        signal_generator: Callable[[pd.DataFrame, int], Tuple[str, float]],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[List[BacktestTrade], List[Dict[str, Any]]]:
        """
        Simulate trades on historical data.

        Args:
            df: DataFrame with OHLCV and indicators
            time_col: Name of time column
            signal_generator: Function to generate signals
            progress_callback: Progress callback

        Returns:
            Tuple of (trades, equity_curve)
        """
        trades: List[BacktestTrade] = []
        equity_curve: List[Dict[str, Any]] = []

        balance = self.initial_balance
        peak_balance = balance
        current_position: Optional[Dict[str, Any]] = None

        total_bars = len(df)
        report_interval = max(1, total_bars // 100)  # Report every 1%

        for idx in range(total_bars):
            row = df.iloc[idx]
            current_time = pd.to_datetime(row[time_col])
            current_price = float(row['close'])
            atr = float(row.get('atr', current_price * 0.01))

            # Report progress
            if progress_callback and idx % report_interval == 0:
                progress_callback(idx, total_bars, f"Processing bar {idx}/{total_bars}")

            # Check if we have an open position
            if current_position is not None:
                # Update max favorable/adverse excursion
                if current_position['direction'] == 'BUY':
                    current_position['max_price'] = max(
                        current_position['max_price'],
                        float(row['high'])
                    )
                    current_position['min_price'] = min(
                        current_position['min_price'],
                        float(row['low'])
                    )
                else:
                    current_position['max_price'] = max(
                        current_position['max_price'],
                        float(row['high'])
                    )
                    current_position['min_price'] = min(
                        current_position['min_price'],
                        float(row['low'])
                    )

                # Check stop loss
                if current_position['direction'] == 'BUY':
                    if float(row['low']) <= current_position['stop_loss']:
                        # Stop loss hit
                        exit_price = current_position['stop_loss']
                        trade = self._close_position(
                            current_position, current_time, exit_price, "stop_loss", idx
                        )
                        trades.append(trade)
                        balance += trade.pnl
                        current_position = None

                    elif float(row['high']) >= current_position['take_profit']:
                        # Take profit hit
                        exit_price = current_position['take_profit']
                        trade = self._close_position(
                            current_position, current_time, exit_price, "take_profit", idx
                        )
                        trades.append(trade)
                        balance += trade.pnl
                        current_position = None

                else:  # SELL position
                    if float(row['high']) >= current_position['stop_loss']:
                        # Stop loss hit
                        exit_price = current_position['stop_loss']
                        trade = self._close_position(
                            current_position, current_time, exit_price, "stop_loss", idx
                        )
                        trades.append(trade)
                        balance += trade.pnl
                        current_position = None

                    elif float(row['low']) <= current_position['take_profit']:
                        # Take profit hit
                        exit_price = current_position['take_profit']
                        trade = self._close_position(
                            current_position, current_time, exit_price, "take_profit", idx
                        )
                        trades.append(trade)
                        balance += trade.pnl
                        current_position = None

            # Generate signal if no position
            if current_position is None:
                signal, confidence = signal_generator(df, idx)

                if signal in ("BUY", "SELL") and confidence >= 0.4:
                    # Calculate position size
                    position_value = balance * (self.position_size_percent / 100)
                    size = position_value / current_price

                    # Apply slippage
                    if signal == "BUY":
                        entry_price = current_price * (1 + self.slippage_percent / 100)
                        stop_loss = entry_price - (atr * self.stop_loss_atr_multiplier)
                        take_profit = entry_price + (atr * self.take_profit_atr_multiplier)
                    else:
                        entry_price = current_price * (1 - self.slippage_percent / 100)
                        stop_loss = entry_price + (atr * self.stop_loss_atr_multiplier)
                        take_profit = entry_price - (atr * self.take_profit_atr_multiplier)

                    # Open position
                    current_position = {
                        'direction': signal,
                        'entry_time': current_time,
                        'entry_price': entry_price,
                        'size': size,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'confidence': confidence,
                        'entry_idx': idx,
                        'max_price': current_price,
                        'min_price': current_price,
                    }

                    # Deduct commission
                    commission = position_value * (self.commission_percent / 100)
                    balance -= commission

            # Record equity
            unrealized_pnl = 0.0
            if current_position is not None:
                if current_position['direction'] == 'BUY':
                    unrealized_pnl = (current_price - current_position['entry_price']) * current_position['size']
                else:
                    unrealized_pnl = (current_position['entry_price'] - current_price) * current_position['size']

            current_equity = balance + unrealized_pnl
            peak_balance = max(peak_balance, current_equity)
            drawdown = (peak_balance - current_equity) / peak_balance * 100 if peak_balance > 0 else 0

            equity_curve.append({
                "time": current_time.isoformat(),
                "balance": round(balance, 2),
                "equity": round(current_equity, 2),
                "drawdown_percent": round(drawdown, 2),
                "position": current_position['direction'] if current_position else None,
            })

        # Close any remaining position at end of data
        if current_position is not None:
            exit_price = float(df.iloc[-1]['close'])
            trade = self._close_position(
                current_position,
                pd.to_datetime(df.iloc[-1][time_col]),
                exit_price,
                "end_of_data",
                len(df) - 1
            )
            trades.append(trade)
            balance += trade.pnl

        return trades, equity_curve

    def _close_position(
        self,
        position: Dict[str, Any],
        exit_time: datetime,
        exit_price: float,
        exit_reason: str,
        current_idx: int,
    ) -> BacktestTrade:
        """
        Close a position and create a trade record.

        Args:
            position: Position dictionary
            exit_time: Exit timestamp
            exit_price: Exit price
            exit_reason: Reason for exit
            current_idx: Current bar index

        Returns:
            BacktestTrade object
        """
        direction = position['direction']
        entry_price = position['entry_price']
        size = position['size']

        # Calculate P&L
        if direction == 'BUY':
            pnl = (exit_price - entry_price) * size
            max_favorable = (position['max_price'] - entry_price) * size
            max_adverse = (entry_price - position['min_price']) * size
        else:
            pnl = (entry_price - exit_price) * size
            max_favorable = (entry_price - position['min_price']) * size
            max_adverse = (position['max_price'] - entry_price) * size

        # Apply commission on exit
        position_value = exit_price * size
        commission = position_value * (self.commission_percent / 100)
        pnl -= commission

        pnl_percent = pnl / (entry_price * size) * 100

        bars_held = current_idx - position['entry_idx']

        return BacktestTrade(
            entry_time=position['entry_time'],
            exit_time=exit_time,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            size=size,
            pnl=pnl,
            pnl_percent=pnl_percent,
            signal_confidence=position['confidence'],
            exit_reason=exit_reason,
            bars_held=bars_held,
            max_favorable=max_favorable,
            max_adverse=max_adverse,
        )

    def _calculate_metrics(
        self,
        trades: List[BacktestTrade],
        equity_curve: List[Dict[str, Any]],
    ) -> BacktestMetrics:
        """
        Calculate performance metrics from trades.

        Args:
            trades: List of completed trades
            equity_curve: Equity curve data

        Returns:
            BacktestMetrics object
        """
        if not trades:
            return BacktestMetrics(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                profit_factor=0.0,
                total_pnl=0.0,
                total_pnl_percent=0.0,
                average_win=0.0,
                average_loss=0.0,
                largest_win=0.0,
                largest_loss=0.0,
                max_drawdown=0.0,
                max_drawdown_percent=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                expectancy=0.0,
                avg_bars_held=0.0,
                avg_winning_bars=0.0,
                avg_losing_bars=0.0,
                consecutive_wins=0,
                consecutive_losses=0,
                recovery_factor=0.0,
                risk_reward_ratio=0.0,
            )

        # Basic stats
        pnls = [t.pnl for t in trades]
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]

        total_pnl = sum(pnls)
        total_pnl_percent = total_pnl / self.initial_balance * 100

        # Win rate
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0

        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.99

        # Average win/loss
        average_win = float(np.mean([t.pnl for t in winning_trades])) if winning_trades else 0.0
        average_loss = float(np.mean([t.pnl for t in losing_trades])) if losing_trades else 0.0

        # Largest win/loss
        largest_win = max([t.pnl for t in winning_trades]) if winning_trades else 0
        largest_loss = min([t.pnl for t in losing_trades]) if losing_trades else 0

        # Maximum drawdown
        max_dd = 0.0
        max_dd_percent = 0.0
        if equity_curve:
            drawdowns = [e['drawdown_percent'] for e in equity_curve]
            max_dd_percent = max(drawdowns) if drawdowns else 0
            peak = self.initial_balance
            for e in equity_curve:
                peak = max(peak, e['equity'])
                dd = peak - e['equity']
                max_dd = max(max_dd, dd)

        # Sharpe ratio (annualized)
        if len(pnls) > 1:
            returns = np.array(pnls) / self.initial_balance
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            # Annualize assuming 252 trading days
            sharpe_ratio = (mean_return * 252) / (std_return * np.sqrt(252)) if std_return > 0 else 0
        else:
            sharpe_ratio = 0

        # Sortino ratio (only downside deviation)
        if len(pnls) > 1:
            returns = np.array(pnls) / self.initial_balance
            negative_returns = returns[returns < 0]
            downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 0
            mean_return = np.mean(returns)
            sortino_ratio = (mean_return * 252) / (downside_std * np.sqrt(252)) if downside_std > 0 else 0
        else:
            sortino_ratio = 0

        # Calmar ratio (return / max drawdown)
        calmar_ratio = (total_pnl_percent / max_dd_percent) if max_dd_percent > 0 else 0

        # Expectancy
        expectancy = (win_rate / 100 * average_win) + ((1 - win_rate / 100) * average_loss)

        # Average bars held
        avg_bars = float(np.mean([t.bars_held for t in trades])) if trades else 0.0
        avg_winning_bars = float(np.mean([t.bars_held for t in winning_trades])) if winning_trades else 0.0
        avg_losing_bars = float(np.mean([t.bars_held for t in losing_trades])) if losing_trades else 0.0

        # Consecutive wins/losses
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        for t in trades:
            if t.pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        # Recovery factor
        recovery_factor = total_pnl / max_dd if max_dd > 0 else 0

        # Risk reward ratio
        risk_reward = float(abs(average_win / average_loss)) if average_loss != 0 else 0.0

        return BacktestMetrics(
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_percent,
            average_win=average_win,
            average_loss=average_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            max_drawdown=max_dd,
            max_drawdown_percent=max_dd_percent,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            expectancy=expectancy,
            avg_bars_held=avg_bars,
            avg_winning_bars=avg_winning_bars,
            avg_losing_bars=avg_losing_bars,
            consecutive_wins=max_wins,
            consecutive_losses=max_losses,
            recovery_factor=recovery_factor,
            risk_reward_ratio=risk_reward,
        )

    def get_status(self) -> Dict[str, Any]:
        """Get engine status and configuration."""
        return {
            "initial_balance": self.initial_balance,
            "position_size_percent": self.position_size_percent,
            "stop_loss_atr_multiplier": self.stop_loss_atr_multiplier,
            "take_profit_atr_multiplier": self.take_profit_atr_multiplier,
            "max_positions": self.max_positions,
            "commission_percent": self.commission_percent,
            "slippage_percent": self.slippage_percent,
            "historical_store_configured": self.historical_store is not None,
        }
