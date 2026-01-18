"""
Position Manager for tracking and managing open positions.
Handles trailing stops and partial take-profit logic.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from enum import Enum

logger = logging.getLogger(__name__)


class PositionDirection(Enum):
    """Position direction enum."""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class TrackedPosition:
    """Represents a tracked position with trailing stop info."""
    deal_id: str
    direction: PositionDirection
    entry_price: float
    current_stop_loss: float
    original_stop_loss: float
    take_profit: Optional[float]
    size: float
    symbol: str
    opened_at: datetime

    # Trailing stop state
    trailing_activated: bool = False
    highest_price: float = 0.0  # For BUY positions
    lowest_price: float = float('inf')  # For SELL positions
    last_stop_update: Optional[datetime] = None

    # ATR at entry for calculations
    entry_atr: float = 0.0

    # Take profit levels for partial closes
    tp_levels: list[dict] = field(default_factory=list)
    tp_levels_hit: list[int] = field(default_factory=list)


@dataclass
class TrailingStopUpdate:
    """Result of trailing stop calculation."""
    should_update: bool
    new_stop_loss: float
    current_price: float
    profit_pips: float
    reason: str


class PositionManager:
    """
    Manages open positions with trailing stop and partial take-profit logic.

    Features:
    - Track multiple open positions
    - Calculate trailing stop updates based on price movement
    - Support configurable activation threshold and step size
    - Calculate partial take-profit levels
    """

    def __init__(
        self,
        trailing_atr_multiplier: float = 1.5,
        activation_pips: float = 10.0,
        step_pips: float = 5.0,
        pip_value: float = 0.01,  # For gold, 1 pip = 0.01
    ):
        """
        Initialize PositionManager.

        Args:
            trailing_atr_multiplier: Distance for trailing stop as ATR multiplier
            activation_pips: Pips in profit before trailing activates
            step_pips: Minimum pips to move stop by
            pip_value: Value of 1 pip for the symbol
        """
        self.trailing_atr_multiplier = trailing_atr_multiplier
        self.activation_pips = activation_pips
        self.step_pips = step_pips
        self.pip_value = pip_value

        # Track positions by deal_id
        self._positions: dict[str, TrackedPosition] = {}

        logger.info(
            f"PositionManager initialized: "
            f"trailing_atr={trailing_atr_multiplier}x, "
            f"activation={activation_pips} pips, "
            f"step={step_pips} pips"
        )

    def add_position(
        self,
        deal_id: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: Optional[float],
        size: float,
        symbol: str,
        entry_atr: float = 0.0,
        tp_levels: Optional[list[dict]] = None,
    ) -> TrackedPosition:
        """
        Add a new position to track.

        Args:
            deal_id: Unique identifier for the position
            direction: "BUY" or "SELL"
            entry_price: Entry price
            stop_loss: Initial stop loss
            take_profit: Take profit level (optional)
            size: Position size
            symbol: Trading symbol
            entry_atr: ATR at entry time for trailing calculations
            tp_levels: List of partial take-profit levels

        Returns:
            TrackedPosition instance
        """
        pos_direction = PositionDirection(direction.upper())

        position = TrackedPosition(
            deal_id=deal_id,
            direction=pos_direction,
            entry_price=entry_price,
            current_stop_loss=stop_loss,
            original_stop_loss=stop_loss,
            take_profit=take_profit,
            size=size,
            symbol=symbol,
            opened_at=datetime.now(timezone.utc),
            entry_atr=entry_atr,
            tp_levels=tp_levels or [],
        )

        # Initialize price tracking
        if pos_direction == PositionDirection.BUY:
            position.highest_price = entry_price
        else:
            position.lowest_price = entry_price

        self._positions[deal_id] = position

        logger.info(
            f"Position added: {deal_id} {direction} {size} {symbol} @ {entry_price}, "
            f"SL={stop_loss}, TP={take_profit}, ATR={entry_atr}"
        )

        return position

    def remove_position(self, deal_id: str) -> Optional[TrackedPosition]:
        """
        Remove a position from tracking.

        Args:
            deal_id: Position identifier

        Returns:
            Removed position or None if not found
        """
        position = self._positions.pop(deal_id, None)
        if position:
            logger.info(f"Position removed: {deal_id}")
        return position

    def get_position(self, deal_id: str) -> Optional[TrackedPosition]:
        """Get a tracked position by deal_id."""
        return self._positions.get(deal_id)

    def get_all_positions(self) -> list[TrackedPosition]:
        """Get all tracked positions."""
        return list(self._positions.values())

    def calculate_trailing_stop(
        self,
        deal_id: str,
        current_price: float,
        current_atr: Optional[float] = None,
    ) -> TrailingStopUpdate:
        """
        Calculate if trailing stop should be updated for a position.

        Args:
            deal_id: Position identifier
            current_price: Current market price
            current_atr: Current ATR value (uses entry_atr if not provided)

        Returns:
            TrailingStopUpdate with recommendation
        """
        position = self._positions.get(deal_id)
        if not position:
            return TrailingStopUpdate(
                should_update=False,
                new_stop_loss=0,
                current_price=current_price,
                profit_pips=0,
                reason="Position not found"
            )

        # Use entry ATR if current not provided
        atr = current_atr or position.entry_atr
        if atr <= 0:
            return TrailingStopUpdate(
                should_update=False,
                new_stop_loss=position.current_stop_loss,
                current_price=current_price,
                profit_pips=0,
                reason="No ATR value available"
            )

        # Calculate profit in pips
        if position.direction == PositionDirection.BUY:
            profit_pips = (current_price - position.entry_price) / self.pip_value
            # Update highest price
            if current_price > position.highest_price:
                position.highest_price = current_price
        else:  # SELL
            profit_pips = (position.entry_price - current_price) / self.pip_value
            # Update lowest price
            if current_price < position.lowest_price:
                position.lowest_price = current_price

        # Check if trailing should activate
        if not position.trailing_activated:
            if profit_pips >= self.activation_pips:
                position.trailing_activated = True
                logger.info(
                    f"Trailing stop activated for {deal_id}: "
                    f"profit={profit_pips:.1f} pips >= {self.activation_pips} pips"
                )
            else:
                return TrailingStopUpdate(
                    should_update=False,
                    new_stop_loss=position.current_stop_loss,
                    current_price=current_price,
                    profit_pips=profit_pips,
                    reason=f"Waiting for activation: {profit_pips:.1f}/{self.activation_pips} pips"
                )

        # Calculate trailing stop distance
        trailing_distance = atr * self.trailing_atr_multiplier

        # Calculate new stop loss based on direction
        if position.direction == PositionDirection.BUY:
            # For BUY: trail below highest price
            new_stop = position.highest_price - trailing_distance
            # Only move stop up, never down
            if new_stop <= position.current_stop_loss:
                return TrailingStopUpdate(
                    should_update=False,
                    new_stop_loss=position.current_stop_loss,
                    current_price=current_price,
                    profit_pips=profit_pips,
                    reason="New stop would be lower than current"
                )
        else:  # SELL
            # For SELL: trail above lowest price
            new_stop = position.lowest_price + trailing_distance
            # Only move stop down, never up
            if new_stop >= position.current_stop_loss:
                return TrailingStopUpdate(
                    should_update=False,
                    new_stop_loss=position.current_stop_loss,
                    current_price=current_price,
                    profit_pips=profit_pips,
                    reason="New stop would be higher than current"
                )

        # Check minimum step size
        stop_movement_pips = abs(new_stop - position.current_stop_loss) / self.pip_value
        if stop_movement_pips < self.step_pips:
            return TrailingStopUpdate(
                should_update=False,
                new_stop_loss=position.current_stop_loss,
                current_price=current_price,
                profit_pips=profit_pips,
                reason=f"Movement {stop_movement_pips:.1f} pips < step {self.step_pips} pips"
            )

        # Update should happen
        return TrailingStopUpdate(
            should_update=True,
            new_stop_loss=round(new_stop, 2),
            current_price=current_price,
            profit_pips=profit_pips,
            reason=f"Moving stop from {position.current_stop_loss} to {new_stop:.2f}"
        )

    def apply_stop_update(self, deal_id: str, new_stop_loss: float) -> bool:
        """
        Apply a stop loss update to a tracked position.

        Args:
            deal_id: Position identifier
            new_stop_loss: New stop loss value

        Returns:
            True if update was applied
        """
        position = self._positions.get(deal_id)
        if not position:
            return False

        old_stop = position.current_stop_loss
        position.current_stop_loss = new_stop_loss
        position.last_stop_update = datetime.now(timezone.utc)

        logger.info(f"Stop loss updated for {deal_id}: {old_stop} -> {new_stop_loss}")
        return True

    def calculate_tp_levels(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        levels: Optional[list[tuple[float, float]]] = None,
    ) -> list[dict]:
        """
        Calculate partial take-profit levels.

        Args:
            entry_price: Entry price
            direction: "BUY" or "SELL"
            atr: ATR value for calculations
            levels: List of (atr_multiplier, close_percentage) tuples
                    Default: [(1.0, 0.5), (2.0, 0.3), (3.0, 0.2)]

        Returns:
            List of TP level dictionaries
        """
        if levels is None:
            levels = [
                (1.0, 0.5),  # TP1: 1x ATR, close 50%
                (2.0, 0.3),  # TP2: 2x ATR, close 30%
                (3.0, 0.2),  # TP3: 3x ATR, close 20%
            ]

        tp_levels = []
        is_buy = direction.upper() == "BUY"

        for i, (atr_mult, close_pct) in enumerate(levels, 1):
            if is_buy:
                price = entry_price + (atr * atr_mult)
            else:
                price = entry_price - (atr * atr_mult)

            tp_levels.append({
                "level": i,
                "price": round(price, 2),
                "atr_multiplier": atr_mult,
                "close_percentage": close_pct,
                "hit": False,
            })

        return tp_levels

    def check_tp_levels(
        self,
        deal_id: str,
        current_price: float,
    ) -> list[dict]:
        """
        Check if any TP levels have been hit for a position.

        Args:
            deal_id: Position identifier
            current_price: Current market price

        Returns:
            List of TP levels that were just hit
        """
        position = self._positions.get(deal_id)
        if not position or not position.tp_levels:
            return []

        newly_hit = []
        is_buy = position.direction == PositionDirection.BUY

        for tp in position.tp_levels:
            if tp["hit"]:
                continue

            level_hit = False
            if is_buy and current_price >= tp["price"]:
                level_hit = True
            elif not is_buy and current_price <= tp["price"]:
                level_hit = True

            if level_hit:
                tp["hit"] = True
                position.tp_levels_hit.append(tp["level"])
                newly_hit.append(tp)
                logger.info(
                    f"TP{tp['level']} hit for {deal_id} @ {current_price}, "
                    f"close {tp['close_percentage']*100}%"
                )

        return newly_hit

    def get_position_status(self, deal_id: str, current_price: float) -> dict:
        """
        Get comprehensive status of a tracked position.

        Args:
            deal_id: Position identifier
            current_price: Current market price

        Returns:
            Status dictionary with all position info
        """
        position = self._positions.get(deal_id)
        if not position:
            return {"error": "Position not found"}

        # Calculate P&L
        if position.direction == PositionDirection.BUY:
            pnl_pips = (current_price - position.entry_price) / self.pip_value
        else:
            pnl_pips = (position.entry_price - current_price) / self.pip_value

        return {
            "deal_id": position.deal_id,
            "direction": position.direction.value,
            "symbol": position.symbol,
            "size": position.size,
            "entry_price": position.entry_price,
            "current_price": current_price,
            "current_stop_loss": position.current_stop_loss,
            "original_stop_loss": position.original_stop_loss,
            "take_profit": position.take_profit,
            "pnl_pips": round(pnl_pips, 1),
            "trailing_activated": position.trailing_activated,
            "highest_price": position.highest_price if position.direction == PositionDirection.BUY else None,
            "lowest_price": position.lowest_price if position.direction == PositionDirection.SELL else None,
            "tp_levels": position.tp_levels,
            "tp_levels_hit": position.tp_levels_hit,
            "opened_at": position.opened_at.isoformat(),
            "last_stop_update": position.last_stop_update.isoformat() if position.last_stop_update else None,
        }

    def get_config(self) -> dict:
        """Get current configuration."""
        return {
            "trailing_atr_multiplier": self.trailing_atr_multiplier,
            "activation_pips": self.activation_pips,
            "step_pips": self.step_pips,
            "pip_value": self.pip_value,
            "tracked_positions_count": len(self._positions),
        }
