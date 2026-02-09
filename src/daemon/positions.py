"""Position lifecycle management for daemon."""

from datetime import UTC, datetime

from loguru import logger
from pydantic import BaseModel

from src.daemon.config import PositionManagementConfig
from src.data.broker import AlpacaBroker, BrokerPosition
from src.workflows.types import TradingWorkflowResult


class PositionRecord(BaseModel):
    """Active position requiring management."""

    symbol: str
    entry_timestamp: datetime
    entry_price: float
    entry_signal: str
    entry_confidence: float
    current_qty: float
    current_stop_loss: float
    initial_stop_loss: float
    stop_loss_order_id: str | None = None
    profit_targets: list[float]
    days_held: int = 0
    last_updated: datetime
    trailing_stop_activated: bool = False
    breakeven_activated: bool = False
    high_water_mark: float | None = None


class PositionManagementAction(BaseModel):
    """Action taken by position manager."""

    symbol: str
    action_type: str
    timestamp: datetime
    old_stop_loss: float | None = None
    new_stop_loss: float | None = None
    qty_sold: float | None = None
    price: float
    reason: str
    executed: bool
    order_id: str | None = None


class PositionManager:
    """Manage position lifecycle (trailing stops, profit-taking, time exits, conviction scaling)."""

    def __init__(self, broker: AlpacaBroker, config: PositionManagementConfig) -> None:
        """Initialize position manager.

        Args:
            broker: Alpaca broker for order execution
            config: Position management configuration
        """
        self.broker = broker
        self.config = config
        logger.info(f"PositionManager initialized: {config}")

    def sync_with_broker(
        self,
        state_positions: dict[str, PositionRecord],
    ) -> tuple[list[PositionRecord], list[PositionRecord], list[str]]:
        """Sync state positions with broker positions.

        Args:
            state_positions: Current positions in daemon state

        Returns:
            Tuple of (new_positions, updated_positions, closed_symbols)
        """
        logger.info("Syncing positions with broker")
        broker_info = self.broker.get_account_info()
        broker_positions = broker_info.positions

        new_positions: list[PositionRecord] = []
        updated_positions: list[PositionRecord] = []
        closed_symbols: list[str] = []

        # Find new positions
        for symbol, broker_pos in broker_positions.items():
            if symbol not in state_positions:
                logger.info(f"New position detected: {symbol}")
                new_pos = self._create_position_from_broker(symbol, broker_pos)
                new_positions.append(new_pos)
            else:
                # Update quantity if changed
                existing = state_positions[symbol]
                if existing.current_qty != broker_pos.qty:
                    logger.info(f"Position qty changed: {symbol} {existing.current_qty} → {broker_pos.qty}")
                    existing.current_qty = broker_pos.qty
                    existing.last_updated = datetime.now(UTC)
                    updated_positions.append(existing)

        # Find closed positions
        for symbol in state_positions:
            if symbol not in broker_positions:
                logger.info(f"Position closed: {symbol}")
                closed_symbols.append(symbol)

        return new_positions, updated_positions, closed_symbols

    def _create_position_from_broker(self, symbol: str, broker_pos: BrokerPosition) -> PositionRecord:
        """Create PositionRecord from broker position.

        Args:
            symbol: Stock ticker
            broker_pos: BrokerPosition from broker API

        Returns:
            New PositionRecord
        """
        entry_price = float(broker_pos.avg_entry_price)
        profit_targets = self._calculate_profit_targets(entry_price)
        initial_stop = self._calculate_initial_stop_loss(entry_price)

        # Using defaults: timestamp=now(), confidence=0.75 (#272)
        return PositionRecord(
            symbol=symbol,
            entry_timestamp=datetime.now(UTC),
            entry_price=entry_price,
            entry_signal="BUY",
            entry_confidence=0.75,
            current_qty=broker_pos.qty,
            current_stop_loss=initial_stop,
            initial_stop_loss=initial_stop,
            profit_targets=profit_targets,
            last_updated=datetime.now(UTC),
        )

    def _calculate_profit_targets(self, entry_price: float) -> list[float]:
        """Calculate profit target prices.

        Args:
            entry_price: Position entry price

        Returns:
            List of target prices
        """
        targets = []
        if self.config.partial_profit_enabled:
            targets.append(entry_price * (1 + self.config.profit_target_1_percent / 100))
            targets.append(entry_price * (1 + self.config.profit_target_2_percent / 100))
        return targets

    def _calculate_initial_stop_loss(self, entry_price: float) -> float:
        """Calculate initial stop-loss price.

        Args:
            entry_price: Position entry price

        Returns:
            Stop-loss price
        """
        return entry_price * (1 - self.config.trailing_stop_percent / 100)

    def _execute_stop_loss_action(self, position: PositionRecord, action: PositionManagementAction) -> None:
        """Execute stop-loss update action."""
        order_id = self._update_stop_loss(position, action.new_stop_loss)
        if order_id:
            action.executed = True
            action.order_id = order_id
        else:
            action.executed = False

    def _execute_sell_action(self, position: PositionRecord, action: PositionManagementAction) -> None:
        """Execute sell order action."""
        try:
            order = self.broker.submit_order(
                symbol=position.symbol,
                qty=int(action.qty_sold),
                side="sell",
            )
            action.executed = True
            action.order_id = order.order_id
            logger.info(f"Executed {action.action_type}: {position.symbol} x{action.qty_sold}")
        except Exception as e:
            logger.error(f"Failed to execute {action.action_type} for {position.symbol}: {e}")
            action.executed = False

    def _execute_actions(self, position: PositionRecord, actions: list[PositionManagementAction]) -> None:
        """Execute position management actions."""
        for action in actions:
            if action.action_type in ("TRAILING_STOP", "BREAKEVEN"):
                self._execute_stop_loss_action(position, action)
            elif action.action_type in ("PARTIAL_PROFIT", "TIME_EXIT", "CONVICTION_SCALE"):
                self._execute_sell_action(position, action)

    def review_position(
        self,
        position: PositionRecord,
        current_price: float,
        latest_analysis: TradingWorkflowResult | None = None,
    ) -> list[PositionManagementAction]:
        """Review position and generate management actions.

        Args:
            position: Position to review
            current_price: Current market price
            latest_analysis: Latest trading analysis (for conviction scaling)

        Returns:
            List of actions to execute
        """
        actions: list[PositionManagementAction] = []

        # Priority 1: Partial profit-taking
        if self.config.partial_profit_enabled:
            actions.extend(self._check_profit_targets(position, current_price))

        # Priority 2: Break-even stop activation
        if self.config.breakeven_enabled:
            action = self._check_breakeven_activation(position, current_price)
            if action:
                actions.append(action)

        # Priority 3: Trailing stop update
        if self.config.trailing_stop_enabled:
            action = self._check_trailing_stop(position, current_price)
            if action:
                actions.append(action)

        # Priority 4: Time-based exit
        if self.config.time_exit_enabled:
            action = self._check_time_exit(position, current_price)
            if action:
                actions.append(action)

        # Priority 5: Conviction-based scaling
        if self.config.conviction_scaling_enabled and latest_analysis:
            action = self._check_conviction_scaling(position, latest_analysis)
            if action:
                actions.append(action)

        self._execute_actions(position, actions)
        return actions

    def _check_profit_targets(
        self,
        position: PositionRecord,
        current_price: float,
    ) -> list[PositionManagementAction]:
        """Check if profit targets are hit.

        Args:
            position: Position to check
            current_price: Current market price

        Returns:
            List of profit-taking actions
        """
        actions: list[PositionManagementAction] = []

        if not position.profit_targets:
            return actions

        targets_hit = [t for t in position.profit_targets if current_price >= t]
        if not targets_hit:
            return actions

        remaining_qty = position.current_qty

        for target in targets_hit:
            # Determine sell percentage based on target index
            target_idx = position.profit_targets.index(target)
            sell_pct = (
                self.config.profit_target_1_sell_pct
                if target_idx == 0
                else self.config.profit_target_2_sell_pct
            )

            qty_to_sell = remaining_qty * sell_pct
            if qty_to_sell < 1:
                continue

            gain_pct = ((current_price - position.entry_price) / position.entry_price) * 100

            action = PositionManagementAction(
                symbol=position.symbol,
                action_type="PARTIAL_PROFIT",
                timestamp=datetime.now(UTC),
                qty_sold=qty_to_sell,
                price=current_price,
                reason=f"Hit profit target {target:.2f} (+{gain_pct:.1f}%)",
                executed=False,
            )
            actions.append(action)
            remaining_qty -= qty_to_sell

            # Remove hit target
            position.profit_targets.remove(target)

        return actions

    def _check_breakeven_activation(
        self,
        position: PositionRecord,
        current_price: float,
    ) -> PositionManagementAction | None:
        """Check if break-even stop should be activated.

        Args:
            position: Position to check
            current_price: Current market price

        Returns:
            Break-even action or None
        """
        if position.breakeven_activated:
            return None

        gain_pct = ((current_price - position.entry_price) / position.entry_price) * 100
        if gain_pct < self.config.breakeven_activation_percent:
            return None

        # Move stop to entry price
        new_stop = position.entry_price
        if new_stop <= position.current_stop_loss:
            return None

        action = PositionManagementAction(
            symbol=position.symbol,
            action_type="BREAKEVEN",
            timestamp=datetime.now(UTC),
            old_stop_loss=position.current_stop_loss,
            new_stop_loss=new_stop,
            price=current_price,
            reason=f"Break-even activated at +{gain_pct:.1f}%",
            executed=False,
        )

        position.breakeven_activated = True
        return action

    def _check_trailing_stop(
        self,
        position: PositionRecord,
        current_price: float,
    ) -> PositionManagementAction | None:
        """Check if trailing stop should be updated.

        Args:
            position: Position to check
            current_price: Current market price

        Returns:
            Trailing stop action or None
        """
        # Update high water mark
        if position.high_water_mark is None or current_price > position.high_water_mark:
            position.high_water_mark = current_price

        # Calculate new stop from high water mark
        new_stop = position.high_water_mark * (1 - self.config.trailing_stop_percent / 100)

        # Only update if new stop is higher
        if new_stop <= position.current_stop_loss:
            return None

        action = PositionManagementAction(
            symbol=position.symbol,
            action_type="TRAILING_STOP",
            timestamp=datetime.now(UTC),
            old_stop_loss=position.current_stop_loss,
            new_stop_loss=new_stop,
            price=current_price,
            reason=f"Trailing stop update (HWM=${position.high_water_mark:.2f})",
            executed=False,
        )

        position.trailing_stop_activated = True
        return action

    def _check_time_exit(
        self,
        position: PositionRecord,
        current_price: float,
    ) -> PositionManagementAction | None:
        """Check if time-based exit should trigger.

        Args:
            position: Position to check
            current_price: Current market price

        Returns:
            Time exit action or None
        """
        days_held = (datetime.now(UTC) - position.entry_timestamp).days
        position.days_held = days_held

        if days_held < self.config.max_holding_days:
            return None

        return PositionManagementAction(
            symbol=position.symbol,
            action_type="TIME_EXIT",
            timestamp=datetime.now(UTC),
            qty_sold=position.current_qty,
            price=current_price,
            reason=f"Held for {days_held} days (max={self.config.max_holding_days})",
            executed=False,
        )

    def _check_conviction_scaling(
        self,
        position: PositionRecord,
        latest_analysis: TradingWorkflowResult,
    ) -> PositionManagementAction | None:
        """Check if conviction-based scaling should trigger.

        Args:
            position: Position to check
            latest_analysis: Latest trading analysis

        Returns:
            Conviction scaling action or None
        """
        if latest_analysis.decision.action.value != "REDUCE":
            return None

        confidence_drop = position.entry_confidence - latest_analysis.decision.confidence
        if confidence_drop < self.config.conviction_decrease_threshold:
            return None

        qty_to_sell = position.current_qty * self.config.conviction_scale_out_percent
        if qty_to_sell < 1:
            return None

        entry_conf = position.entry_confidence
        current_conf = latest_analysis.decision.confidence
        reason = f"Conviction dropped {confidence_drop:.2f} (entry={entry_conf:.2f}, now={current_conf:.2f})"

        return PositionManagementAction(
            symbol=position.symbol,
            action_type="CONVICTION_SCALE",
            timestamp=datetime.now(UTC),
            qty_sold=qty_to_sell,
            price=latest_analysis.risk.current_price,
            reason=reason,
            executed=False,
        )

    def _update_stop_loss(self, position: PositionRecord, new_stop_loss: float) -> str | None:
        """Update stop-loss order (cancel old, submit new).

        Args:
            position: Position to update
            new_stop_loss: New stop-loss price

        Returns:
            Order ID if successful, None if failed
        """
        # Cancel old stop-loss order if exists
        if position.stop_loss_order_id:
            try:
                self.broker.cancel_order(position.stop_loss_order_id)
                logger.info(f"Cancelled old stop-loss order: {position.stop_loss_order_id}")
            except Exception as e:
                logger.warning(f"Failed to cancel old stop-loss order: {e}")

        # Verify position still exists
        broker_info = self.broker.get_account_info()
        if position.symbol not in broker_info.positions:
            logger.warning(f"Position closed during stop update: {position.symbol}")
            return None

        # Submit new stop-loss order
        try:
            order = self.broker.submit_order(
                symbol=position.symbol,
                qty=int(position.current_qty),
                side="sell",
                stop_loss_price=new_stop_loss,
            )
            position.stop_loss_order_id = order.order_id
            position.current_stop_loss = new_stop_loss
            position.last_updated = datetime.now(UTC)
            logger.info(f"Updated stop-loss: {position.symbol} → ${new_stop_loss:.2f}")
            return order.order_id
        except Exception as e:
            logger.error(f"Failed to submit new stop-loss: {e}")
            return None

    def __repr__(self) -> str:
        """Return string representation."""
        return f"PositionManager(config={self.config})"
