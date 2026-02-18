"""Position lifecycle management for daemon."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import BaseModel, Field

from src.daemon.config import PositionManagementConfig
from src.data.broker import AlpacaBroker, BrokerPosition
from src.workflows.types import TradingWorkflowResult

if TYPE_CHECKING:
    from src.database.engine import DatabaseEngine


def _make_task_cleanup_callback(task_set: set[asyncio.Task[Any]]) -> Callable[[asyncio.Task[object]], None]:
    """Create callback that removes task from set and logs exceptions."""

    def _cleanup_and_log(task: asyncio.Task[object]) -> None:
        """Log exceptions and remove task from tracking set."""
        task_set.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.opt(exception=exc).error("Background task failed")

    return _cleanup_and_log


class PositionContext(BaseModel):
    """Position context for trader decisions."""

    entry_price: float = Field(gt=0.0, description="Entry price for the position")
    days_held: int = Field(ge=0, description="Number of days position has been held")
    current_stop_loss: float = Field(gt=0.0, description="Current stop loss price")
    profit_targets: list[float] = Field(default_factory=list, description="Profit target prices")
    trailing_activated: bool = Field(default=False, description="Whether trailing stop is activated")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PositionContext(entry={self.entry_price:.2f}, "
            f"days={self.days_held}, stop={self.current_stop_loss:.2f})"
        )


class MarketEvent(BaseModel):
    """Market event record."""

    timestamp: datetime = Field(description="Event timestamp")
    event_type: str = Field(description="Event type (HALT, NEWS, EARNINGS)")
    symbol: str = Field(description="Stock ticker symbol")
    description: str = Field(description="Event description")

    def __repr__(self) -> str:
        """String representation."""
        return f"MarketEvent(type={self.event_type}, symbol={self.symbol})"


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

    def __init__(
        self,
        broker: AlpacaBroker | None,
        config: PositionManagementConfig,
        database_engine: DatabaseEngine | None = None,
    ) -> None:
        """Initialize position manager.

        Args:
            broker: Alpaca broker for order execution (None during init, set via set_broker())
            config: Position management configuration
            database_engine: Optional database engine for creating per-task sessions
        """
        self.broker = broker
        self.config = config
        self._database_engine = database_engine
        self._pending_tasks: set[asyncio.Task[Any]] = set()  # Track background tasks
        logger.info(f"PositionManager initialized: {config}")

    def set_broker(self, broker: AlpacaBroker) -> None:
        """Set broker after initialization (deferred to avoid event loop issues)."""
        self.broker = broker
        logger.debug("PositionManager broker updated")

    def _ensure_broker(self) -> AlpacaBroker:
        """Ensure broker is initialized, raise if not."""
        if self.broker is None:
            msg = "Broker not initialized - call set_broker() first"
            raise RuntimeError(msg)
        return self.broker

    def set_database(self, database_engine: DatabaseEngine) -> None:
        """Set database engine after initialization.

        Called during lifecycle startup to set database components after event loop is running.

        Args:
            database_engine: Database engine for creating per-task sessions
        """
        self._database_engine = database_engine
        logger.info("PositionManager database components set")

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
        broker = self._ensure_broker()
        broker_info = broker.get_account_info()
        broker_positions = broker_info.positions

        new_positions = self._find_new_positions(state_positions, broker_positions)
        updated_positions = self._find_updated_positions(state_positions, broker_positions)
        closed_symbols = self._find_closed_positions(state_positions, broker_positions)

        return new_positions, updated_positions, closed_symbols

    def _find_new_positions(
        self, state_positions: dict[str, PositionRecord], broker_positions: dict[str, BrokerPosition]
    ) -> list[PositionRecord]:
        """Find new positions not in state."""
        new_positions: list[PositionRecord] = []
        for symbol, broker_pos in broker_positions.items():
            if symbol not in state_positions:
                logger.info(f"New position detected: {symbol}")
                new_pos = self._create_position_from_broker(symbol, broker_pos)
                new_positions.append(new_pos)
                self._persist_position_create(new_pos)
        return new_positions

    def _find_updated_positions(
        self, state_positions: dict[str, PositionRecord], broker_positions: dict[str, BrokerPosition]
    ) -> list[PositionRecord]:
        """Find positions with updated quantities."""
        updated_positions: list[PositionRecord] = []
        for symbol, broker_pos in broker_positions.items():
            if symbol in state_positions:
                existing = state_positions[symbol]
                if existing.current_qty != broker_pos.qty:
                    logger.info(f"Position qty changed: {symbol} {existing.current_qty} → {broker_pos.qty}")
                    existing.current_qty = broker_pos.qty
                    existing.last_updated = datetime.now(UTC)
                    updated_positions.append(existing)
                    self._persist_position_update(existing)
        return updated_positions

    def _find_closed_positions(
        self, state_positions: dict[str, PositionRecord], broker_positions: dict[str, BrokerPosition]
    ) -> list[str]:
        """Find positions closed at broker."""
        closed_symbols: list[str] = []
        for symbol in state_positions:
            if symbol not in broker_positions:
                logger.info(f"Position closed: {symbol}")
                closed_symbols.append(symbol)
                self._persist_position_delete(symbol)
        return closed_symbols

    def _persist_position_create(self, position: PositionRecord) -> None:
        """Persist new position to database."""
        if self._database_engine:
            try:
                task = asyncio.create_task(self._async_persist_position_create(position))
                self._pending_tasks.add(task)
                task.add_done_callback(_make_task_cleanup_callback(self._pending_tasks))
            except Exception as e:
                logger.opt(exception=True).error(f"Failed to persist new position to database: {e}")
                raise

    async def _async_persist_position_create(self, position: PositionRecord) -> None:
        """Async helper to persist position with fresh session."""
        from src.database.repositories.position import PositionRecordRepository

        if self._database_engine is None:
            msg = "Database engine not initialized"
            raise RuntimeError(msg)
        async with self._database_engine.session() as session:
            repository = PositionRecordRepository(session)
            await repository.create(position)

    def _persist_position_update(self, position: PositionRecord) -> None:
        """Persist position update to database."""
        if self._database_engine:
            try:
                task = asyncio.create_task(self._async_persist_position_update(position))
                self._pending_tasks.add(task)
                task.add_done_callback(_make_task_cleanup_callback(self._pending_tasks))
            except Exception as e:
                logger.opt(exception=True).error(f"Failed to update position in database: {e}")
                raise

    async def _async_persist_position_update(self, position: PositionRecord) -> None:
        """Async helper to persist position update with fresh session."""
        from src.database.repositories.position import PositionRecordRepository

        if self._database_engine is None:
            msg = "Database engine not initialized"
            raise RuntimeError(msg)
        async with self._database_engine.session() as session:
            repository = PositionRecordRepository(session)
            await repository.update(position)

    def _persist_position_delete(self, symbol: str) -> None:
        """Delete position from database."""
        if self._database_engine:
            try:
                task = asyncio.create_task(self._async_persist_position_delete(symbol))
                self._pending_tasks.add(task)
                task.add_done_callback(_make_task_cleanup_callback(self._pending_tasks))
            except Exception as e:
                logger.opt(exception=True).error(f"Failed to delete position from database: {e}")
                raise

    async def _async_persist_position_delete(self, symbol: str) -> None:
        """Async helper to delete position with fresh session."""
        from src.database.repositories.position import PositionRecordRepository

        if self._database_engine is None:
            msg = "Database engine not initialized"
            raise RuntimeError(msg)
        async with self._database_engine.session() as session:
            repository = PositionRecordRepository(session)
            await repository.delete_by_symbol(symbol)

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

        # Load entry metadata from trades table if available (#272)
        entry_timestamp, entry_confidence, entry_signal = self._load_entry_metadata(symbol)

        return PositionRecord(
            symbol=symbol,
            entry_timestamp=entry_timestamp,
            entry_price=entry_price,
            entry_signal=entry_signal,
            entry_confidence=entry_confidence,
            current_qty=broker_pos.qty,
            current_stop_loss=initial_stop,
            initial_stop_loss=initial_stop,
            profit_targets=profit_targets,
            last_updated=datetime.now(UTC),
        )

    def _load_entry_metadata(self, symbol: str) -> tuple[datetime, float, str]:
        """Load entry metadata from trades table.

        Returns defaults when called from a running event loop (the normal case),
        because asyncio.run() cannot be used inside an existing loop. Falls back
        to asyncio.run() only when no event loop is active (e.g. sync test context).

        Args:
            symbol: Stock ticker

        Returns:
            Tuple of (entry_timestamp, entry_confidence, entry_signal)
        """
        if not self._database_engine:
            logger.warning(f"No database engine available, using defaults for {symbol}")
            return datetime.now(UTC), 0.75, "BUY"

        try:
            asyncio.get_running_loop()
            # Event loop running — cannot call asyncio.run(); use defaults
            logger.debug(f"Event loop running, using defaults for entry metadata for {symbol}")
            return datetime.now(UTC), 0.75, "BUY"
        except RuntimeError:
            pass

        try:
            return asyncio.run(self._async_load_entry_metadata(symbol))
        except Exception as e:
            logger.opt(exception=True).error(
                f"Failed to load entry metadata for {symbol}: {e}, using defaults"
            )
            return datetime.now(UTC), 0.75, "BUY"

    async def _async_load_entry_metadata(self, symbol: str) -> tuple[datetime, float, str]:
        """Async helper to load entry metadata with a fresh session."""
        from src.database.repositories.trade import TradeRepository

        if self._database_engine is None:
            msg = "Database engine not initialized"
            raise RuntimeError(msg)
        async with self._database_engine.session() as session:
            repo = TradeRepository(session)
            entry_trade = await repo.get_entry_trade(symbol)

        if entry_trade:
            logger.info(
                f"Loaded entry metadata for {symbol}: "
                f"timestamp={entry_trade.timestamp}, "
                f"confidence={entry_trade.confidence:.2f}, "
                f"signal={entry_trade.action.value}"
            )
            return entry_trade.timestamp, entry_trade.confidence, entry_trade.action.value

        logger.warning(f"No entry trade found for {symbol}, using defaults")
        return datetime.now(UTC), 0.75, "BUY"

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
        if action.new_stop_loss is None:
            action.executed = False
            return
        order_id = self._update_stop_loss(position, action.new_stop_loss)
        if order_id:
            action.executed = True
            action.order_id = order_id
        else:
            action.executed = False

    def _execute_sell_action(self, position: PositionRecord, action: PositionManagementAction) -> None:
        """Execute sell order action."""
        if action.qty_sold is None:
            action.executed = False
            return
        try:
            broker = self._ensure_broker()
            order = broker.submit_order(
                symbol=position.symbol,
                qty=int(action.qty_sold),
                side="sell",
            )
            action.executed = True
            action.order_id = order.order_id
            logger.info(f"Executed {action.action_type}: {position.symbol} x{action.qty_sold}")
        except Exception as e:
            logger.opt(exception=True).error(
                f"Failed to execute {action.action_type} for {position.symbol}: {e}"
            )
            action.executed = False

    def _execute_actions(self, position: PositionRecord, actions: list[PositionManagementAction]) -> None:
        """Execute position management actions."""
        for action in actions:
            if action.action_type in ("TRAILING_STOP", "BREAKEVEN"):
                self._execute_stop_loss_action(position, action)
            elif action.action_type in ("PARTIAL_PROFIT", "TIME_EXIT", "CONVICTION_SCALE"):
                self._execute_sell_action(position, action)

            # Persist action to database if database engine available
            if self._database_engine:
                try:
                    task = asyncio.create_task(self._async_persist_action(action))
                    self._pending_tasks.add(task)
                    task.add_done_callback(_make_task_cleanup_callback(self._pending_tasks))
                    logger.debug(
                        f"Persisted position action to database: {action.symbol} {action.action_type}"
                    )
                except Exception as e:
                    logger.opt(exception=True).error(f"Failed to persist position action to database: {e}")
                    raise  # Fail fast per user requirement

    async def _async_persist_action(self, action: PositionManagementAction) -> None:
        """Async helper to persist action with fresh session."""
        from src.database.repositories.position_action import PositionManagementActionRepository

        if self._database_engine is None:
            msg = "Database engine not initialized"
            raise RuntimeError(msg)
        async with self._database_engine.session() as session:
            repository = PositionManagementActionRepository(session)
            await repository.create(action)

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
        broker = self._ensure_broker()

        # Cancel old stop-loss order if exists
        if position.stop_loss_order_id:
            try:
                broker.cancel_order(position.stop_loss_order_id)
                logger.info(f"Cancelled old stop-loss order: {position.stop_loss_order_id}")
            except Exception as e:
                logger.opt(exception=True).error(
                    f"Failed to cancel old stop-loss order {position.stop_loss_order_id}: {e}"
                )
                return None

        # Verify position still exists and get current price
        broker_info = broker.get_account_info()
        if position.symbol not in broker_info.positions:
            logger.warning(f"Position closed during stop update: {position.symbol}")
            return None

        broker_pos = broker_info.positions[position.symbol]
        current_price = broker_pos.market_value / broker_pos.qty if broker_pos.qty > 0 else 0.0

        if current_price <= 0:
            logger.warning(f"Invalid current price for {position.symbol}: {current_price}")
            return None

        # Enforce minimum gap between stop and current price
        min_gap = self.config.min_stop_gap_dollars
        max_allowed_stop = current_price - min_gap

        if new_stop_loss > max_allowed_stop:
            logger.warning(
                f"Stop price ${new_stop_loss:.2f} too close to current ${current_price:.2f} "
                f"(min gap ${min_gap:.2f}). Adjusting to ${max_allowed_stop:.2f}"
            )
            new_stop_loss = max_allowed_stop

        # Verify adjusted stop is still higher than current stop
        if new_stop_loss <= position.current_stop_loss:
            logger.debug(
                f"Adjusted stop ${new_stop_loss:.2f} not higher than current "
                f"${position.current_stop_loss:.2f}, skipping update"
            )
            return None

        # Submit new stop-loss order
        try:
            order = broker.submit_stop_order(
                symbol=position.symbol,
                qty=int(position.current_qty),
                stop_price=new_stop_loss,
            )
            position.stop_loss_order_id = order.order_id
            position.current_stop_loss = new_stop_loss
            position.last_updated = datetime.now(UTC)
            logger.info(f"Updated stop-loss: {position.symbol} → ${new_stop_loss:.2f}")
            return order.order_id
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to submit new stop-loss: {e}")
            return None

    async def wait_for_pending_tasks(self, timeout_seconds: float = 5.0) -> None:
        """Wait for all pending background tasks to complete.

        Args:
            timeout_seconds: Maximum seconds to wait for tasks

        Called during daemon shutdown to ensure database operations complete cleanly.
        """
        if not self._pending_tasks:
            return

        logger.info(f"Waiting for {len(self._pending_tasks)} pending position persistence tasks...")
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._pending_tasks, return_exceptions=True),  # type: ignore[bad-argument-type]
                timeout_seconds,
            )
            logger.info("All pending position persistence tasks completed")
        except TimeoutError:
            logger.opt(exception=True).warning(
                f"Position persistence tasks timed out after {timeout_seconds}s, cancelling..."
            )
            for task in self._pending_tasks:
                if not task.done():
                    task.cancel()
            # Wait briefly for cancellations to propagate
            await asyncio.sleep(0.1)
            self._pending_tasks.clear()

    def __repr__(self) -> str:
        """Return string representation."""
        return f"PositionManager(config={self.config})"
