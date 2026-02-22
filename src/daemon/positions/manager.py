"""Position manager — orchestrates checks, execution, and persistence."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger

from src.daemon.config import PositionManagementConfig
from src.daemon.positions.checks import PositionCheckRunner
from src.daemon.positions.models import PositionManagementAction, PositionRecord
from src.daemon.positions.persistence import PositionPersistenceManager
from src.v1.trades.brokers import Broker, BrokerPosition
from src.workflows.types import TradingWorkflowResult

if TYPE_CHECKING:
    from src.database.engine import DatabaseEngine


class PositionManager:
    """Manage position lifecycle (trailing stops, profit-taking, time exits, conviction scaling)."""

    def __init__(
        self,
        broker: Broker | None,
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
        self._persistence = PositionPersistenceManager(database_engine)
        self._checks = PositionCheckRunner(config)
        logger.info(f"PositionManager initialized: {config}")

    def set_broker(self, broker: Broker) -> None:
        """Set broker after initialization (deferred to avoid event loop issues)."""
        self.broker = broker
        logger.debug("PositionManager broker updated")

    def _ensure_broker(self) -> Broker:
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
        self._persistence.set_database(database_engine)
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
                self._persistence.persist_position_create(new_pos)
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
                    self._persistence.persist_position_update(existing)
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
                self._persistence.persist_position_delete(symbol)
        return closed_symbols

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
            initial_risk_per_share=entry_price - initial_stop,
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
        if not self._persistence.database_engine:
            logger.warning(f"No database engine available, using defaults for {symbol}")
            return datetime.now(UTC), 0.75, "BUY"

        try:
            asyncio.get_running_loop()
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

        db_engine = self._persistence.database_engine
        if db_engine is None:
            msg = "Database engine not initialized"
            raise RuntimeError(msg)
        async with db_engine.session() as session:
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
        targets: list[float] = []
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

        # Priority 0: Circuit breaker — non-overridable, short-circuits all other checks
        cb_action = self._checks.check_circuit_breaker(position, current_price)
        if cb_action:
            actions.append(cb_action)
            self._execute_actions(position, actions)
            return actions

        atr_value: float | None = None
        adx_value: float | None = None
        if latest_analysis and latest_analysis.technical:
            atr_value = latest_analysis.technical.atr_14
            adx_value = latest_analysis.technical.adx

        if latest_analysis:
            self._refresh_conviction(position, latest_analysis)

        # ADX filter: suppress soft exits in choppy (low-trend) markets
        suppress_soft = self._checks.should_suppress_exit(adx_value)

        # Soft exits: conviction decay, profit targets, conviction scaling
        # ADX filter suppresses these in choppy (low-trend) markets
        soft_sell_actions: list[PositionManagementAction] = []

        if not suppress_soft:
            if self.config.conviction_decay_enabled:
                action = self._checks.check_conviction_decay(position, current_price)
                if action:
                    soft_sell_actions.append(action)

            if self.config.use_r_multiple_targets and position.initial_risk_per_share:
                soft_sell_actions.extend(self._checks.check_r_multiple_targets(position, current_price))
            elif self.config.partial_profit_enabled:
                soft_sell_actions.extend(self._checks.check_profit_targets(position, current_price))

            if self.config.conviction_scaling_enabled and latest_analysis:
                action = self._checks.check_conviction_scaling(position, latest_analysis)
                if action:
                    soft_sell_actions.append(action)

        # Sell confirmation: require N consecutive cycles before executing soft sells
        has_soft_sell = len(soft_sell_actions) > 0
        if self._checks.check_sell_confirmation(position, has_soft_sell):
            actions.extend(soft_sell_actions)

        # Hard exits: breakeven, trailing stop, time exit bypass ADX filter + confirmation
        if self.config.breakeven_enabled:
            action = self._checks.check_breakeven_activation(position, current_price)
            if action:
                actions.append(action)

        if self.config.trailing_stop_enabled:
            action = self._checks.check_trailing_stop(position, current_price, atr_value)
            if action:
                actions.append(action)

        if self.config.time_exit_enabled:
            action = self._checks.check_time_exit(position, current_price)
            if action:
                actions.append(action)

        self._execute_actions(position, actions)
        return actions

    def _refresh_conviction(
        self,
        position: PositionRecord,
        latest_analysis: TradingWorkflowResult,
    ) -> None:
        """Refresh conviction from latest analysis and maintain history.

        Args:
            position: Position to update
            latest_analysis: Latest trading analysis
        """
        new_conviction = latest_analysis.decision.confidence
        position.current_conviction = new_conviction
        position.last_analysis_timestamp = datetime.now(UTC)

        position.conviction_history.append(new_conviction)
        max_len = self.config.conviction_history_length
        if len(position.conviction_history) > max_len:
            position.conviction_history = position.conviction_history[-max_len:]

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
            elif action.action_type in (
                "PARTIAL_PROFIT",
                "TIME_EXIT",
                "CONVICTION_SCALE",
                "CONVICTION_DECAY",
                "CIRCUIT_BREAKER",
            ):
                self._execute_sell_action(position, action)

            self._persistence.persist_action(action)

    def _update_stop_loss(self, position: PositionRecord, new_stop_loss: float) -> str | None:
        """Update stop-loss order (cancel old, submit new).

        Args:
            position: Position to update
            new_stop_loss: New stop-loss price

        Returns:
            Order ID if successful, None if failed
        """
        broker = self._ensure_broker()

        if position.stop_loss_order_id:
            try:
                broker.cancel_order(position.stop_loss_order_id)
                logger.info(f"Cancelled old stop-loss order: {position.stop_loss_order_id}")
            except Exception as e:
                if "order pending cancel" in str(e).lower():
                    logger.info(
                        f"Stop-loss order {position.stop_loss_order_id} already pending cancel, proceeding"
                    )
                else:
                    logger.opt(exception=True).error(
                        f"Failed to cancel old stop-loss order {position.stop_loss_order_id}: {e}"
                    )
                    return None

        broker_info = broker.get_account_info()
        if position.symbol not in broker_info.positions:
            logger.warning(f"Position closed during stop update: {position.symbol}")
            return None

        broker_pos = broker_info.positions[position.symbol]
        current_price = broker_pos.market_value / broker_pos.qty if broker_pos.qty > 0 else 0.0

        if current_price <= 0:
            logger.warning(f"Invalid current price for {position.symbol}: {current_price}")
            return None

        min_gap = self.config.min_stop_gap_dollars
        max_allowed_stop = current_price - min_gap

        if new_stop_loss > max_allowed_stop:
            logger.warning(
                f"Stop price ${new_stop_loss:.2f} too close to current ${current_price:.2f} "
                f"(min gap ${min_gap:.2f}). Adjusting to ${max_allowed_stop:.2f}"
            )
            new_stop_loss = max_allowed_stop

        if new_stop_loss <= position.current_stop_loss:
            logger.debug(
                f"Adjusted stop ${new_stop_loss:.2f} not higher than current "
                f"${position.current_stop_loss:.2f}, skipping update"
            )
            return None

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
        """
        await self._persistence.wait_for_pending_tasks(timeout_seconds)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"PositionManager(config={self.config})"
