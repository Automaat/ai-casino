"""Execute trade tool for coordinator."""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final

from loguru import logger

from src.daemon.config.base import TradingMode
from src.strategies.signal import Signal
from src.tools.base import BaseTool
from src.tools.models import ToolDefinition, ToolFunction, ToolParameter, ToolParametersSchema

if TYPE_CHECKING:
    from src.daemon.config import DaemonConfig
    from src.daemon.threshold_adapter import AdaptiveThresholdManager
    from src.data.broker import AlpacaBroker, OrderStatus
    from src.database.engine import DatabaseEngine
    from src.v1.coordinator.confirmation import TradeConfirmationHandler
    from src.v1.notifications.service import NotificationService

MIN_RATIONALE_LENGTH: Final[int] = 10
DEFAULT_STOP_LOSS_PCT: Final[float] = 0.05
CONFIDENCE_LOW_RISK: Final[float] = 0.75
CONFIDENCE_MEDIUM_RISK: Final[float] = 0.5


@dataclass
class ExecuteTradeServices:
    """Optional services for ExecuteTradeTool."""

    confirmation_handler: TradeConfirmationHandler | None = None
    adaptive_threshold_manager: AdaptiveThresholdManager | None = None
    database_engine: DatabaseEngine | None = None
    notification_service: NotificationService | None = None


class ExecuteTradeTool(BaseTool):
    """Tool to execute trades via broker."""

    def __init__(
        self,
        broker: AlpacaBroker,
        daemon_config: DaemonConfig,
        services: ExecuteTradeServices | None = None,
    ) -> None:
        """Initialize tool with broker and config.

        Args:
            broker: Alpaca broker instance
            daemon_config: Daemon configuration for trading mode
            services: Optional services (confirmation, thresholds, DB, notifications)
        """
        self._broker = broker
        self._daemon_config = daemon_config
        svc = services or ExecuteTradeServices()
        self._confirmation_handler = svc.confirmation_handler
        self._adaptive_threshold_manager = svc.adaptive_threshold_manager
        self._database_engine = svc.database_engine
        self._notification_service = svc.notification_service

    @property
    def name(self) -> str:
        """Tool name."""
        return "execute_trade"

    @property
    def requires_confirmation(self) -> bool:
        """Requires confirmation in LIVE mode, auto-execute in PAPER mode.

        Returns:
            True if trading mode is LIVE, False if PAPER
        """
        return self._daemon_config.trading_mode == TradingMode.LIVE

    def get_tool_definition(self) -> ToolDefinition:
        """Get tool definition in LiteLLM/OpenAI format.

        Returns:
            Tool definition for LLM function calling
        """
        return ToolDefinition(
            function=ToolFunction(
                name=self.name,
                description=(
                    "Execute a market order (BUY or SELL) with optional stop loss. "
                    "Requires confidence (0.0-1.0) for threshold validation and rationale for "
                    "trade decision. In LIVE mode, requires user confirmation. In PAPER mode, "
                    "executes automatically."
                ),
                parameters=ToolParametersSchema(
                    properties={
                        "symbol": ToolParameter(
                            type="string",
                            description="Stock ticker symbol (e.g., AAPL, TSLA)",
                        ),
                        "action": ToolParameter(
                            type="string",
                            description="Order action: BUY or SELL",
                        ),
                        "quantity": ToolParameter(
                            type="integer",
                            description="Number of shares to trade (minimum: 1)",
                        ),
                        "confidence": ToolParameter(
                            type="number",
                            description="Decision confidence (0.0-1.0) for threshold validation",
                        ),
                        "stop_loss_price": ToolParameter(
                            type="number",
                            description="Optional stop loss price",
                        ),
                        "rationale": ToolParameter(
                            type="string",
                            description="Trading rationale (minimum 10 characters)",
                        ),
                    },
                    required=["symbol", "action", "quantity", "confidence", "rationale"],
                ),
            ),
        )

    def execute(self, **kwargs: str | int | float | bool) -> str:
        """Execute trade order (sync version).

        Args:
            **kwargs: Tool arguments (symbol: str, action: str, quantity: int,
                     confidence: float, stop_loss_price: float | None, rationale: str)

        Returns:
            Order status with details
        """
        symbol = str(kwargs["symbol"]).upper()
        action = str(kwargs["action"]).upper()
        quantity = int(kwargs["quantity"])
        stop_loss_price = kwargs.get("stop_loss_price")
        rationale = str(kwargs["rationale"])

        # Validate inputs
        if quantity <= 0:
            return "Error: Quantity must be positive"

        if action not in ["BUY", "SELL"]:
            return f"Error: Invalid action '{action}'. Must be BUY or SELL"

        if len(rationale) < MIN_RATIONALE_LENGTH:
            return f"Error: Rationale must be at least {MIN_RATIONALE_LENGTH} characters"

        # Convert stop loss to float if provided
        stop_loss_float = float(stop_loss_price) if stop_loss_price is not None else None

        logger.info(f"Executing {action} order: {quantity} {symbol} (stop_loss={stop_loss_float})")

        try:
            # Submit order
            order_status = self._broker.submit_order(
                symbol=symbol,
                qty=quantity,
                side=action.lower(),
                stop_loss_price=stop_loss_float,
            )

            # Format output
            lines = [
                "# Trade Executed",
                "",
                f"**Order ID:** {order_status.order_id}",
                f"**Symbol:** {order_status.symbol}",
                f"**Action:** {order_status.side.upper()}",
                f"**Quantity:** {order_status.qty}",
                f"**Status:** {order_status.status}",
                f"**Submitted:** {order_status.submitted_at.strftime('%Y-%m-%d %H:%M:%S')}",
            ]

            if stop_loss_float is not None:
                lines.append(f"**Stop Loss:** ${stop_loss_float:.2f}")

            lines.extend(
                [
                    "",
                    "## Rationale",
                    rationale,
                ]
            )

            return "\n".join(lines)

        except Exception as e:
            logger.opt(exception=True).error(f"Trade execution failed: {e}")
            return f"Failed to execute trade: {e}"

    async def aexecute(self, **kwargs: str | int | float | bool) -> str:
        """Execute trade order with async confirmation support.

        Args:
            **kwargs: Tool arguments (symbol: str, action: str, quantity: int,
                     confidence: float, stop_loss_price: float | None, rationale: str)

        Returns:
            Order status with details or confirmation status
        """
        symbol = str(kwargs["symbol"]).upper()
        action = str(kwargs["action"]).upper()
        quantity = int(kwargs["quantity"])
        confidence = float(kwargs["confidence"])
        stop_loss_price = kwargs.get("stop_loss_price")
        rationale = str(kwargs["rationale"])

        # Validate inputs (consolidated)
        if validation_error := self._validate_inputs(quantity, action, rationale):
            return validation_error

        # Validate confidence against adaptive threshold
        if threshold_error := await self._check_adaptive_threshold(action, confidence):
            return threshold_error

        # Guard against duplicate BUY when open position already exists in DB
        if action == "BUY" and (duplicate_error := await self._check_duplicate_position(symbol)):
            return duplicate_error

        # Convert stop loss to float if provided
        stop_loss_float = float(stop_loss_price) if stop_loss_price is not None else None

        # Check if confirmation required
        if confirmation_error := await self._handle_confirmation(
            symbol, action, quantity, stop_loss_float, rationale
        ):
            return confirmation_error

        # Execute trade
        logger.info(f"Executing {action} order: {quantity} {symbol} (stop_loss={stop_loss_float})")
        result, order_status = await self._submit_order(symbol, action, quantity, stop_loss_float, rationale)

        # Post-trade side effects (persist + notify)
        if order_status:
            await self._persist_trade(order_status, confidence, stop_loss_float)
            await self._notify_trade(order_status, confidence, rationale)

        return result

    async def _check_duplicate_position(self, symbol: str) -> str | None:
        """Reject BUY if an open position already exists in DB.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Error message if duplicate, None if safe to proceed
        """
        if not self._database_engine:
            return None

        try:
            from src.database.repositories.trade import TradeRepository

            async with self._database_engine.session() as session:
                repo = TradeRepository(session)
                existing = await repo.get_entry_trade(symbol)

            if existing:
                logger.info(f"Blocked duplicate BUY {symbol}: open position since {existing.timestamp}")
                return (
                    f"Skipped: already hold open BUY position in {symbol} "
                    f"({existing.shares} shares since {existing.timestamp.strftime('%Y-%m-%d %H:%M')}). "
                    f"SELL first before buying again."
                )
        except Exception as e:
            logger.opt(exception=True).warning(f"Duplicate position check failed for {symbol}: {e}")
            return (
                f"Skipped: could not verify existing position for {symbol} due to an internal error. "
                f"Trade not executed to avoid potential duplicate BUY."
            )

        return None

    def _validate_inputs(self, quantity: int, action: str, rationale: str) -> str | None:
        """Validate trade inputs.

        Args:
            quantity: Number of shares
            action: Order action (BUY/SELL)
            rationale: Trading rationale

        Returns:
            Error message if validation fails, None if valid
        """
        if quantity <= 0:
            return "Error: Quantity must be positive"
        if action not in ["BUY", "SELL"]:
            return f"Error: Invalid action '{action}'. Must be BUY or SELL"
        if len(rationale) < MIN_RATIONALE_LENGTH:
            return f"Error: Rationale must be at least {MIN_RATIONALE_LENGTH} characters"
        return None

    async def _check_adaptive_threshold(
        self,
        signal_type: str,
        confidence: float,
    ) -> str | None:
        """Validate confidence against adaptive threshold.

        Args:
            signal_type: BUY/SELL
            confidence: Decision confidence (0.0-1.0)

        Returns:
            Error message if rejected, None if passes
        """
        if not self._adaptive_threshold_manager:
            # Fallback to static threshold
            min_conf = self._daemon_config.coordinator.min_confidence_to_trade
            if confidence < min_conf:
                return f"Confidence {confidence:.0%} below threshold {min_conf:.0%}"
            return None

        threshold = self._adaptive_threshold_manager.get_threshold(signal_type)
        if confidence < threshold:
            logger.info(
                f"Rejected {signal_type}: confidence {confidence:.0%} "
                f"below adaptive threshold {threshold:.0%}"
            )
            return (
                f"Confidence {confidence:.0%} below adaptive {signal_type} "
                f"threshold {threshold:.0%} (adapts based on recent accuracy)"
            )

        return None

    async def _handle_confirmation(
        self,
        symbol: str,
        action: str,
        quantity: int,
        stop_loss_price: float | None,
        rationale: str,
    ) -> str | None:
        """Handle manual trade confirmation if required.

        Args:
            symbol: Stock ticker
            action: Order action
            quantity: Number of shares
            stop_loss_price: Optional stop loss price
            rationale: Trading rationale

        Returns:
            Error message if confirmation fails, None if approved or not required
        """
        if not (self.requires_confirmation and self._daemon_config.coordinator.confirmation_mode == "manual"):
            return None

        if not self._confirmation_handler:
            return "Error: Manual confirmation mode enabled but no handler configured"

        # Request approval via Telegram
        logger.info(f"Requesting manual approval for {action} {quantity} {symbol}")
        approved = await self._confirmation_handler.request_approval(
            symbol=symbol,
            action=action,
            quantity=quantity,
            stop_loss_price=stop_loss_price,
            rationale=rationale,
        )

        if not approved:
            logger.info(f"Trade {action} {quantity} {symbol} rejected or timed out")
            return f"Trade {action} {quantity} {symbol} rejected by user or timed out"

        logger.info(f"Trade {action} {quantity} {symbol} approved")
        return None

    async def _submit_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        stop_loss_price: float | None,
        rationale: str,
    ) -> tuple[str, OrderStatus | None]:
        """Submit order to broker.

        Args:
            symbol: Stock ticker
            action: Order action
            quantity: Number of shares
            stop_loss_price: Optional stop loss price
            rationale: Trading rationale

        Returns:
            Tuple of (formatted output, order_status or None on failure)
        """
        try:
            order_status = await asyncio.to_thread(
                self._broker.submit_order,
                symbol=symbol,
                qty=quantity,
                side=action.lower(),
                stop_loss_price=stop_loss_price,
            )

            lines = [
                "# Trade Executed",
                "",
                f"**Order ID:** {order_status.order_id}",
                f"**Symbol:** {order_status.symbol}",
                f"**Action:** {order_status.side.upper()}",
                f"**Quantity:** {order_status.qty}",
                f"**Status:** {order_status.status}",
                f"**Submitted:** {order_status.submitted_at.strftime('%Y-%m-%d %H:%M:%S')}",
            ]

            if stop_loss_price is not None:
                lines.append(f"**Stop Loss:** ${stop_loss_price:.2f}")

            lines.extend(["", "## Rationale", rationale])

            return "\n".join(lines), order_status

        except Exception as e:
            logger.opt(exception=True).error(f"Trade execution failed: {e}")
            return f"Failed to execute trade: {e}", None

    async def _persist_trade(
        self,
        order_status: OrderStatus,
        confidence: float,
        stop_loss_price: float | None,
    ) -> None:
        """Persist executed trade to database.

        Args:
            order_status: Broker order status
            confidence: Decision confidence (0.0-1.0)
            stop_loss_price: Stop loss price (or default ±5%)
        """
        if not self._database_engine:
            return

        try:
            from src.database.repositories.trade import TradeRepository
            from src.metrics.tracker import TradeRecord

            entry_price = order_status.filled_avg_price or 0.0
            effective_stop = stop_loss_price or self._default_stop_loss(entry_price, order_status.side)
            is_paper = self._daemon_config.trading_mode == TradingMode.PAPER

            trade = TradeRecord(
                timestamp=datetime.now(UTC),
                symbol=order_status.symbol,
                action=Signal(order_status.side.upper()),
                entry_price=entry_price,
                exit_price=None,
                shares=int(order_status.qty),
                stop_loss_price=effective_stop,
                confidence=confidence,
                risk_level=self._derive_risk_level(confidence),
                status="OPEN",
                pnl=None,
                pnl_percent=None,
                strategy_name="coordinator",
                broker_order_id=order_status.order_id,
                is_paper_trade=is_paper,
            )

            async with self._database_engine.session() as session:
                repo = TradeRepository(session)
                await repo.create(trade)

            logger.info(f"Persisted coordinator trade: {order_status.symbol} {order_status.side}")
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to persist trade to DB: {e}")

    async def _notify_trade(
        self,
        order_status: OrderStatus,
        confidence: float,
        rationale: str,
    ) -> None:
        """Send trade execution notification.

        Args:
            order_status: Broker order status
            confidence: Decision confidence (0.0-1.0)
            rationale: Trading rationale
        """
        if not self._notification_service:
            return

        try:
            from src.v1.notifications.models import NotificationMessage, NotificationSeverity

            side = order_status.side.upper()
            symbol = order_status.symbol
            price = order_status.filled_avg_price or 0.0

            message = NotificationMessage(
                title=f"{side} {symbol} x{int(order_status.qty)}",
                body=rationale,
                severity=NotificationSeverity.WARNING,
                metadata={
                    "symbol": symbol,
                    "action": side,
                    "quantity": int(order_status.qty),
                    "price": price,
                    "confidence": confidence,
                },
                timestamp=datetime.now(UTC),
            )

            await self._notification_service.notify(message)
        except Exception as e:
            logger.opt(exception=True).warning(f"Trade notification failed: {e}")

    @staticmethod
    def _derive_risk_level(confidence: float) -> str:
        """Derive risk level from confidence per domain rules."""
        if confidence >= CONFIDENCE_LOW_RISK:
            return "LOW"
        if confidence >= CONFIDENCE_MEDIUM_RISK:
            return "MEDIUM"
        return "HIGH"

    @staticmethod
    def _default_stop_loss(entry_price: float, side: str) -> float:
        """Calculate default stop loss ±5% when LLM doesn't provide one."""
        if side.upper() == "SELL":
            return entry_price * (1 + DEFAULT_STOP_LOSS_PCT)
        return entry_price * (1 - DEFAULT_STOP_LOSS_PCT)

    def __repr__(self) -> str:
        """String representation."""
        return "ExecuteTradeTool()"
