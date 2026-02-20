"""Unified trading service for trade execution."""

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger

from src.daemon.config.base import TradingMode
from src.strategies.signal import Signal
from src.v1.trades.models import (
    CONFIDENCE_LOW_RISK,
    CONFIDENCE_MEDIUM_RISK,
    DEFAULT_STOP_LOSS_PCT,
    TradeAction,
    TradeRejection,
    TradeRejectionReason,
    TradeRequest,
    TradeResult,
)

if TYPE_CHECKING:
    from src.daemon.config import DaemonConfig
    from src.database.engine import DatabaseEngine
    from src.v1.coordinator.confirmation import TradeConfirmationHandler
    from src.v1.notifications.service import NotificationService
    from src.v1.trades.brokers import Broker, OrderStatus


class TradingService:
    """Unified service for executing, persisting, and notifying trades."""

    def __init__(
        self,
        broker: Broker,
        daemon_config: DaemonConfig,
        database_engine: DatabaseEngine | None = None,
        notification_service: NotificationService | None = None,
        confirmation_handler: TradeConfirmationHandler | None = None,
    ) -> None:
        """Initialize trading service.

        Args:
            broker: Broker instance
            daemon_config: Daemon configuration
            database_engine: Optional database engine for persistence
            notification_service: Optional notification service
            confirmation_handler: Optional trade confirmation handler
        """
        self._broker = broker
        self._daemon_config = daemon_config
        self._database_engine = database_engine
        self._notification_service = notification_service
        self._confirmation_handler = confirmation_handler

    async def execute(self, request: TradeRequest) -> TradeResult:
        """Execute a trade through validation, submission, persistence, and notification.

        Args:
            request: Trade request with all parameters

        Returns:
            TradeResult with execution outcome
        """
        # Pre-trade validations
        if rejection := self._check_threshold(request):
            return self._rejected_result(request, rejection)

        if request.action == TradeAction.BUY and (rejection := await self._check_duplicate(request.symbol)):
            return self._rejected_result(request, rejection)

        if rejection := await self._handle_confirmation(request):
            return self._rejected_result(request, rejection)

        # Submit order
        order_status = await self._submit_order(request)
        if order_status is None:
            return self._rejected_result(
                request,
                TradeRejection(
                    reason=TradeRejectionReason.BROKER_ERROR,
                    message=f"Failed to submit {request.action} order for {request.symbol}",
                ),
            )

        result = TradeResult(
            executed=True,
            order_id=order_status.order_id,
            symbol=order_status.symbol,
            action=request.action,
            quantity=int(order_status.qty),
            status=order_status.status,
            filled_avg_price=order_status.filled_avg_price,
            submitted_at=order_status.submitted_at,
            stop_loss_price=request.stop_loss_price,
        )

        # Non-blocking side effects
        await self._persist_trade(order_status, request)
        await self._notify_trade(order_status, request)

        return result

    def _check_threshold(self, request: TradeRequest) -> TradeRejection | None:
        """Check confidence against static threshold from config."""
        min_conf = self._daemon_config.coordinator.min_confidence_to_trade
        if request.confidence < min_conf:
            return TradeRejection(
                reason=TradeRejectionReason.BELOW_THRESHOLD,
                message=f"Confidence {request.confidence:.0%} below threshold {min_conf:.0%}",
            )
        return None

    async def _check_duplicate(self, symbol: str) -> TradeRejection | None:
        """Reject BUY if open position already exists in DB."""
        if not self._database_engine:
            return None

        try:
            from src.database.repositories.trade import TradeRepository

            async with self._database_engine.session() as session:
                repo = TradeRepository(session)
                existing = await repo.get_entry_trade(symbol)

            if existing:
                logger.info(f"Blocked duplicate BUY {symbol}: open position since {existing.timestamp}")
                return TradeRejection(
                    reason=TradeRejectionReason.DUPLICATE_POSITION,
                    message=(
                        f"Already hold open BUY position in {symbol} "
                        f"({existing.shares} shares since "
                        f"{existing.timestamp.strftime('%Y-%m-%d %H:%M')}). "
                        f"SELL first before buying again."
                    ),
                )
        except Exception as e:
            logger.opt(exception=True).warning(f"Duplicate position check failed for {symbol}: {e}")
            return TradeRejection(
                reason=TradeRejectionReason.DUPLICATE_POSITION,
                message=(
                    f"Could not verify existing position for {symbol} due to an internal error. "
                    f"Trade not executed to avoid potential duplicate BUY."
                ),
            )

        return None

    async def _handle_confirmation(self, request: TradeRequest) -> TradeRejection | None:
        """Handle manual trade confirmation if required."""
        is_live = self._daemon_config.trading_mode == TradingMode.LIVE
        is_manual = self._daemon_config.coordinator.confirmation_mode == "manual"

        if not (is_live and is_manual):
            return None

        if not self._confirmation_handler:
            return TradeRejection(
                reason=TradeRejectionReason.CONFIRMATION_REJECTED,
                message="Manual confirmation mode enabled but no handler configured",
            )

        logger.info(f"Requesting manual approval for {request.action} {request.quantity} {request.symbol}")
        approved = await self._confirmation_handler.request_approval(
            symbol=request.symbol,
            action=request.action.value,
            quantity=request.quantity,
            stop_loss_price=request.stop_loss_price,
            rationale=request.rationale,
        )

        if not approved:
            logger.info(f"Trade {request.action} {request.quantity} {request.symbol} rejected or timed out")
            return TradeRejection(
                reason=TradeRejectionReason.CONFIRMATION_REJECTED,
                message=(
                    f"Trade {request.action} {request.quantity} {request.symbol} "
                    f"rejected by user or timed out"
                ),
            )

        logger.info(f"Trade {request.action} {request.quantity} {request.symbol} approved")
        return None

    async def _submit_order(self, request: TradeRequest) -> OrderStatus | None:
        """Submit order to broker."""
        try:
            logger.info(
                f"Executing {request.action} order: {request.quantity} {request.symbol} "
                f"(stop_loss={request.stop_loss_price})"
            )
            return await asyncio.to_thread(
                self._broker.submit_order,
                symbol=request.symbol,
                qty=request.quantity,
                side=request.action.value.lower(),
                stop_loss_price=request.stop_loss_price,
            )
        except Exception as e:
            logger.opt(exception=True).error(f"Trade execution failed: {e}")
            return None

    async def _persist_trade(self, order_status: OrderStatus, request: TradeRequest) -> None:
        """Persist executed trade to database (non-blocking on failure)."""
        if not self._database_engine:
            return

        try:
            from src.database.repositories.trade import TradeRepository
            from src.metrics.tracker import TradeRecord

            entry_price = order_status.filled_avg_price or 0.0
            effective_stop = request.stop_loss_price or self._default_stop_loss(
                entry_price, order_status.side
            )
            is_paper = self._daemon_config.trading_mode == TradingMode.PAPER

            trade = TradeRecord(
                timestamp=datetime.now(UTC),
                symbol=order_status.symbol,
                action=Signal(order_status.side.upper()),
                entry_price=entry_price,
                exit_price=None,
                shares=int(order_status.qty),
                stop_loss_price=effective_stop,
                confidence=request.confidence,
                risk_level=self._derive_risk_level(request.confidence),
                status="OPEN",
                pnl=None,
                pnl_percent=None,
                strategy_name=request.strategy_name,
                broker_order_id=order_status.order_id,
                is_paper_trade=is_paper,
            )

            async with self._database_engine.session() as session:
                repo = TradeRepository(session)
                await repo.create(trade)

            logger.info(f"Persisted trade: {order_status.symbol} {order_status.side}")
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to persist trade to DB: {e}")

    async def _notify_trade(self, order_status: OrderStatus, request: TradeRequest) -> None:
        """Send trade execution notification (non-blocking on failure)."""
        if not self._notification_service:
            return

        try:
            from src.v1.notifications.models import NotificationMessage, NotificationSeverity

            side = order_status.side.upper()
            symbol = order_status.symbol
            price = order_status.filled_avg_price or 0.0

            if side == "BUY":
                emoji = "💰"
            elif side == "SELL":
                emoji = "🔴"
            else:
                emoji = "⚪"
                logger.warning(f"Unknown order side for notification: {order_status.side!r}")
            message = NotificationMessage(
                title=f"{emoji} {side} {symbol} x{int(order_status.qty)}",
                body=request.rationale,
                severity=NotificationSeverity.WARNING,
                metadata={
                    "symbol": symbol,
                    "action": side,
                    "quantity": int(order_status.qty),
                    "price": price,
                    "confidence": request.confidence,
                },
                timestamp=datetime.now(UTC),
            )

            await self._notification_service.notify(message)
        except Exception as e:
            logger.opt(exception=True).warning(f"Trade notification failed: {e}")

    @staticmethod
    def _rejected_result(request: TradeRequest, rejection: TradeRejection) -> TradeResult:
        """Build a rejected TradeResult."""
        return TradeResult(
            executed=False,
            symbol=request.symbol,
            action=request.action,
            quantity=request.quantity,
            status="rejected",
            rejection=rejection,
        )

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
        """Calculate default stop loss +/-5% when not provided."""
        if side.upper() == "SELL":
            return entry_price * (1 + DEFAULT_STOP_LOSS_PCT)
        return entry_price * (1 - DEFAULT_STOP_LOSS_PCT)

    def __repr__(self) -> str:
        """String representation."""
        return "TradingService()"
