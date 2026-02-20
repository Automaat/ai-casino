"""Trade confirmation handler for manual approval via Telegram."""

import asyncio
from datetime import UTC, datetime, timedelta

from loguru import logger
from pydantic import BaseModel, Field

from src.v1.notifications.channels.telegram import TelegramChannel
from src.v1.notifications.models import NotificationMessage, NotificationSeverity


class TradeConfirmationRequest(BaseModel):
    """Trade pending manual approval."""

    trade_id: str = Field(description="Unique trade ID (symbol + timestamp)")
    symbol: str
    action: str
    quantity: int
    stop_loss_price: float | None
    rationale: str
    requested_at: datetime
    expires_at: datetime


class TradeApprovalResponse(BaseModel):
    """User approval/rejection."""

    trade_id: str
    approved: bool
    responded_at: datetime


class TradeConfirmationHandler:
    """Telegram-based trade approval handler."""

    def __init__(
        self,
        telegram_channel: TelegramChannel,
        approval_timeout_seconds: int = 60,
    ) -> None:
        """Initialize confirmation handler.

        Args:
            telegram_channel: Telegram channel for notifications
            approval_timeout_seconds: Timeout for approval response (seconds)
        """
        self._telegram = telegram_channel
        self._timeout = approval_timeout_seconds
        self._pending: dict[str, TradeConfirmationRequest] = {}

    async def request_approval(
        self,
        symbol: str,
        action: str,
        quantity: int,
        stop_loss_price: float | None,
        rationale: str,
    ) -> bool:
        """Send approval request via Telegram, poll for response.

        Args:
            symbol: Stock ticker symbol
            action: Order action (BUY/SELL)
            quantity: Number of shares
            stop_loss_price: Optional stop loss price
            rationale: Trading rationale

        Returns:
            True if approved, False if rejected or timeout
        """
        trade_id = f"{symbol}_{datetime.now(UTC).timestamp()}"
        request = TradeConfirmationRequest(
            trade_id=trade_id,
            symbol=symbol,
            action=action,
            quantity=quantity,
            stop_loss_price=stop_loss_price,
            rationale=rationale,
            requested_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(seconds=self._timeout),
        )

        # Store pending
        self._pending[trade_id] = request

        # Send Telegram notification
        message = self._format_approval_message(request)
        sent = await self._telegram.send(message)

        # Fail fast if Telegram send failed
        if not sent:
            logger.error(f"Failed to send approval request for {symbol} via Telegram")
            del self._pending[trade_id]
            return False

        # Poll for response
        deadline = request.expires_at
        while datetime.now(UTC) < deadline:
            response = await self._check_approval(trade_id)
            if response:
                del self._pending[trade_id]
                return response.approved
            await asyncio.sleep(5)

        # Timeout - auto-reject
        logger.warning(f"Trade approval timeout for {symbol}, auto-rejecting")
        del self._pending[trade_id]
        return False

    def _format_approval_message(self, request: TradeConfirmationRequest) -> NotificationMessage:
        """Format Telegram approval request.

        Args:
            request: Trade confirmation request

        Returns:
            Notification message for Telegram
        """
        stop_loss_text = f"${request.stop_loss_price:.2f}" if request.stop_loss_price else "None"
        text = (
            f"🚨 **TRADE APPROVAL REQUIRED** 🚨\n\n"
            f"**Action:** {request.action}\n"
            f"**Symbol:** {request.symbol}\n"
            f"**Quantity:** {request.quantity}\n"
            f"**Stop Loss:** {stop_loss_text}\n\n"
            f"**Rationale:** {request.rationale}\n\n"
            f"**Respond within {self._timeout}s:**\n"
            f"`/approve {request.symbol}` or `/reject {request.symbol}`"
        )
        return NotificationMessage(
            title="Trade Approval",
            body=text,
            severity=NotificationSeverity.CRITICAL,
            metadata={
                "symbol": request.symbol,
                "action": request.action,
                "quantity": request.quantity,
            },
            timestamp=datetime.now(UTC),
        )

    async def _check_approval(self, _trade_id: str) -> TradeApprovalResponse | None:
        """Check if user responded (poll Telegram for commands).

        Implementation: Parse recent Telegram messages for /approve or /reject commands.

        Args:
            _trade_id: Trade ID to check (currently unused - placeholder for future implementation)

        Returns:
            Response if found, None otherwise

        Note:
            Current implementation placeholder - requires Telegram Bot API integration:
            - Option 1: Use getUpdates endpoint to poll for new messages
            - Option 2: Implement webhook receiver for real-time updates
            - Option 3: Use in-memory command queue populated by bot instance
        """
        # TODO(Phase 2): Implement Telegram message polling
        # For now, return None (always timeout)
        return None

    def __repr__(self) -> str:
        """String representation."""
        return f"TradeConfirmationHandler(timeout={self._timeout}s, pending={len(self._pending)})"
