"""Core notification service for trading daemon."""

import asyncio
from abc import ABC, abstractmethod
from datetime import UTC, datetime

from loguru import logger
from pydantic import BaseModel

from src.daemon.config import NotificationsConfig, NotificationTrigger


class NotificationMessage(BaseModel):
    """Notification message."""

    trigger: NotificationTrigger
    title: str
    body: str
    metadata: dict[str, object]
    timestamp: datetime


class NotificationChannel(ABC):
    """Base interface for notification channels."""

    @abstractmethod
    async def send(self, message: NotificationMessage) -> bool:
        """Send notification message.

        Args:
            message: Notification message to send

        Returns:
            True if sent successfully, False otherwise
        """
        ...

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if channel is properly configured.

        Returns:
            True if configured, False otherwise
        """
        ...


class NotificationRateLimiter:
    """Rate limiter for notifications."""

    def __init__(self, limit_minutes: int) -> None:
        """Initialize rate limiter.

        Args:
            limit_minutes: Minimum minutes between notifications per symbol:trigger
        """
        self.limit_minutes = limit_minutes
        self._last_notified: dict[str, datetime] = {}

    def can_notify(self, symbol: str, trigger: NotificationTrigger) -> bool:
        """Check if notification is allowed by rate limit.

        Args:
            symbol: Stock symbol
            trigger: Notification trigger type

        Returns:
            True if notification allowed, False if rate-limited
        """
        # Critical triggers always allowed
        if trigger in [NotificationTrigger.PORTFOLIO_VAR_BREACH, NotificationTrigger.HEALTH_FAILURE]:
            return True

        key = f"{symbol}:{trigger.value}"
        last_time = self._last_notified.get(key)
        if last_time is None:
            return True

        elapsed = (datetime.now(UTC) - last_time).total_seconds() / 60
        return elapsed >= self.limit_minutes

    def record_notification(self, symbol: str, trigger: NotificationTrigger) -> None:
        """Record notification timestamp.

        Args:
            symbol: Stock symbol
            trigger: Notification trigger type
        """
        key = f"{symbol}:{trigger.value}"
        self._last_notified[key] = datetime.now(UTC)


class NotificationService:
    """Centralized notification service."""

    def __init__(self, config: NotificationsConfig) -> None:
        """Initialize notification service.

        Args:
            config: Notification configuration
        """
        self.config = config
        self.rate_limiter = NotificationRateLimiter(config.rate_limit_per_symbol_minutes)
        self.channels: dict[str, NotificationChannel] = {}
        self._init_channels()

    def _init_channels(self) -> None:
        """Initialize configured notification channels."""
        if not self.config.enabled:
            return
        from src.daemon.notification_channels import TelegramChannel

        if "telegram" in self.config.channels:
            channel = TelegramChannel(self.config.telegram)
            if channel.is_configured():
                self.channels["telegram"] = channel  # type: ignore[unsupported-operation]
                logger.info("Enabled telegram notification channel")
            else:
                logger.warning("Telegram channel not configured (missing bot_token or chat_id)")

    async def notify(self, trigger: NotificationTrigger, message: NotificationMessage) -> None:
        """Send notification to all channels.

        Args:
            trigger: Notification trigger type
            message: Notification message
        """
        if not self.config.enabled or trigger not in self.config.notify_on:
            return

        symbol = str(message.metadata.get("symbol", "UNKNOWN"))
        if self.config.rate_limit_enabled and not self.rate_limiter.can_notify(symbol, trigger):
            logger.debug(f"Rate limit: skipping {trigger.value} notification for {symbol}")
            return

        # Send to all channels
        tasks = [self._send_to_channel(name, ch, message) for name, ch in self.channels.items()]
        await asyncio.gather(*tasks, return_exceptions=True)

        if self.config.rate_limit_enabled and self.channels:
            self.rate_limiter.record_notification(symbol, trigger)

    async def _send_to_channel(
        self, name: str, channel: NotificationChannel, message: NotificationMessage
    ) -> None:
        """Send message to single channel.

        Args:
            name: Channel name
            channel: Channel instance
            message: Notification message
        """
        try:
            success = await channel.send(message)
            if success:
                logger.info(f"Notification sent via {name}: {message.title}")
            else:
                logger.warning(f"Failed to send notification via {name}")
        except Exception as e:
            logger.error(f"Error sending notification via {name}: {e}")

    def __repr__(self) -> str:
        """Return string representation."""
        return f"NotificationService(enabled={self.config.enabled}, channels={list(self.channels.keys())})"
