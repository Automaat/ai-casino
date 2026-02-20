"""Centralized notification service."""

import asyncio

from loguru import logger

from src.daemon.config.notifications import NotificationsConfig
from src.notifications.base import NotificationChannel
from src.notifications.models import NotificationMessage


class NotificationService:
    """Sends notifications to all configured channels."""

    def __init__(self, config: NotificationsConfig) -> None:
        """Initialize notification service.

        Args:
            config: Notification configuration
        """
        self.config = config
        self.channels: dict[str, NotificationChannel] = {}
        self._init_channels()

    def _init_channels(self) -> None:
        """Initialize configured notification channels."""
        if not self.config.enabled:
            return
        from src.notifications.channels.telegram import TelegramChannel

        if "telegram" in self.config.channels:
            channel = TelegramChannel(self.config.telegram)
            if channel.is_configured():
                self.channels["telegram"] = channel
                logger.info("Enabled telegram notification channel")
            else:
                logger.warning("Telegram channel not configured (missing bot_token or chat_id)")

    async def notify(self, message: NotificationMessage) -> None:
        """Send notification to all channels.

        Args:
            message: Notification message
        """
        if not self.config.enabled:
            return

        if not self.channels:
            return

        async with asyncio.TaskGroup() as tg:
            for name, ch in self.channels.items():
                tg.create_task(self._safe_send(name, ch, message))

    async def _safe_send(self, name: str, ch: NotificationChannel, message: NotificationMessage) -> None:
        """Send to channel, swallowing errors.

        Args:
            name: Channel name
            ch: Channel instance
            message: Notification message
        """
        try:
            success = await ch.send(message)
            if success:
                logger.info(f"Notification sent via {name}: {message.title}")
            else:
                logger.warning(f"Failed to send notification via {name}")
        except Exception as e:
            logger.opt(exception=True).error(f"Error sending notification via {name}: {e}")

    def __repr__(self) -> str:
        """Return string representation."""
        return f"NotificationService(enabled={self.config.enabled}, channels={list(self.channels.keys())})"
