"""Notification channel implementations."""

import os

import httpx
from loguru import logger

from src.daemon.config import TelegramNotificationConfig
from src.daemon.notifications import NotificationChannel, NotificationMessage
from src.models.providers.retry import retry


class TelegramChannel(NotificationChannel):
    """Telegram bot notification channel."""

    def __init__(self, config: TelegramNotificationConfig) -> None:
        """Initialize Telegram channel.

        Args:
            config: Telegram configuration
        """
        self.bot_token = config.bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = config.chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    def is_configured(self) -> bool:
        """Check if channel is properly configured.

        Returns:
            True if bot_token and chat_id are set, False otherwise
        """
        return bool(self.bot_token and self.chat_id)

    @retry(max_attempts=2, delay=1.0, exceptions=(httpx.HTTPError,))
    async def send(self, message: NotificationMessage) -> bool:
        """Send notification via Telegram.

        Args:
            message: Notification message

        Returns:
            True if sent successfully, False otherwise
        """
        from src.daemon.notification_formatter import NotificationFormatter

        formatted = NotificationFormatter.format_for_telegram(message)

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{self.base_url}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": formatted,
                    "parse_mode": "Markdown",
                },
            )
            response.raise_for_status()
            logger.debug(f"Telegram API HTTP response status: {response.status_code}")

            try:
                data = response.json()
            except ValueError:
                logger.error(f"Telegram API returned non-JSON response: {response.text!r}")
                return False

            ok = data.get("ok")
            if ok is True:
                logger.debug("Telegram API reported successful delivery.")
                return True

            description = data.get("description")
            if description:
                logger.error(f"Telegram API error: ok={ok}, description={description}")
            else:
                logger.error(f"Telegram API error: ok={ok}, response={data}")
            return False

    def __repr__(self) -> str:
        """Return string representation."""
        return f"TelegramChannel(configured={self.is_configured()})"
