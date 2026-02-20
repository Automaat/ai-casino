"""Telegram notification channel."""

import httpx
from loguru import logger

from src.daemon.config.notifications import TelegramNotificationConfig
from src.models.providers.retry import retry
from src.v1.notifications.base import NotificationChannel
from src.v1.notifications.models import NotificationMessage, NotificationSeverity

SEVERITY_EMOJI = {
    NotificationSeverity.CRITICAL: "🚨",
    NotificationSeverity.ERROR: "❌",
    NotificationSeverity.WARNING: "⚠️",
    NotificationSeverity.INFO: "\U0001f4ac",
}


def _escape(text: str) -> str:
    """Escape markdown special characters for Telegram Markdown (legacy) mode.

    Args:
        text: Raw text

    Returns:
        Escaped text safe for Telegram Markdown
    """
    special_chars = ["_", "*", "[", "]", "(", ")", "~", "`", ">", "#", "+", "-", "=", "|", "{", "}", "!"]
    for char in special_chars:
        text = text.replace(char, f"\\{char}")
    return text


def _format(msg: NotificationMessage) -> str:
    """Format notification for Telegram Markdown.

    Args:
        msg: Notification message

    Returns:
        Formatted markdown string
    """
    emoji = SEVERITY_EMOJI[msg.severity]
    lines = [f"{emoji} *{_escape(msg.title)}*", "", _escape(msg.body)]
    if msg.metadata:
        lines.append("")
        for k, v in msg.metadata.items():
            lines.append(f"• *{_escape(str(k))}:* {_escape(str(v))}")
    return "\n".join(lines)


class TelegramChannel(NotificationChannel):
    """Telegram bot notification channel."""

    def __init__(self, config: TelegramNotificationConfig) -> None:
        """Initialize Telegram channel.

        Args:
            config: Telegram configuration
        """
        self.bot_token = config.bot_token
        self.chat_id = config.chat_id
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
        formatted = _format(message)

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
                logger.opt(exception=True).error(
                    f"Telegram API returned non-JSON response: {response.text!r}"
                )
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
