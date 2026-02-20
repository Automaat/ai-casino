"""Abstract base for notification channels."""

from abc import ABC, abstractmethod

from src.notifications.models import NotificationMessage


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
