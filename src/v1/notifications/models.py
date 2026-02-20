"""Notification domain models."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


class NotificationSeverity(StrEnum):
    """Severity levels for notifications (PagerDuty-style)."""

    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

    @classmethod
    def from_priority(cls, priority: str) -> NotificationSeverity:
        """Map tool priority string to severity.

        Args:
            priority: Priority string (LOW/MEDIUM/HIGH/CRITICAL)

        Returns:
            Corresponding NotificationSeverity
        """
        return {
            "LOW": cls.INFO,
            "MEDIUM": cls.WARNING,
            "HIGH": cls.ERROR,
            "CRITICAL": cls.CRITICAL,
        }.get(priority.upper(), cls.INFO)


class NotificationMessage(BaseModel):
    """Channel-agnostic notification message."""

    title: str
    body: str
    severity: NotificationSeverity = NotificationSeverity.INFO
    metadata: dict[str, str | int | float | bool]
    timestamp: datetime

    def __repr__(self) -> str:
        """Return string representation."""
        return f"NotificationMessage(title={self.title!r}, severity={self.severity})"
