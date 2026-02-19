"""Notification package — channel-agnostic messaging."""

from src.notifications.models import NotificationMessage, NotificationSeverity
from src.notifications.service import NotificationService

__all__ = [
    "NotificationMessage",
    "NotificationService",
    "NotificationSeverity",
]
