"""Notification package — channel-agnostic messaging."""

from src.v1.notifications.models import NotificationMessage, NotificationSeverity
from src.v1.notifications.service import NotificationService

__all__ = [
    "NotificationMessage",
    "NotificationService",
    "NotificationSeverity",
]
