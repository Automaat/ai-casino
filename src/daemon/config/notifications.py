"""Notification system configuration."""

from pydantic import BaseModel, Field


class TelegramNotificationConfig(BaseModel):
    """Telegram notification channel configuration."""

    bot_token: str | None = None
    chat_id: str | None = None


class NotificationsConfig(BaseModel):
    """Notification system configuration."""

    enabled: bool = False
    channels: list[str] = Field(default_factory=lambda: ["telegram"])
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    telegram: TelegramNotificationConfig = Field(default_factory=TelegramNotificationConfig)
