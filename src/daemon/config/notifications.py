"""Notification system configuration."""

from pydantic import BaseModel, Field

from src.daemon.config.base import NotificationTrigger


class TelegramNotificationConfig(BaseModel):
    """Telegram notification channel configuration."""

    bot_token: str | None = None
    chat_id: str | None = None


class NotificationsConfig(BaseModel):
    """Notification system configuration."""

    enabled: bool = False
    channels: list[str] = Field(default_factory=lambda: ["telegram"])
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    notify_on: list[NotificationTrigger] = Field(
        default_factory=lambda: [
            NotificationTrigger.SIGNAL,
            NotificationTrigger.RISK_REJECTION,
            NotificationTrigger.PORTFOLIO_VAR_BREACH,
            NotificationTrigger.HEALTH_FAILURE,
        ]
    )
    rate_limit_enabled: bool = True
    rate_limit_per_symbol_minutes: int = 60
    telegram: TelegramNotificationConfig = Field(default_factory=TelegramNotificationConfig)
