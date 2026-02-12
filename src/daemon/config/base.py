"""Base enums for daemon configuration."""

from enum import StrEnum


class TradingMode(StrEnum):
    """Trading mode for broker execution."""

    PAPER = "paper"
    LIVE = "live"


class NotificationTrigger(StrEnum):
    """Trigger types for notifications."""

    SIGNAL = "signal"
    RISK_REJECTION = "risk_rejection"
    PORTFOLIO_VAR_BREACH = "portfolio_var_breach"
    HEALTH_FAILURE = "health_failure"
    PAPER_TRADING_READY = "paper_trading_ready"
    AGENT_ALERT = "agent_alert"
