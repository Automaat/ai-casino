"""Base enums for daemon configuration."""

from enum import StrEnum


class TradingMode(StrEnum):
    """Trading mode for broker execution."""

    PAPER = "paper"
    LIVE = "live"
