"""Trading session enum."""

from enum import StrEnum


class TradingSession(StrEnum):
    """Trading session type."""

    REGULAR = "REGULAR"
    PRE_MARKET = "PRE_MARKET"
    AFTER_HOURS = "AFTER_HOURS"
