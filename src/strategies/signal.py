"""Trading signal enum."""

from enum import StrEnum


class Signal(StrEnum):
    """Trading signal."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
