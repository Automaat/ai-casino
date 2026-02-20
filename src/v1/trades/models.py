"""Trade execution models."""

from datetime import datetime
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, Field

MIN_RATIONALE_LENGTH: Final[int] = 10
CONFIDENCE_LOW_RISK: Final[float] = 0.75
CONFIDENCE_MEDIUM_RISK: Final[float] = 0.5


class TradeAction(StrEnum):
    """Trade action type."""

    BUY = "BUY"
    SELL = "SELL"


class TradeRejectionReason(StrEnum):
    """Reason a trade was rejected."""

    VALIDATION_FAILED = "validation_failed"
    BELOW_THRESHOLD = "below_threshold"
    DUPLICATE_POSITION = "duplicate_position"
    CONFIRMATION_REJECTED = "confirmation_rejected"
    BROKER_ERROR = "broker_error"
    RISK_REJECTED = "risk_rejected"


class TradeRequest(BaseModel):
    """Request to execute a trade."""

    symbol: str
    action: TradeAction
    quantity: int = Field(gt=0)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=MIN_RATIONALE_LENGTH)
    stop_loss_price: float | None = None
    strategy_name: str = "coordinator"

    def __repr__(self) -> str:
        """String representation."""
        return f"TradeRequest({self.action} {self.quantity}x {self.symbol} conf={self.confidence:.0%})"


class TradeRejection(BaseModel):
    """Details about why a trade was rejected."""

    reason: TradeRejectionReason
    message: str

    def __repr__(self) -> str:
        """String representation."""
        return f"TradeRejection({self.reason}: {self.message})"


class TradeResult(BaseModel):
    """Result of a trade execution attempt."""

    executed: bool
    order_id: str | None = None
    symbol: str
    action: TradeAction
    quantity: int
    status: str
    filled_avg_price: float | None = None
    submitted_at: datetime | None = None
    stop_loss_price: float | None = None
    rejection: TradeRejection | None = None
    requested_quantity: int | None = None
    risk_capped: bool = False

    def __repr__(self) -> str:
        """String representation."""
        if self.executed:
            return f"TradeResult(executed {self.action} {self.quantity}x {self.symbol})"
        reason = self.rejection.reason if self.rejection else "unknown"
        return f"TradeResult(rejected {self.action} {self.symbol}: {reason})"
