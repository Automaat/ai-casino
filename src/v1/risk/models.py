"""Risk decision models for trade execution gating."""

from enum import StrEnum

from pydantic import BaseModel, Field


class RiskLevel(StrEnum):
    """Risk level for trade decisions."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RiskDecision(BaseModel):
    """Risk assessment result consumed by TradingService for execution gating."""

    approved: bool
    risk_level: RiskLevel
    recommended_shares: int = Field(ge=0)
    stop_loss_price: float = Field(ge=0)
    take_profit_price: float | None = None
    position_value: float = Field(ge=0)
    risk_percent: float = Field(ge=0, le=100)
    warnings: list[str] = Field(default_factory=list)
    reasoning: str

    def __repr__(self) -> str:
        """String representation."""
        status = "APPROVED" if self.approved else "REJECTED"
        return f"RiskDecision({status}, shares={self.recommended_shares}, risk={self.risk_level})"
