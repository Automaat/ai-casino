"""Risk decision models for trade execution gating."""

from pydantic import BaseModel, Field


class RiskDecision(BaseModel):
    """Risk assessment result consumed by TradingService for execution gating."""

    approved: bool
    risk_level: str
    recommended_shares: int = Field(ge=0)
    stop_loss_price: float
    take_profit_price: float | None = None
    position_value: float
    risk_percent: float
    warnings: list[str] = Field(default_factory=list)
    reasoning: str

    def __repr__(self) -> str:
        """String representation."""
        status = "APPROVED" if self.approved else "REJECTED"
        return f"RiskDecision({status}, shares={self.recommended_shares}, risk={self.risk_level})"
