"""Domain models for coordinator decision history queries."""

from datetime import datetime

from pydantic import BaseModel, Field


class DecisionQueryResult(BaseModel):
    """Single decision query result with outcome."""

    symbol: str
    timestamp: datetime
    signal: str
    confidence: float = Field(ge=0.0, le=1.0)
    price_at_signal: float
    price_at_outcome: float | None
    return_pct: float | None = Field(description="Return percentage at outcome horizon")
    hit_miss: str | None = Field(description="HIT/MISS/PENDING based on outcome")
    regime: str | None
    strategy_used: str | None
    trading_session: str


class SuccessRateStats(BaseModel):
    """Success rate statistics for decision queries."""

    total_decisions: int
    hit_count: int
    miss_count: int
    pending_count: int = Field(default=0, description="Decisions without outcomes yet")
    success_rate: float = Field(ge=0.0, le=1.0, description="hit / (hit + miss)")
    avg_return: float | None = Field(description="Average return for completed decisions")
    avg_confidence: float = Field(ge=0.0, le=1.0)
