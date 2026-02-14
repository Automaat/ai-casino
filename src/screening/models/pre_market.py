"""Pre-market screening models."""

from datetime import datetime

from pydantic import BaseModel, Field


class PreMarketCandidate(BaseModel):
    """Pre-market gap candidate."""

    symbol: str
    name: str
    sector: str

    prev_close: float
    current_open: float
    gap_percent: float

    yesterday_volume: int
    avg_volume_20d: float
    volume_ratio: float

    has_earnings: bool
    earnings_date: datetime | None = None
    news_count: int
    news_titles: list[str] = Field(default_factory=list)

    gap_score: float
    volume_score: float
    catalyst_score: float
    composite_score: float

    priority: int = Field(ge=1, le=5)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"PreMarketCandidate(symbol={self.symbol}, gap={self.gap_percent:.1f}%, "
            f"score={self.composite_score:.2f}, P{self.priority})"
        )


class PreMarketResult(BaseModel):
    """Pre-market screening result."""

    candidates: list[PreMarketCandidate]
    total_screened: int
    filtered_count: int
    screened_at: datetime
    expires_at: datetime

    gap_plays_count: int
    volume_spike_count: int
    catalyst_count: int

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"PreMarketResult(candidates={len(self.candidates)}, "
            f"screened={self.total_screened}, expires={self.expires_at.strftime('%H:%M')})"
        )
