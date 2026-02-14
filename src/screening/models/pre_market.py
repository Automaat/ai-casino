"""Pre-market screening models."""

from datetime import datetime

from pydantic import BaseModel, Field


class ScreeningWeights(BaseModel):
    """Weights for composite score calculation."""

    gap: float = Field(default=0.50, ge=0.0, le=1.0)
    volume: float = Field(default=0.30, ge=0.0, le=1.0)
    catalyst: float = Field(default=0.20, ge=0.0, le=1.0)


class ScreeningParams(BaseModel):
    """Parameters for pre-market screening."""

    universe: str = "NASDAQ100"
    top_n: int = Field(default=20, ge=1)
    gap_threshold: float = Field(default=3.0, ge=0.0)
    min_volume_ratio: float = Field(default=1.5, ge=0.0)
    min_score: float = Field(default=0.60, ge=0.0, le=1.0)
    timeout_seconds: int = Field(default=60, ge=1)
    earnings_lookahead_days: int = Field(default=7, ge=1)
    overnight_news_hours: int = Field(default=14, ge=1)
    weights: ScreeningWeights = Field(default_factory=ScreeningWeights)


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
