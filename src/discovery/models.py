"""Discovery data models."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class DiscoverySource(StrEnum):
    """Source of stock discovery."""

    TECHNICAL_SCREENING = "technical_screening"
    REDDIT_TRENDING = "reddit_trending"
    EARNINGS_UPCOMING = "earnings_upcoming"
    SECTOR_ROTATION = "sector_rotation"
    VOLUME_SPIKE = "volume_spike"
    PRICE_GAP = "price_gap"
    NEWS_TRENDING = "news_trending"
    PRE_MARKET = "pre_market"


class DiscoveryCandidate(BaseModel):
    """Stock candidate from discovery process."""

    symbol: str
    name: str
    sector: str
    sources: list[DiscoverySource]
    composite_score: float  # 0-1
    source_scores: dict[str, float] = Field(default_factory=dict)
    discovery_timestamp: datetime
    metadata: dict[str, object] = Field(default_factory=dict)
    ttl_expires_at: datetime

    def __repr__(self) -> str:
        """Return string representation."""
        return f"DiscoveryCandidate(symbol={self.symbol}, score={self.composite_score:.2f})"


class DiscoveryResult(BaseModel):
    """Result of discovery run."""

    candidates: list[DiscoveryCandidate]
    total_discovered: int
    filtered_count: int
    discovered_at: datetime
    source_breakdown: dict[str, int] = Field(default_factory=dict)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"DiscoveryResult(candidates={len(self.candidates)}, total={self.total_discovered})"


class DiscoverySourceDetail(BaseModel):
    """Detailed discovery source information."""

    source_type: str
    weight: float
    metadata: dict[str, object] = Field(default_factory=dict)


class ActiveDiscoveryCandidate(BaseModel):
    """Active discovery candidate in volatile state."""

    symbol: str
    discovered_at: datetime
    composite_score: float
    sources: list[DiscoverySourceDetail]
    ttl_expires_at: datetime

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ActiveDiscoveryCandidate(symbol={self.symbol}, score={self.composite_score:.2f})"
