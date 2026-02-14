"""Pre-market screening configuration."""

from pydantic import BaseModel, Field, model_validator


class PreMarketScreeningConfig(BaseModel):
    """Configuration for pre-market screening (7:00-9:30 AM ET)."""

    enabled: bool = False
    screening_time: str = "07:00"
    universe: str = "NASDAQ100"
    top_n: int = Field(default=20, ge=1, le=50)
    gap_threshold_percent: float = Field(default=3.0, ge=0.0)
    min_volume_ratio: float = Field(default=1.5, ge=1.0)
    min_composite_score: float = Field(default=0.60, ge=0.0, le=1.0)
    timeout_seconds: int = Field(default=60, ge=10, le=300)
    earnings_lookahead_days: int = Field(default=7, ge=1, le=14)
    overnight_news_hours: int = Field(default=14, ge=1, le=24)

    gap_weight: float = Field(default=0.50, ge=0.0, le=1.0)
    volume_weight: float = Field(default=0.30, ge=0.0, le=1.0)
    catalyst_weight: float = Field(default=0.20, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_weights(self) -> PreMarketScreeningConfig:
        """Validate scoring weights sum to 1.0."""
        total = self.gap_weight + self.volume_weight + self.catalyst_weight
        if abs(total - 1.0) > 1e-6:
            msg = f"Scoring weights must sum to 1.0, got {total:.3f}"
            raise ValueError(msg)
        return self
