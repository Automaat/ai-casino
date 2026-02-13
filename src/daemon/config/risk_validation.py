"""Configuration for pre-decision risk validation."""

from pydantic import BaseModel, Field


class RiskValidationConfig(BaseModel):
    """Configuration for pre-decision risk validation."""

    enabled: bool = True

    # Confidence thresholds
    min_overall_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    min_technical_confidence: float = Field(default=0.4, ge=0.0, le=1.0)
    min_sentiment_confidence: float = Field(default=0.4, ge=0.0, le=1.0)
    min_news_confidence: float = Field(default=0.4, ge=0.0, le=1.0)
    min_research_confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # Signal consistency
    allow_conflicting_signals: bool = True
    max_conflicting_signals: int = Field(default=2, ge=0, le=5)

    # Market conditions
    max_volatility_threshold: float | None = None

    # Session rules
    pre_market_min_confidence: float = Field(default=0.7, ge=0.5, le=1.0)

    # Data freshness
    max_data_age_minutes: int = Field(default=60, ge=1, le=1440)
