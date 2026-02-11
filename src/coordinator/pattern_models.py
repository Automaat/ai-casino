"""Pattern detection models for coordinator learning."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class PatternType(StrEnum):
    """Types of patterns detected."""

    SYMBOL_PERFORMANCE = "symbol_performance"
    CONFIDENCE_CALIBRATION = "confidence_calibration"
    TIMING = "timing"
    TECHNICAL_INDICATOR = "technical_indicator"
    EXECUTION_GAP = "execution_gap"


class PatternInsight(BaseModel):
    """Detected pattern with actionable insight."""

    pattern_type: PatternType
    symbol: str | None = None
    confidence: float = Field(ge=0.0, le=1.0, description="Pattern confidence")
    sample_size: int = Field(ge=0, description="Number of observations")
    insight_text: str = Field(description="Human-readable insight")
    recommendation: str = Field(description="Actionable recommendation")
    detected_at: datetime

    def __repr__(self) -> str:
        """String representation."""
        symbol_text = f", {self.symbol}" if self.symbol else ""
        return f"PatternInsight({self.pattern_type}{symbol_text}, confidence={self.confidence:.2f})"
