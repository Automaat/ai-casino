"""News analysis models."""

from pydantic import BaseModel, Field


class NewsAnalysis(BaseModel):
    """News analysis result."""

    key_themes: list[str]
    impact_assessment: str
    recommendation: str
    confidence: float = Field(description="Confidence score (0.0-1.0)", ge=0.0, le=1.0)
