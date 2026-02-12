"""API request/response models for FinBERT service."""

from pydantic import BaseModel, Field


class SentimentScore(BaseModel):
    """Sentiment analysis result (backward-compatible with src/models/sentiment.py)."""

    positive: float = Field(ge=0.0, le=1.0, description="Positive sentiment probability")
    negative: float = Field(ge=0.0, le=1.0, description="Negative sentiment probability")
    neutral: float = Field(ge=0.0, le=1.0, description="Neutral sentiment probability")

    @property
    def dominant(self) -> str:
        """Get dominant sentiment label."""
        if self.positive > self.negative and self.positive > self.neutral:
            return "positive"
        if self.negative > self.positive and self.negative > self.neutral:
            return "negative"
        return "neutral"

    @property
    def score(self) -> float:
        """Get overall sentiment score (-1 to 1)."""
        return self.positive - self.negative


class BatchRequest(BaseModel):
    """Batch sentiment analysis request."""

    texts: list[str] = Field(min_length=1, max_length=100, description="Texts to analyze (1-100)")


class BatchResponse(BaseModel):
    """Batch sentiment analysis response."""

    scores: list[SentimentScore]
    inference_time_ms: float
    batch_size: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    device: str
    uptime_seconds: float
