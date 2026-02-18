"""Sentiment analysis models."""

from pydantic import BaseModel, Field


class SentimentAnalysis(BaseModel):
    """Sentiment analysis result."""

    overall_sentiment: str
    sentiment_score: float
    positive_ratio: float
    negative_ratio: float
    neutral_ratio: float
    article_count: int
    summary: str
    confidence: float = Field(description="Confidence score (0.0-1.0)", ge=0.0, le=1.0)
