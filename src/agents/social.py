"""Social sentiment analysis models."""

from pydantic import BaseModel


class SocialSentimentAnalysis(BaseModel):
    """Social sentiment analysis result."""

    finnhub_sentiment: float | None  # -1 to 1
    reddit_sentiment: float | None  # -1 to 1
    overall_social_score: float  # -1 to 1 weighted average
    social_momentum: str  # rising/falling/stable
    wsb_mentions_24h: int
    sentiment_label: str  # BULLISH/BEARISH/NEUTRAL from LLM
    interpretation: str  # From LLM
    confidence: float  # 0.0-1.0 multi-factor
