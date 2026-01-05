"""Sentiment Analysis Agent."""

from loguru import logger
from pydantic import BaseModel

from src.data.news import NewsArticle
from src.models.sentiment import FinBERTSentiment, SentimentScore


class SentimentAnalysis(BaseModel):
    """Sentiment analysis result."""

    overall_sentiment: str
    sentiment_score: float
    positive_ratio: float
    negative_ratio: float
    neutral_ratio: float
    article_count: int
    summary: str


class SentimentAnalyst:
    """Agent for analyzing sentiment from news articles."""

    POSITIVE_THRESHOLD = 0.2
    NEGATIVE_THRESHOLD = -0.2

    def __init__(self, finbert: FinBERTSentiment) -> None:
        """Initialize sentiment analyst.

        Args:
            finbert: FinBERT sentiment model
        """
        self.finbert = finbert
        logger.info("Initialized SentimentAnalyst")

    def analyze(self, symbol: str, articles: list[NewsArticle]) -> SentimentAnalysis:
        """Analyze sentiment from news articles.

        Args:
            symbol: Stock ticker symbol
            articles: List of news articles

        Returns:
            SentimentAnalysis with aggregated sentiment
        """
        logger.info(f"Analyzing sentiment for {symbol} from {len(articles)} articles")

        if not articles:
            logger.warning("No articles provided for sentiment analysis")
            return SentimentAnalysis(
                overall_sentiment="neutral",
                sentiment_score=0.0,
                positive_ratio=0.0,
                negative_ratio=0.0,
                neutral_ratio=1.0,
                article_count=0,
                summary="No news articles available for analysis",
            )

        texts = [f"{article.title}. {article.description}" for article in articles]
        scores = self.finbert.analyze_batch(texts)

        overall_score = self._aggregate_sentiment(scores)
        sentiment_label = self._get_sentiment_label(overall_score)

        positive_count = sum(1 for s in scores if s.dominant == "positive")
        negative_count = sum(1 for s in scores if s.dominant == "negative")
        neutral_count = sum(1 for s in scores if s.dominant == "neutral")
        total = len(scores)

        summary = self._generate_summary(
            symbol,
            sentiment_label,
            overall_score,
            positive_count,
            negative_count,
            total,
        )

        logger.info(
            f"Sentiment: {sentiment_label} (score={overall_score:.2f}, "
            f"pos={positive_count}, neg={negative_count})"
        )

        return SentimentAnalysis(
            overall_sentiment=sentiment_label,
            sentiment_score=overall_score,
            positive_ratio=positive_count / total,
            negative_ratio=negative_count / total,
            neutral_ratio=neutral_count / total,
            article_count=total,
            summary=summary,
        )

    def _aggregate_sentiment(self, scores: list[SentimentScore]) -> float:
        """Aggregate individual sentiment scores.

        Args:
            scores: List of sentiment scores

        Returns:
            Overall sentiment score (-1 to 1)
        """
        if not scores:
            return 0.0

        return sum(s.score for s in scores) / len(scores)

    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label.

        Args:
            score: Sentiment score (-1 to 1)

        Returns:
            Sentiment label (positive/negative/neutral)
        """
        if score > self.POSITIVE_THRESHOLD:
            return "positive"
        if score < self.NEGATIVE_THRESHOLD:
            return "negative"
        return "neutral"

    def _generate_summary(
        self,
        symbol: str,
        sentiment: str,
        score: float,
        positive: int,
        negative: int,
        total: int,
    ) -> str:
        """Generate human-readable summary.

        Args:
            symbol: Stock ticker
            sentiment: Overall sentiment label
            score: Sentiment score
            positive: Number of positive articles
            negative: Number of negative articles
            total: Total articles

        Returns:
            Summary text
        """
        return (
            f"News sentiment for {symbol} is {sentiment} (score: {score:.2f}). "
            f"Out of {total} articles analyzed: {positive} positive, "
            f"{negative} negative, {total - positive - negative} neutral."
        )

    def __repr__(self) -> str:
        """String representation."""
        return "SentimentAnalyst(model=FinBERT)"
