"""Sentiment Analysis Worker - Pydantic AI migration POC."""

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

from src.agents.sentiment import SentimentAnalysis
from src.data.news import NewsArticle
from src.models.sentiment import _analyze_batch_worker, get_finbert_executor
from src.tools.models import ToolDefinition, ToolFunction, ToolParameter, ToolParametersSchema

if TYPE_CHECKING:
    from src.models.sentiment import SentimentScore


class SentimentWorker:
    """Sentiment analysis worker - Pydantic AI migration POC."""

    POSITIVE_THRESHOLD = 0.2
    NEGATIVE_THRESHOLD = -0.2

    def __init__(self, finbert: object) -> None:
        """Initialize sentiment worker.

        Args:
            finbert: FinBERT sentiment analyzer (local or remote, provides analyze_batch method)
        """
        self.finbert = finbert
        logger.info("Initialized SentimentWorker (POC)")

    async def analyze(self, symbol: str, articles: list[NewsArticle]) -> SentimentAnalysis:
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
                confidence=0.0,
            )

        texts = [f"{article.title}. {article.description}" for article in articles]

        loop = asyncio.get_running_loop()
        # Use ProcessPoolExecutor for true parallelism (avoids GIL)
        device = getattr(self.finbert, "device", "cpu")
        executor = get_finbert_executor()
        score_dicts = await loop.run_in_executor(executor, _analyze_batch_worker, texts, device)

        # Import here to avoid circular import at module level
        from src.models.sentiment import SentimentScore

        scores = [SentimentScore(**s) for s in score_dicts]

        overall_score = self._aggregate_sentiment(scores)
        sentiment_label = self._get_sentiment_label(overall_score)

        positive_count = sum(1 for s in scores if s.dominant == "positive")
        negative_count = sum(1 for s in scores if s.dominant == "negative")
        neutral_count = sum(1 for s in scores if s.dominant == "neutral")
        total = len(scores)

        summary = self._generate_summary(symbol, sentiment_label, overall_score, scores)
        confidence = min(abs(overall_score), 1.0)

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
            confidence=confidence,
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
        self, symbol: str, sentiment: str, score: float, scores: list[SentimentScore]
    ) -> str:
        """Generate human-readable summary.

        Args:
            symbol: Stock ticker
            sentiment: Overall sentiment label
            score: Sentiment score
            scores: List of sentiment scores

        Returns:
            Summary text
        """
        positive = sum(1 for s in scores if s.dominant == "positive")
        negative = sum(1 for s in scores if s.dominant == "negative")
        total = len(scores)
        return (
            f"News sentiment for {symbol} is {sentiment} (score: {score:.2f}). "
            f"Out of {total} articles analyzed: {positive} positive, "
            f"{negative} negative, {total - positive - negative} neutral."
        )

    def get_tool_definition(self) -> ToolDefinition:
        """Get tool definition for supervisor integration.

        Returns:
            Tool definition
        """
        return ToolDefinition(
            type="function",
            function=ToolFunction(
                name="analyze_sentiment",
                description="Analyze sentiment from news articles using FinBERT",
                parameters=ToolParametersSchema(
                    type="object",
                    properties={
                        "symbol": ToolParameter(
                            type="string",
                            description="Stock ticker symbol",
                        ),
                    },
                    required=["symbol"],
                ),
            ),
        )

    def __repr__(self) -> str:
        """String representation."""
        return "SentimentWorker(model=FinBERT)"
