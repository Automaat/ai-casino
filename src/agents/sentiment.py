"""Sentiment Analysis Agent."""

import asyncio
import hashlib
import time
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel, Field

from src.data.news import NewsArticle
from src.execution_tracking import track_agent
from src.models.sentiment import _analyze_batch_worker, get_finbert_executor

if TYPE_CHECKING:
    from src.models.sentiment import SentimentScore


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


class SentimentAnalyst:
    """Agent for analyzing sentiment from news articles."""

    POSITIVE_THRESHOLD = 0.2
    NEGATIVE_THRESHOLD = -0.2
    CACHE_TTL_SECONDS = 3600

    def __init__(self, finbert: object) -> None:
        """Initialize sentiment analyst.

        Args:
            finbert: FinBERT sentiment analyzer (local or remote, provides analyze_batch method)
        """
        self.finbert = finbert
        self._cache: dict[str, tuple[SentimentAnalysis, float]] = {}
        logger.info("Initialized SentimentAnalyst")

    def _get_cache_key(self, symbol: str, articles: list[NewsArticle]) -> str:
        """Generate cache key from symbol and article URLs.

        Args:
            symbol: Stock ticker symbol
            articles: List of news articles

        Returns:
            SHA256 hash of symbol + sorted article URLs
        """
        urls = sorted(article.url for article in articles)
        content = f"{symbol}:{','.join(urls)}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> SentimentAnalysis | None:
        """Get cached sentiment result if fresh.

        Args:
            cache_key: Cache key

        Returns:
            Cached SentimentAnalysis or None if expired/missing
        """
        if cache_key in self._cache:
            result, timestamp = self._cache[cache_key]
            age = time.time() - timestamp
            if age < self.CACHE_TTL_SECONDS:
                logger.debug(f"Sentiment cache hit (age={age:.1f}s)")
                return result
            logger.debug(f"Sentiment cache expired (age={age:.1f}s)")
            del self._cache[cache_key]
        return None

    def _store_cached_result(self, cache_key: str, result: SentimentAnalysis) -> None:
        """Store sentiment result in cache.

        Args:
            cache_key: Cache key
            result: Sentiment analysis result
        """
        self._cache[cache_key] = (result, time.time())
        logger.debug(f"Stored sentiment in cache ({len(self._cache)} entries)")

    @track_agent
    async def analyze(self, symbol: str, articles: list[NewsArticle]) -> SentimentAnalysis:
        """Analyze sentiment from news articles.

        Args:
            symbol: Stock ticker symbol
            articles: List of news articles

        Returns:
            SentimentAnalysis with aggregated sentiment
        """
        logger.info(f"Analyzing sentiment for {symbol} from {len(articles)} articles")

        cache_key = self._get_cache_key(symbol, articles)
        cached = self._get_cached_result(cache_key)
        if cached:
            return cached

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

        # Check if using remote FinBERT service or local model
        if hasattr(self.finbert, "analyze_batch_async"):
            # Remote service: use async HTTP call directly
            scores = await self.finbert.analyze_batch_async(texts)
        else:
            # Local model: use ProcessPoolExecutor for true parallelism (avoids GIL)
            loop = asyncio.get_running_loop()
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

        # Calculate confidence from sentiment strength (stronger sentiment = higher confidence)
        confidence = abs(overall_score)

        result = SentimentAnalysis(
            overall_sentiment=sentiment_label,
            sentiment_score=overall_score,
            positive_ratio=positive_count / total,
            negative_ratio=negative_count / total,
            neutral_ratio=neutral_count / total,
            article_count=total,
            summary=summary,
            confidence=confidence,
        )

        self._store_cached_result(cache_key, result)
        return result

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
