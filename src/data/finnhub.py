"""Finnhub social sentiment data fetcher."""

import hashlib
import os
from datetime import datetime
from pathlib import Path

import httpx
from diskcache import Cache
from loguru import logger
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Cache TTL in seconds
FINNHUB_CACHE_TTL = 3600  # 1 hour

HTTP_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception) & retry_if_not_exception_type(ValueError),
    reraise=True,
    before_sleep=lambda retry_state: logger.warning(
        f"Retry {retry_state.attempt_number} after {retry_state.outcome.exception()}"
    ),
)


class SocialSentimentEntry(BaseModel):
    """Single social sentiment data point."""

    at_time: datetime
    mention: int
    score: float


class SocialSentimentData(BaseModel):
    """Social sentiment data from Reddit and Twitter."""

    symbol: str
    reddit: list[SocialSentimentEntry]
    twitter: list[SocialSentimentEntry]
    fetched_at: datetime


class BuzzData(BaseModel):
    """News buzz metrics."""

    articles_in_last_week: int
    buzz: float
    weekly_average: float


class SentimentBreakdown(BaseModel):
    """Bearish/bullish sentiment breakdown."""

    bearish_percent: float
    bullish_percent: float


class NewsSentimentData(BaseModel):
    """News sentiment indicator data."""

    symbol: str
    buzz: BuzzData
    company_news_score: float
    sector_avg_bullish_percent: float
    sector_avg_news_score: float
    sentiment: SentimentBreakdown
    fetched_at: datetime


class FinnhubFetcher:
    """Fetch social sentiment data from Finnhub API."""

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(self, api_key: str | None = None, cache_dir: str | None = None) -> None:
        """Initialize Finnhub fetcher.

        Args:
            api_key: Finnhub API key
            cache_dir: Cache directory path
        """
        self._api_key = api_key or os.getenv("FINNHUB_API_KEY")

        self._cache_dir = Path(cache_dir or "data/cache/finnhub")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = Cache(str(self._cache_dir))

        if not self._api_key:
            logger.warning("Finnhub API key not set - API calls will fail")
        else:
            logger.info(f"Initialized FinnhubFetcher (cache_dir={self._cache_dir})")

    def _cache_key(self, prefix: str, *args: str) -> str:
        """Generate cache key.

        Args:
            prefix: Cache key prefix
            args: Additional key components

        Returns:
            Cache key string
        """
        raw = f"{prefix}:{':'.join(str(a) for a in args)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    @HTTP_RETRY
    def fetch_social_sentiment(
        self, symbol: str, from_date: str | None = None, to_date: str | None = None
    ) -> SocialSentimentData:
        """Fetch social sentiment data for a symbol.

        Args:
            symbol: Stock ticker symbol
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)

        Returns:
            SocialSentimentData with Reddit and Twitter sentiment
        """
        logger.info(f"Fetching Finnhub social sentiment for {symbol}")

        cache_key = self._cache_key("social", symbol, from_date or "", to_date or "")
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for {symbol} social sentiment")
            return SocialSentimentData.model_validate(cached)

        if not self._api_key:
            msg = "Finnhub API key not configured"
            raise ValueError(msg)

        try:
            params = {"symbol": symbol, "token": self._api_key}
            if from_date:
                params["from"] = from_date
            if to_date:
                params["to"] = to_date

            with httpx.Client(timeout=30.0) as client:
                response = client.get(f"{self.BASE_URL}/stock/social-sentiment", params=params)
                response.raise_for_status()
                data = response.json()

            reddit_entries = [
                SocialSentimentEntry(
                    at_time=datetime.fromisoformat(entry["atTime"]),
                    mention=entry["mention"],
                    score=entry["score"],
                )
                for entry in data.get("reddit", [])
            ]

            twitter_entries = [
                SocialSentimentEntry(
                    at_time=datetime.fromisoformat(entry["atTime"]),
                    mention=entry["mention"],
                    score=entry["score"],
                )
                for entry in data.get("twitter", [])
            ]

            result = SocialSentimentData(
                symbol=symbol,
                reddit=reddit_entries,
                twitter=twitter_entries,
                fetched_at=datetime.now(),
            )

            self._cache.set(cache_key, result.model_dump(mode="json"), expire=FINNHUB_CACHE_TTL)
            r_count, t_count = len(reddit_entries), len(twitter_entries)
            logger.info(f"Fetched {symbol} social sentiment: {r_count} reddit, {t_count} twitter")
            return result

        except Exception as e:
            logger.error(f"Finnhub social sentiment fetch failed: {e}")
            raise

    @HTTP_RETRY
    def fetch_sentiment_indicator(self, symbol: str) -> NewsSentimentData:
        """Fetch news sentiment indicator for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            NewsSentimentData with buzz and sentiment metrics
        """
        logger.info(f"Fetching Finnhub sentiment indicator for {symbol}")

        cache_key = self._cache_key("indicator", symbol)
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for {symbol} sentiment indicator")
            return NewsSentimentData.model_validate(cached)

        if not self._api_key:
            msg = "Finnhub API key not configured"
            raise ValueError(msg)

        try:
            params = {"symbol": symbol, "token": self._api_key}

            with httpx.Client(timeout=30.0) as client:
                response = client.get(f"{self.BASE_URL}/news-sentiment", params=params)
                response.raise_for_status()
                data = response.json()

            buzz_data = data.get("buzz", {})
            sentiment_data = data.get("sentiment", {})

            result = NewsSentimentData(
                symbol=symbol,
                buzz=BuzzData(
                    articles_in_last_week=buzz_data.get("articlesInLastWeek", 0),
                    buzz=buzz_data.get("buzz", 0.0),
                    weekly_average=buzz_data.get("weeklyAverage", 0.0),
                ),
                company_news_score=data.get("companyNewsScore", 0.0),
                sector_avg_bullish_percent=data.get("sectorAverageBullishPercent", 0.0),
                sector_avg_news_score=data.get("sectorAverageNewsScore", 0.0),
                sentiment=SentimentBreakdown(
                    bearish_percent=sentiment_data.get("bearishPercent", 0.0),
                    bullish_percent=sentiment_data.get("bullishPercent", 0.0),
                ),
                fetched_at=datetime.now(),
            )

            self._cache.set(cache_key, result.model_dump(mode="json"), expire=FINNHUB_CACHE_TTL)
            logger.info(f"Fetched sentiment indicator for {symbol}")
            return result

        except Exception as e:
            logger.error(f"Finnhub sentiment indicator fetch failed: {e}")
            raise

    def clear_cache(self) -> None:
        """Clear all cached Finnhub data."""
        self._cache.clear()
        logger.info("Cleared Finnhub cache")

    def __repr__(self) -> str:
        """String representation."""
        authenticated = bool(self._api_key)
        return f"FinnhubFetcher(authenticated={authenticated}, cache_dir={self._cache_dir})"
