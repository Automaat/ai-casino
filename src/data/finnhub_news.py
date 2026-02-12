"""Finnhub news fetcher."""

import asyncio
import os
from datetime import UTC, datetime

import httpx
from dotenv import load_dotenv
from loguru import logger

from src.cache.historical import HistoricalCache
from src.data.news import HTTP_RETRY, NewsArticle

load_dotenv()


class FinnhubNewsFetcher:
    """Fetch news from Finnhub API (free tier)."""

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(
        self,
        api_key: str | None = None,
        historical_cache: HistoricalCache | None = None,
    ) -> None:
        """Initialize Finnhub news fetcher.

        Args:
            api_key: Finnhub API key
            historical_cache: Optional permanent cache for news articles
        """
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY", "")
        self._cache = historical_cache
        if not self.api_key:
            logger.warning("FINNHUB_API_KEY not set - API calls may be limited")

    async def afetch_company_news(self, symbol: str, limit: int = 10) -> list[NewsArticle]:
        """Fetch company-specific news asynchronously.

        Args:
            symbol: Stock ticker symbol
            limit: Maximum number of articles

        Returns:
            List of NewsArticle objects
        """
        return await asyncio.to_thread(self._fetch_company_sync, symbol, limit)

    async def afetch_market_news(self, limit: int = 20) -> list[NewsArticle]:
        """Fetch general market news asynchronously.

        Args:
            limit: Maximum number of articles

        Returns:
            List of NewsArticle objects
        """
        return await asyncio.to_thread(self._fetch_market_sync, limit)

    def get_source_name(self) -> str:
        """Return source identifier."""
        return "finnhub"

    @HTTP_RETRY
    def _fetch_company_sync(self, symbol: str, limit: int) -> list[NewsArticle]:
        """Fetch company-specific news (sync implementation).

        Args:
            symbol: Stock ticker symbol
            limit: Maximum number of articles

        Returns:
            List of NewsArticle objects
        """
        logger.info(f"Fetching {limit} news articles for {symbol} from Finnhub")

        url = f"{self.BASE_URL}/company-news"
        params = {
            "symbol": symbol,
            "token": self.api_key,
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(url, params=params)
                response.raise_for_status()

                data = response.json()
                articles = []

                for item in data[:limit]:
                    if not item.get("url") or not item.get("headline"):
                        continue

                    # Finnhub returns unix timestamp
                    published_at = datetime.fromtimestamp(item.get("datetime", 0), tz=UTC)

                    articles.append(
                        NewsArticle(
                            title=item.get("headline", ""),
                            description=item.get("summary", ""),
                            url=item.get("url", ""),
                            published_at=published_at,
                            source=item.get("source", "finnhub"),
                        )
                    )

                logger.info(f"Fetched {len(articles)} articles from Finnhub")

                if self._cache and articles:
                    self._cache.store_news_articles(symbol, articles)

                return articles

        except httpx.HTTPError as e:
            logger.error(f"Finnhub fetch failed: {e}")
            raise

    @HTTP_RETRY
    def _fetch_market_sync(self, limit: int) -> list[NewsArticle]:
        """Fetch general market news (sync implementation).

        Args:
            limit: Maximum number of articles

        Returns:
            List of NewsArticle objects
        """
        logger.info(f"Fetching {limit} general market news from Finnhub")

        url = f"{self.BASE_URL}/news"
        params = {
            "category": "general",
            "token": self.api_key,
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(url, params=params)
                response.raise_for_status()

                data = response.json()
                articles = []

                for item in data[:limit]:
                    if not item.get("url") or not item.get("headline"):
                        continue

                    from datetime import UTC

                    published_at = datetime.fromtimestamp(item.get("datetime", 0), tz=UTC)

                    articles.append(
                        NewsArticle(
                            title=item.get("headline", ""),
                            description=item.get("summary", ""),
                            url=item.get("url", ""),
                            published_at=published_at,
                            source=item.get("source", "finnhub"),
                        )
                    )

                logger.info(f"Fetched {len(articles)} articles from Finnhub")
                return articles

        except httpx.HTTPError as e:
            logger.error(f"Finnhub market news fetch failed: {e}")
            raise

    def __repr__(self) -> str:
        """String representation."""
        has_key = bool(self.api_key)
        return f"FinnhubNewsFetcher(authenticated={has_key})"
