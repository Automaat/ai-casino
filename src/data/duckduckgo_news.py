"""DuckDuckGo news fetcher."""

import asyncio
from datetime import UTC, datetime

from ddgs import DDGS
from loguru import logger

from src.cache.historical import HistoricalCache
from src.data.news import NewsArticle


class DuckDuckGoNewsFetcher:
    """Fetch news via DuckDuckGo search (no API key, unlimited)."""

    def __init__(self, historical_cache: HistoricalCache | None = None) -> None:
        """Initialize DuckDuckGo news fetcher.

        Args:
            historical_cache: Optional permanent cache for news articles
        """
        self._cache = historical_cache

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

    def _fetch_company_sync(self, symbol: str, limit: int) -> list[NewsArticle]:
        """Fetch company-specific news (sync implementation).

        Args:
            symbol: Stock ticker symbol
            limit: Maximum number of articles

        Returns:
            List of NewsArticle objects
        """
        logger.info(f"Fetching {limit} news articles for {symbol} from DuckDuckGo")

        try:
            with DDGS() as ddgs:
                results = list(ddgs.news(query=f"{symbol} stock", max_results=limit))

            articles = []
            for item in results:
                if not item.get("url") or not item.get("title"):
                    continue

                # Parse date - DDG returns ISO string or timestamp
                published_at = self._parse_date(item.get("date"))

                articles.append(
                    NewsArticle(
                        title=item.get("title", ""),
                        description=item.get("body", ""),
                        url=item.get("url", ""),
                        published_at=published_at,
                        source=item.get("source", "duckduckgo"),
                    )
                )

            logger.info(f"Fetched {len(articles)} articles from DuckDuckGo")

            if self._cache and articles:
                self._cache.store_news_articles(symbol, articles)

            return articles

        except Exception as e:
            logger.error(f"DuckDuckGo fetch failed: {e}")
            raise

    def _fetch_market_sync(self, limit: int) -> list[NewsArticle]:
        """Fetch general market news (sync implementation).

        Args:
            limit: Maximum number of articles

        Returns:
            List of NewsArticle objects
        """
        logger.info(f"Fetching {limit} general market news from DuckDuckGo")

        try:
            with DDGS() as ddgs:
                results = list(ddgs.news(query="stock market finance", max_results=limit))

            articles = []
            for item in results:
                if not item.get("url") or not item.get("title"):
                    continue

                published_at = self._parse_date(item.get("date"))

                articles.append(
                    NewsArticle(
                        title=item.get("title", ""),
                        description=item.get("body", ""),
                        url=item.get("url", ""),
                        published_at=published_at,
                        source=item.get("source", "duckduckgo"),
                    )
                )

            logger.info(f"Fetched {len(articles)} articles from DuckDuckGo")
            return articles

        except Exception as e:
            logger.error(f"DuckDuckGo market news fetch failed: {e}")
            raise

    def get_source_name(self) -> str:
        """Return source identifier."""
        return "duckduckgo"

    def _parse_date(self, date_str: str | None) -> datetime:
        """Parse date from various formats.

        Args:
            date_str: Date string (ISO format or timestamp)

        Returns:
            Parsed datetime object with fallback to now()
        """
        if not date_str:
            return datetime.now(UTC)

        try:
            # Try ISO format
            return datetime.fromisoformat(date_str)
        except (ValueError, AttributeError):
            pass

        try:
            # Try timestamp
            return datetime.fromtimestamp(float(date_str), tz=UTC)
        except (ValueError, TypeError):
            logger.warning(f"Could not parse date: {date_str}, using now()")
            return datetime.now(UTC)

    def __repr__(self) -> str:
        """String representation."""
        return "DuckDuckGoNewsFetcher()"
