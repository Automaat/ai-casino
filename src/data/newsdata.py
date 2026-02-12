"""NewsData.io news fetcher."""

from datetime import UTC, datetime

import httpx
from dotenv import load_dotenv
from loguru import logger

from src.cache.historical import HistoricalCache
from src.data.news import HTTP_RETRY, NewsArticle

load_dotenv()


class NewsDataFetcher:
    """Fetch news from NewsData.io API (200 credits/day free tier)."""

    BASE_URL = "https://newsdata.io/api/1/news"

    def __init__(
        self,
        api_key: str | None = None,
        historical_cache: HistoricalCache | None = None,
    ) -> None:
        """Initialize NewsData fetcher.

        Args:
            api_key: NewsData.io API key
            historical_cache: Optional permanent cache for news articles
        """
        self.api_key = api_key or ""
        self._cache = historical_cache
        if not self.api_key:
            logger.warning("newsdata_api_key not set in config - API calls may be limited")

    @HTTP_RETRY
    async def afetch_company_news(self, symbol: str, limit: int = 10) -> list[NewsArticle]:
        """Fetch company-specific news asynchronously.

        Args:
            symbol: Stock ticker symbol
            limit: Maximum number of articles

        Returns:
            List of NewsArticle objects (max 10 per API limitation)
        """
        logger.info(f"Fetching {limit} news articles for {symbol} from NewsData.io")

        params = {
            "apikey": self.api_key,
            "q": f"{symbol} stock",
            "language": "en",
            "size": min(limit, 10),  # NewsData.io max 10 per request
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(self.BASE_URL, params=params)
                response.raise_for_status()

                data = response.json()
                articles = []

                for item in data.get("results", []):
                    if not item.get("link") or not item.get("title"):
                        continue

                    try:
                        published_at = datetime.fromisoformat(item.get("pubDate", ""))
                    except (ValueError, AttributeError, TypeError):
                        published_at = datetime.now(UTC)
                        logger.warning(f"Invalid date format: {item.get('pubDate')}")

                    articles.append(
                        NewsArticle(
                            title=item.get("title", ""),
                            description=item.get("description", ""),
                            url=item.get("link", ""),
                            published_at=published_at,
                            source=item.get("source_name", "newsdata"),
                        )
                    )

                logger.info(f"Fetched {len(articles)} articles from NewsData.io")

                if self._cache and articles:
                    self._cache.store_news_articles(symbol, articles)

                return articles

        except httpx.HTTPError as e:
            logger.error(f"NewsData.io fetch failed: {e}")
            raise

    @HTTP_RETRY
    async def afetch_market_news(self, limit: int = 20) -> list[NewsArticle]:
        """Fetch general market news asynchronously.

        Args:
            limit: Maximum number of articles

        Returns:
            List of NewsArticle objects (max 10 per API limitation)
        """
        logger.info(f"Fetching {limit} general market news from NewsData.io")

        params = {
            "apikey": self.api_key,
            "q": "stock market OR finance",
            "language": "en",
            "size": min(limit, 10),
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(self.BASE_URL, params=params)
                response.raise_for_status()

                data = response.json()
                articles = []

                for item in data.get("results", []):
                    if not item.get("link") or not item.get("title"):
                        continue

                    try:
                        published_at = datetime.fromisoformat(item.get("pubDate", ""))
                    except (ValueError, AttributeError, TypeError):
                        published_at = datetime.now(UTC)
                        logger.warning(f"Invalid date format: {item.get('pubDate')}")

                    articles.append(
                        NewsArticle(
                            title=item.get("title", ""),
                            description=item.get("description", ""),
                            url=item.get("link", ""),
                            published_at=published_at,
                            source=item.get("source_name", "newsdata"),
                        )
                    )

                logger.info(f"Fetched {len(articles)} articles from NewsData.io")
                return articles

        except httpx.HTTPError as e:
            logger.error(f"NewsData.io market news fetch failed: {e}")
            raise

    def get_source_name(self) -> str:
        """Return source identifier."""
        return "newsdata"

    def __repr__(self) -> str:
        """String representation."""
        has_key = bool(self.api_key)
        return f"NewsDataFetcher(authenticated={has_key})"
