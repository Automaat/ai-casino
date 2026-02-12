"""News data fetcher for financial news."""

from datetime import datetime

import httpx
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.cache.historical import HistoricalCache
from src.metrics.execution import timed_operation

load_dotenv()

HTTP_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(
        (
            httpx.ReadTimeout,
            httpx.ConnectError,
            httpx.TimeoutException,
        )
    ),
    reraise=True,
    before_sleep=lambda retry_state: logger.warning(
        f"Retry {retry_state.attempt_number} after {retry_state.outcome.exception()}"
    ),
)


class NewsArticle(BaseModel):
    """Single news article."""

    title: str
    description: str
    url: str
    published_at: datetime
    source: str


class NewsFetcher:
    """Fetch financial news from Marketaux API."""

    BASE_URL = "https://api.marketaux.com/v1/news/all"

    def __init__(
        self,
        api_key: str | None = None,
        historical_cache: HistoricalCache | None = None,
    ) -> None:
        """Initialize news fetcher.

        Args:
            api_key: Marketaux API key. Defaults to env variable.
            historical_cache: Optional permanent cache for news articles
        """
        self.api_key = api_key or ""
        self._cache = historical_cache
        if not self.api_key:
            logger.warning("marketaux_api_key not set in config - API calls may be limited")

    def _log_rate_limit_headers(self, response: httpx.Response, symbol: str = "") -> None:
        """Log Marketaux rate limit headers.

        Args:
            response: httpx Response object
            symbol: Stock symbol (for context)
        """
        usage_limit = response.headers.get("X-UsageLimit-Limit")
        rate_limit = response.headers.get("X-RateLimit-Limit")

        if usage_limit:
            logger.warning(f"Marketaux usage limit header (symbol={symbol}): {usage_limit}")
        if rate_limit:
            logger.info(f"Marketaux rate limit header (symbol={symbol}): {rate_limit}")

    @HTTP_RETRY
    async def afetch_company_news(
        self,
        symbol: str,
        limit: int = 10,
    ) -> list[NewsArticle]:
        """Fetch recent news for a company asynchronously.

        Args:
            symbol: Stock ticker symbol
            limit: Maximum number of articles

        Returns:
            List of NewsArticle objects
        """
        logger.info(f"Fetching {limit} news articles for {symbol}")

        params = {
            "symbols": symbol,
            "filter_entities": "true",
            "limit": limit,
            "language": "en",
        }

        if self.api_key:
            params["api_token"] = self.api_key

        with timed_operation("news_fetch", source="marketaux"):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(self.BASE_URL, params=params)

                    # Log rate limit headers before raising
                    self._log_rate_limit_headers(response, symbol)

                    response.raise_for_status()

                    data = response.json()
                    articles = []

                    for item in data.get("data", []):
                        articles.append(
                            NewsArticle(
                                title=item.get("title", ""),
                                description=item.get("description", ""),
                                url=item.get("url", ""),
                                published_at=datetime.fromisoformat(item.get("published_at", "")),
                                source=item.get("source", ""),
                            )
                        )

                    logger.info(f"Fetched {len(articles)} articles")

                    if self._cache and articles:
                        self._cache.store_news_articles(symbol, articles)

                    return articles

            except httpx.HTTPStatusError as e:
                self._log_rate_limit_headers(e.response, symbol)
                logger.opt(exception=True).error(f"News fetch failed: {e}")
                raise
            except httpx.HTTPError as e:
                logger.opt(exception=True).error(f"News fetch failed: {e}")
                raise

    @HTTP_RETRY
    async def afetch_market_news(self, limit: int = 20) -> list[NewsArticle]:
        """Fetch general market news asynchronously.

        Args:
            limit: Maximum number of articles

        Returns:
            List of NewsArticle objects
        """
        logger.info(f"Fetching {limit} general market news articles")

        params = {
            "filter_entities": "true",
            "limit": limit,
            "language": "en",
        }

        if self.api_key:
            params["api_token"] = self.api_key

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(self.BASE_URL, params=params)
                response.raise_for_status()

                data = response.json()
                articles = []

                for item in data.get("data", []):
                    articles.append(
                        NewsArticle(
                            title=item.get("title", ""),
                            description=item.get("description", ""),
                            url=item.get("url", ""),
                            published_at=datetime.fromisoformat(item.get("published_at", "")),
                            source=item.get("source", ""),
                        )
                    )

                logger.info(f"Fetched {len(articles)} articles")
                return articles

        except httpx.HTTPError as e:
            logger.opt(exception=True).error(f"Market news fetch failed: {e}")
            raise

    def get_source_name(self) -> str:
        """Return source identifier."""
        return "marketaux"

    def __repr__(self) -> str:
        """String representation."""
        has_key = bool(self.api_key)
        return f"NewsFetcher(authenticated={has_key})"
