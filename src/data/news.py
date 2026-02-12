"""News data fetcher for financial news."""

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING

import httpx
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.cache.historical import HistoricalCache
from src.metrics.execution import timed_operation

if TYPE_CHECKING:
    from src.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerRegistry

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
        circuit_breaker: CircuitBreaker | None = None,
        circuit_breaker_registry: CircuitBreakerRegistry | None = None,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
    ) -> None:
        """Initialize news fetcher.

        Args:
            api_key: Marketaux API key. Defaults to env variable.
            historical_cache: Optional permanent cache for news articles
            circuit_breaker: Optional circuit breaker instance (direct injection)
            circuit_breaker_registry: Optional registry for lazy breaker creation
            circuit_breaker_config: Optional config for lazy breaker creation
        """
        self.api_key = api_key or ""
        self._cache = historical_cache
        self._circuit_breaker = circuit_breaker
        self._circuit_breaker_registry = circuit_breaker_registry
        self._circuit_breaker_config = circuit_breaker_config
        if not self.api_key:
            logger.warning("marketaux_api_key not set in config - API calls may be limited")

    async def _ensure_circuit_breaker(self) -> None:
        """Lazily initialize circuit breaker from registry."""
        if (
            self._circuit_breaker is None
            and self._circuit_breaker_registry is not None
            and self._circuit_breaker_config is not None
        ):
            self._circuit_breaker = await self._circuit_breaker_registry.get_breaker(
                "marketaux",
                self._circuit_breaker_config,
            )
            logger.debug("Initialized circuit breaker for marketaux")

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
        await self._ensure_circuit_breaker()

        if self._circuit_breaker:
            return await self._circuit_breaker.call(
                self._fetch_with_breaker_company,
                symbol,
                limit,
            )
        return await asyncio.to_thread(self._fetch_company_news_sync, symbol, limit)

    async def afetch_market_news(self, limit: int = 20) -> list[NewsArticle]:
        """Fetch general market news asynchronously.

        Args:
            limit: Maximum number of articles

        Returns:
            List of NewsArticle objects
        """
        await self._ensure_circuit_breaker()

        if self._circuit_breaker:
            return await self._circuit_breaker.call(
                self._fetch_with_breaker_market,
                limit,
            )
        return await asyncio.to_thread(self._fetch_market_news_sync, limit)

    async def _fetch_with_breaker_company(self, symbol: str, limit: int) -> list[NewsArticle]:
        """Wrapper for circuit breaker - company news."""
        return await asyncio.to_thread(self._fetch_company_news_sync, symbol, limit)

    async def _fetch_with_breaker_market(self, limit: int) -> list[NewsArticle]:
        """Wrapper for circuit breaker - market news."""
        return await asyncio.to_thread(self._fetch_market_news_sync, limit)

    def get_source_name(self) -> str:
        """Return source identifier."""
        return "marketaux"

    @HTTP_RETRY
    def _fetch_company_news_sync(
        self,
        symbol: str,
        limit: int = 10,
    ) -> list[NewsArticle]:
        """Fetch recent news for a company (sync implementation).

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
                with httpx.Client(timeout=30.0) as client:
                    response = client.get(self.BASE_URL, params=params)

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

    def fetch_company_news(
        self,
        symbol: str,
        limit: int = 10,
    ) -> list[NewsArticle]:
        """Fetch recent news for a company (deprecated, use afetch_company_news).

        Args:
            symbol: Stock ticker symbol
            limit: Maximum number of articles

        Returns:
            List of NewsArticle objects
        """
        return self._fetch_company_news_sync(symbol, limit)

    @HTTP_RETRY
    def _fetch_market_news_sync(self, limit: int = 20) -> list[NewsArticle]:
        """Fetch general market news (sync implementation).

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
            with httpx.Client(timeout=30.0) as client:
                response = client.get(self.BASE_URL, params=params)
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

    def fetch_market_news(self, limit: int = 20) -> list[NewsArticle]:
        """Fetch general market news (deprecated, use afetch_market_news).

        Args:
            limit: Maximum number of articles

        Returns:
            List of NewsArticle objects
        """
        return self._fetch_market_news_sync(limit)

    def __repr__(self) -> str:
        """String representation."""
        has_key = bool(self.api_key)
        return f"NewsFetcher(authenticated={has_key})"
