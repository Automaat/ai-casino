"""Web search fetcher using DuckDuckGo."""

import contextlib
import hashlib
from datetime import datetime
from enum import StrEnum

from ddgs import DDGS
from loguru import logger
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.cache.memory import MemoryTTLCache

HTTP_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception) & retry_if_not_exception_type(ValueError),
    reraise=True,
    before_sleep=lambda retry_state: logger.warning(
        f"Retry {retry_state.attempt_number} after {retry_state.outcome.exception()}"
    ),
)

# Cache TTLs in seconds
NEWS_CACHE_TTL = 3600  # 1 hour
GENERAL_CACHE_TTL = 86400  # 24 hours


class SearchType(StrEnum):
    """Type of web search."""

    GENERAL = "general"
    NEWS = "news"


class WebSearchResult(BaseModel):
    """Single web search result."""

    title: str
    url: str
    body: str
    source: str | None = None
    published_at: datetime | None = None


class WebSearchResponse(BaseModel):
    """Web search response with multiple results."""

    query: str
    search_type: SearchType
    results: list[WebSearchResult]
    fetched_at: datetime


class WebSearchFetcher:
    """Fetch web search results using DuckDuckGo."""

    def __init__(self) -> None:
        """Initialize web search fetcher."""
        self._cache = MemoryTTLCache()
        logger.info("Initialized WebSearchFetcher")

    def _cache_key(self, query: str, search_type: SearchType) -> str:
        """Generate cache key from query and search type.

        Args:
            query: Search query
            search_type: Type of search

        Returns:
            Cache key string
        """
        raw = f"{query}:{search_type.value}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def _get_ttl(self, search_type: SearchType) -> int:
        """Get cache TTL for search type.

        Args:
            search_type: Type of search

        Returns:
            TTL in seconds
        """
        return NEWS_CACHE_TTL if search_type == SearchType.NEWS else GENERAL_CACHE_TTL

    @HTTP_RETRY
    def search(self, query: str, max_results: int = 10) -> WebSearchResponse:
        """Perform general web search.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            WebSearchResponse with results
        """
        logger.info(f"Web search: '{query}' (max_results={max_results})")

        cache_key = self._cache_key(query, SearchType.GENERAL)
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for query: '{query}'")
            return WebSearchResponse.model_validate(cached)

        try:
            with DDGS() as ddgs:
                raw_results = list(ddgs.text(query, max_results=max_results))

            results = [
                WebSearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    body=r.get("body", ""),
                )
                for r in raw_results
            ]

            response = WebSearchResponse(
                query=query,
                search_type=SearchType.GENERAL,
                results=results,
                fetched_at=datetime.now(),
            )

            self._cache.set(
                cache_key, response.model_dump(mode="json"), expire=self._get_ttl(SearchType.GENERAL)
            )
            logger.info(f"Fetched {len(results)} search results")
            return response

        except Exception as e:
            logger.opt(exception=True).error(f"Web search failed: {e}")
            raise

    @HTTP_RETRY
    def search_news(self, query: str, max_results: int = 10) -> WebSearchResponse:
        """Perform news-specific web search.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            WebSearchResponse with news results
        """
        logger.info(f"News search: '{query}' (max_results={max_results})")

        cache_key = self._cache_key(query, SearchType.NEWS)
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for news query: '{query}'")
            return WebSearchResponse.model_validate(cached)

        try:
            with DDGS() as ddgs:
                raw_results = list(ddgs.news(query, max_results=max_results))

            results = []
            for r in raw_results:
                published_at = None
                if date_str := r.get("date"):
                    with contextlib.suppress(ValueError):
                        published_at = datetime.fromisoformat(date_str)

                results.append(
                    WebSearchResult(
                        title=r.get("title", ""),
                        url=r.get("url", ""),
                        body=r.get("body", ""),
                        source=r.get("source"),
                        published_at=published_at,
                    )
                )

            response = WebSearchResponse(
                query=query,
                search_type=SearchType.NEWS,
                results=results,
                fetched_at=datetime.now(),
            )

            self._cache.set(
                cache_key, response.model_dump(mode="json"), expire=self._get_ttl(SearchType.NEWS)
            )
            logger.info(f"Fetched {len(results)} news results")
            return response

        except Exception as e:
            logger.opt(exception=True).error(f"News search failed: {e}")
            raise

    def clear_cache(self) -> None:
        """Clear all cached search results."""
        self._cache.clear()
        logger.info("Cleared web search cache")

    def __repr__(self) -> str:
        """String representation."""
        return "WebSearchFetcher()"
