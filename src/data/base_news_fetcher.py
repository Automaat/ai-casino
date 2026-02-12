"""Base protocol for news fetchers."""

from typing import Protocol

from src.data.news import NewsArticle


class BaseNewsFetcher(Protocol):
    """Protocol for news fetchers."""

    async def afetch_company_news(self, symbol: str, limit: int) -> list[NewsArticle]:
        """Fetch company-specific news asynchronously."""
        ...

    async def afetch_market_news(self, limit: int) -> list[NewsArticle]:
        """Fetch general market news asynchronously."""
        ...

    def get_source_name(self) -> str:
        """Return source identifier (e.g., 'marketaux', 'finnhub')."""
        ...
