"""Tests for FetchNewsHeadlinesTool."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from src.data.news import NewsArticle
from src.tools.game_plan.news_headlines import FetchNewsHeadlinesTool


@pytest.fixture
def mock_news_fetcher() -> AsyncMock:
    """Create mock news fetcher."""
    fetcher = AsyncMock()
    fetcher.afetch_market_news.return_value = [
        NewsArticle(
            title="Markets rally on jobs data",
            description="desc",
            url="https://example.com",
            published_at=datetime.now(UTC),
            source="Reuters",
        ),
    ]
    fetcher.afetch_company_news.return_value = [
        NewsArticle(
            title="AAPL beats earnings",
            description="desc",
            url="https://example.com",
            published_at=datetime.now(UTC),
            source="Bloomberg",
        ),
    ]
    return fetcher


class TestFetchNewsHeadlinesTool:
    """Tests for FetchNewsHeadlinesTool."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_fetches_market_news(self, mock_news_fetcher: AsyncMock) -> None:
        """Fetches and formats market news."""
        tool = FetchNewsHeadlinesTool(mock_news_fetcher)

        result = await tool.aexecute()

        assert "Markets rally" in result
        assert "Reuters" in result
        mock_news_fetcher.afetch_market_news.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_fetches_company_news(self, mock_news_fetcher: AsyncMock) -> None:
        """Fetches company-specific news when symbol given."""
        tool = FetchNewsHeadlinesTool(mock_news_fetcher)

        result = await tool.aexecute(symbol="AAPL")

        assert "AAPL" in result
        assert "beats earnings" in result
        mock_news_fetcher.afetch_company_news.assert_awaited_once_with("AAPL", limit=5)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_handles_fetch_failure(self, mock_news_fetcher: AsyncMock) -> None:
        """Gracefully handles news fetch failure."""
        mock_news_fetcher.afetch_market_news.side_effect = RuntimeError("API down")
        tool = FetchNewsHeadlinesTool(mock_news_fetcher)

        result = await tool.aexecute()

        assert "unavailable" in result.lower()

    @pytest.mark.unit
    def test_tool_definition(self, mock_news_fetcher: AsyncMock) -> None:
        """Tool definition has correct name."""
        tool = FetchNewsHeadlinesTool(mock_news_fetcher)
        defn = tool.get_tool_definition()

        assert defn.function.name == "fetch_news_headlines"
