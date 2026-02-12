"""Tests for GetNewsTool."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.data.news import NewsArticle
from src.tools.news import GetNewsTool


@pytest.fixture
def sample_articles():
    """Create sample news articles."""
    return [
        NewsArticle(
            title="Apple Announces New Product",
            description="Apple Inc. unveiled its latest innovative product today.",
            url="https://news.example.com/apple-product",
            published_at=datetime(2024, 1, 15, 10, 30),
            source="TechNews",
        ),
        NewsArticle(
            title="Apple Stock Rises",
            description="Apple shares climbed 3% following strong earnings report.",
            url="https://finance.example.com/aapl-stock",
            published_at=datetime(2024, 1, 14, 9, 0),
            source="MarketWatch",
        ),
    ]


class TestGetNewsTool:
    """Tests for GetNewsTool."""

    def test_name(self, test_container_full):
        """Test tool name."""
        tool = GetNewsTool(container=test_container_full)
        assert tool.name == "get_news"

    def test_requires_confirmation(self, test_container_full):
        """Test that tool doesn't require confirmation."""
        tool = GetNewsTool(container=test_container_full)
        assert tool.requires_confirmation is False

    def test_get_tool_definition(self, test_container_full):
        """Test tool definition format."""
        tool = GetNewsTool(container=test_container_full)
        definition = tool.get_tool_definition().model_dump(mode="json", by_alias=True, exclude_none=True)

        assert definition["type"] == "function"
        assert definition["function"]["name"] == "get_news"
        assert "description" in definition["function"]

        params = definition["function"]["parameters"]
        assert "symbol" in params["properties"]
        assert "limit" in params["properties"]
        assert "symbol" in params["required"]

    def test_execute_success(self, test_container_full, sample_articles):
        """Test successful execution."""
        tool = GetNewsTool(container=test_container_full)

        mock_fetcher = MagicMock()
        mock_fetcher.afetch_company_news = AsyncMock(return_value=sample_articles)
        test_container_full.news_fetcher.override(mock_fetcher)

        result = tool.execute(symbol="AAPL", limit=5)

        assert "AAPL" in result
        assert "Apple Announces New Product" in result
        assert "TechNews" in result
        mock_fetcher.afetch_company_news.assert_called_once_with("AAPL", limit=5)

    def test_execute_default_limit(self, test_container_full, sample_articles):
        """Test execution with default limit."""
        tool = GetNewsTool(container=test_container_full)

        mock_fetcher = MagicMock()
        mock_fetcher.afetch_company_news = AsyncMock(return_value=sample_articles)
        test_container_full.news_fetcher.override(mock_fetcher)

        tool.execute(symbol="AAPL")

        mock_fetcher.afetch_company_news.assert_called_once_with("AAPL", limit=5)

    def test_execute_uppercase_symbol(self, test_container_full, sample_articles):
        """Test that symbol is uppercased."""
        tool = GetNewsTool(container=test_container_full)

        mock_fetcher = MagicMock()
        mock_fetcher.afetch_company_news = AsyncMock(return_value=sample_articles)
        test_container_full.news_fetcher.override(mock_fetcher)

        tool.execute(symbol="aapl", limit=5)

        mock_fetcher.afetch_company_news.assert_called_once_with("AAPL", limit=5)

    def test_execute_empty_results(self, test_container_full):
        """Test handling empty news results."""
        tool = GetNewsTool(container=test_container_full)

        mock_fetcher = MagicMock()
        mock_fetcher.afetch_company_news = AsyncMock(return_value=[])
        test_container_full.news_fetcher.override(mock_fetcher)

        result = tool.execute(symbol="AAPL")

        assert "No recent news found" in result

    def test_execute_error_handling(self, test_container_full):
        """Test error handling on fetch failure."""
        tool = GetNewsTool(container=test_container_full)

        mock_fetcher = MagicMock()
        mock_fetcher.afetch_company_news = AsyncMock(side_effect=Exception("API error"))
        test_container_full.news_fetcher.override(mock_fetcher)

        result = tool.execute(symbol="INVALID")

        assert "Failed to fetch news" in result
        assert "API error" in result

    def test_format_articles_truncates_long_description(self, test_container_full):
        """Test that long descriptions are truncated."""
        tool = GetNewsTool(container=test_container_full)
        long_description = "A" * 500
        articles = [
            NewsArticle(
                title="Test Article",
                description=long_description,
                url="https://example.com",
                published_at=datetime.now(),
                source="TestSource",
            ),
        ]

        result = tool._format_articles("AAPL", articles)

        assert "..." in result
        assert len(result) < len(long_description) + 200

    def test_format_articles_content(self, test_container_full, sample_articles):
        """Test formatted articles content."""
        tool = GetNewsTool(container=test_container_full)
        result = tool._format_articles("AAPL", sample_articles)

        assert "# AAPL Recent News" in result
        assert "## 1. Apple Announces New Product" in result
        assert "## 2. Apple Stock Rises" in result
        assert "TechNews" in result
        assert "MarketWatch" in result

    def test_repr(self, test_container_full):
        """Test string representation."""
        tool = GetNewsTool(container=test_container_full)
        repr_str = repr(tool)
        assert "GetNewsTool" in repr_str
