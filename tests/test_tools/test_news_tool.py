"""Tests for GetNewsTool."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.data.news import NewsArticle
from src.tools.news import GetNewsTool


@pytest.fixture
def tool():
    """Create GetNewsTool."""
    return GetNewsTool()


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

    def test_name(self, tool):
        """Test tool name."""
        assert tool.name == "get_news"

    def test_requires_confirmation(self, tool):
        """Test that tool doesn't require confirmation."""
        assert tool.requires_confirmation is False

    def test_get_tool_definition(self, tool):
        """Test tool definition format."""
        definition = tool.get_tool_definition()

        assert definition["type"] == "function"
        assert definition["function"]["name"] == "get_news"
        assert "description" in definition["function"]

        params = definition["function"]["parameters"]
        assert "symbol" in params["properties"]
        assert "limit" in params["properties"]
        assert "symbol" in params["required"]

    def test_execute_success(self, tool, sample_articles):
        """Test successful execution."""
        with patch("src.data.news.NewsFetcher") as mock_fetcher_cls:
            mock_instance = MagicMock()
            mock_instance.fetch_company_news.return_value = sample_articles
            mock_fetcher_cls.return_value = mock_instance

            result = tool.execute("AAPL", limit=5)

            assert "AAPL" in result
            assert "Apple Announces New Product" in result
            assert "TechNews" in result
            mock_instance.fetch_company_news.assert_called_once_with("AAPL", limit=5)

    def test_execute_default_limit(self, tool, sample_articles):
        """Test execution with default limit."""
        with patch("src.data.news.NewsFetcher") as mock_fetcher_cls:
            mock_instance = MagicMock()
            mock_instance.fetch_company_news.return_value = sample_articles
            mock_fetcher_cls.return_value = mock_instance

            tool.execute("AAPL")

            mock_instance.fetch_company_news.assert_called_once_with("AAPL", limit=5)

    def test_execute_uppercase_symbol(self, tool, sample_articles):
        """Test that symbol is uppercased."""
        with patch("src.data.news.NewsFetcher") as mock_fetcher_cls:
            mock_instance = MagicMock()
            mock_instance.fetch_company_news.return_value = sample_articles
            mock_fetcher_cls.return_value = mock_instance

            tool.execute("aapl", limit=5)

            mock_instance.fetch_company_news.assert_called_once_with("AAPL", limit=5)

    def test_execute_empty_results(self, tool):
        """Test handling empty news results."""
        with patch("src.data.news.NewsFetcher") as mock_fetcher_cls:
            mock_instance = MagicMock()
            mock_instance.fetch_company_news.return_value = []
            mock_fetcher_cls.return_value = mock_instance

            result = tool.execute("AAPL")

            assert "No recent news found" in result

    def test_execute_error_handling(self, tool):
        """Test error handling on fetch failure."""
        with patch("src.data.news.NewsFetcher") as mock_fetcher_cls:
            mock_instance = MagicMock()
            mock_instance.fetch_company_news.side_effect = Exception("API error")
            mock_fetcher_cls.return_value = mock_instance

            result = tool.execute("INVALID")

            assert "Failed to fetch news" in result
            assert "API error" in result

    def test_format_articles_truncates_long_description(self, tool):
        """Test that long descriptions are truncated."""
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

    def test_format_articles_content(self, tool, sample_articles):
        """Test formatted articles content."""
        result = tool._format_articles("AAPL", sample_articles)

        assert "# AAPL Recent News" in result
        assert "## 1. Apple Announces New Product" in result
        assert "## 2. Apple Stock Rises" in result
        assert "TechNews" in result
        assert "MarketWatch" in result

    def test_repr(self, tool):
        """Test string representation."""
        repr_str = repr(tool)
        assert "GetNewsTool" in repr_str
