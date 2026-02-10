"""Tests for WebSearchTool."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.data.websearch import SearchType, WebSearchResponse, WebSearchResult
from src.tools.websearch import WebSearchTool


@pytest.fixture
def mock_general_response():
    """Mock general search response."""
    return WebSearchResponse(
        query="AAPL stock",
        search_type=SearchType.GENERAL,
        results=[
            WebSearchResult(
                title="Apple Stock Analysis",
                url="https://example.com/aapl",
                body="Detailed analysis of Apple Inc stock performance",
            ),
            WebSearchResult(
                title="AAPL Price Target",
                url="https://example.com/target",
                body="Analysts raise price target for Apple",
            ),
        ],
        fetched_at=datetime.now(),
    )


@pytest.fixture
def mock_news_response():
    """Mock news search response."""
    return WebSearchResponse(
        query="AAPL news",
        search_type=SearchType.NEWS,
        results=[
            WebSearchResult(
                title="Apple announces new product",
                url="https://news.example.com/apple",
                body="Apple unveils new product line",
                source="Reuters",
                published_at=datetime(2024, 1, 15, 10, 0),
            ),
        ],
        fetched_at=datetime.now(),
    )


class TestWebSearchTool:
    """Tests for WebSearchTool."""

    def test_tool_name(self, test_container_full):
        """Test tool name property."""
        tool = WebSearchTool(container=test_container_full)
        assert tool.name == "web_search"

    def test_get_tool_definition(self, test_container_full):
        """Test tool definition format."""
        tool = WebSearchTool(container=test_container_full)
        definition = tool.get_tool_definition()

        assert definition["type"] == "function"
        assert definition["function"]["name"] == "web_search"
        assert "description" in definition["function"]
        assert "parameters" in definition["function"]

        params = definition["function"]["parameters"]
        assert "query" in params["properties"]
        assert "search_type" in params["properties"]
        assert params["properties"]["search_type"]["enum"] == ["general", "news"]

    def test_execute_general_search(self, test_container_full, mock_general_response):
        """Test executing general search."""
        mock_fetcher = MagicMock()
        mock_fetcher.search.return_value = mock_general_response
        test_container_full.websearch_fetcher.override(mock_fetcher)

        tool = WebSearchTool(container=test_container_full)

        result = tool.execute(query="AAPL stock", search_type="general", max_results=5)

        mock_fetcher.search.assert_called_once_with("AAPL stock", max_results=5)
        assert "Search results for 'AAPL stock'" in result
        assert "Apple Stock Analysis" in result
        assert "https://example.com/aapl" in result

    def test_execute_news_search(self, test_container_full, mock_news_response):
        """Test executing news search."""
        mock_fetcher = MagicMock()
        mock_fetcher.search_news.return_value = mock_news_response
        test_container_full.websearch_fetcher.override(mock_fetcher)

        tool = WebSearchTool(container=test_container_full)

        result = tool.execute(query="AAPL news", search_type="news", max_results=5)

        mock_fetcher.search_news.assert_called_once_with("AAPL news", max_results=5)
        assert "news" in result
        assert "Apple announces new product" in result
        assert "Reuters" in result

    def test_execute_empty_results(self, test_container_full):
        """Test handling empty search results."""
        empty_response = WebSearchResponse(
            query="nonexistent",
            search_type=SearchType.GENERAL,
            results=[],
            fetched_at=datetime.now(),
        )
        mock_fetcher = MagicMock()
        mock_fetcher.search.return_value = empty_response
        test_container_full.websearch_fetcher.override(mock_fetcher)

        tool = WebSearchTool(container=test_container_full)
        result = tool.execute(query="nonexistent query", search_type="general")

        assert "No results found" in result

    def test_execute_truncates_long_body(self, test_container_full):
        """Test that long body text is truncated."""
        long_body = "A" * 500
        long_response = WebSearchResponse(
            query="test",
            search_type=SearchType.GENERAL,
            results=[
                WebSearchResult(
                    title="Test",
                    url="https://example.com",
                    body=long_body,
                ),
            ],
            fetched_at=datetime.now(),
        )
        mock_fetcher = MagicMock()
        mock_fetcher.search.return_value = long_response
        test_container_full.websearch_fetcher.override(mock_fetcher)

        tool = WebSearchTool(container=test_container_full)

        result = tool.execute(query="test", search_type="general")

        assert "..." in result
        assert len(result) < len(long_body) + 200

    def test_repr(self, test_container_full):
        """Test string representation."""
        tool = WebSearchTool(container=test_container_full)
        repr_str = repr(tool)
        assert "WebSearchTool" in repr_str
