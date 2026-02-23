"""Tests for WebSearchFetcher."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.data.websearch import (
    SearchType,
    WebSearchFetcher,
    WebSearchResponse,
    WebSearchResult,
)


@pytest.fixture
def mock_ddgs():
    """Mock DuckDuckGo search client."""
    with patch("src.data.websearch.DDGS") as mock:
        mock_instance = MagicMock()
        mock.__enter__ = MagicMock(return_value=mock_instance)
        mock.__exit__ = MagicMock(return_value=False)
        mock.return_value.__enter__ = MagicMock(return_value=mock_instance)
        mock.return_value.__exit__ = MagicMock(return_value=False)
        yield mock_instance


@pytest.fixture
def fetcher():
    """Create WebSearchFetcher."""
    return WebSearchFetcher()


class TestWebSearchFetcher:
    """Tests for WebSearchFetcher."""

    def test_cache_key_deterministic(self, fetcher):
        """Test cache key generation is deterministic."""
        key1 = fetcher._cache_key("AAPL stock", SearchType.GENERAL)
        key2 = fetcher._cache_key("AAPL stock", SearchType.GENERAL)
        key3 = fetcher._cache_key("AAPL stock", SearchType.NEWS)

        assert key1 == key2
        assert key1 != key3

    def test_search_returns_results(self, fetcher, mock_ddgs):
        """Test general web search returns results."""
        mock_ddgs.text.return_value = [
            {
                "title": "AAPL News",
                "href": "https://example.com/aapl",
                "body": "Apple Inc stock analysis",
            },
            {
                "title": "Apple Earnings",
                "href": "https://example.com/earnings",
                "body": "Quarterly report details",
            },
        ]

        response = fetcher.search("AAPL stock", max_results=5)

        assert isinstance(response, WebSearchResponse)
        assert response.query == "AAPL stock"
        assert response.search_type == SearchType.GENERAL
        assert len(response.results) == 2
        assert response.results[0].title == "AAPL News"
        assert response.results[0].url == "https://example.com/aapl"

    def test_search_news_returns_results(self, fetcher, mock_ddgs):
        """Test news search returns results with dates."""
        mock_ddgs.news.return_value = [
            {
                "title": "Breaking: Apple announcement",
                "url": "https://news.example.com/apple",
                "body": "New product launch",
                "source": "Reuters",
                "date": "2024-01-15T10:00:00+00:00",
            },
        ]

        response = fetcher.search_news("AAPL news", max_results=5)

        assert isinstance(response, WebSearchResponse)
        assert response.search_type == SearchType.NEWS
        assert len(response.results) == 1
        assert response.results[0].source == "Reuters"
        assert response.results[0].published_at is not None

    def test_search_uses_cache(self, fetcher, mock_ddgs):
        """Test that repeated searches use cache."""
        mock_ddgs.text.return_value = [{"title": "Result", "href": "https://example.com", "body": "Content"}]

        fetcher.search("AAPL", max_results=5)
        fetcher.search("AAPL", max_results=5)

        mock_ddgs.text.assert_called_once()

    def test_search_empty_results(self, fetcher, mock_ddgs):
        """Test handling of empty search results."""
        mock_ddgs.text.return_value = []

        response = fetcher.search("nonexistent query xyz", max_results=5)

        assert len(response.results) == 0

    def test_clear_cache(self, fetcher, mock_ddgs):
        """Test cache clearing."""
        mock_ddgs.text.return_value = [{"title": "Result", "href": "https://example.com", "body": "Content"}]

        fetcher.search("AAPL", max_results=5)
        fetcher.clear_cache()
        fetcher.search("AAPL", max_results=5)

        assert mock_ddgs.text.call_count == 2

    def test_repr(self, fetcher):
        """Test string representation."""
        repr_str = repr(fetcher)
        assert "WebSearchFetcher" in repr_str


class TestWebSearchResult:
    """Tests for WebSearchResult model."""

    def test_create_minimal(self):
        """Test creating result with minimal fields."""
        result = WebSearchResult(
            title="Test",
            url="https://example.com",
            body="Content",
        )
        assert result.source is None
        assert result.published_at is None

    def test_create_full(self):
        """Test creating result with all fields."""
        result = WebSearchResult(
            title="Test",
            url="https://example.com",
            body="Content",
            source="Reuters",
            published_at=datetime(2024, 1, 15, 10, 0),
        )
        assert result.source == "Reuters"
        assert result.published_at is not None
        assert result.published_at.year == 2024


class TestWebSearchResponse:
    """Tests for WebSearchResponse model."""

    def test_create_response(self):
        """Test creating response."""
        response = WebSearchResponse(
            query="AAPL",
            search_type=SearchType.GENERAL,
            results=[WebSearchResult(title="Test", url="https://example.com", body="Content")],
            fetched_at=datetime.now(),
        )
        assert response.query == "AAPL"
        assert len(response.results) == 1
