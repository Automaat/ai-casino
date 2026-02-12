"""Tests for DuckDuckGo news fetcher."""

from datetime import UTC, datetime
from unittest.mock import Mock, patch

import pytest

from src.data.duckduckgo_news import DuckDuckGoNewsFetcher
from src.data.news import NewsArticle


@pytest.fixture
def sample_ddg_response():
    return [
        {
            "title": "Apple stock surges",
            "body": "Apple shares hit new high",
            "url": "https://example.com/article1",
            "date": "2024-01-15T10:30:00Z",
            "source": "TechNews",
        },
        {
            "title": "Tech sector analysis",
            "body": "Technology stocks see gains",
            "url": "https://example.com/article2",
            "date": "2024-01-15T12:00:00Z",
            "source": "FinanceDaily",
        },
    ]


def test_get_source_name():
    fetcher = DuckDuckGoNewsFetcher()
    assert fetcher.get_source_name() == "duckduckgo"


@pytest.mark.asyncio
async def test_afetch_company_news(sample_ddg_response):
    with patch("src.data.duckduckgo_news.DDGS") as mock_ddgs_class:
        mock_ddgs = Mock()
        mock_ddgs.news.return_value = sample_ddg_response
        mock_ddgs.__enter__ = Mock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = Mock(return_value=False)
        mock_ddgs_class.return_value = mock_ddgs

        fetcher = DuckDuckGoNewsFetcher()
        articles = await fetcher.afetch_company_news("AAPL", 10)

        assert len(articles) == 2
        assert all(isinstance(a, NewsArticle) for a in articles)
        assert articles[0].title == "Apple stock surges"
        assert articles[1].source == "FinanceDaily"


@pytest.mark.asyncio
async def test_afetch_market_news(sample_ddg_response):
    with patch("src.data.duckduckgo_news.DDGS") as mock_ddgs_class:
        mock_ddgs = Mock()
        mock_ddgs.news.return_value = sample_ddg_response
        mock_ddgs.__enter__ = Mock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = Mock(return_value=False)
        mock_ddgs_class.return_value = mock_ddgs

        fetcher = DuckDuckGoNewsFetcher()
        articles = await fetcher.afetch_market_news(limit=20)

        assert len(articles) == 2
        assert all(isinstance(a, NewsArticle) for a in articles)


def test_fetch_company_news_error():
    with patch("src.data.duckduckgo_news.DDGS") as mock_ddgs_class:
        mock_ddgs = Mock()
        mock_ddgs.news.side_effect = RuntimeError("Search error")
        mock_ddgs.__enter__ = Mock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = Mock(return_value=False)
        mock_ddgs_class.return_value = mock_ddgs

        fetcher = DuckDuckGoNewsFetcher()

        with pytest.raises(RuntimeError, match="Search error"):
            fetcher._fetch_company_sync("AAPL", 10)


def test_fetch_market_news_error():
    with patch("src.data.duckduckgo_news.DDGS") as mock_ddgs_class:
        mock_ddgs = Mock()
        mock_ddgs.news.side_effect = RuntimeError("Search error")
        mock_ddgs.__enter__ = Mock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = Mock(return_value=False)
        mock_ddgs_class.return_value = mock_ddgs

        fetcher = DuckDuckGoNewsFetcher()

        with pytest.raises(RuntimeError, match="Search error"):
            fetcher._fetch_market_sync(20)


def test_repr():
    fetcher = DuckDuckGoNewsFetcher()
    assert repr(fetcher) == "DuckDuckGoNewsFetcher()"


def test_fetch_skips_invalid_items():
    with patch("src.data.duckduckgo_news.DDGS") as mock_ddgs_class:
        mock_ddgs = Mock()
        mock_ddgs.news.return_value = [
            {"title": "Valid", "url": "https://example.com", "date": "2024-01-15T10:30:00Z"},
            {"title": "No URL", "date": "2024-01-15T10:30:00Z"},  # Missing URL
            {"url": "https://example.com/no-title", "date": "2024-01-15T10:30:00Z"},  # Missing title
            {"title": "Another valid", "url": "https://example.com/2", "date": "2024-01-15T10:30:00Z"},
        ]
        mock_ddgs.__enter__ = Mock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = Mock(return_value=False)
        mock_ddgs_class.return_value = mock_ddgs

        fetcher = DuckDuckGoNewsFetcher()
        articles = fetcher._fetch_company_sync("AAPL", 10)

        assert len(articles) == 2


def test_parse_date_iso_format():
    fetcher = DuckDuckGoNewsFetcher()
    result = fetcher._parse_date("2024-01-15T10:30:00Z")
    assert isinstance(result, datetime)
    assert result.year == 2024
    assert result.month == 1


def test_parse_date_timestamp():
    fetcher = DuckDuckGoNewsFetcher()
    result = fetcher._parse_date("1705317000")
    assert isinstance(result, datetime)


def test_parse_date_invalid():
    fetcher = DuckDuckGoNewsFetcher()
    result = fetcher._parse_date("invalid-date")
    assert isinstance(result, datetime)
    # Should fallback to now()
    assert (datetime.now(UTC) - result).total_seconds() < 2


def test_parse_date_none():
    fetcher = DuckDuckGoNewsFetcher()
    result = fetcher._parse_date(None)
    assert isinstance(result, datetime)
    # Should fallback to now()
    assert (datetime.now(UTC) - result).total_seconds() < 2
