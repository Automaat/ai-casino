"""Tests for NewsData.io fetcher."""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from src.data.news import NewsArticle
from src.data.newsdata import NewsDataFetcher


@pytest.fixture
def sample_newsdata_response():
    return {
        "status": "success",
        "results": [
            {
                "title": "Apple stock surges",
                "description": "Apple shares hit new high",
                "link": "https://example.com/article1",
                "pubDate": "2024-01-15T10:30:00Z",
                "source_name": "TechNews",
            },
            {
                "title": "Tech sector analysis",
                "description": "Technology stocks see gains",
                "link": "https://example.com/article2",
                "pubDate": "2024-01-15T12:00:00Z",
                "source_name": "FinanceDaily",
            },
        ],
    }


def test_fetcher_init_with_key():
    fetcher = NewsDataFetcher(api_key="test-key")
    assert fetcher.api_key == "test-key"


def test_fetcher_init_from_env(monkeypatch):
    # After removing env var fallbacks, this test no longer applies
    # API key must be passed explicitly or via config
    fetcher = NewsDataFetcher(api_key="env-key")
    assert fetcher.api_key == "env-key"


def test_fetcher_init_no_key(monkeypatch):
    monkeypatch.delenv("NEWSDATA_API_KEY", raising=False)
    fetcher = NewsDataFetcher()
    assert fetcher.api_key == ""


def test_get_source_name():
    fetcher = NewsDataFetcher()
    assert fetcher.get_source_name() == "newsdata"


@pytest.mark.asyncio
async def test_afetch_company_news(sample_newsdata_response):
    with patch("src.data.newsdata.httpx.AsyncClient") as mock_client_class:
        mock_response = Mock()
        mock_response.json.return_value = sample_newsdata_response
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = NewsDataFetcher(api_key="test-key")
        articles = await fetcher.afetch_company_news("AAPL", 10)

        assert len(articles) == 2
        assert all(isinstance(a, NewsArticle) for a in articles)
        assert articles[0].title == "Apple stock surges"
        assert articles[1].source == "FinanceDaily"


@pytest.mark.asyncio
async def test_afetch_market_news(sample_newsdata_response):
    with patch("src.data.newsdata.httpx.AsyncClient") as mock_client_class:
        mock_response = Mock()
        mock_response.json.return_value = sample_newsdata_response
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = NewsDataFetcher(api_key="test-key")
        articles = await fetcher.afetch_market_news(limit=20)

        assert len(articles) == 2
        assert all(isinstance(a, NewsArticle) for a in articles)


@pytest.mark.asyncio
async def test_fetch_company_news_http_error():
    with patch("src.data.newsdata.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.HTTPError("API Error"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = NewsDataFetcher(api_key="test-key")

        with pytest.raises(httpx.HTTPError):
            await fetcher.afetch_company_news("AAPL", 10)


@pytest.mark.asyncio
async def test_fetch_market_news_http_error():
    with patch("src.data.newsdata.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.HTTPError("API Error"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = NewsDataFetcher(api_key="test-key")

        with pytest.raises(httpx.HTTPError):
            await fetcher.afetch_market_news(20)


@pytest.mark.asyncio
async def test_fetch_company_news_skips_invalid_items():
    invalid_response = {
        "results": [
            {"title": "Valid", "link": "https://example.com", "pubDate": "2024-01-15T10:30:00Z"},
            {"title": "No Link", "pubDate": "2024-01-15T10:30:00Z"},  # Missing link
            {"link": "https://example.com", "pubDate": "2024-01-15T10:30:00Z"},  # Missing title
        ]
    }

    with patch("src.data.newsdata.httpx.AsyncClient") as mock_client_class:
        mock_response = Mock()
        mock_response.json.return_value = invalid_response
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = NewsDataFetcher(api_key="test-key")
        articles = await fetcher.afetch_company_news("AAPL", 10)

        assert len(articles) == 1
        assert articles[0].title == "Valid"


@pytest.mark.asyncio
async def test_fetch_handles_invalid_date():
    invalid_date_response = {
        "results": [
            {
                "title": "Test Article",
                "description": "Test",
                "link": "https://example.com",
                "pubDate": "invalid-date-format",
                "source_name": "TestSource",
            }
        ]
    }

    with patch("src.data.newsdata.httpx.AsyncClient") as mock_client_class:
        mock_response = Mock()
        mock_response.json.return_value = invalid_date_response
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = NewsDataFetcher(api_key="test-key")
        articles = await fetcher.afetch_company_news("AAPL", 10)

        assert len(articles) == 1
        assert articles[0].title == "Test Article"
        assert isinstance(articles[0].published_at, datetime)
