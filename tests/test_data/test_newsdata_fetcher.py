"""Tests for NewsData.io fetcher."""

from datetime import datetime
from unittest.mock import Mock, patch

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


def test_fetcher_init_no_key(monkeypatch):
    monkeypatch.delenv("NEWSDATA_API_KEY", raising=False)
    fetcher = NewsDataFetcher()
    assert fetcher.api_key == ""


def test_get_source_name():
    fetcher = NewsDataFetcher()
    assert fetcher.get_source_name() == "newsdata"


@pytest.mark.asyncio
async def test_afetch_company_news(sample_newsdata_response):
    with patch("src.data.newsdata.httpx.Client") as mock_client_class:
        mock_response = Mock()
        mock_response.json.return_value = sample_newsdata_response
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = NewsDataFetcher(api_key="test-key")
        articles = await fetcher.afetch_company_news("AAPL", 10)

        assert len(articles) == 2
        assert all(isinstance(a, NewsArticle) for a in articles)
        assert articles[0].title == "Apple stock surges"
        assert articles[1].source == "FinanceDaily"


@pytest.mark.asyncio
async def test_afetch_market_news(sample_newsdata_response):
    with patch("src.data.newsdata.httpx.Client") as mock_client_class:
        mock_response = Mock()
        mock_response.json.return_value = sample_newsdata_response
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = NewsDataFetcher(api_key="test-key")
        articles = await fetcher.afetch_market_news(limit=20)

        assert len(articles) == 2
        assert all(isinstance(a, NewsArticle) for a in articles)


def test_fetch_company_news_http_error():
    with patch("src.data.newsdata.httpx.Client") as mock_client_class:
        mock_client = Mock()
        mock_client.get.side_effect = httpx.HTTPError("API Error")
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = NewsDataFetcher(api_key="test-key")

        with pytest.raises(httpx.HTTPError):
            fetcher._fetch_company_sync("AAPL", 10)


def test_fetch_market_news_http_error():
    with patch("src.data.newsdata.httpx.Client") as mock_client_class:
        mock_client = Mock()
        mock_client.get.side_effect = httpx.HTTPError("API Error")
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = NewsDataFetcher(api_key="test-key")

        with pytest.raises(httpx.HTTPError):
            fetcher._fetch_market_sync(20)


def test_repr(monkeypatch):
    monkeypatch.delenv("NEWSDATA_API_KEY", raising=False)

    fetcher = NewsDataFetcher(api_key="test-key")
    assert repr(fetcher) == "NewsDataFetcher(authenticated=True)"

    fetcher_no_key = NewsDataFetcher(api_key="")
    assert repr(fetcher_no_key) == "NewsDataFetcher(authenticated=False)"


def test_fetch_company_news_skips_invalid_items():
    with patch("src.data.newsdata.httpx.Client") as mock_client_class:
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [
                {"title": "Valid article", "link": "https://example.com", "pubDate": "2024-01-15T10:30:00Z"},
                {"title": "No link"},  # Missing link
                {"link": "https://example.com/no-title"},  # Missing title
                {
                    "title": "Another valid",
                    "link": "https://example.com/2",
                    "pubDate": "2024-01-15T11:00:00Z",
                },
            ]
        }
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = NewsDataFetcher(api_key="test-key")
        articles = fetcher._fetch_company_sync("AAPL", 10)

        assert len(articles) == 2  # Only valid items


def test_fetch_handles_invalid_date():
    with patch("src.data.newsdata.httpx.Client") as mock_client_class:
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [
                {"title": "Test", "link": "https://example.com", "pubDate": "invalid-date"},
            ]
        }
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = NewsDataFetcher(api_key="test-key")
        articles = fetcher._fetch_company_sync("AAPL", 10)

        assert len(articles) == 1
        assert isinstance(articles[0].published_at, datetime)
