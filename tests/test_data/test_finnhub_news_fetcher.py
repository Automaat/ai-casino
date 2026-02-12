"""Tests for Finnhub news fetcher."""

from unittest.mock import Mock, patch

import httpx
import pytest

from src.data.finnhub_news import FinnhubNewsFetcher
from src.data.news import NewsArticle


@pytest.fixture
def sample_finnhub_response():
    return [
        {
            "headline": "Apple announces new product",
            "summary": "Apple released a new iPhone model",
            "url": "https://example.com/article1",
            "datetime": 1705317000,  # Unix timestamp
            "source": "TechCrunch",
        },
        {
            "headline": "Tech stocks surge",
            "summary": "Technology sector sees gains",
            "url": "https://example.com/article2",
            "datetime": 1705322400,
            "source": "Bloomberg",
        },
    ]


def test_fetcher_init_with_key():
    fetcher = FinnhubNewsFetcher(api_key="test-key")
    assert fetcher.api_key == "test-key"


def test_fetcher_init_no_key(monkeypatch):
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    fetcher = FinnhubNewsFetcher()
    assert fetcher.api_key == ""


def test_get_source_name():
    fetcher = FinnhubNewsFetcher()
    assert fetcher.get_source_name() == "finnhub"


@pytest.mark.asyncio
async def test_afetch_company_news(sample_finnhub_response):
    with patch("src.data.finnhub_news.httpx.Client") as mock_client_class:
        mock_response = Mock()
        mock_response.json.return_value = sample_finnhub_response
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = FinnhubNewsFetcher(api_key="test-key")
        articles = await fetcher.afetch_company_news("AAPL", 10)

        assert len(articles) == 2
        assert all(isinstance(a, NewsArticle) for a in articles)
        assert articles[0].title == "Apple announces new product"
        assert articles[1].source == "Bloomberg"


@pytest.mark.asyncio
async def test_afetch_market_news(sample_finnhub_response):
    with patch("src.data.finnhub_news.httpx.Client") as mock_client_class:
        mock_response = Mock()
        mock_response.json.return_value = sample_finnhub_response
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = FinnhubNewsFetcher(api_key="test-key")
        articles = await fetcher.afetch_market_news(limit=20)

        assert len(articles) == 2
        assert all(isinstance(a, NewsArticle) for a in articles)


def test_fetch_company_news_http_error():
    with patch("src.data.finnhub_news.httpx.Client") as mock_client_class:
        mock_client = Mock()
        mock_client.get.side_effect = httpx.HTTPError("API Error")
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = FinnhubNewsFetcher(api_key="test-key")

        with pytest.raises(httpx.HTTPError):
            fetcher._fetch_company_sync("AAPL", 10)


def test_fetch_market_news_http_error():
    with patch("src.data.finnhub_news.httpx.Client") as mock_client_class:
        mock_client = Mock()
        mock_client.get.side_effect = httpx.HTTPError("API Error")
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = FinnhubNewsFetcher(api_key="test-key")

        with pytest.raises(httpx.HTTPError):
            fetcher._fetch_market_sync(20)


def test_repr(monkeypatch):
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)

    fetcher = FinnhubNewsFetcher(api_key="test-key")
    assert repr(fetcher) == "FinnhubNewsFetcher(authenticated=True)"

    fetcher_no_key = FinnhubNewsFetcher(api_key="")
    assert repr(fetcher_no_key) == "FinnhubNewsFetcher(authenticated=False)"


def test_fetch_company_news_skips_invalid_items():
    with patch("src.data.finnhub_news.httpx.Client") as mock_client_class:
        mock_response = Mock()
        mock_response.json.return_value = [
            {"headline": "Valid", "url": "https://example.com", "datetime": 1705317000},
            {"headline": "No URL", "datetime": 1705317000},  # Missing URL
            {"url": "https://example.com/no-headline", "datetime": 1705317000},  # Missing headline
            {"headline": "Another valid", "url": "https://example.com/2", "datetime": 1705317000},
        ]
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = FinnhubNewsFetcher(api_key="test-key")
        articles = fetcher._fetch_company_sync("AAPL", 10)

        assert len(articles) == 2


def test_fetch_respects_limit():
    with patch("src.data.finnhub_news.httpx.Client") as mock_client_class:
        mock_response = Mock()
        mock_response.json.return_value = [
            {"headline": f"Article {i}", "url": f"https://example.com/{i}", "datetime": 1705317000}
            for i in range(50)
        ]
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = FinnhubNewsFetcher(api_key="test-key")
        articles = fetcher._fetch_company_sync("AAPL", 10)

        assert len(articles) == 10
