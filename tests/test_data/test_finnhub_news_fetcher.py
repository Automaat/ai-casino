"""Tests for Finnhub news fetcher."""

from unittest.mock import AsyncMock, Mock, patch

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


def test_fetcher_init_from_env():
    fetcher = FinnhubNewsFetcher(api_key="env-key")
    assert fetcher.api_key == "env-key"


def test_fetcher_init_no_key():
    fetcher = FinnhubNewsFetcher()
    assert fetcher.api_key == ""


def test_get_source_name():
    fetcher = FinnhubNewsFetcher()
    assert fetcher.get_source_name() == "finnhub"


@pytest.mark.asyncio
async def test_afetch_company_news(sample_finnhub_response):
    with patch("src.data.finnhub_news.httpx.AsyncClient") as mock_client_class:
        mock_response = Mock()
        mock_response.json.return_value = sample_finnhub_response
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = FinnhubNewsFetcher(api_key="test-key")
        articles = await fetcher.afetch_company_news("AAPL", 10)

        assert len(articles) == 2
        assert all(isinstance(a, NewsArticle) for a in articles)
        assert articles[0].title == "Apple announces new product"
        assert articles[1].source == "Bloomberg"


@pytest.mark.asyncio
async def test_afetch_market_news(sample_finnhub_response):
    with patch("src.data.finnhub_news.httpx.AsyncClient") as mock_client_class:
        mock_response = Mock()
        mock_response.json.return_value = sample_finnhub_response
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = FinnhubNewsFetcher(api_key="test-key")
        articles = await fetcher.afetch_market_news(limit=20)

        assert len(articles) == 2
        assert all(isinstance(a, NewsArticle) for a in articles)


@pytest.mark.asyncio
async def test_fetch_company_news_http_error():
    with patch("src.data.finnhub_news.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.HTTPError("API Error"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = FinnhubNewsFetcher(api_key="test-key")

        with pytest.raises(httpx.HTTPError):
            await fetcher.afetch_company_news("AAPL", 10)


@pytest.mark.asyncio
async def test_fetch_market_news_http_error():
    with patch("src.data.finnhub_news.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.HTTPError("API Error"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = FinnhubNewsFetcher(api_key="test-key")

        with pytest.raises(httpx.HTTPError):
            await fetcher.afetch_market_news(20)


@pytest.mark.asyncio
async def test_fetch_company_news_skips_invalid_items():
    invalid_response = [
        {"headline": "Valid", "url": "https://example.com", "datetime": 1705317000, "summary": "Test"},
        {"headline": "No URL", "datetime": 1705317000},  # Missing URL
        {"url": "https://example.com", "datetime": 1705317000},  # Missing headline
        {"headline": "", "url": "https://example.com", "datetime": 1705317000},  # Empty headline
    ]

    with patch("src.data.finnhub_news.httpx.AsyncClient") as mock_client_class:
        mock_response = Mock()
        mock_response.json.return_value = invalid_response
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = FinnhubNewsFetcher(api_key="test-key")
        articles = await fetcher.afetch_company_news("AAPL", 10)

        assert len(articles) == 1
        assert articles[0].title == "Valid"


@pytest.mark.asyncio
async def test_fetch_respects_limit(sample_finnhub_response):
    # Add more items than limit
    extended_response = sample_finnhub_response + [
        {"headline": f"Article {i}", "url": f"https://example.com/{i}", "datetime": 1705317000, "summary": "Test"}
        for i in range(3, 20)
    ]

    with patch("src.data.finnhub_news.httpx.AsyncClient") as mock_client_class:
        mock_response = Mock()
        mock_response.json.return_value = extended_response
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = FinnhubNewsFetcher(api_key="test-key")
        articles = await fetcher.afetch_company_news("AAPL", limit=5)

        assert len(articles) == 5
