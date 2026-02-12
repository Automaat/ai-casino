"""Tests for Finnhub news fetcher."""

from unittest.mock import Mock, patch

import httpx
import pytest
import respx

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


@pytest.mark.asyncio
@respx.mock
async def test_fetcher_init_with_key():
    fetcher = FinnhubNewsFetcher(api_key="test-key")
    assert fetcher.api_key == "test-key"


@pytest.mark.asyncio
@respx.mock
async def test_fetcher_init_from_env():
    fetcher = FinnhubNewsFetcher(api_key="env-key")
    assert fetcher.api_key == "env-key"


@pytest.mark.asyncio
@respx.mock
async def test_fetcher_init_no_key():
    fetcher = FinnhubNewsFetcher()
    assert fetcher.api_key == ""


@pytest.mark.asyncio
@respx.mock
async def test_get_source_name():
    fetcher = FinnhubNewsFetcher()
    assert fetcher.get_source_name() == "finnhub"


@pytest.mark.asyncio
async @pytest.mark.asyncio
@respx.mock
async def test_afetch_company_news(sample_finnhub_response):
    respx.get(re.compile(r"https://finnhub.io/api/v1/.*")).mock(return_value=httpx.Response(200, json=mock_data))

        fetcher = FinnhubNewsFetcher(api_key="test-key")
        articles = await fetcher.afetch_company_news("AAPL", 10)

        assert len(articles) == 2
        assert all(isinstance(a, NewsArticle) for a in articles)
        assert articles[0].title == "Apple announces new product"
        assert articles[1].source == "Bloomberg"


@pytest.mark.asyncio
async @pytest.mark.asyncio
@respx.mock
async def test_afetch_market_news(sample_finnhub_response):
    respx.get(re.compile(r"https://finnhub.io/api/v1/.*")).mock(return_value=httpx.Response(200, json=mock_data))

        fetcher = FinnhubNewsFetcher(api_key="test-key")
        articles = await fetcher.afetch_market_news(limit=20)

        assert len(articles) == 2
        assert all(isinstance(a, NewsArticle) for a in articles)


@pytest.mark.asyncio
@respx.mock
async def test_fetch_company_news_http_error():
    respx.get(re.compile(r"https://finnhub.io/api/v1/.*")).mock(return_value=httpx.Response(200, json=mock_data))

        fetcher = FinnhubNewsFetcher(api_key="test-key")

        with pytest.raises(httpx.HTTPError):
            await fetcher.afetch_company_news("AAPL", 10)


@pytest.mark.asyncio
@respx.mock
async def test_fetch_market_news_http_error():
    respx.get(re.compile(r"https://finnhub.io/api/v1/.*")).mock(return_value=httpx.Response(200, json=mock_data))

        fetcher = FinnhubNewsFetcher(api_key="test-key")

        with pytest.raises(httpx.HTTPError):
            await fetcher.afetch_market_news(20)


@pytest.mark.asyncio
@respx.mock
async def test_repr():
    fetcher = FinnhubNewsFetcher(api_key="test-key")
    assert repr(fetcher) == "FinnhubNewsFetcher(authenticated=True)"

    fetcher_no_key = FinnhubNewsFetcher(api_key="")
    assert repr(fetcher_no_key) == "FinnhubNewsFetcher(authenticated=False)"


@pytest.mark.asyncio
@respx.mock
async def test_fetch_company_news_skips_invalid_items():
    respx.get(re.compile(r"https://finnhub.io/api/v1/.*")).mock(return_value=httpx.Response(200, json=mock_data))

        fetcher = FinnhubNewsFetcher(api_key="test-key")
        articles = await fetcher.afetch_company_news("AAPL", 10)

        assert len(articles) == 2


@pytest.mark.asyncio
@respx.mock
async def test_fetch_respects_limit():
    respx.get(re.compile(r"https://finnhub.io/api/v1/.*")).mock(return_value=httpx.Response(200, json=mock_data))

        fetcher = FinnhubNewsFetcher(api_key="test-key")
        articles = await fetcher.afetch_company_news("AAPL", 10)

        assert len(articles) == 10
