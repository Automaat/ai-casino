"""Tests for news fetcher."""

from datetime import datetime
from unittest.mock import Mock, patch

import httpx
import pytest

from src.data.news import NewsArticle, NewsFetcher


@pytest.fixture
def sample_news_response():
    return {
        "data": [
            {
                "title": "Apple announces new product",
                "description": "Apple released a new iPhone model",
                "url": "https://example.com/article1",
                "published_at": "2024-01-15T10:30:00Z",
                "source": "TechCrunch",
            },
            {
                "title": "Tech stocks surge",
                "description": "Technology sector sees gains",
                "url": "https://example.com/article2",
                "published_at": "2024-01-15T12:00:00Z",
                "source": "Bloomberg",
            },
        ]
    }


def test_news_article_creation():
    article = NewsArticle(
        title="Test Title",
        description="Test Description",
        url="https://example.com",
        published_at=datetime(2024, 1, 15, 10, 30),
        source="TestSource",
    )

    assert article.title == "Test Title"
    assert article.description == "Test Description"
    assert article.url == "https://example.com"
    assert article.source == "TestSource"


def test_fetcher_init_with_key():
    fetcher = NewsFetcher(api_key="test-key")
    assert fetcher.api_key == "test-key"


def test_fetcher_init_no_key(monkeypatch):
    monkeypatch.delenv("MARKETAUX_API_KEY", raising=False)
    fetcher = NewsFetcher()
    assert fetcher.api_key == ""


def test_fetch_company_news(sample_news_response):
    with patch("src.data.news.httpx.Client") as mock_client_class:
        mock_response = Mock()
        mock_response.json.return_value = sample_news_response
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = NewsFetcher(api_key="test-key")
        articles = fetcher.fetch_company_news("AAPL", limit=10)

        assert len(articles) == 2
        assert all(isinstance(a, NewsArticle) for a in articles)
        assert articles[0].title == "Apple announces new product"
        assert articles[1].source == "Bloomberg"

        mock_client.get.assert_called_once()
        call_args = mock_client.get.call_args
        assert call_args.kwargs["params"]["symbols"] == "AAPL"
        assert call_args.kwargs["params"]["limit"] == 10


def test_fetch_market_news(sample_news_response):
    with patch("src.data.news.httpx.Client") as mock_client_class:
        mock_response = Mock()
        mock_response.json.return_value = sample_news_response
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = NewsFetcher(api_key="test-key")
        articles = fetcher.fetch_market_news(limit=20)

        assert len(articles) == 2
        assert all(isinstance(a, NewsArticle) for a in articles)

        call_args = mock_client.get.call_args
        assert call_args.kwargs["params"]["limit"] == 20
        assert "symbols" not in call_args.kwargs["params"]


def test_fetch_company_news_no_api_key(monkeypatch):
    monkeypatch.delenv("MARKETAUX_API_KEY", raising=False)

    with patch("src.data.news.httpx.Client") as mock_client_class:
        mock_response = Mock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = NewsFetcher(api_key="")
        fetcher.fetch_company_news("AAPL")

        call_args = mock_client.get.call_args
        assert "api_token" not in call_args.kwargs["params"]


def test_fetch_company_news_http_error():
    with patch("src.data.news.httpx.Client") as mock_client_class:
        mock_client = Mock()
        mock_client.get.side_effect = httpx.HTTPError("API Error")
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = NewsFetcher(api_key="test-key")

        with pytest.raises(httpx.HTTPError):
            fetcher.fetch_company_news("AAPL")


def test_fetch_market_news_http_error():
    with patch("src.data.news.httpx.Client") as mock_client_class:
        mock_client = Mock()
        mock_client.get.side_effect = httpx.HTTPError("API Error")
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = NewsFetcher(api_key="test-key")

        with pytest.raises(httpx.HTTPError):
            fetcher.fetch_market_news()


def test_repr(monkeypatch):
    monkeypatch.delenv("MARKETAUX_API_KEY", raising=False)

    fetcher = NewsFetcher(api_key="test-key")
    assert repr(fetcher) == "NewsFetcher(authenticated=True)"

    fetcher_no_key = NewsFetcher(api_key="")
    assert repr(fetcher_no_key) == "NewsFetcher(authenticated=False)"


def test_fetch_company_news_retries_on_timeout(sample_news_response):
    with patch("src.data.news.httpx.Client") as mock_client_class:
        mock_response = Mock()
        mock_response.json.return_value = sample_news_response
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.get.side_effect = [
            httpx.TimeoutException("timeout"),
            mock_response,
        ]
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = NewsFetcher(api_key="test-key")
        articles = fetcher.fetch_company_news("AAPL")

        assert len(articles) == 2
        assert mock_client.get.call_count == 2


def test_fetch_company_news_retries_on_connection_error(sample_news_response):
    with patch("src.data.news.httpx.Client") as mock_client_class:
        mock_response = Mock()
        mock_response.json.return_value = sample_news_response
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.get.side_effect = [
            httpx.ConnectError("connection failed"),
            httpx.ConnectError("connection failed again"),
            mock_response,
        ]
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = NewsFetcher(api_key="test-key")
        articles = fetcher.fetch_company_news("AAPL")

        assert len(articles) == 2
        assert mock_client.get.call_count == 3


def test_fetch_company_news_exhausts_retries():
    with patch("src.data.news.httpx.Client") as mock_client_class:
        mock_client = Mock()
        mock_client.get.side_effect = httpx.TimeoutException("timeout")
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = NewsFetcher(api_key="test-key")

        with pytest.raises(httpx.TimeoutException):
            fetcher.fetch_company_news("AAPL")

        assert mock_client.get.call_count == 3


def test_fetch_market_news_retries_on_timeout(sample_news_response):
    with patch("src.data.news.httpx.Client") as mock_client_class:
        mock_response = Mock()
        mock_response.json.return_value = sample_news_response
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.get.side_effect = [
            httpx.ReadTimeout("read timeout"),
            mock_response,
        ]
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = NewsFetcher(api_key="test-key")
        articles = fetcher.fetch_market_news()

        assert len(articles) == 2
        assert mock_client.get.call_count == 2


@pytest.mark.asyncio
async def test_afetch_company_news(sample_news_response):
    with patch("src.data.news.httpx.Client") as mock_client_class:
        mock_response = Mock()
        mock_response.json.return_value = sample_news_response
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = NewsFetcher(api_key="test-key")
        articles = await fetcher.afetch_company_news("AAPL", 10)

        assert len(articles) == 2
        assert all(isinstance(a, NewsArticle) for a in articles)
        assert articles[0].title == "Apple announces new product"
        assert articles[1].source == "Bloomberg"


@pytest.mark.asyncio
async def test_afetch_market_news(sample_news_response):
    with patch("src.data.news.httpx.Client") as mock_client_class:
        mock_response = Mock()
        mock_response.json.return_value = sample_news_response
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        fetcher = NewsFetcher(api_key="test-key")
        articles = await fetcher.afetch_market_news(limit=20)

        assert len(articles) == 2
        assert all(isinstance(a, NewsArticle) for a in articles)


def test_get_source_name():
    fetcher = NewsFetcher()
    assert fetcher.get_source_name() == "marketaux"
