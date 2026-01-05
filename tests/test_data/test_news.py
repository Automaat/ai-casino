"""Tests for news fetcher."""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest
import requests

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


def test_fetcher_init_from_env(monkeypatch):
    monkeypatch.setenv("MARKETAUX_API_KEY", "env-key")
    fetcher = NewsFetcher()
    assert fetcher.api_key == "env-key"


def test_fetcher_init_no_key(monkeypatch):
    monkeypatch.delenv("MARKETAUX_API_KEY", raising=False)
    fetcher = NewsFetcher()
    assert fetcher.api_key == ""


def test_fetch_company_news(sample_news_response):
    with patch("src.data.news.requests.get") as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = sample_news_response
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        fetcher = NewsFetcher(api_key="test-key")
        articles = fetcher.fetch_company_news("AAPL", limit=10)

        assert len(articles) == 2
        assert all(isinstance(a, NewsArticle) for a in articles)
        assert articles[0].title == "Apple announces new product"
        assert articles[1].source == "Bloomberg"

        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args.kwargs["params"]["symbols"] == "AAPL"
        assert call_args.kwargs["params"]["limit"] == 10


def test_fetch_market_news(sample_news_response):
    with patch("src.data.news.requests.get") as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = sample_news_response
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        fetcher = NewsFetcher(api_key="test-key")
        articles = fetcher.fetch_market_news(limit=20)

        assert len(articles) == 2
        assert all(isinstance(a, NewsArticle) for a in articles)

        call_args = mock_get.call_args
        assert call_args.kwargs["params"]["limit"] == 20
        assert "symbols" not in call_args.kwargs["params"]


def test_fetch_company_news_no_api_key():
    with patch("src.data.news.requests.get") as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        fetcher = NewsFetcher(api_key="")
        fetcher.fetch_company_news("AAPL")

        call_args = mock_get.call_args
        assert "api_token" not in call_args.kwargs["params"]


def test_fetch_company_news_http_error():
    with patch("src.data.news.requests.get") as mock_get:
        mock_get.side_effect = requests.exceptions.RequestException("API Error")

        fetcher = NewsFetcher(api_key="test-key")

        with pytest.raises(requests.exceptions.RequestException):
            fetcher.fetch_company_news("AAPL")


def test_fetch_market_news_http_error():
    with patch("src.data.news.requests.get") as mock_get:
        mock_get.side_effect = requests.exceptions.RequestException("API Error")

        fetcher = NewsFetcher(api_key="test-key")

        with pytest.raises(requests.exceptions.RequestException):
            fetcher.fetch_market_news()


def test_repr():
    fetcher = NewsFetcher(api_key="test-key")
    assert repr(fetcher) == "NewsFetcher(authenticated=True)"

    fetcher_no_key = NewsFetcher(api_key="")
    assert repr(fetcher_no_key) == "NewsFetcher(authenticated=False)"
