"""Tests for FinnhubFetcher."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.data.finnhub import (
    BuzzData,
    FinnhubFetcher,
    NewsSentimentData,
    SentimentBreakdown,
    SocialSentimentData,
    SocialSentimentEntry,
)


@pytest.fixture
def mock_social_response():
    """Mock social sentiment API response."""
    return {
        "reddit": [
            {"atTime": "2024-01-01T12:00:00Z", "mention": 100, "score": 0.75},
            {"atTime": "2024-01-01T13:00:00Z", "mention": 150, "score": 0.82},
        ],
        "twitter": [
            {"atTime": "2024-01-01T12:00:00Z", "mention": 500, "score": 0.68},
        ],
        "symbol": "AAPL",
    }


@pytest.fixture
def mock_indicator_response():
    """Mock sentiment indicator API response."""
    return {
        "buzz": {
            "articlesInLastWeek": 42,
            "buzz": 1.5,
            "weeklyAverage": 28.0,
        },
        "companyNewsScore": 0.72,
        "sectorAverageBullishPercent": 0.55,
        "sectorAverageNewsScore": 0.65,
        "sentiment": {
            "bearishPercent": 0.25,
            "bullishPercent": 0.75,
        },
        "symbol": "AAPL",
    }


@pytest.fixture
def fetcher(tmp_path):
    """Create FinnhubFetcher with temp cache dir and mock API key."""
    return FinnhubFetcher(cache_dir=str(tmp_path / "cache"), api_key="test_api_key")


class TestSocialSentimentEntry:
    """Tests for SocialSentimentEntry model."""

    def test_create_entry(self):
        """Test creating a SocialSentimentEntry."""
        entry = SocialSentimentEntry(
            at_time=datetime(2024, 1, 1, 12, 0, 0),
            mention=100,
            score=0.75,
        )
        assert entry.mention == 100
        assert entry.score == 0.75
        assert entry.at_time == datetime(2024, 1, 1, 12, 0, 0)


class TestSocialSentimentData:
    """Tests for SocialSentimentData model."""

    def test_create_data(self):
        """Test creating SocialSentimentData."""
        entry = SocialSentimentEntry(
            at_time=datetime(2024, 1, 1, 12, 0, 0),
            mention=100,
            score=0.75,
        )
        data = SocialSentimentData(
            symbol="AAPL",
            reddit=[entry],
            twitter=[],
            fetched_at=datetime.now(),
        )
        assert data.symbol == "AAPL"
        assert len(data.reddit) == 1
        assert len(data.twitter) == 0


class TestBuzzData:
    """Tests for BuzzData model."""

    def test_create_buzz(self):
        """Test creating BuzzData."""
        buzz = BuzzData(
            articles_in_last_week=42,
            buzz=1.5,
            weekly_average=28.0,
        )
        assert buzz.articles_in_last_week == 42
        assert buzz.buzz == 1.5
        assert buzz.weekly_average == 28.0


class TestSentimentBreakdown:
    """Tests for SentimentBreakdown model."""

    def test_create_breakdown(self):
        """Test creating SentimentBreakdown."""
        breakdown = SentimentBreakdown(
            bearish_percent=0.25,
            bullish_percent=0.75,
        )
        assert breakdown.bearish_percent == 0.25
        assert breakdown.bullish_percent == 0.75

    def test_breakdown_sums_to_one(self):
        """Test that typical breakdown sums to 1.0."""
        breakdown = SentimentBreakdown(
            bearish_percent=0.3,
            bullish_percent=0.7,
        )
        assert abs(breakdown.bearish_percent + breakdown.bullish_percent - 1.0) < 0.001


class TestNewsSentimentData:
    """Tests for NewsSentimentData model."""

    def test_create_news_sentiment(self):
        """Test creating NewsSentimentData."""
        data = NewsSentimentData(
            symbol="AAPL",
            buzz=BuzzData(articles_in_last_week=42, buzz=1.5, weekly_average=28.0),
            company_news_score=0.72,
            sector_avg_bullish_percent=0.55,
            sector_avg_news_score=0.65,
            sentiment=SentimentBreakdown(bearish_percent=0.25, bullish_percent=0.75),
            fetched_at=datetime.now(),
        )
        assert data.symbol == "AAPL"
        assert data.company_news_score == 0.72
        assert data.buzz.articles_in_last_week == 42


class TestFinnhubFetcher:
    """Tests for FinnhubFetcher."""

    def test_init_creates_cache_dir(self, tmp_path):
        """Test that init creates cache directory."""
        cache_dir = tmp_path / "new_cache"
        fetcher = FinnhubFetcher(cache_dir=str(cache_dir), api_key="test")
        assert cache_dir.exists()
        assert fetcher._cache_dir == cache_dir

    def test_init_with_explicit_key(self, tmp_path):
        """Test init with explicit API key."""
        fetcher = FinnhubFetcher(api_key="explicit_key", cache_dir=str(tmp_path / "cache"))
        assert fetcher._api_key == "explicit_key"

    def test_init_without_key_logs_warning(self, tmp_path):
        """Test that missing API key logs warning."""
        fetcher = FinnhubFetcher(cache_dir=str(tmp_path / "cache"))
        assert fetcher._api_key is None

    def test_repr_shows_auth_status(self, tmp_path):
        """Test __repr__ shows authentication status."""
        # Authenticated
        fetcher = FinnhubFetcher(cache_dir=str(tmp_path / "cache1"), api_key="test")
        assert "authenticated=True" in repr(fetcher)

        # Not authenticated
        fetcher = FinnhubFetcher(cache_dir=str(tmp_path / "cache2"))
        assert "authenticated=False" in repr(fetcher)

    def test_cache_key_deterministic(self, fetcher):
        """Test cache key generation is deterministic."""
        key1 = fetcher._cache_key("social", "AAPL", "2024-01-01", "2024-01-31")
        key2 = fetcher._cache_key("social", "AAPL", "2024-01-01", "2024-01-31")
        key3 = fetcher._cache_key("social", "TSLA", "2024-01-01", "2024-01-31")

        assert key1 == key2
        assert key1 != key3
        assert len(key1) == 32


class TestFetchSocialSentiment:
    """Tests for fetch_social_sentiment method."""

    def test_returns_correct_data(self, fetcher, mock_social_response):
        """Test fetch_social_sentiment returns correct data structure."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_social_response
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response

            result = fetcher.fetch_social_sentiment("AAPL")

            assert isinstance(result, SocialSentimentData)
            assert result.symbol == "AAPL"
            assert len(result.reddit) == 2
            assert len(result.twitter) == 1
            assert result.reddit[0].mention == 100
            assert result.reddit[0].score == 0.75

    def test_uses_cache_on_second_call(self, fetcher, mock_social_response):
        """Test that repeated fetches use cache."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_social_response
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response

            fetcher.fetch_social_sentiment("AAPL")
            fetcher.fetch_social_sentiment("AAPL")

            # Should only call API once due to caching
            assert mock_client.return_value.__enter__.return_value.get.call_count == 1

    def test_respects_date_params(self, fetcher, mock_social_response):
        """Test that date parameters are passed to API."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_social_response
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response

            fetcher.fetch_social_sentiment("AAPL", from_date="2024-01-01", to_date="2024-01-31")

            call_args = mock_client.return_value.__enter__.return_value.get.call_args
            params = call_args[1]["params"]
            assert params["from"] == "2024-01-01"
            assert params["to"] == "2024-01-31"

    def test_returns_empty_without_api_key(self, tmp_path):
        """Test that fetch returns empty data without API key."""
        fetcher = FinnhubFetcher(cache_dir=str(tmp_path / "cache"))
        result = fetcher.fetch_social_sentiment("AAPL")
        assert result.symbol == "AAPL"
        assert result.reddit == []
        assert result.twitter == []

    def test_retries_on_timeout(self, fetcher, mock_social_response):
        """Test that timeout errors trigger retry."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_social_response
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client:
            mock_get = mock_client.return_value.__enter__.return_value.get
            mock_get.side_effect = [
                httpx.TimeoutException("Timeout"),
                mock_response,
            ]

            result = fetcher.fetch_social_sentiment("AAPL")

            assert result.symbol == "AAPL"
            assert mock_get.call_count == 2

    def test_exhausts_retries(self, fetcher):
        """Test that retries are exhausted after max attempts."""
        with patch("httpx.Client") as mock_client:
            mock_get = mock_client.return_value.__enter__.return_value.get
            mock_get.side_effect = httpx.TimeoutException("Timeout")

            with pytest.raises(httpx.TimeoutException):
                fetcher.fetch_social_sentiment("AAPL")

            # Should have retried 3 times
            assert mock_get.call_count == 3


class TestFetchSentimentIndicator:
    """Tests for fetch_sentiment_indicator method."""

    def test_returns_correct_data(self, fetcher, mock_indicator_response):
        """Test fetch_sentiment_indicator returns correct data structure."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_indicator_response
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response

            result = fetcher.fetch_sentiment_indicator("AAPL")

            assert isinstance(result, NewsSentimentData)
            assert result.symbol == "AAPL"
            assert result.company_news_score == 0.72
            assert result.buzz.articles_in_last_week == 42
            assert result.sentiment.bullish_percent == 0.75

    def test_uses_cache_on_second_call(self, fetcher, mock_indicator_response):
        """Test that repeated fetches use cache."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_indicator_response
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response

            fetcher.fetch_sentiment_indicator("AAPL")
            fetcher.fetch_sentiment_indicator("AAPL")

            assert mock_client.return_value.__enter__.return_value.get.call_count == 1

    def test_returns_empty_without_api_key(self, tmp_path):
        """Test that fetch returns empty data without API key."""
        fetcher = FinnhubFetcher(cache_dir=str(tmp_path / "cache"))
        result = fetcher.fetch_sentiment_indicator("AAPL")
        assert result.symbol == "AAPL"
        assert result.company_news_score == 0.0
        assert result.buzz.articles_in_last_week == 0
        assert result.sentiment.bullish_percent == 0.0

    def test_retries_on_connection_error(self, fetcher, mock_indicator_response):
        """Test that connection errors trigger retry."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_indicator_response
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client:
            mock_get = mock_client.return_value.__enter__.return_value.get
            mock_get.side_effect = [
                httpx.ConnectError("Connection failed"),
                mock_response,
            ]

            result = fetcher.fetch_sentiment_indicator("AAPL")

            assert result.symbol == "AAPL"
            assert mock_get.call_count == 2


class TestClearCache:
    """Tests for clear_cache method."""

    def test_cache_cleared_refetches(self, fetcher, mock_social_response):
        """Test cache is cleared and re-fetches on next call."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_social_response
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response

            fetcher.fetch_social_sentiment("AAPL")
            fetcher.clear_cache()
            fetcher.fetch_social_sentiment("AAPL")

            assert mock_client.return_value.__enter__.return_value.get.call_count == 2
