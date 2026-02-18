"""Tests for ApeWisdom API fetcher."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.data.apewisdom import ApeWisdomFetcher, ApeWisdomTicker


@pytest.fixture
def sample_api_response() -> dict:
    """Sample ApeWisdom API response."""
    return {
        "results": [
            {
                "rank": 1,
                "ticker": "GME",
                "name": "GameStop Corp",
                "mentions": 500,
                "upvotes": 12000,
                "rank_24h_ago": 2,
                "mentions_24h_ago": 300,
            },
            {
                "rank": 2,
                "ticker": "AAPL",
                "name": "Apple Inc",
                "mentions": 350,
                "upvotes": 8000,
                "rank_24h_ago": 1,
                "mentions_24h_ago": 400,
            },
            {
                "rank": 3,
                "ticker": "TSLA",
                "name": "Tesla Inc",
                "mentions": 200,
                "upvotes": 5000,
                "rank_24h_ago": 5,
                "mentions_24h_ago": 100,
            },
        ]
    }


@pytest.fixture
def fetcher() -> ApeWisdomFetcher:
    """Create fetcher with short TTL for testing."""
    return ApeWisdomFetcher(cache_ttl=300)


@pytest.mark.unit
class TestApeWisdomFetcher:
    """Tests for ApeWisdomFetcher."""

    def test_fetch_trending_parses_response(
        self, fetcher: ApeWisdomFetcher, sample_api_response: dict
    ) -> None:
        """Verify API response is parsed into ApeWisdomTicker models."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = sample_api_response
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            tickers = fetcher.fetch_trending()

        assert len(tickers) == 3
        assert tickers[0].ticker == "GME"
        assert tickers[0].rank == 1
        assert tickers[0].mentions == 500
        assert tickers[1].ticker == "AAPL"
        assert tickers[2].mentions_24h_ago == 100

    def test_fetch_trending_uses_cache(self, fetcher: ApeWisdomFetcher, sample_api_response: dict) -> None:
        """Verify second call uses cache, no HTTP request."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = sample_api_response
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            fetcher.fetch_trending()
            fetcher.fetch_trending()

            # Only one HTTP call
            assert mock_client.get.call_count == 1

    def test_fetch_trending_empty_results(self, fetcher: ApeWisdomFetcher) -> None:
        """Verify empty results return empty list."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            tickers = fetcher.fetch_trending()

        assert tickers == []

    def test_fetch_trending_network_error_returns_stale_cache(self, fetcher: ApeWisdomFetcher) -> None:
        """Verify network errors return stale cache gracefully."""
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = httpx.ConnectError("Connection refused")
            mock_client_cls.return_value = mock_client

            tickers = fetcher.fetch_trending()

        # No stale cache, so empty
        assert tickers == []

    def test_get_ticker_found(self, fetcher: ApeWisdomFetcher, sample_api_response: dict) -> None:
        """Verify get_ticker returns matching ticker."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = sample_api_response
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            ticker = fetcher.get_ticker("gme")  # Case insensitive

        assert ticker is not None
        assert ticker.ticker == "GME"
        assert ticker.rank == 1

    def test_get_ticker_not_found(self, fetcher: ApeWisdomFetcher, sample_api_response: dict) -> None:
        """Verify get_ticker returns None for unknown symbol."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = sample_api_response
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            ticker = fetcher.get_ticker("UNKNOWN")

        assert ticker is None


@pytest.mark.unit
class TestApeWisdomTicker:
    """Tests for ApeWisdomTicker model."""

    def test_model_creation(self) -> None:
        """Verify model fields are set correctly."""
        ticker = ApeWisdomTicker(
            rank=1,
            ticker="GME",
            name="GameStop",
            mentions=500,
            upvotes=12000,
            rank_24h_ago=2,
            mentions_24h_ago=300,
        )
        assert ticker.rank == 1
        assert ticker.ticker == "GME"
        assert ticker.mentions == 500

    def test_repr(self) -> None:
        """Verify repr output."""
        ticker = ApeWisdomTicker(
            rank=5,
            ticker="AAPL",
            name="Apple",
            mentions=200,
            upvotes=5000,
            rank_24h_ago=3,
            mentions_24h_ago=150,
        )
        assert "AAPL" in repr(ticker)
        assert "rank=5" in repr(ticker)
