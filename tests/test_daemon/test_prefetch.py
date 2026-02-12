"""Tests for after-hours data prefetching."""

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest

from src.daemon.prefetch import (
    AV_RATE_LIMIT_SLEEP,
    FUNDAMENTALS_TTL,
    MARKET_DATA_TTL,
    NEWS_TTL,
    DataPrefetcher,
    PrefetchReport,
    PrefetchResult,
)
from src.data.market import MarketData
from src.data.news import NewsArticle


@pytest.fixture
def sample_market_data() -> MarketData:
    """Create sample market data."""
    df = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [105.0, 106.0, 107.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [103.0, 104.0, 105.0],
            "Volume": [1000000, 1100000, 1200000],
        }
    )
    return MarketData(symbol="AAPL", data=df, last_updated=datetime(2024, 1, 15))


@pytest.fixture
def sample_articles() -> list[NewsArticle]:
    """Create sample news articles."""
    return [
        NewsArticle(
            title="Apple reports strong earnings",
            description="Q4 results exceed expectations",
            url="https://example.com/1",
            published_at=datetime(2024, 1, 15),
            source="Reuters",
        ),
    ]


@pytest.fixture
def mock_market_fetcher(sample_market_data: MarketData) -> Mock:
    """Create mock market data fetcher."""
    fetcher = Mock()
    fetcher.fetch_daily.return_value = sample_market_data
    return fetcher


@pytest.fixture
def mock_news_fetcher(sample_articles: list[NewsArticle]) -> Mock:
    """Create mock news fetcher."""
    fetcher = Mock()
    fetcher.afetch_company_news = AsyncMock(return_value=sample_articles)
    return fetcher


@pytest.fixture
def mock_fundamental_fetcher() -> Mock:
    """Create mock fundamental data fetcher."""
    fetcher = Mock()
    fetcher.fetch_overview.return_value = {
        "Symbol": "AAPL",
        "PERatio": "25.5",
        "MarketCap": "3000000000000",
    }
    return fetcher


@pytest.fixture
def prefetcher(
    tmp_path: Path,
    mock_market_fetcher: Mock,
    mock_news_fetcher: Mock,
    mock_fundamental_fetcher: Mock,
) -> DataPrefetcher:
    """Create prefetcher with mocked fetchers and tmp cache."""
    return DataPrefetcher(
        market_fetcher=mock_market_fetcher,
        news_fetcher=mock_news_fetcher,
        fundamental_fetcher=mock_fundamental_fetcher,
        cache_dir=str(tmp_path / "prefetch_cache"),
    )


class TestPrefetchSymbol:
    def test_caches_all_data(self, prefetcher: DataPrefetcher) -> None:
        """Verify all three data types are cached."""
        with patch("src.daemon.prefetch.time.sleep"):
            result = prefetcher.prefetch_symbol("AAPL")

        assert result.symbol == "AAPL"
        assert result.market_data is True
        assert result.news is True
        assert result.fundamentals is True
        assert result.duration_ms >= 0

        assert prefetcher.get_cached_market_data("AAPL") is not None
        assert prefetcher.get_cached_news("AAPL") is not None
        assert prefetcher.get_cached_fundamentals("AAPL") is not None

    def test_cache_ttls(self, prefetcher: DataPrefetcher) -> None:
        """Verify cache entries use correct TTL values."""
        cache_set = prefetcher._cache.set
        with (
            patch("src.daemon.prefetch.time.sleep"),
            patch.object(prefetcher._cache, "set", wraps=cache_set) as mock_set,
        ):
            prefetcher.prefetch_symbol("AAPL")

        ttls = [c.kwargs.get("expire") for c in mock_set.call_args_list]
        assert MARKET_DATA_TTL in ttls
        assert NEWS_TTL in ttls
        assert FUNDAMENTALS_TTL in ttls

    def test_handles_market_data_failure(self, prefetcher: DataPrefetcher, mock_market_fetcher: Mock) -> None:
        """Verify partial failure doesn't stop other fetches."""
        mock_market_fetcher.fetch_daily.side_effect = RuntimeError("down")

        with patch("src.daemon.prefetch.time.sleep"):
            result = prefetcher.prefetch_symbol("AAPL")

        assert result.market_data is False
        assert result.news is True
        assert result.fundamentals is True

    def test_handles_news_failure(self, prefetcher: DataPrefetcher, mock_news_fetcher: Mock) -> None:
        """Verify news failure doesn't stop fundamentals."""
        mock_news_fetcher.afetch_company_news = AsyncMock(side_effect=RuntimeError("down"))

        with patch("src.daemon.prefetch.time.sleep"):
            result = prefetcher.prefetch_symbol("AAPL")

        assert result.market_data is True
        assert result.news is False
        assert result.fundamentals is True

    def test_handles_fundamentals_failure(
        self,
        prefetcher: DataPrefetcher,
        mock_fundamental_fetcher: Mock,
    ) -> None:
        """Verify fundamentals failure is isolated."""
        mock_fundamental_fetcher.fetch_overview.side_effect = RuntimeError("down")

        with patch("src.daemon.prefetch.time.sleep"):
            result = prefetcher.prefetch_symbol("AAPL")

        assert result.market_data is True
        assert result.news is True
        assert result.fundamentals is False


class TestPrefetchWatchlist:
    def test_sequential_execution(self, prefetcher: DataPrefetcher) -> None:
        """Verify symbols are prefetched sequentially."""
        call_order: list[str] = []
        original = prefetcher.prefetch_symbol

        def tracking(symbol: str) -> PrefetchResult:
            call_order.append(symbol)
            return original(symbol)

        with (
            patch.object(prefetcher, "prefetch_symbol", side_effect=tracking),
            patch("src.daemon.prefetch.time.sleep"),
        ):
            report = prefetcher.prefetch_watchlist(["AAPL", "TSLA", "GOOGL"])

        assert call_order == ["AAPL", "TSLA", "GOOGL"]
        assert len(report.results) == 3
        assert report.total_duration_seconds > 0

    def test_rate_limiting(self, prefetcher: DataPrefetcher) -> None:
        """Verify AV staggering sleeps between symbols."""
        sleep_calls: list[float] = []

        def record_sleep(s: float) -> None:
            sleep_calls.append(s)

        with patch("src.daemon.prefetch.time.sleep", side_effect=record_sleep):
            prefetcher.prefetch_watchlist(["AAPL", "TSLA"])

        assert any(s == AV_RATE_LIMIT_SLEEP for s in sleep_calls)


class TestCachedDataRetrieval:
    def test_get_cached_market_data(self, prefetcher: DataPrefetcher) -> None:
        """Verify cached market data can be retrieved."""
        with patch("src.daemon.prefetch.time.sleep"):
            prefetcher.prefetch_symbol("AAPL")

        cached = prefetcher.get_cached_market_data("AAPL")
        assert cached is not None
        assert cached.symbol == "AAPL"
        assert len(cached.data) == 3

    def test_get_cached_news(self, prefetcher: DataPrefetcher) -> None:
        """Verify cached news can be retrieved."""
        with patch("src.daemon.prefetch.time.sleep"):
            prefetcher.prefetch_symbol("AAPL")

        cached = prefetcher.get_cached_news("AAPL")
        assert cached is not None
        assert len(cached) == 1
        assert cached[0].title == "Apple reports strong earnings"

    def test_get_cached_fundamentals(self, prefetcher: DataPrefetcher) -> None:
        """Verify cached fundamentals can be retrieved."""
        with patch("src.daemon.prefetch.time.sleep"):
            prefetcher.prefetch_symbol("AAPL")

        cached = prefetcher.get_cached_fundamentals("AAPL")
        assert cached is not None
        assert cached["Symbol"] == "AAPL"
        assert cached["PERatio"] == "25.5"

    def test_cache_miss_returns_none(self, prefetcher: DataPrefetcher) -> None:
        """Verify cache miss returns None."""
        assert prefetcher.get_cached_market_data("UNKNOWN") is None
        assert prefetcher.get_cached_news("UNKNOWN") is None
        assert prefetcher.get_cached_fundamentals("UNKNOWN") is None

    def test_clear_cache(self, prefetcher: DataPrefetcher) -> None:
        """Verify cache can be cleared."""
        with patch("src.daemon.prefetch.time.sleep"):
            prefetcher.prefetch_symbol("AAPL")

        prefetcher.clear_cache()

        assert prefetcher.get_cached_market_data("AAPL") is None
        assert prefetcher.get_cached_news("AAPL") is None
        assert prefetcher.get_cached_fundamentals("AAPL") is None


class TestWarmFinbert:
    def test_warm_finbert_success(self, prefetcher: DataPrefetcher) -> None:
        """Verify FinBERT warming returns True on success."""
        with patch("src.models.sentiment.get_finbert_sentiment") as mock_finbert:
            mock_finbert.return_value = Mock()
            result = prefetcher.warm_finbert()

        assert result is True
        mock_finbert.assert_called_once()

    def test_warm_finbert_failure(self, prefetcher: DataPrefetcher) -> None:
        """Verify FinBERT warming returns False on failure."""
        with patch(
            "src.models.sentiment.get_finbert_sentiment",
            side_effect=RuntimeError("model error"),
        ):
            result = prefetcher.warm_finbert()

        assert result is False


class TestPrefetchHandlesFailures:
    def test_all_failures_still_returns_result(
        self,
        prefetcher: DataPrefetcher,
        mock_market_fetcher: Mock,
        mock_news_fetcher: Mock,
        mock_fundamental_fetcher: Mock,
    ) -> None:
        """Verify complete failure returns result with all False."""
        mock_market_fetcher.fetch_daily.side_effect = RuntimeError("x")
        mock_news_fetcher.afetch_company_news = AsyncMock(side_effect=RuntimeError("x"))
        mock_fundamental_fetcher.fetch_overview.side_effect = RuntimeError("x")

        result = prefetcher.prefetch_symbol("AAPL")

        assert result.market_data is False
        assert result.news is False
        assert result.fundamentals is False

    def test_watchlist_continues_after_symbol_failure(
        self,
        prefetcher: DataPrefetcher,
        mock_market_fetcher: Mock,
    ) -> None:
        """Verify watchlist prefetch continues past failing symbols."""
        call_count = 0

        def fail_first(symbol: str, **_kwargs: object) -> MarketData:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                msg = "First symbol fails"
                raise RuntimeError(msg)
            return MarketData(
                symbol=symbol,
                data=pd.DataFrame({"Close": [100.0], "Volume": [1000000]}),
                last_updated=datetime(2024, 1, 15),
            )

        mock_market_fetcher.fetch_daily.side_effect = fail_first

        with patch("src.daemon.prefetch.time.sleep"):
            report = prefetcher.prefetch_watchlist(["AAPL", "TSLA"])

        assert len(report.results) == 2
        assert report.results[0].market_data is False
        assert report.results[1].market_data is True


class TestPrefetchModels:
    def test_prefetch_result_model(self) -> None:
        """Test PrefetchResult model."""
        result = PrefetchResult(
            symbol="AAPL",
            market_data=True,
            news=True,
            fundamentals=False,
            duration_ms=1500.0,
        )

        assert result.symbol == "AAPL"
        assert result.fundamentals is False
        assert result.duration_ms == 1500.0

    def test_prefetch_report_model(self) -> None:
        """Test PrefetchReport model."""
        report = PrefetchReport(
            timestamp=datetime(2024, 1, 15),
            results=[
                PrefetchResult(
                    symbol="AAPL",
                    market_data=True,
                    news=True,
                    fundamentals=True,
                    duration_ms=1000.0,
                ),
            ],
            finbert_ready=True,
            api_connectivity={
                "alpha_vantage": True,
                "marketaux": True,
            },
            total_duration_seconds=30.0,
        )

        assert len(report.results) == 1
        assert report.finbert_ready is True
        assert report.total_duration_seconds == 30.0


class TestApiKeyPresence:
    def test_check_with_keys(self, prefetcher: DataPrefetcher) -> None:
        """Verify presence reports True when keys present."""
        env = {
            "ALPHA_VANTAGE_API_KEY": "test",
            "MARKETAUX_API_KEY": "test",
        }
        with patch.dict("os.environ", env):
            result = prefetcher.check_api_key_presence()

        assert result["alpha_vantage"] is True
        assert result["marketaux"] is True

    def test_check_without_keys(self, prefetcher: DataPrefetcher) -> None:
        """Verify presence reports False when keys missing."""
        with patch.dict("os.environ", {}, clear=True):
            result = prefetcher.check_api_key_presence()

        assert result["alpha_vantage"] is False
        assert result["marketaux"] is False
