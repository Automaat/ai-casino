"""Tests for comparative data fetcher."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.data.comparative import (
    ComparativeData,
    ComparativeDataFetcher,
    PerformanceData,
    Sector,
)


@pytest.fixture
def mock_yfinance_ticker():
    """Mock yfinance Ticker with realistic data."""
    mock = MagicMock()
    mock.info = {
        "symbol": "AAPL",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "trailingPE": 28.5,
        "forwardPE": 25.0,
        "priceToBook": 45.2,
    }
    return mock


@pytest.fixture
def mock_yfinance_history():
    """Mock yfinance history DataFrame."""
    import pandas as pd

    return pd.DataFrame({"Close": [150.0, 155.0, 160.0, 165.0, 170.0]})


@pytest.fixture
def fetcher():
    """ComparativeDataFetcher instance."""
    return ComparativeDataFetcher()


class TestSectorMapping:
    """Test sector to ETF mapping."""

    def test_technology_sector(self, fetcher):
        """Technology maps to XLK."""
        assert fetcher._get_sector_etf("Technology") == "XLK"

    def test_healthcare_sector(self, fetcher):
        """Healthcare maps to XLV."""
        assert fetcher._get_sector_etf("Healthcare") == "XLV"

    def test_financial_services_sector(self, fetcher):
        """Financial Services maps to XLF."""
        assert fetcher._get_sector_etf("Financial Services") == "XLF"

    def test_consumer_cyclical_sector(self, fetcher):
        """Consumer Cyclical maps to XLY."""
        assert fetcher._get_sector_etf("Consumer Cyclical") == "XLY"

    def test_unknown_sector_fallback(self, fetcher):
        """Unknown sector falls back to SPY."""
        assert fetcher._get_sector_etf("Unknown Sector") == "SPY"

    def test_none_sector_fallback(self, fetcher):
        """None sector falls back to SPY."""
        assert fetcher._get_sector_etf(None) == "SPY"


class TestSafeFloat:
    """Test safe float conversion."""

    def test_valid_float(self, fetcher):
        """Valid float value."""
        assert fetcher._safe_float(28.5) == 28.5

    def test_valid_string(self, fetcher):
        """Valid string converts to float."""
        assert fetcher._safe_float("28.5") == 28.5

    def test_none_value(self, fetcher):
        """None returns None."""
        assert fetcher._safe_float(None) is None

    def test_negative_pe_filtered(self, fetcher):
        """Negative P/E filtered out."""
        assert fetcher._safe_float(-5.0) is None

    def test_extreme_pe_filtered(self, fetcher):
        """Extreme P/E (>1000) filtered out."""
        assert fetcher._safe_float(1500.0) is None

    def test_invalid_string(self, fetcher):
        """Invalid string returns None."""
        assert fetcher._safe_float("not-a-number") is None


class TestFetchStockInfo:
    """Test stock info fetching."""

    @patch("src.data.comparative.yf.Ticker")
    def test_fetch_stock_info_success(self, mock_ticker_class, fetcher, mock_yfinance_ticker):
        """Successfully fetch stock info."""
        mock_ticker_class.return_value = mock_yfinance_ticker

        info = fetcher._fetch_stock_info("AAPL")

        assert info.symbol == "AAPL"
        assert info.sector == "Technology"
        assert info.industry == "Consumer Electronics"
        assert info.pe_ratio == 28.5
        assert info.price_to_book == 45.2

    @patch("src.data.comparative.yf.Ticker")
    def test_fetch_stock_info_no_data(self, mock_ticker_class, fetcher):
        """Raise ValueError when no data available."""
        mock_ticker = MagicMock()
        mock_ticker.info = {}
        mock_ticker_class.return_value = mock_ticker

        with pytest.raises(ValueError, match="No data available"):
            fetcher._fetch_stock_info("INVALID")


class TestFetchPerformance:
    """Test performance fetching."""

    @patch("src.data.comparative.yf.Ticker")
    def test_fetch_performance_success(self, mock_ticker_class, fetcher, mock_yfinance_history):
        """Successfully fetch performance data."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_yfinance_history
        mock_ticker_class.return_value = mock_ticker

        perf = fetcher._fetch_performance("AAPL")

        assert isinstance(perf, PerformanceData)
        # Returns should be calculated
        assert perf.ytd_return is not None or perf.three_month_return is not None

    @patch("src.data.comparative.yf.Ticker")
    def test_fetch_performance_empty_data(self, mock_ticker_class, fetcher):
        """Handle empty history data gracefully."""
        import pandas as pd

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_class.return_value = mock_ticker

        perf = fetcher._fetch_performance("AAPL")

        assert perf.ytd_return is None
        assert perf.three_month_return is None


class TestFetchComparativeData:
    """Test full comparative data fetching."""

    @patch("src.data.comparative.yf.Ticker")
    def test_fetch_comparative_data_success(self, mock_ticker_class, fetcher):
        """Successfully fetch comparative data."""
        import pandas as pd

        # Setup mock that returns different data for different symbols
        def create_mock_ticker(symbol):
            mock = MagicMock()
            if symbol == "AAPL":
                mock.info = {
                    "symbol": "AAPL",
                    "sector": "Technology",
                    "industry": "Consumer Electronics",
                    "trailingPE": 28.5,
                    "priceToBook": 45.2,
                }
            else:
                # Sector ETF or SPY
                mock.info = {"symbol": symbol, "trailingPE": 25.0}

            mock.history.return_value = pd.DataFrame({"Close": [100.0, 105.0, 110.0]})
            return mock

        mock_ticker_class.side_effect = create_mock_ticker

        data = fetcher.fetch_comparative_data("AAPL")

        assert isinstance(data, ComparativeData)
        assert data.stock_info.symbol == "AAPL"
        assert data.sector_etf == "XLK"
        assert isinstance(data.fetched_at, datetime)


class TestSectorEnum:
    """Test Sector enum values."""

    def test_all_sectors_have_etf_tickers(self):
        """All sectors map to valid ETF tickers."""
        for sector in Sector:
            assert len(sector.value) in (3, 4)  # All are 3-4 char ETFs
