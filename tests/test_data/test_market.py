"""Tests for market data fetcher."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.market import MarketData, MarketDataFetcher


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [105.0, 106.0, 107.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [104.0, 105.0, 106.0],
            "Volume": [1000000, 1100000, 1200000],
        }
    )


def test_market_data_latest_close(sample_df):
    market_data = MarketData(
        symbol="AAPL",
        data=sample_df,
        last_updated=datetime.now(),
    )

    assert market_data.latest_close == 106.0


def test_market_data_latest_volume(sample_df):
    market_data = MarketData(
        symbol="AAPL",
        data=sample_df,
        last_updated=datetime.now(),
    )

    assert market_data.latest_volume == 1200000


def test_fetcher_init_alpha_vantage():
    with patch("src.data.market.TimeSeries") as mock_ts:
        fetcher = MarketDataFetcher(use_alpha_vantage=True, api_key="test-key")

        assert fetcher.use_alpha_vantage is True
        mock_ts.assert_called_once_with(key="test-key", output_format="pandas")


def test_fetcher_init_alpha_vantage_no_key():
    with pytest.raises(ValueError, match="alpha_vantage_api_key not set in config"):
        MarketDataFetcher(use_alpha_vantage=True)


def test_fetcher_init_yfinance():
    fetcher = MarketDataFetcher(use_alpha_vantage=False)
    assert fetcher.use_alpha_vantage is False


def test_fetch_daily_alpha_vantage(sample_df):
    with patch("src.data.market.TimeSeries") as mock_ts:
        mock_instance = MagicMock()
        mock_ts.return_value = mock_instance
        mock_instance.get_daily.return_value = (sample_df.copy(), {})

        fetcher = MarketDataFetcher(use_alpha_vantage=True, api_key="test-key")
        result = fetcher.fetch_daily("AAPL")

        assert result.symbol == "AAPL"
        assert isinstance(result.data, pd.DataFrame)
        assert len(result.data) == 3
        mock_instance.get_daily.assert_called_once_with(symbol="AAPL", outputsize="compact")


def test_fetch_daily_yfinance(sample_df):
    with patch("src.data.market.yf.Ticker") as mock_ticker:
        mock_instance = MagicMock()
        mock_ticker.return_value = mock_instance
        mock_instance.history.return_value = sample_df.copy()

        fetcher = MarketDataFetcher(use_alpha_vantage=False)
        result = fetcher.fetch_daily("AAPL", period_days=90)

        assert result.symbol == "AAPL"
        assert isinstance(result.data, pd.DataFrame)
        assert len(result.data) == 3
        mock_instance.history.assert_called_once()


def test_fetch_daily_yfinance_empty():
    with patch("src.data.market.yf.Ticker") as mock_ticker:
        mock_instance = MagicMock()
        mock_ticker.return_value = mock_instance
        mock_instance.history.return_value = pd.DataFrame()

        fetcher = MarketDataFetcher(use_alpha_vantage=False)

        with pytest.raises(ValueError, match="No data returned for AAPL"):
            fetcher.fetch_daily("AAPL")


def test_fetch_intraday(sample_df):
    with patch("src.data.market.TimeSeries") as mock_ts:
        mock_instance = MagicMock()
        mock_ts.return_value = mock_instance
        mock_instance.get_intraday.return_value = (sample_df.copy(), {})

        fetcher = MarketDataFetcher(use_alpha_vantage=True, api_key="test-key")
        result = fetcher.fetch_intraday("AAPL", interval="5min")

        assert result.symbol == "AAPL"
        assert isinstance(result.data, pd.DataFrame)
        mock_instance.get_intraday.assert_called_once_with(
            symbol="AAPL",
            interval="5min",
            outputsize="compact",
        )


def test_fetch_intraday_yfinance_not_supported():
    fetcher = MarketDataFetcher(use_alpha_vantage=False)

    with pytest.raises(ValueError, match="Intraday data only available with Alpha Vantage"):
        fetcher.fetch_intraday("AAPL")


def test_repr():
    with patch("src.data.market.TimeSeries"):
        fetcher = MarketDataFetcher(use_alpha_vantage=False)
        assert repr(fetcher) == "MarketDataFetcher(source=yfinance)"


def test_fetch_alpha_vantage_retries_on_error(sample_df):
    with patch("src.data.market.TimeSeries") as mock_ts:
        mock_instance = MagicMock()
        mock_ts.return_value = mock_instance
        mock_instance.get_daily.side_effect = [
            Exception("API error"),
            (sample_df.copy(), {}),
        ]

        fetcher = MarketDataFetcher(use_alpha_vantage=True, api_key="test-key")
        result = fetcher.fetch_daily("AAPL")

        assert result.symbol == "AAPL"
        assert mock_instance.get_daily.call_count == 2


def test_fetch_yfinance_retries_on_error(sample_df):
    with patch("src.data.market.yf.Ticker") as mock_ticker:
        mock_instance = MagicMock()
        mock_ticker.return_value = mock_instance
        mock_instance.history.side_effect = [
            Exception("Network error"),
            sample_df.copy(),
        ]

        fetcher = MarketDataFetcher(use_alpha_vantage=False)
        result = fetcher.fetch_daily("AAPL", period_days=90)

        assert result.symbol == "AAPL"
        assert mock_instance.history.call_count == 2


def test_fetch_yfinance_no_retry_on_value_error():
    with patch("src.data.market.yf.Ticker") as mock_ticker:
        mock_instance = MagicMock()
        mock_ticker.return_value = mock_instance
        mock_instance.history.return_value = pd.DataFrame()

        fetcher = MarketDataFetcher(use_alpha_vantage=False)

        with pytest.raises(ValueError, match="No data returned for AAPL"):
            fetcher.fetch_daily("AAPL")

        assert mock_instance.history.call_count == 1


def test_fetch_intraday_retries_on_error(sample_df):
    with patch("src.data.market.TimeSeries") as mock_ts:
        mock_instance = MagicMock()
        mock_ts.return_value = mock_instance
        mock_instance.get_intraday.side_effect = [
            Exception("API error"),
            (sample_df.copy(), {}),
        ]

        fetcher = MarketDataFetcher(use_alpha_vantage=True, api_key="test-key")
        result = fetcher.fetch_intraday("AAPL", interval="5min")

        assert result.symbol == "AAPL"
        assert mock_instance.get_intraday.call_count == 2


def test_fetch_alpha_vantage_exhausts_retries():
    with patch("src.data.market.TimeSeries") as mock_ts:
        mock_instance = MagicMock()
        mock_ts.return_value = mock_instance
        mock_instance.get_daily.side_effect = Exception("API error")

        fetcher = MarketDataFetcher(use_alpha_vantage=True, api_key="test-key")

        with pytest.raises(Exception, match="API error"):
            fetcher.fetch_daily("AAPL")

        assert mock_instance.get_daily.call_count == 3


def test_fetch_overnight_futures_valid():
    """Test fetch_overnight_futures with valid data."""
    with patch("src.data.market.yf.Ticker") as mock_ticker:
        mock_instance = MagicMock()
        mock_ticker.return_value = mock_instance
        mock_df = pd.DataFrame({"Close": [100.0, 102.0]})
        mock_instance.history.return_value = mock_df

        fetcher = MarketDataFetcher(use_alpha_vantage=False)
        result = fetcher.fetch_overnight_futures(["ES=F", "NQ=F"])

        assert result == {"ES=F": 2.0, "NQ=F": 2.0}
        assert mock_instance.history.call_count == 2


def test_fetch_overnight_futures_insufficient_data():
    """Test fetch_overnight_futures with insufficient data returns empty dict."""
    with patch("src.data.market.yf.Ticker") as mock_ticker:
        mock_instance = MagicMock()
        mock_ticker.return_value = mock_instance
        mock_df = pd.DataFrame({"Close": [100.0]})
        mock_instance.history.return_value = mock_df

        fetcher = MarketDataFetcher(use_alpha_vantage=False)

        result = fetcher.fetch_overnight_futures(["ES=F"])
        assert result == {}


def test_fetch_overnight_futures_all_fail():
    """Test fetch_overnight_futures when all symbols fail returns empty dict."""
    with patch("src.data.market.yf.Ticker") as mock_ticker:
        mock_instance = MagicMock()
        mock_ticker.return_value = mock_instance
        mock_instance.history.side_effect = Exception("Network error")

        fetcher = MarketDataFetcher(use_alpha_vantage=False)

        result = fetcher.fetch_overnight_futures(["ES=F", "NQ=F"])
        assert result == {}
