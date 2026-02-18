"""Unit tests for OptionsFlowFetcher."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.options_flow import OptionsFlowFetcher


def _make_chain_df(
    strikes: list[float],
    volumes: list[int],
    last_prices: list[float],
) -> pd.DataFrame:
    """Build a minimal options chain dataframe."""
    return pd.DataFrame(
        {
            "strike": strikes,
            "lastPrice": last_prices,
            "bid": [p * 0.95 for p in last_prices],
            "ask": [p * 1.05 for p in last_prices],
            "volume": volumes,
            "openInterest": [v * 10 for v in volumes],
            "impliedVolatility": [0.3] * len(strikes),
            "inTheMoney": [s < 180 for s in strikes],
        }
    )


@pytest.fixture
def fetcher() -> OptionsFlowFetcher:
    """Create fetcher instance."""
    return OptionsFlowFetcher(max_expirations=2)


@pytest.mark.unit
def test_fetch_options_chain_success(fetcher: OptionsFlowFetcher) -> None:
    """Fetch chains from 2 expirations, aggregate volume/OI."""
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    next_week = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")

    calls_df = _make_chain_df([170, 180, 190], [100, 200, 50], [5.0, 3.0, 1.0])
    puts_df = _make_chain_df([170, 180, 190], [80, 150, 30], [2.0, 4.0, 6.0])
    chain_mock = (calls_df, puts_df)

    ticker_mock = MagicMock()
    ticker_mock.options = (tomorrow, next_week)
    ticker_mock.option_chain.return_value = chain_mock

    with patch("yfinance.Ticker", return_value=ticker_mock):
        snapshot = fetcher.fetch_options_chain("AAPL")

    assert snapshot.symbol == "AAPL"
    # 2 expirations * (100+200+50) = 700
    assert snapshot.total_call_volume == 700
    assert snapshot.total_put_volume == 520
    assert len(snapshot.calls) == 6  # 3 strikes * 2 expirations
    assert len(snapshot.puts) == 6
    assert snapshot.near_term_expiry is not None
    assert not snapshot.is_empty


@pytest.mark.unit
def test_fetch_no_expirations(fetcher: OptionsFlowFetcher) -> None:
    """Return empty snapshot when no expirations available."""
    ticker_mock = MagicMock()
    ticker_mock.options = ()

    with patch("yfinance.Ticker", return_value=ticker_mock):
        snapshot = fetcher.fetch_options_chain("AAPL")

    assert snapshot.is_empty
    assert snapshot.symbol == "AAPL"


@pytest.mark.unit
def test_fetch_error_returns_empty(fetcher: OptionsFlowFetcher) -> None:
    """Return empty snapshot on exception."""
    with patch("yfinance.Ticker", side_effect=Exception("API error")):
        snapshot = fetcher.fetch_options_chain("AAPL")

    assert snapshot.is_empty
    assert snapshot.symbol == "AAPL"


@pytest.mark.unit
def test_fetch_skips_today_expiry(fetcher: OptionsFlowFetcher) -> None:
    """Today's expiration is skipped (stale data)."""
    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    calls_df = _make_chain_df([180], [100], [5.0])
    puts_df = _make_chain_df([180], [50], [3.0])
    chain_mock = (calls_df, puts_df)

    ticker_mock = MagicMock()
    ticker_mock.options = (today, tomorrow)
    ticker_mock.option_chain.return_value = chain_mock

    with patch("yfinance.Ticker", return_value=ticker_mock):
        snapshot = fetcher.fetch_options_chain("AAPL")

    # Only tomorrow's expiration should be fetched
    ticker_mock.option_chain.assert_called_once_with(tomorrow)


@pytest.mark.unit
def test_nan_handling(fetcher: OptionsFlowFetcher) -> None:
    """NaN values in volume/OI are handled via fillna(0)."""
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    calls_df = pd.DataFrame(
        {
            "strike": [180.0],
            "lastPrice": [5.0],
            "bid": [float("nan")],
            "ask": [float("nan")],
            "volume": [float("nan")],
            "openInterest": [float("nan")],
            "impliedVolatility": [0.3],
            "inTheMoney": [True],
        }
    )
    puts_df = _make_chain_df([180], [0], [0.0])
    chain_mock = (calls_df, puts_df)

    ticker_mock = MagicMock()
    ticker_mock.options = (tomorrow,)
    ticker_mock.option_chain.return_value = chain_mock

    with patch("yfinance.Ticker", return_value=ticker_mock):
        snapshot = fetcher.fetch_options_chain("AAPL")

    # NaN volume should become 0
    assert snapshot.total_call_volume == 0
    assert snapshot.calls[0].volume == 0
