"""Tests for vectorized backtesting runner."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.backtesting.vectorbt_runner import MultiAssetBacktest, VectorBTResult, VectorBTRunner


@pytest.fixture
def sample_backtest_data() -> pd.DataFrame:
    """Generate sample OHLCV data with RSI/MACD-triggering patterns."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

    # Create price pattern that triggers buy/sell signals
    close_prices = []
    for i in range(100):
        if i < 30:
            close_prices.append(100.0 - i * 0.5)  # Downtrend (RSI drops)
        elif i < 60:
            close_prices.append(85.0 + (i - 30) * 1.0)  # Uptrend (RSI rises)
        else:
            close_prices.append(115.0 - (i - 60) * 0.3)  # Mild downtrend

    return pd.DataFrame(
        {
            "Open": [p - 0.5 for p in close_prices],
            "High": [p + 1.0 for p in close_prices],
            "Low": [p - 1.0 for p in close_prices],
            "Close": close_prices,
            "Volume": [1000000] * 100,
        },
        index=dates,
    )


def test_vectorbt_runner_init():
    """VectorBTRunner initializes with default values."""
    runner = VectorBTRunner()

    assert runner.cash == 100_000.0
    assert runner.commission == 0.002


def test_vectorbt_runner_init_custom_values():
    """VectorBTRunner accepts custom cash and commission."""
    runner = VectorBTRunner(cash=50_000.0, commission=0.001)

    assert runner.cash == 50_000.0
    assert runner.commission == 0.001


def test_vectorbt_runner_repr():
    """VectorBTRunner has readable repr."""
    runner = VectorBTRunner(cash=100_000.0, commission=0.002)

    assert repr(runner) == "VectorBTRunner(cash=$100,000.00, commission=0.20%)"


@patch("src.backtesting.vectorbt_runner.yf.Ticker")
def test_fetch_data(mock_ticker, sample_backtest_data):
    """_fetch_data retrieves historical data via yfinance."""
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.return_value = sample_backtest_data
    mock_ticker.return_value = mock_ticker_instance

    runner = VectorBTRunner()
    data = runner._fetch_data("AAPL", datetime(2023, 1, 1), datetime(2023, 4, 10))

    assert len(data) == 100
    assert list(data.columns) == ["Open", "High", "Low", "Close", "Volume"]
    mock_ticker_instance.history.assert_called_once()


@patch("src.backtesting.vectorbt_runner.yf.Ticker")
def test_fetch_data_empty(mock_ticker):
    """_fetch_data raises ValueError on empty data."""
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.return_value = pd.DataFrame()
    mock_ticker.return_value = mock_ticker_instance

    runner = VectorBTRunner()

    with pytest.raises(ValueError, match="No data available"):
        runner._fetch_data("INVALID", datetime(2023, 1, 1), datetime(2023, 4, 10))


@patch("src.backtesting.vectorbt_runner.VectorBTRunner._fetch_data")
def test_run_backtest(mock_fetch_data, sample_backtest_data):
    """run_backtest returns VectorBTResult with correct fields."""
    mock_fetch_data.return_value = sample_backtest_data

    runner = VectorBTRunner()
    result = runner.run_backtest("AAPL", "2023-01-01", "2023-04-10")

    assert isinstance(result, VectorBTResult)
    assert result.symbol == "AAPL"
    assert isinstance(result.total_return, float)
    assert isinstance(result.sharpe_ratio, float)
    assert isinstance(result.sortino_ratio, float)
    assert isinstance(result.max_drawdown, float)
    assert isinstance(result.calmar_ratio, float)
    assert 0.0 <= result.win_rate <= 1.0
    assert isinstance(result.profit_factor, float)
    assert isinstance(result.total_trades, int)
    assert isinstance(result.equity_curve, list)
    assert len(result.equity_curve) == 100
    assert isinstance(result.equity_dates, list)
    assert len(result.equity_dates) == 100
    assert all(isinstance(d, datetime) for d in result.equity_dates)
    assert result.start_date == datetime(2023, 1, 1)
    assert result.end_date == datetime(2023, 4, 10)


@patch("src.backtesting.vectorbt_runner.VectorBTRunner._fetch_data")
def test_run_backtest_with_datetime(mock_fetch_data, sample_backtest_data):
    """run_backtest accepts datetime objects."""
    mock_fetch_data.return_value = sample_backtest_data

    runner = VectorBTRunner()
    result = runner.run_backtest("AAPL", datetime(2023, 1, 1), datetime(2023, 4, 10))

    assert result.start_date == datetime(2023, 1, 1)
    assert result.end_date == datetime(2023, 4, 10)


@patch("src.backtesting.vectorbt_runner.VectorBTRunner._fetch_data")
def test_run_portfolio_backtest(mock_fetch_data, sample_backtest_data):
    """run_portfolio_backtest returns MultiAssetBacktest."""
    mock_fetch_data.return_value = sample_backtest_data

    runner = VectorBTRunner()
    result = runner.run_portfolio_backtest(["AAPL", "MSFT"], "2023-01-01", "2023-04-10")

    assert isinstance(result, MultiAssetBacktest)
    assert result.symbols == ["AAPL", "MSFT"]
    assert len(result.results) == 2
    assert isinstance(result.portfolio_sharpe, float)
    assert isinstance(result.portfolio_return, float)
    assert isinstance(result.portfolio_max_drawdown, float)
    assert "AAPL" in result.correlation_matrix
    assert "MSFT" in result.correlation_matrix["AAPL"]


def test_generate_signals(sample_backtest_data):
    """_generate_signals returns boolean entry/exit arrays."""
    runner = VectorBTRunner()
    entries, exits = runner._generate_signals(sample_backtest_data)

    assert len(entries) == 100
    assert len(exits) == 100
    assert entries.dtype == bool
    assert exits.dtype == bool


def test_vectorbt_result_model():
    """VectorBTResult validates fields correctly."""
    result = VectorBTResult(
        total_return=0.15,
        sharpe_ratio=1.5,
        sortino_ratio=2.0,
        max_drawdown=-0.1,
        calmar_ratio=1.5,
        win_rate=0.6,
        profit_factor=2.0,
        total_trades=10,
        equity_curve=[100000.0, 100500.0, 101000.0],
        equity_dates=[datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)],
        symbol="AAPL",
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 4, 10),
    )

    assert result.total_return == 0.15
    assert result.symbol == "AAPL"
    assert len(result.equity_curve) == 3
    assert len(result.equity_dates) == 3
