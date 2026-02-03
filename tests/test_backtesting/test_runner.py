"""Tests for backtesting runner."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.backtesting.runner import BacktestResult, BacktestRunner


@pytest.fixture
def sample_backtest_data() -> pd.DataFrame:
    """Generate sample OHLCV data."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

    return pd.DataFrame(
        {
            "Open": [100.0 + i * 0.5 for i in range(100)],
            "High": [101.0 + i * 0.5 for i in range(100)],
            "Low": [99.0 + i * 0.5 for i in range(100)],
            "Close": [100.5 + i * 0.5 for i in range(100)],
            "Volume": [1000000] * 100,
        },
        index=dates,
    )


def test_backtest_runner_init():
    """BacktestRunner initializes with default values."""
    runner = BacktestRunner()

    assert runner.cash == 100000.0
    assert runner.commission == 0.002


def test_backtest_runner_init_custom_values():
    """BacktestRunner accepts custom cash and commission."""
    runner = BacktestRunner(cash=50000.0, commission=0.001)

    assert runner.cash == 50000.0
    assert runner.commission == 0.001


def test_backtest_runner_repr():
    """BacktestRunner has readable repr."""
    runner = BacktestRunner(cash=100000.0, commission=0.002)

    assert repr(runner) == "BacktestRunner(cash=$100,000.00, commission=0.20%)"


@patch("src.backtesting.runner.yf.Ticker")
def test_fetch_data(mock_ticker, sample_backtest_data):
    """_fetch_data retrieves historical data via yfinance."""
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.return_value = sample_backtest_data
    mock_ticker.return_value = mock_ticker_instance

    runner = BacktestRunner()
    data = runner._fetch_data("AAPL", datetime(2023, 1, 1), datetime(2023, 4, 10))

    assert len(data) == 100
    assert list(data.columns) == ["Open", "High", "Low", "Close", "Volume"]
    mock_ticker_instance.history.assert_called_once()


@patch("src.backtesting.runner.yf.Ticker")
def test_fetch_data_empty(mock_ticker):
    """_fetch_data raises ValueError on empty data."""
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.return_value = pd.DataFrame()
    mock_ticker.return_value = mock_ticker_instance

    runner = BacktestRunner()

    with pytest.raises(ValueError, match="No data available"):
        runner._fetch_data("INVALID", datetime(2023, 1, 1), datetime(2023, 4, 10))


@patch("src.backtesting.runner.BacktestRunner._fetch_data")
@patch("src.backtesting.runner.Backtest")
def test_run_backtest(mock_backtest, mock_fetch_data, sample_backtest_data):
    """run_backtest executes and returns BacktestResult."""
    mock_fetch_data.return_value = sample_backtest_data

    mock_stats = MagicMock()
    mock_stats.__getitem__ = lambda _, key: {
        "Return [%]": 15.5,
        "Sharpe Ratio": 1.2,
        "Max. Drawdown [%]": -8.3,
        "Win Rate [%]": 60.0,
        "# Trades": 10,
        "Avg. Trade [%]": 1.5,
    }[key]
    mock_stats._trades = []

    mock_backtest_instance = MagicMock()
    mock_backtest_instance.run.return_value = mock_stats
    mock_backtest.return_value = mock_backtest_instance

    runner = BacktestRunner()
    result = runner.run_backtest("AAPL", "2023-01-01", "2023-04-10")

    assert isinstance(result, BacktestResult)
    assert result.symbol == "AAPL"
    assert result.total_return == 0.155
    assert result.sharpe_ratio == 1.2
    assert result.max_drawdown == -0.083
    assert result.win_rate == 0.6
    assert result.total_trades == 10
    assert result.avg_return_per_trade == 0.015


@patch("src.backtesting.runner.BacktestRunner._fetch_data")
@patch("src.backtesting.runner.Backtest")
def test_run_backtest_with_datetime(mock_backtest, mock_fetch_data, sample_backtest_data):
    """run_backtest accepts datetime objects."""
    mock_fetch_data.return_value = sample_backtest_data

    mock_stats = MagicMock()
    mock_stats.__getitem__ = lambda _, key: {
        "Return [%]": 10.0,
        "Sharpe Ratio": 1.0,
        "Max. Drawdown [%]": -5.0,
        "Win Rate [%]": 55.0,
        "# Trades": 5,
        "Avg. Trade [%]": 2.0,
    }[key]
    mock_stats._trades = []

    mock_backtest_instance = MagicMock()
    mock_backtest_instance.run.return_value = mock_stats
    mock_backtest.return_value = mock_backtest_instance

    runner = BacktestRunner()
    result = runner.run_backtest("AAPL", datetime(2023, 1, 1), datetime(2023, 4, 10))

    assert result.start_date == datetime(2023, 1, 1)
    assert result.end_date == datetime(2023, 4, 10)


def test_convert_trades():
    """_convert_trades transforms backtesting.py trades."""
    runner = BacktestRunner()

    mock_trade_1 = MagicMock()
    mock_trade_1.Size = 10
    mock_trade_1.EntryPrice = 100.0
    mock_trade_1.ExitPrice = 110.0
    mock_trade_1.EntryTime = datetime(2023, 1, 1)
    mock_trade_1.PnL = 100.0

    mock_trade_2 = MagicMock()
    mock_trade_2.Size = -5
    mock_trade_2.EntryPrice = 120.0
    mock_trade_2.ExitPrice = None
    mock_trade_2.EntryTime = datetime(2023, 1, 5)
    mock_trade_2.PnL = 0.0

    mock_stats = MagicMock()
    mock_stats._trades = [mock_trade_1, mock_trade_2]

    records = runner._convert_trades(mock_stats, "AAPL")

    assert len(records) == 2

    assert records[0].symbol == "AAPL"
    assert records[0].action.value == "BUY"
    assert records[0].shares == 10
    assert records[0].entry_price == 100.0
    assert records[0].exit_price == 110.0
    assert records[0].pnl == 100.0
    assert records[0].status == "CLOSED"

    assert records[1].action.value == "SELL"
    assert records[1].shares == 5
    assert records[1].exit_price is None
    assert records[1].status == "OPEN"
