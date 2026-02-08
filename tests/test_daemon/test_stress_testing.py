"""Tests for Monte Carlo stress testing daemon integration."""

import zlib
from datetime import UTC, datetime
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.daemon.config import MonteCarloConfig
from src.daemon.state import MonteCarloRecord
from src.daemon.stress_testing import DaemonStressTester
from src.data.broker import BrokerPosition
from src.data.market import MarketData
from src.metrics.monte_carlo import MonteCarloResult, SimulationMethod


@pytest.fixture
def monte_carlo_config():
    """Monte Carlo configuration for testing."""
    return MonteCarloConfig(
        enabled=True,
        schedule_time="17:00",
        schedule_days=["sun"],
        num_simulations=1000,
        horizon_days=252,
        simulation_method="PARAMETRIC",
        min_historical_days=90,
    )


@pytest.fixture
def mock_broker(mocker):
    """Mock broker with sample positions."""
    from src.data.broker import BrokerAccountInfo

    broker = mocker.Mock()
    broker.get_account_info.return_value = BrokerAccountInfo(
        balance=10000.0,
        available_cash=5000.0,
        positions={
            "AAPL": BrokerPosition(
                symbol="AAPL",
                qty=10,
                market_value=1500.0,
                avg_entry_price=140.0,
                unrealized_pnl=100.0,
                unrealized_pnl_percent=0.067,
            ),
            "MSFT": BrokerPosition(
                symbol="MSFT",
                qty=5,
                market_value=1000.0,
                avg_entry_price=190.0,
                unrealized_pnl=50.0,
                unrealized_pnl_percent=0.053,
            ),
        },
        total_exposure=2500.0,
        portfolio_value=10000.0,
    )
    return broker


@pytest.fixture
def mock_market_fetcher(mocker):
    """Mock market data fetcher."""
    fetcher = mocker.Mock()
    # Use a shared date range for all symbols to ensure alignment
    shared_dates = pd.date_range(end=datetime.now(UTC), periods=100, freq="D")

    def fetch_daily(symbol, period_days):
        # Generate sample historical data with slight variations per symbol
        np.random.seed(zlib.crc32(symbol.encode()) % 2**32)  # Stable hash per symbol
        base_price = 150.0
        close_prices = base_price + np.random.normal(0, 2, 100).cumsum()

        df = pd.DataFrame(
            {
                "open": close_prices,
                "high": close_prices + 2,
                "low": close_prices - 2,
                "close": close_prices,
                "volume": 1000000,
            },
            index=shared_dates,  # Use shared dates for alignment
        )
        return MarketData(symbol=symbol, data=df, last_updated=datetime.now(UTC))

    fetcher.fetch_daily = Mock(side_effect=fetch_daily)
    return fetcher


def test_executor_end_to_end(mock_broker, mock_market_fetcher, monte_carlo_config):
    """Test full execution flow."""
    executor = DaemonStressTester(mock_broker, mock_market_fetcher, monte_carlo_config)
    record = executor.execute()

    assert isinstance(record, MonteCarloRecord)
    assert record.portfolio_symbols == ["AAPL", "MSFT"]
    assert record.num_simulations == 1000
    assert record.horizon_days == 252
    assert record.simulation_method == "PARAMETRIC"
    assert 0.0 <= record.prob_loss_gt_threshold <= 1.0
    assert record.cvar_95 <= record.var_95  # CVaR more negative than VaR
    assert record.total_market_value == 2500.0


def test_executor_handles_no_positions(mock_broker, mock_market_fetcher, monte_carlo_config):
    """Test error when portfolio is empty."""
    from src.data.broker import BrokerAccountInfo

    mock_broker.get_account_info.return_value = BrokerAccountInfo(
        balance=10000.0,
        available_cash=10000.0,
        positions={},
        total_exposure=0.0,
        portfolio_value=10000.0,
    )

    executor = DaemonStressTester(mock_broker, mock_market_fetcher, monte_carlo_config)
    with pytest.raises(ValueError, match="No positions in portfolio"):
        executor.execute()


def test_executor_handles_insufficient_data(mock_broker, mock_market_fetcher, monte_carlo_config):
    """Test error when historical data insufficient."""

    def fetch_daily_limited(symbol, period_days):
        # Only 20 days of data
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(UTC), periods=20, freq="D")
        base_price = 150.0
        close_prices = base_price + np.random.normal(0, 2, 20).cumsum()

        df = pd.DataFrame(
            {
                "open": close_prices,
                "high": close_prices + 2,
                "low": close_prices - 2,
                "close": close_prices,
                "volume": 1000000,
            },
            index=dates,
        )
        return MarketData(symbol=symbol, data=df, last_updated=datetime.now(UTC))

    mock_market_fetcher.fetch_daily = Mock(side_effect=fetch_daily_limited)

    executor = DaemonStressTester(mock_broker, mock_market_fetcher, monte_carlo_config)
    with pytest.raises(ValueError, match=r"Only .* days available"):
        executor.execute()


def test_executor_risk_threshold_alert(mock_broker, mock_market_fetcher, mocker):
    """Test alert when risk exceeds threshold."""
    from src.metrics.monte_carlo import MonteCarloSimulator

    config = MonteCarloConfig(
        enabled=True,
        num_simulations=100,
        max_acceptable_prob=0.05,  # Very low threshold
        random_seed=42,
    )

    executor = DaemonStressTester(mock_broker, mock_market_fetcher, config)

    # Patch simulator result to force high probability exceeding threshold
    mock_result = MonteCarloResult(
        simulation_method=SimulationMethod.PARAMETRIC,
        num_simulations=100,
        horizon_days=252,
        prob_loss_gt_threshold=0.20,  # Exceeds 5% threshold
        expected_worst_drawdown=-0.15,
        var_95=-0.08,
        cvar_95=-0.12,
        median_recovery_days=30.0,
        mean_return=0.05,
        std_return=0.15,
        min_return=-0.25,
        max_return=0.35,
        simulated_returns=[],
    )
    mocker.patch.object(MonteCarloSimulator, "simulate", return_value=mock_result)

    record = executor.execute()

    # Now we can assert deterministically
    assert isinstance(record, MonteCarloRecord)
    assert record.exceeds_risk_tolerance is True
    assert record.alert_message is not None
    assert "exceeds threshold" in record.alert_message.lower()


def test_executor_fetches_correct_lookback(mock_broker, mock_market_fetcher, monte_carlo_config):
    """Test that executor fetches correct lookback period."""
    executor = DaemonStressTester(mock_broker, mock_market_fetcher, monte_carlo_config)
    executor.execute()

    # Should fetch max(min_historical_days, horizon_days * 2) = max(90, 504) = 504
    expected_lookback = max(monte_carlo_config.min_historical_days, monte_carlo_config.horizon_days * 2)
    mock_market_fetcher.fetch_daily.assert_any_call("AAPL", period_days=expected_lookback)
    mock_market_fetcher.fetch_daily.assert_any_call("MSFT", period_days=expected_lookback)


def test_executor_bootstrap_method(mock_broker, mock_market_fetcher):
    """Test bootstrap simulation method."""
    config = MonteCarloConfig(enabled=True, num_simulations=1000, simulation_method="BOOTSTRAP")

    executor = DaemonStressTester(mock_broker, mock_market_fetcher, config)
    record = executor.execute()

    assert record.simulation_method == "BOOTSTRAP"
    assert 0.0 <= record.prob_loss_gt_threshold <= 1.0


def test_executor_gbm_method(mock_broker, mock_market_fetcher):
    """Test GBM simulation method."""
    config = MonteCarloConfig(enabled=True, num_simulations=1000, simulation_method="GBM")

    executor = DaemonStressTester(mock_broker, mock_market_fetcher, config)
    record = executor.execute()

    assert record.simulation_method == "GBM"
    assert 0.0 <= record.prob_loss_gt_threshold <= 1.0
