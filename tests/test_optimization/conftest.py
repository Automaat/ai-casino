"""Optimization-specific test fixtures."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.backtesting.runner import BacktestResult
from src.optimization.search_space import ParamRange, SearchSpace, StrategyType


@pytest.fixture
def mock_backtest_runner():
    """Mock backtest runner."""
    mock = MagicMock()
    mock.run_backtest.return_value = BacktestResult(
        symbol="AAPL",
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2024, 1, 1),
        total_return=0.15,
        sharpe_ratio=1.2,
        max_drawdown=-0.08,
        win_rate=0.55,
        total_trades=10,
        avg_return_per_trade=0.015,
        trades=[],
    )
    return mock


@pytest.fixture
def sample_momentum_search_space():
    """Sample momentum search space."""
    return SearchSpace(
        strategy=StrategyType.MOMENTUM,
        params=[
            ParamRange(name="rsi_period", low=10, high=20, step=1, is_int=True),
            ParamRange(name="rsi_oversold", low=25, high=35, step=5),
            ParamRange(name="rsi_overbought", low=65, high=75, step=5),
        ],
    )


@pytest.fixture
def sample_dates():
    """Sample date range."""
    return {
        "start_date": "2023-01-01",
        "end_date": "2024-01-01",
    }
