"""Tests for backtesting strategies."""

import pandas as pd
import pytest

from src.backtesting.strategies import MomentumBacktestStrategy


@pytest.fixture
def sample_backtest_data() -> pd.DataFrame:
    """Generate sample OHLCV data for backtesting."""
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


def test_momentum_strategy_has_tunable_parameters():
    """MomentumBacktestStrategy has configurable parameters."""
    assert hasattr(MomentumBacktestStrategy, "rsi_period")
    assert hasattr(MomentumBacktestStrategy, "rsi_oversold")
    assert hasattr(MomentumBacktestStrategy, "rsi_overbought")
    assert hasattr(MomentumBacktestStrategy, "macd_fast")
    assert hasattr(MomentumBacktestStrategy, "macd_slow")
    assert hasattr(MomentumBacktestStrategy, "macd_signal")


def test_momentum_strategy_default_parameters():
    """Default parameter values are sensible."""
    assert MomentumBacktestStrategy.rsi_period == 14
    assert MomentumBacktestStrategy.rsi_oversold == 30
    assert MomentumBacktestStrategy.rsi_overbought == 70
    assert MomentumBacktestStrategy.macd_fast == 12
    assert MomentumBacktestStrategy.macd_slow == 26
    assert MomentumBacktestStrategy.macd_signal == 9
