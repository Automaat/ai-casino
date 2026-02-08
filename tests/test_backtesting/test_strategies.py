"""Tests for backtesting strategies."""

from unittest.mock import patch

import pandas as pd
import pytest

from backtesting import Backtest
from src.backtesting.strategies import (
    EnsembleBacktestStrategy,
    MomentumBacktestStrategy,
    TrendFollowingBacktestStrategy,
)


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


@patch("pandas_ta_classic.rsi")
@patch("pandas_ta_classic.macd")
def test_momentum_strategy_rsi_none_fallback(mock_macd, mock_rsi, sample_backtest_data):
    """MomentumBacktestStrategy handles RSI None fallback without error."""
    mock_rsi.return_value = None
    mock_macd.return_value = pd.DataFrame({"MACDh_12_26_9": [0.0] * 100})

    bt = Backtest(sample_backtest_data, MomentumBacktestStrategy, commission=0.002, exclusive_orders=True)
    stats = bt.run()

    assert stats is not None
    assert "Return [%]" in stats


@patch("pandas_ta_classic.sma")
@patch("pandas_ta_classic.adx")
def test_trend_following_strategy_sma_none_fallback(mock_adx, mock_sma, sample_backtest_data):
    """TrendFollowingBacktestStrategy handles SMA None fallback without error."""
    mock_sma.return_value = None
    mock_adx.return_value = pd.DataFrame(
        {"ADX_14": [0.0] * 100, "DMP_14": [0.0] * 100, "DMN_14": [0.0] * 100}
    )

    bt = Backtest(
        sample_backtest_data, TrendFollowingBacktestStrategy, commission=0.002, exclusive_orders=True
    )
    stats = bt.run()

    assert stats is not None
    assert "Return [%]" in stats


@patch("pandas_ta_classic.rsi")
@patch("pandas_ta_classic.macd")
@patch("pandas_ta_classic.sma")
@patch("pandas_ta_classic.adx")
@patch("pandas_ta_classic.bbands")
def test_ensemble_strategy_rsi_none_fallback(
    mock_bbands, mock_adx, mock_sma, mock_macd, mock_rsi, sample_backtest_data
):
    """EnsembleBacktestStrategy handles RSI None fallback without error."""
    mock_rsi.return_value = None
    mock_macd.return_value = pd.DataFrame({"MACDh_12_26_9": [0.0] * 100})
    mock_sma.return_value = pd.Series([100.0] * 100)
    mock_adx.return_value = pd.DataFrame(
        {"ADX_14": [0.0] * 100, "DMP_14": [0.0] * 100, "DMN_14": [0.0] * 100}
    )
    mock_bbands.return_value = pd.DataFrame({"BBL_20_2.0": [95.0] * 100, "BBU_20_2.0": [105.0] * 100})

    bt = Backtest(sample_backtest_data, EnsembleBacktestStrategy, commission=0.002, exclusive_orders=True)
    stats = bt.run()

    assert stats is not None
    assert "Return [%]" in stats


@patch("pandas_ta_classic.rsi")
@patch("pandas_ta_classic.macd")
@patch("pandas_ta_classic.sma")
@patch("pandas_ta_classic.adx")
@patch("pandas_ta_classic.bbands")
def test_ensemble_strategy_sma_fast_none_fallback(
    mock_bbands, mock_adx, mock_sma, mock_macd, mock_rsi, sample_backtest_data
):
    """EnsembleBacktestStrategy handles fast SMA None fallback without error."""
    mock_rsi.return_value = pd.Series([50.0] * 100)
    mock_macd.return_value = pd.DataFrame({"MACDh_12_26_9": [0.0] * 100})
    mock_adx.return_value = pd.DataFrame(
        {"ADX_14": [0.0] * 100, "DMP_14": [0.0] * 100, "DMN_14": [0.0] * 100}
    )
    mock_bbands.return_value = pd.DataFrame({"BBL_20_2.0": [95.0] * 100, "BBU_20_2.0": [105.0] * 100})

    # Return None only on first call (fast SMA), Series on second (slow SMA)
    mock_sma.side_effect = [None, pd.Series([100.0] * 100)]

    bt = Backtest(sample_backtest_data, EnsembleBacktestStrategy, commission=0.002, exclusive_orders=True)
    stats = bt.run()

    assert stats is not None
    assert "Return [%]" in stats


@patch("pandas_ta_classic.rsi")
@patch("pandas_ta_classic.macd")
@patch("pandas_ta_classic.sma")
@patch("pandas_ta_classic.adx")
@patch("pandas_ta_classic.bbands")
def test_ensemble_strategy_sma_slow_none_fallback(
    mock_bbands, mock_adx, mock_sma, mock_macd, mock_rsi, sample_backtest_data
):
    """EnsembleBacktestStrategy handles slow SMA None fallback without error."""
    mock_rsi.return_value = pd.Series([50.0] * 100)
    mock_macd.return_value = pd.DataFrame({"MACDh_12_26_9": [0.0] * 100})
    mock_adx.return_value = pd.DataFrame(
        {"ADX_14": [0.0] * 100, "DMP_14": [0.0] * 100, "DMN_14": [0.0] * 100}
    )
    mock_bbands.return_value = pd.DataFrame({"BBL_20_2.0": [95.0] * 100, "BBU_20_2.0": [105.0] * 100})

    # Return Series on first call (fast SMA), None on second (slow SMA)
    mock_sma.side_effect = [pd.Series([100.0] * 100), None]

    bt = Backtest(sample_backtest_data, EnsembleBacktestStrategy, commission=0.002, exclusive_orders=True)
    stats = bt.run()

    assert stats is not None
    assert "Return [%]" in stats
