"""Tests for momentum strategy."""

import pandas as pd
import pytest

from src.strategies.momentum import MomentumIndicators, MomentumStrategy, Signal


@pytest.fixture
def sample_ohlcv():
    return pd.DataFrame(
        {
            "Open": [100 + i for i in range(50)],
            "High": [105 + i for i in range(50)],
            "Low": [99 + i for i in range(50)],
            "Close": [104 + i for i in range(50)],
            "Volume": [1000000] * 50,
        }
    )


def test_momentum_strategy_init():
    strategy = MomentumStrategy(
        rsi_period=14,
        rsi_oversold=30.0,
        rsi_overbought=70.0,
    )

    assert strategy.rsi_period == 14
    assert strategy.rsi_oversold == 30.0
    assert strategy.rsi_overbought == 70.0
    assert strategy.macd_fast == 12
    assert strategy.macd_slow == 26
    assert strategy.macd_signal == 9


def test_calculate_indicators(sample_ohlcv):
    strategy = MomentumStrategy()
    result = strategy.calculate_indicators(sample_ohlcv)

    assert "RSI_14" in result.columns
    assert "MACD_12_26_9" in result.columns
    assert "MACDs_12_26_9" in result.columns
    assert "MACDh_12_26_9" in result.columns
    assert len(result) == len(sample_ohlcv)


def test_get_latest_indicators(sample_ohlcv):
    strategy = MomentumStrategy()
    df_with_indicators = strategy.calculate_indicators(sample_ohlcv)
    indicators = strategy.get_latest_indicators(df_with_indicators)

    assert isinstance(indicators, MomentumIndicators)
    assert isinstance(indicators.rsi, float)
    assert isinstance(indicators.macd, float)
    assert isinstance(indicators.rsi_oversold, bool)
    assert isinstance(indicators.macd_bullish, bool)


def test_momentum_indicators_properties():
    indicators = MomentumIndicators(
        rsi=25.0,
        rsi_oversold=True,
        rsi_overbought=False,
        macd=0.5,
        macd_signal=0.3,
        macd_hist=0.2,
        macd_bullish=True,
        macd_bearish=False,
    )

    assert indicators.rsi == 25.0
    assert indicators.rsi_oversold is True
    assert indicators.macd_bullish is True


def test_generate_signal_buy(sample_ohlcv):
    declining_prices = pd.DataFrame(
        {
            "Open": [150 - i * 2 for i in range(50)],
            "High": [155 - i * 2 for i in range(50)],
            "Low": [149 - i * 2 for i in range(50)],
            "Close": [154 - i * 2 for i in range(50)],
            "Volume": [1000000] * 50,
        }
    )

    strategy = MomentumStrategy()
    signal, indicators = strategy.generate_signal(declining_prices)

    assert isinstance(signal, Signal)
    assert isinstance(indicators, MomentumIndicators)
    assert indicators.rsi < 70


def test_generate_signal_returns_tuple(sample_ohlcv):
    strategy = MomentumStrategy()
    result = strategy.generate_signal(sample_ohlcv)

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], Signal)
    assert isinstance(result[1], MomentumIndicators)


def test_repr():
    strategy = MomentumStrategy(rsi_period=14, rsi_oversold=30.0, rsi_overbought=70.0)
    expected = "MomentumStrategy(rsi_period=14, oversold=30.0, overbought=70.0)"
    assert repr(strategy) == expected


def test_signal_enum_values():
    assert Signal.BUY.value == "BUY"
    assert Signal.SELL.value == "SELL"
    assert Signal.HOLD.value == "HOLD"


def test_calculate_indicators_preserves_original(sample_ohlcv):
    strategy = MomentumStrategy()
    original_len = len(sample_ohlcv.columns)

    strategy.calculate_indicators(sample_ohlcv)

    assert len(sample_ohlcv.columns) == original_len
