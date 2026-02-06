"""Tests for mean reversion strategy."""

import pandas as pd
import pytest

from src.strategies.mean_reversion import MeanReversionIndicators, MeanReversionStrategy
from src.strategies.signal import Signal


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


def test_strategy_init():
    strategy = MeanReversionStrategy(bb_period=20, bb_std=2.0)

    assert strategy.bb_period == 20
    assert strategy.bb_std == 2.0


def test_strategy_init_defaults():
    strategy = MeanReversionStrategy()

    assert strategy.bb_period == 20
    assert strategy.bb_std == 2.0


def test_calculate_indicators(sample_ohlcv):
    strategy = MeanReversionStrategy()
    result = strategy.calculate_indicators(sample_ohlcv)

    # pandas-ta bbands uses format: BB{L,M,U,B,P}_{length}_{std}... (float formatting varies)
    bb_period = strategy.bb_period
    assert any(col.startswith(f"BBL_{bb_period}") for col in result.columns)
    assert any(col.startswith(f"BBM_{bb_period}") for col in result.columns)
    assert any(col.startswith(f"BBU_{bb_period}") for col in result.columns)
    assert any(col.startswith(f"BBB_{bb_period}") for col in result.columns)
    assert any(col.startswith(f"BBP_{bb_period}") for col in result.columns)
    assert len(result) == len(sample_ohlcv)


def test_get_latest_indicators(sample_ohlcv):
    strategy = MeanReversionStrategy()
    df_with_indicators = strategy.calculate_indicators(sample_ohlcv)
    indicators = strategy.get_latest_indicators(df_with_indicators)

    assert isinstance(indicators, MeanReversionIndicators)
    assert isinstance(indicators.close, float)
    assert isinstance(indicators.bb_upper, float)
    assert isinstance(indicators.bb_middle, float)
    assert isinstance(indicators.bb_lower, float)
    assert isinstance(indicators.bb_width, float)
    assert isinstance(indicators.bb_percent, float)
    assert isinstance(indicators.oversold, bool)
    assert isinstance(indicators.overbought, bool)


def test_generate_signal_oversold():
    # Stable prices then sudden drop - last price breaches lower band
    prices = [100.0] * 40 + [100.0, 99.0, 97.0, 93.0, 88.0, 82.0, 75.0, 67.0, 58.0, 48.0]
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": [p + 1 for p in prices],
            "Low": [p - 1 for p in prices],
            "Close": prices,
            "Volume": [1000000] * 50,
        }
    )

    strategy = MeanReversionStrategy()
    signal, indicators = strategy.generate_signal(df)

    assert isinstance(signal, Signal)
    assert isinstance(indicators, MeanReversionIndicators)
    # Last close should be below lower band
    assert indicators.close < indicators.bb_lower
    assert indicators.oversold is True
    assert signal == Signal.BUY


def test_generate_signal_overbought():
    # Stable prices then sudden spike - last price breaches upper band
    prices = [100.0] * 40 + [100.0, 102.0, 105.0, 110.0, 118.0, 128.0, 140.0, 155.0, 172.0, 192.0]
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": [p + 1 for p in prices],
            "Low": [p - 1 for p in prices],
            "Close": prices,
            "Volume": [1000000] * 50,
        }
    )

    strategy = MeanReversionStrategy()
    signal, indicators = strategy.generate_signal(df)

    assert isinstance(signal, Signal)
    assert isinstance(indicators, MeanReversionIndicators)
    # Last close should be above upper band
    assert indicators.close > indicators.bb_upper
    assert indicators.overbought is True
    assert signal == Signal.SELL


def test_generate_signal_hold(sample_ohlcv):
    # Steady prices should stay within bands
    strategy = MeanReversionStrategy()
    signal, indicators = strategy.generate_signal(sample_ohlcv)

    assert isinstance(signal, Signal)
    assert isinstance(indicators, MeanReversionIndicators)
    # With steady upward trend, price should be within bands
    assert indicators.bb_lower <= indicators.close <= indicators.bb_upper
    assert indicators.oversold is False
    assert indicators.overbought is False
    assert signal == Signal.HOLD


def test_generate_signal_returns_tuple(sample_ohlcv):
    strategy = MeanReversionStrategy()
    result = strategy.generate_signal(sample_ohlcv)

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], Signal)
    assert isinstance(result[1], MeanReversionIndicators)


def test_repr():
    strategy = MeanReversionStrategy(bb_period=20, bb_std=2.0)
    expected = "MeanReversionStrategy(bb_period=20, bb_std=2.0)"
    assert repr(strategy) == expected


def test_calculate_indicators_preserves_original(sample_ohlcv):
    strategy = MeanReversionStrategy()
    original_len = len(sample_ohlcv.columns)

    strategy.calculate_indicators(sample_ohlcv)

    assert len(sample_ohlcv.columns) == original_len


def test_indicators_model_properties():
    indicators = MeanReversionIndicators(
        close=100.0,
        bb_upper=110.0,
        bb_middle=100.0,
        bb_lower=90.0,
        bb_width=20.0,
        bb_percent=0.5,
        oversold=False,
        overbought=False,
    )

    assert indicators.close == 100.0
    assert indicators.bb_upper == 110.0
    assert indicators.bb_lower == 90.0
    assert indicators.oversold is False
    assert indicators.overbought is False
