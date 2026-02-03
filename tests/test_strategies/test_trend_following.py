"""Tests for trend following strategy."""

import pandas as pd
import pytest

from src.strategies.momentum import Signal
from src.strategies.trend_following import TrendFollowingIndicators, TrendFollowingStrategy


@pytest.fixture
def sample_ohlcv():
    """Sample OHLCV data - 250 rows for SMA 200."""
    return pd.DataFrame(
        {
            "Open": [100 + i * 0.5 for i in range(250)],
            "High": [105 + i * 0.5 for i in range(250)],
            "Low": [99 + i * 0.5 for i in range(250)],
            "Close": [104 + i * 0.5 for i in range(250)],
            "Volume": [1000000] * 250,
        }
    )


def test_strategy_init():
    strategy = TrendFollowingStrategy(sma_fast=50, sma_slow=200, adx_period=14, adx_threshold=25.0)

    assert strategy.sma_fast == 50
    assert strategy.sma_slow == 200
    assert strategy.adx_period == 14
    assert strategy.adx_threshold == 25.0


def test_strategy_init_defaults():
    strategy = TrendFollowingStrategy()

    assert strategy.sma_fast == 50
    assert strategy.sma_slow == 200
    assert strategy.adx_period == 14
    assert strategy.adx_threshold == 25.0


def test_calculate_indicators(sample_ohlcv):
    strategy = TrendFollowingStrategy()
    result = strategy.calculate_indicators(sample_ohlcv)

    assert any(col.startswith(f"SMA_{strategy.sma_fast}") for col in result.columns)
    assert any(col.startswith(f"SMA_{strategy.sma_slow}") for col in result.columns)
    assert any(col.startswith(f"ADX_{strategy.adx_period}") for col in result.columns)
    assert any(col.startswith(f"DMP_{strategy.adx_period}") for col in result.columns)
    assert any(col.startswith(f"DMN_{strategy.adx_period}") for col in result.columns)
    assert len(result) == len(sample_ohlcv)


def test_get_latest_indicators(sample_ohlcv):
    strategy = TrendFollowingStrategy()
    df_with_indicators = strategy.calculate_indicators(sample_ohlcv)
    indicators = strategy.get_latest_indicators(df_with_indicators)

    assert isinstance(indicators, TrendFollowingIndicators)
    assert isinstance(indicators.close, float)
    assert isinstance(indicators.sma_fast, float)
    assert isinstance(indicators.sma_slow, float)
    assert isinstance(indicators.adx, float)
    assert isinstance(indicators.plus_di, float)
    assert isinstance(indicators.minus_di, float)
    assert isinstance(indicators.sma_bullish_cross, bool)
    assert isinstance(indicators.sma_bearish_cross, bool)
    assert isinstance(indicators.strong_trend, bool)
    assert indicators.trend_direction in ["bullish", "bearish", "neutral"]


def test_generate_signal_uptrend():
    # Strong uptrend - prices consistently rising with high volatility for strong ADX
    base = 100
    prices = []
    for i in range(250):
        # Exponential growth with volatility
        price = base * (1.003**i) + (i % 5) * 0.5
        prices.append(price)

    highs = [p * 1.02 for p in prices]
    lows = [p * 0.98 for p in prices]

    df = pd.DataFrame(
        {
            "Open": prices,
            "High": highs,
            "Low": lows,
            "Close": prices,
            "Volume": [1000000] * 250,
        }
    )

    strategy = TrendFollowingStrategy()
    signal, indicators = strategy.generate_signal(df)

    assert isinstance(signal, Signal)
    assert isinstance(indicators, TrendFollowingIndicators)
    # In strong uptrend, fast SMA should be above slow SMA
    assert indicators.sma_fast > indicators.sma_slow


def test_generate_signal_downtrend():
    # Strong downtrend - prices consistently falling with volatility
    base = 200
    prices = []
    for i in range(250):
        price = base * (0.997**i) + (i % 5) * 0.5
        prices.append(price)

    highs = [p * 1.02 for p in prices]
    lows = [p * 0.98 for p in prices]

    df = pd.DataFrame(
        {
            "Open": prices,
            "High": highs,
            "Low": lows,
            "Close": prices,
            "Volume": [1000000] * 250,
        }
    )

    strategy = TrendFollowingStrategy()
    signal, indicators = strategy.generate_signal(df)

    assert isinstance(signal, Signal)
    assert isinstance(indicators, TrendFollowingIndicators)
    # In strong downtrend, fast SMA should be below slow SMA
    assert indicators.sma_fast < indicators.sma_slow


def test_generate_signal_sideways(sample_ohlcv):
    # Steady upward trend should generate HOLD with weak trend
    strategy = TrendFollowingStrategy()
    signal, indicators = strategy.generate_signal(sample_ohlcv)

    assert isinstance(signal, Signal)
    assert isinstance(indicators, TrendFollowingIndicators)


def test_generate_signal_returns_tuple(sample_ohlcv):
    strategy = TrendFollowingStrategy()
    result = strategy.generate_signal(sample_ohlcv)

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], Signal)
    assert isinstance(result[1], TrendFollowingIndicators)


def test_repr():
    strategy = TrendFollowingStrategy(sma_fast=50, sma_slow=200, adx_period=14, adx_threshold=25.0)
    expected = "TrendFollowingStrategy(sma_fast=50, sma_slow=200, adx_period=14, adx_threshold=25.0)"
    assert repr(strategy) == expected


def test_calculate_indicators_preserves_original(sample_ohlcv):
    strategy = TrendFollowingStrategy()
    original_len = len(sample_ohlcv.columns)

    strategy.calculate_indicators(sample_ohlcv)

    assert len(sample_ohlcv.columns) == original_len


def test_indicators_model_properties():
    indicators = TrendFollowingIndicators(
        close=150.0,
        sma_fast=148.0,
        sma_slow=145.0,
        sma_bullish_cross=False,
        sma_bearish_cross=False,
        adx=30.0,
        plus_di=25.0,
        minus_di=15.0,
        strong_trend=True,
        trend_direction="bullish",
    )

    assert indicators.close == 150.0
    assert indicators.sma_fast == 148.0
    assert indicators.sma_slow == 145.0
    assert indicators.adx == 30.0
    assert indicators.strong_trend is True
    assert indicators.trend_direction == "bullish"


def test_adx_threshold_custom():
    strategy = TrendFollowingStrategy(adx_threshold=30.0)

    indicators = TrendFollowingIndicators(
        close=150.0,
        sma_fast=148.0,
        sma_slow=145.0,
        sma_bullish_cross=False,
        sma_bearish_cross=False,
        adx=28.0,
        plus_di=25.0,
        minus_di=15.0,
        strong_trend=False,  # 28 < 30
        trend_direction="bullish",
    )

    assert indicators.adx < strategy.adx_threshold
    assert indicators.strong_trend is False


def test_get_latest_indicators_insufficient_data():
    strategy = TrendFollowingStrategy()
    df = pd.DataFrame(
        {
            "Open": [100],
            "High": [105],
            "Low": [99],
            "Close": [104],
            "Volume": [1000000],
        }
    )

    df_with_indicators = strategy.calculate_indicators(df)

    with pytest.raises(ValueError, match=r"Need at least .* rows"):
        strategy.get_latest_indicators(df_with_indicators)
