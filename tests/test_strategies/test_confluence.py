"""Tests for multi-timeframe confluence calculator."""

import pytest

from src.strategies.confluence import ConfluenceCalculator
from src.strategies.signal import Signal
from src.strategies.timeframe import Timeframe, TimeframeResult


@pytest.fixture
def sample_timeframe_results():
    """Sample timeframe results for testing."""
    return {
        Timeframe.DAILY: TimeframeResult(
            timeframe=Timeframe.DAILY,
            signal=Signal.BUY,
            rsi=35.0,
            macd_hist=0.5,
            interpretation="Strong momentum on daily timeframe",
            confidence=0.75,
            indicators={"rsi": 35.0, "macd_hist": 0.5},
        ),
        Timeframe.HOURLY: TimeframeResult(
            timeframe=Timeframe.HOURLY,
            signal=Signal.BUY,
            rsi=40.0,
            macd_hist=0.3,
            interpretation="Bullish hourly momentum",
            confidence=0.65,
            indicators={"rsi": 40.0, "macd_hist": 0.3},
        ),
    }


def test_confluence_all_agree_buy(sample_timeframe_results):
    """Test confluence when all timeframes agree on BUY."""
    calculator = ConfluenceCalculator()
    signal, confluence, conflict = calculator.calculate_confluence(sample_timeframe_results)

    assert signal == Signal.BUY
    assert confluence > 0.9
    assert conflict is False


def test_confluence_conflict_daily_buy_hourly_sell():
    """Test conflict detection when daily=BUY, hourly=SELL."""
    results = {
        Timeframe.DAILY: TimeframeResult(
            timeframe=Timeframe.DAILY,
            signal=Signal.BUY,
            rsi=35.0,
            macd_hist=0.5,
            interpretation="Strong daily buy",
            confidence=0.75,
            indicators={},
        ),
        Timeframe.HOURLY: TimeframeResult(
            timeframe=Timeframe.HOURLY,
            signal=Signal.SELL,
            rsi=75.0,
            macd_hist=-0.3,
            interpretation="Hourly sell signal",
            confidence=0.65,
            indicators={},
        ),
    }

    calculator = ConfluenceCalculator()
    signal, confluence, conflict = calculator.calculate_confluence(results)

    assert conflict is True
    # New weights: weighted avg 0.14 < 0.15 margin threshold, so HOLD forced
    assert signal == Signal.HOLD
    # Confluence 0.50 (both timeframes disagree with HOLD)
    assert 0.4 < confluence < 0.6


def test_confluence_daily_buy_hourly_hold():
    """Test when daily=BUY, hourly=HOLD (partial agreement)."""
    results = {
        Timeframe.DAILY: TimeframeResult(
            timeframe=Timeframe.DAILY,
            signal=Signal.BUY,
            rsi=35.0,
            macd_hist=0.5,
            interpretation="Daily buy",
            confidence=0.75,
            indicators={},
        ),
        Timeframe.HOURLY: TimeframeResult(
            timeframe=Timeframe.HOURLY,
            signal=Signal.HOLD,
            rsi=50.0,
            macd_hist=0.0,
            interpretation="Neutral hourly",
            confidence=0.55,
            indicators={},
        ),
    }

    calculator = ConfluenceCalculator()
    signal, confluence, conflict = calculator.calculate_confluence(results)

    assert signal == Signal.BUY
    # Confluence 0.60 = 52% daily agrees, 35% hourly HOLD treated as partial agreement (13% from 15min unused)
    assert 0.55 <= confluence <= 0.85
    assert conflict is False


def test_confluence_all_sell():
    """Test confluence when all timeframes agree on SELL."""
    results = {
        Timeframe.DAILY: TimeframeResult(
            timeframe=Timeframe.DAILY,
            signal=Signal.SELL,
            rsi=75.0,
            macd_hist=-0.5,
            interpretation="Daily sell",
            confidence=0.8,
            indicators={},
        ),
        Timeframe.HOURLY: TimeframeResult(
            timeframe=Timeframe.HOURLY,
            signal=Signal.SELL,
            rsi=78.0,
            macd_hist=-0.4,
            interpretation="Hourly sell",
            confidence=0.7,
            indicators={},
        ),
    }

    calculator = ConfluenceCalculator()
    signal, confluence, conflict = calculator.calculate_confluence(results)

    assert signal == Signal.SELL
    assert confluence > 0.9
    assert conflict is False


def test_confidence_adjustment_high_confluence():
    """Test confidence boost for high confluence."""
    calculator = ConfluenceCalculator()
    adjusted = calculator.adjust_confidence(base_confidence=0.70, confluence_score=0.85)

    assert abs(adjusted - 0.80) < 0.001
    assert adjusted > 0.70


def test_confidence_adjustment_low_confluence():
    """Test confidence penalty for low confluence."""
    calculator = ConfluenceCalculator()
    adjusted = calculator.adjust_confidence(base_confidence=0.70, confluence_score=0.40)

    assert abs(adjusted - 0.50) < 0.001
    assert adjusted < 0.70


def test_confidence_adjustment_medium_confluence():
    """Test no adjustment for medium confluence."""
    calculator = ConfluenceCalculator()
    adjusted = calculator.adjust_confidence(base_confidence=0.70, confluence_score=0.65)

    assert adjusted == 0.70


def test_confluence_empty_results():
    """Test error handling for empty results."""
    calculator = ConfluenceCalculator()

    with pytest.raises(ValueError, match="No timeframe results"):
        calculator.calculate_confluence({})


def test_confluence_5_timeframes_all_agree():
    """Test all 5 timeframes agree on BUY - high confluence."""
    results = {
        Timeframe.DAILY: TimeframeResult(
            timeframe=Timeframe.DAILY,
            signal=Signal.BUY,
            rsi=35.0,
            macd_hist=0.5,
            interpretation="Strong daily momentum",
            confidence=0.80,
            indicators={},
        ),
        Timeframe.HOURLY: TimeframeResult(
            timeframe=Timeframe.HOURLY,
            signal=Signal.BUY,
            rsi=38.0,
            macd_hist=0.4,
            interpretation="Bullish hourly",
            confidence=0.75,
            indicators={},
        ),
        Timeframe.FIFTEEN_MIN: TimeframeResult(
            timeframe=Timeframe.FIFTEEN_MIN,
            signal=Signal.BUY,
            rsi=40.0,
            macd_hist=0.3,
            interpretation="15min bullish",
            confidence=0.70,
            indicators={},
        ),
        Timeframe.FIVE_MIN: TimeframeResult(
            timeframe=Timeframe.FIVE_MIN,
            signal=Signal.BUY,
            rsi=42.0,
            macd_hist=0.2,
            interpretation="5min buy",
            confidence=0.65,
            indicators={},
        ),
        Timeframe.ONE_MIN: TimeframeResult(
            timeframe=Timeframe.ONE_MIN,
            signal=Signal.BUY,
            rsi=45.0,
            macd_hist=0.1,
            interpretation="1min buy",
            confidence=0.60,
            indicators={},
        ),
    }

    calculator = ConfluenceCalculator()
    signal, confluence, conflict = calculator.calculate_confluence(results)

    assert signal == Signal.BUY
    assert confluence > 0.95
    assert conflict is False


def test_confluence_5_timeframes_4_agree():
    """Test 4/5 agree (daily/hourly/15m/5m=BUY, 1m=SELL)."""
    results = {
        Timeframe.DAILY: TimeframeResult(
            timeframe=Timeframe.DAILY,
            signal=Signal.BUY,
            rsi=35.0,
            macd_hist=0.5,
            interpretation="Daily buy",
            confidence=0.80,
            indicators={},
        ),
        Timeframe.HOURLY: TimeframeResult(
            timeframe=Timeframe.HOURLY,
            signal=Signal.BUY,
            rsi=38.0,
            macd_hist=0.4,
            interpretation="Hourly buy",
            confidence=0.75,
            indicators={},
        ),
        Timeframe.FIFTEEN_MIN: TimeframeResult(
            timeframe=Timeframe.FIFTEEN_MIN,
            signal=Signal.BUY,
            rsi=40.0,
            macd_hist=0.3,
            interpretation="15min buy",
            confidence=0.70,
            indicators={},
        ),
        Timeframe.FIVE_MIN: TimeframeResult(
            timeframe=Timeframe.FIVE_MIN,
            signal=Signal.BUY,
            rsi=42.0,
            macd_hist=0.2,
            interpretation="5min buy",
            confidence=0.65,
            indicators={},
        ),
        Timeframe.ONE_MIN: TimeframeResult(
            timeframe=Timeframe.ONE_MIN,
            signal=Signal.SELL,
            rsi=75.0,
            macd_hist=-0.1,
            interpretation="1min sell (noise)",
            confidence=0.50,
            indicators={},
        ),
    }

    calculator = ConfluenceCalculator()
    signal, confluence, conflict = calculator.calculate_confluence(results)

    # 1min only has 5% weight, should not flip signal
    assert signal == Signal.BUY
    # Confluence ~0.95 (95% of weight agrees with BUY)
    assert 0.90 <= confluence <= 1.0
    # Conflict detected (BUY and SELL both present)
    assert conflict is True


def test_confluence_5_timeframes_split():
    """Test split decision (daily/hourly=BUY, 15m/5m/1m=SELL)."""
    results = {
        Timeframe.DAILY: TimeframeResult(
            timeframe=Timeframe.DAILY,
            signal=Signal.BUY,
            rsi=35.0,
            macd_hist=0.5,
            interpretation="Daily buy",
            confidence=0.80,
            indicators={},
        ),
        Timeframe.HOURLY: TimeframeResult(
            timeframe=Timeframe.HOURLY,
            signal=Signal.BUY,
            rsi=38.0,
            macd_hist=0.4,
            interpretation="Hourly buy",
            confidence=0.75,
            indicators={},
        ),
        Timeframe.FIFTEEN_MIN: TimeframeResult(
            timeframe=Timeframe.FIFTEEN_MIN,
            signal=Signal.SELL,
            rsi=72.0,
            macd_hist=-0.3,
            interpretation="15min sell",
            confidence=0.70,
            indicators={},
        ),
        Timeframe.FIVE_MIN: TimeframeResult(
            timeframe=Timeframe.FIVE_MIN,
            signal=Signal.SELL,
            rsi=74.0,
            macd_hist=-0.2,
            interpretation="5min sell",
            confidence=0.65,
            indicators={},
        ),
        Timeframe.ONE_MIN: TimeframeResult(
            timeframe=Timeframe.ONE_MIN,
            signal=Signal.SELL,
            rsi=76.0,
            macd_hist=-0.1,
            interpretation="1min sell",
            confidence=0.60,
            indicators={},
        ),
    }

    calculator = ConfluenceCalculator()
    signal, confluence, conflict = calculator.calculate_confluence(results)

    # Daily+hourly = 70% weight → BUY wins
    assert signal == Signal.BUY
    # Confluence lower due to disagreement (~0.70, only 70% agrees)
    assert 0.65 <= confluence <= 0.75
    # Conflict detected
    assert conflict is True


def test_dominant_timeframe_selection():
    """Test dominant timeframe = highest weight agreeing with final signal."""
    results = {
        Timeframe.DAILY: TimeframeResult(
            timeframe=Timeframe.DAILY,
            signal=Signal.BUY,
            rsi=35.0,
            macd_hist=0.5,
            interpretation="Daily buy",
            confidence=0.80,
            indicators={},
        ),
        Timeframe.HOURLY: TimeframeResult(
            timeframe=Timeframe.HOURLY,
            signal=Signal.SELL,
            rsi=75.0,
            macd_hist=-0.3,
            interpretation="Hourly sell",
            confidence=0.70,
            indicators={},
        ),
        Timeframe.FIFTEEN_MIN: TimeframeResult(
            timeframe=Timeframe.FIFTEEN_MIN,
            signal=Signal.BUY,
            rsi=40.0,
            macd_hist=0.2,
            interpretation="15min buy",
            confidence=0.65,
            indicators={},
        ),
    }

    calculator = ConfluenceCalculator()
    final_signal, _, _ = calculator.calculate_confluence(results)

    # If final signal is BUY, dominant should be DAILY (highest weight with BUY)
    dominant = calculator.select_dominant_timeframe(final_signal, results)
    assert dominant == Timeframe.DAILY

    # Test case where daily doesn't match final signal
    results_hourly_dominant = {
        Timeframe.DAILY: TimeframeResult(
            timeframe=Timeframe.DAILY,
            signal=Signal.SELL,
            rsi=75.0,
            macd_hist=-0.5,
            interpretation="Daily sell",
            confidence=0.80,
            indicators={},
        ),
        Timeframe.HOURLY: TimeframeResult(
            timeframe=Timeframe.HOURLY,
            signal=Signal.BUY,
            rsi=35.0,
            macd_hist=0.4,
            interpretation="Hourly buy",
            confidence=0.75,
            indicators={},
        ),
        Timeframe.FIFTEEN_MIN: TimeframeResult(
            timeframe=Timeframe.FIFTEEN_MIN,
            signal=Signal.BUY,
            rsi=38.0,
            macd_hist=0.3,
            interpretation="15min buy",
            confidence=0.70,
            indicators={},
        ),
        Timeframe.FIVE_MIN: TimeframeResult(
            timeframe=Timeframe.FIVE_MIN,
            signal=Signal.BUY,
            rsi=40.0,
            macd_hist=0.2,
            interpretation="5min buy",
            confidence=0.65,
            indicators={},
        ),
    }

    final_signal2, _, _ = calculator.calculate_confluence(results_hourly_dominant)
    # With new weights: DAILY=0.40*(-1) + HOURLY=0.30*(1) + 15MIN=0.15*(1) + 5MIN=0.10*(1)
    # Weighted sum = -0.40 + 0.30 + 0.15 + 0.10 = 0.15; total_weight = 0.95
    # → weighted_avg = 0.15 / 0.95 ≈ 0.1579 → BUY
    # So dominant should be HOURLY (highest weight agreeing with BUY)
    dominant2 = calculator.select_dominant_timeframe(final_signal2, results_hourly_dominant)
    assert dominant2 == Timeframe.HOURLY


def test_timeframe_weights_sum_to_one():
    """Validate weights sum to 1.0."""
    calculator = ConfluenceCalculator()
    total = sum(calculator.TIMEFRAME_WEIGHTS.values())
    assert abs(total - 1.0) < 0.001
