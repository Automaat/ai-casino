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
    # Conflict detected but weighted avg 0.20 > 0.15, so BUY still wins
    assert signal == Signal.BUY
    # Confluence should be around 0.60 (60% daily weight agrees with final BUY)
    assert 0.5 < confluence < 0.7


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
