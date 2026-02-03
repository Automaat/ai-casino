"""Tests for ensemble strategy."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.strategies.ensemble import (
    AggregationMethod,
    EnsembleResult,
    EnsembleStrategy,
    StrategyResult,
)
from src.strategies.mean_reversion import MeanReversionIndicators, MeanReversionStrategy
from src.strategies.momentum import MomentumIndicators, MomentumStrategy, Signal
from src.strategies.trend_following import TrendFollowingIndicators, TrendFollowingStrategy


@pytest.fixture
def sample_ohlcv():
    """Large sample OHLCV data for trend following (needs 200+ points for SMA_200)."""
    return pd.DataFrame(
        {
            "Open": [100 + i * 0.5 for i in range(250)],
            "High": [105 + i * 0.5 for i in range(250)],
            "Low": [99 + i * 0.5 for i in range(250)],
            "Close": [104 + i * 0.5 for i in range(250)],
            "Volume": [1000000] * 250,
        }
    )


@pytest.fixture
def mock_momentum_indicators():
    return MomentumIndicators(
        rsi=45.0,
        rsi_oversold=False,
        rsi_overbought=False,
        macd=0.5,
        macd_signal=0.3,
        macd_hist=0.2,
        macd_bullish=True,
        macd_bearish=False,
    )


@pytest.fixture
def mock_mean_reversion_indicators():
    return MeanReversionIndicators(
        close=150.0,
        bb_upper=160.0,
        bb_middle=150.0,
        bb_lower=140.0,
        bb_width=13.3,
        bb_percent=0.5,
        oversold=False,
        overbought=False,
    )


@pytest.fixture
def mock_trend_following_indicators():
    return TrendFollowingIndicators(
        close=150.0,
        sma_fast=148.0,
        sma_slow=145.0,
        sma_bullish_cross=False,
        sma_bearish_cross=False,
        adx=30.0,
        plus_di=25.0,
        minus_di=20.0,
        strong_trend=True,
        trend_direction="bullish",
    )


def test_ensemble_strategy_init_defaults():
    strategy = EnsembleStrategy()

    assert isinstance(strategy.momentum, MomentumStrategy)
    assert isinstance(strategy.mean_reversion, MeanReversionStrategy)
    assert isinstance(strategy.trend_following, TrendFollowingStrategy)
    assert strategy.weights["momentum"] == 0.40
    assert strategy.weights["mean_reversion"] == 0.25
    assert strategy.weights["trend_following"] == 0.35
    assert strategy.aggregation == AggregationMethod.WEIGHTED_VOTING


def test_ensemble_strategy_init_custom_weights():
    custom_weights = {"momentum": 0.5, "mean_reversion": 0.3, "trend_following": 0.2}
    strategy = EnsembleStrategy(weights=custom_weights)

    assert strategy.weights["momentum"] == 0.5
    assert strategy.weights["mean_reversion"] == 0.3
    assert strategy.weights["trend_following"] == 0.2


def test_ensemble_strategy_normalizes_weights():
    unnormalized = {"momentum": 2.0, "mean_reversion": 1.0, "trend_following": 1.0}
    strategy = EnsembleStrategy(weights=unnormalized)

    total = sum(strategy.weights.values())
    assert abs(total - 1.0) < 0.01


def test_run_strategies(
    sample_ohlcv, mock_momentum_indicators, mock_mean_reversion_indicators, mock_trend_following_indicators
):
    mock_momentum = MagicMock(spec=MomentumStrategy)
    mock_momentum.generate_signal.return_value = (Signal.BUY, mock_momentum_indicators)

    mock_mean_rev = MagicMock(spec=MeanReversionStrategy)
    mock_mean_rev.generate_signal.return_value = (Signal.HOLD, mock_mean_reversion_indicators)

    mock_trend = MagicMock(spec=TrendFollowingStrategy)
    mock_trend.generate_signal.return_value = (Signal.BUY, mock_trend_following_indicators)

    strategy = EnsembleStrategy(
        momentum=mock_momentum,
        mean_reversion=mock_mean_rev,
        trend_following=mock_trend,
    )

    results = strategy._run_strategies(sample_ohlcv)

    assert len(results) == 3
    assert all(isinstance(r, StrategyResult) for r in results)
    mock_momentum.generate_signal.assert_called_once_with(sample_ohlcv)
    mock_mean_rev.generate_signal.assert_called_once_with(sample_ohlcv)
    mock_trend.generate_signal.assert_called_once_with(sample_ohlcv)


def test_weighted_voting_unanimous(
    mock_momentum_indicators, mock_mean_reversion_indicators, mock_trend_following_indicators
):
    results = [
        StrategyResult(name="momentum", signal=Signal.BUY, weight=0.4, indicators=mock_momentum_indicators),
        StrategyResult(
            name="mean_reversion", signal=Signal.BUY, weight=0.25, indicators=mock_mean_reversion_indicators
        ),
        StrategyResult(
            name="trend_following", signal=Signal.BUY, weight=0.35, indicators=mock_trend_following_indicators
        ),
    ]

    strategy = EnsembleStrategy()
    signal, score, conflict = strategy._weighted_voting(results)

    assert signal == Signal.BUY
    assert score == 1.0
    assert conflict is False


def test_weighted_voting_conflict(
    mock_momentum_indicators, mock_mean_reversion_indicators, mock_trend_following_indicators
):
    """Test weighted voting with disagreement but clear winner (margin > 10%)."""
    results = [
        StrategyResult(name="momentum", signal=Signal.BUY, weight=0.5, indicators=mock_momentum_indicators),
        StrategyResult(
            name="mean_reversion", signal=Signal.SELL, weight=0.2, indicators=mock_mean_reversion_indicators
        ),
        StrategyResult(
            name="trend_following",
            signal=Signal.HOLD,
            weight=0.3,
            indicators=mock_trend_following_indicators,
        ),
    ]

    strategy = EnsembleStrategy()
    signal, score, conflict = strategy._weighted_voting(results)

    assert signal == Signal.BUY
    assert score == 0.5
    assert conflict is False


def test_conflict_resolution_close_margin(
    mock_momentum_indicators, mock_mean_reversion_indicators, mock_trend_following_indicators
):
    results = [
        StrategyResult(name="momentum", signal=Signal.BUY, weight=0.34, indicators=mock_momentum_indicators),
        StrategyResult(
            name="mean_reversion", signal=Signal.SELL, weight=0.33, indicators=mock_mean_reversion_indicators
        ),
        StrategyResult(
            name="trend_following",
            signal=Signal.HOLD,
            weight=0.33,
            indicators=mock_trend_following_indicators,
        ),
    ]

    strategy = EnsembleStrategy()
    signal, _score, conflict = strategy._weighted_voting(results)

    assert signal == Signal.HOLD
    assert conflict is True


def test_conflict_resolution_buy_sell_tie(
    mock_momentum_indicators, mock_mean_reversion_indicators, mock_trend_following_indicators
):
    results = [
        StrategyResult(name="momentum", signal=Signal.BUY, weight=0.35, indicators=mock_momentum_indicators),
        StrategyResult(
            name="mean_reversion", signal=Signal.SELL, weight=0.35, indicators=mock_mean_reversion_indicators
        ),
        StrategyResult(
            name="trend_following",
            signal=Signal.HOLD,
            weight=0.30,
            indicators=mock_trend_following_indicators,
        ),
    ]

    strategy = EnsembleStrategy()
    signal, _score, conflict = strategy._weighted_voting(results)

    assert signal == Signal.HOLD
    assert conflict is True


def test_confidence_calculation():
    strategy = EnsembleStrategy()

    confidence = strategy._calculate_confidence(agreement_ratio=1.0, weighted_score=1.0, signal_strength=1.0)
    assert 0.0 <= confidence <= 1.0
    assert confidence == 1.0

    confidence = strategy._calculate_confidence(agreement_ratio=0.5, weighted_score=0.5, signal_strength=0.5)
    assert confidence == 0.5

    confidence = strategy._calculate_confidence(agreement_ratio=0.0, weighted_score=0.0, signal_strength=0.0)
    assert confidence == 0.0


def test_generate_signal_returns_tuple(sample_ohlcv):
    strategy = EnsembleStrategy()
    result = strategy.generate_signal(sample_ohlcv)

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], Signal)
    assert isinstance(result[1], EnsembleResult)


def test_generate_signal_ensemble_result_fields(sample_ohlcv):
    strategy = EnsembleStrategy()
    _signal, result = strategy.generate_signal(sample_ohlcv)

    assert isinstance(result.signal, Signal)
    assert 0.0 <= result.confidence <= 1.0
    assert 0.0 <= result.agreement_ratio <= 1.0
    assert len(result.strategy_results) == 3
    assert isinstance(result.conflict_resolved, bool)


def test_majority_vote_unanimous(
    mock_momentum_indicators, mock_mean_reversion_indicators, mock_trend_following_indicators
):
    results = [
        StrategyResult(name="momentum", signal=Signal.SELL, weight=0.4, indicators=mock_momentum_indicators),
        StrategyResult(
            name="mean_reversion", signal=Signal.SELL, weight=0.25, indicators=mock_mean_reversion_indicators
        ),
        StrategyResult(
            name="trend_following",
            signal=Signal.SELL,
            weight=0.35,
            indicators=mock_trend_following_indicators,
        ),
    ]

    strategy = EnsembleStrategy(aggregation=AggregationMethod.MAJORITY_VOTE)
    signal, score, conflict = strategy._majority_vote(results)

    assert signal == Signal.SELL
    assert score == 1.0
    assert conflict is False


def test_majority_vote_majority(
    mock_momentum_indicators, mock_mean_reversion_indicators, mock_trend_following_indicators
):
    results = [
        StrategyResult(name="momentum", signal=Signal.BUY, weight=0.4, indicators=mock_momentum_indicators),
        StrategyResult(
            name="mean_reversion", signal=Signal.BUY, weight=0.25, indicators=mock_mean_reversion_indicators
        ),
        StrategyResult(
            name="trend_following",
            signal=Signal.SELL,
            weight=0.35,
            indicators=mock_trend_following_indicators,
        ),
    ]

    strategy = EnsembleStrategy(aggregation=AggregationMethod.MAJORITY_VOTE)
    signal, score, conflict = strategy._majority_vote(results)

    assert signal == Signal.BUY
    assert score == pytest.approx(2 / 3, rel=0.01)
    assert conflict is False


def test_majority_vote_tie(
    mock_momentum_indicators, mock_mean_reversion_indicators, mock_trend_following_indicators
):
    results = [
        StrategyResult(name="momentum", signal=Signal.BUY, weight=0.4, indicators=mock_momentum_indicators),
        StrategyResult(
            name="mean_reversion", signal=Signal.SELL, weight=0.25, indicators=mock_mean_reversion_indicators
        ),
        StrategyResult(
            name="trend_following",
            signal=Signal.HOLD,
            weight=0.35,
            indicators=mock_trend_following_indicators,
        ),
    ]

    strategy = EnsembleStrategy(aggregation=AggregationMethod.MAJORITY_VOTE)
    signal, _score, conflict = strategy._majority_vote(results)

    assert signal == Signal.HOLD
    assert conflict is True


def test_unanimous_all_agree(
    mock_momentum_indicators, mock_mean_reversion_indicators, mock_trend_following_indicators
):
    results = [
        StrategyResult(name="momentum", signal=Signal.BUY, weight=0.4, indicators=mock_momentum_indicators),
        StrategyResult(
            name="mean_reversion", signal=Signal.BUY, weight=0.25, indicators=mock_mean_reversion_indicators
        ),
        StrategyResult(
            name="trend_following", signal=Signal.BUY, weight=0.35, indicators=mock_trend_following_indicators
        ),
    ]

    strategy = EnsembleStrategy(aggregation=AggregationMethod.UNANIMOUS)
    signal, score, conflict = strategy._unanimous(results)

    assert signal == Signal.BUY
    assert score == 1.0
    assert conflict is False


def test_unanimous_disagree(
    mock_momentum_indicators, mock_mean_reversion_indicators, mock_trend_following_indicators
):
    results = [
        StrategyResult(name="momentum", signal=Signal.BUY, weight=0.4, indicators=mock_momentum_indicators),
        StrategyResult(
            name="mean_reversion", signal=Signal.HOLD, weight=0.25, indicators=mock_mean_reversion_indicators
        ),
        StrategyResult(
            name="trend_following", signal=Signal.BUY, weight=0.35, indicators=mock_trend_following_indicators
        ),
    ]

    strategy = EnsembleStrategy(aggregation=AggregationMethod.UNANIMOUS)
    signal, score, conflict = strategy._unanimous(results)

    assert signal == Signal.HOLD
    assert score == 0.0
    assert conflict is True


def test_repr():
    strategy = EnsembleStrategy()
    repr_str = repr(strategy)

    assert "EnsembleStrategy" in repr_str
    assert "WEIGHTED_VOTING" in repr_str
    assert "momentum" in repr_str


def test_aggregation_method_enum():
    assert AggregationMethod.WEIGHTED_VOTING.value == "WEIGHTED_VOTING"
    assert AggregationMethod.MAJORITY_VOTE.value == "MAJORITY_VOTE"
    assert AggregationMethod.UNANIMOUS.value == "UNANIMOUS"


def test_strategy_result_model(mock_momentum_indicators):
    result = StrategyResult(
        name="momentum",
        signal=Signal.BUY,
        weight=0.4,
        indicators=mock_momentum_indicators,
    )

    assert result.name == "momentum"
    assert result.signal == Signal.BUY
    assert result.weight == 0.4
    assert isinstance(result.indicators, MomentumIndicators)


def test_ensemble_result_model(mock_momentum_indicators):
    strategy_results = [
        StrategyResult(name="momentum", signal=Signal.BUY, weight=0.4, indicators=mock_momentum_indicators),
    ]

    result = EnsembleResult(
        signal=Signal.BUY,
        confidence=0.8,
        agreement_ratio=1.0,
        strategy_results=strategy_results,
        conflict_resolved=False,
    )

    assert result.signal == Signal.BUY
    assert result.confidence == 0.8
    assert result.agreement_ratio == 1.0
    assert len(result.strategy_results) == 1
    assert result.conflict_resolved is False


def test_signal_strength_calculation_momentum(mock_momentum_indicators):
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
    results = [StrategyResult(name="momentum", signal=Signal.BUY, weight=1.0, indicators=indicators)]

    strategy = EnsembleStrategy()
    strength = strategy._calculate_signal_strength(results)

    assert 0.0 <= strength <= 1.0
    assert strength > 0.5


def test_agreement_ratio_calculation(
    mock_momentum_indicators, mock_mean_reversion_indicators, mock_trend_following_indicators
):
    results = [
        StrategyResult(name="momentum", signal=Signal.BUY, weight=0.4, indicators=mock_momentum_indicators),
        StrategyResult(
            name="mean_reversion", signal=Signal.BUY, weight=0.25, indicators=mock_mean_reversion_indicators
        ),
        StrategyResult(
            name="trend_following",
            signal=Signal.SELL,
            weight=0.35,
            indicators=mock_trend_following_indicators,
        ),
    ]

    strategy = EnsembleStrategy()
    ratio = strategy._calculate_agreement_ratio(results, Signal.BUY)

    assert ratio == pytest.approx(0.65, rel=0.01)


def test_full_integration(sample_ohlcv):
    strategy = EnsembleStrategy()
    signal, result = strategy.generate_signal(sample_ohlcv)

    assert signal in [Signal.BUY, Signal.SELL, Signal.HOLD]
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.strategy_results) == 3

    for sr in result.strategy_results:
        assert sr.name in ["momentum", "mean_reversion", "trend_following"]
        assert sr.signal in [Signal.BUY, Signal.SELL, Signal.HOLD]
        assert sr.weight > 0
