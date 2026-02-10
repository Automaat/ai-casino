"""Tests for technical analyst agent."""

import pandas as pd
import pytest

from src.agents.technical import TechnicalAnalysis
from src.strategies.ensemble import EnsembleStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.signal import Signal
from src.strategies.trend_following import TrendFollowingStrategy


def test_technical_analyst_init(test_container):
    strategy = MomentumStrategy()
    analyst = test_container.technical_analyst()(strategy)

    assert analyst.llm is not None
    assert analyst.strategy == strategy


async def test_technical_analyst_analyze(test_container, sample_ohlcv_data):
    strategy = MomentumStrategy()
    analyst = test_container.technical_analyst()(strategy)

    result = await analyst.analyze("AAPL", sample_ohlcv_data)

    assert isinstance(result, TechnicalAnalysis)
    assert isinstance(result.signal, Signal)
    assert 0.0 <= result.confidence <= 1.0
    assert result.interpretation


async def test_technical_analyst_analyze_calls_strategy(test_container, sample_ohlcv_data):
    strategy = MomentumStrategy()
    analyst = test_container.technical_analyst()(strategy)

    # Get the mock LLM to verify it was called
    llm_mock = test_container.llm_client()

    await analyst.analyze("AAPL", sample_ohlcv_data)

    call_args = llm_mock.acomplete.call_args
    assert "AAPL" in call_args.args[0]
    assert "RSI" in call_args.args[0]
    assert "MACD" in call_args.args[0]


def test_repr(test_container):
    strategy = MomentumStrategy()
    analyst = test_container.technical_analyst()(strategy)

    repr_str = repr(analyst)

    assert "TechnicalAnalyst" in repr_str
    assert "MomentumStrategy" in repr_str


async def test_technical_analyst_trend_following_strategy(test_container):
    """Test TechnicalAnalyst with TrendFollowingStrategy."""
    # TrendFollowing needs 200+ rows for SMA_200
    large_ohlcv = pd.DataFrame(
        {
            "Open": [100 + i for i in range(250)],
            "High": [105 + i for i in range(250)],
            "Low": [99 + i for i in range(250)],
            "Close": [104 + i for i in range(250)],
            "Volume": [1000000] * 250,
        }
    )

    strategy = TrendFollowingStrategy()
    analyst = test_container.technical_analyst()(strategy)

    result = await analyst.analyze("AAPL", large_ohlcv)

    assert isinstance(result, TechnicalAnalysis)
    assert isinstance(result.signal, Signal)
    assert 0.0 <= result.confidence <= 1.0
    assert result.interpretation


async def test_technical_analyst_mean_reversion_strategy(test_container, sample_ohlcv_data):
    """Test TechnicalAnalyst with MeanReversionStrategy."""
    strategy = MeanReversionStrategy()
    analyst = test_container.technical_analyst()(strategy)

    result = await analyst.analyze("AAPL", sample_ohlcv_data)

    assert isinstance(result, TechnicalAnalysis)
    assert isinstance(result.signal, Signal)
    assert 0.0 <= result.confidence <= 1.0
    assert result.interpretation


async def test_technical_analyst_ensemble_strategy(test_container, sample_ohlcv_data):
    """Test TechnicalAnalyst with EnsembleStrategy."""
    # EnsembleStrategy uses shorter SMAs (20/50) by default - 50 rows sufficient
    strategy = EnsembleStrategy()
    analyst = test_container.technical_analyst()(strategy)

    result = await analyst.analyze("AAPL", sample_ohlcv_data)

    assert isinstance(result, TechnicalAnalysis)
    assert isinstance(result.signal, Signal)
    assert 0.0 <= result.confidence <= 1.0
    assert result.interpretation


async def test_build_prompt_raises_type_error_on_mismatch(test_container, sample_ohlcv_data, mocker):
    """Test _build_prompt raises TypeError when strategy/indicators mismatch."""
    strategy = MomentumStrategy()
    analyst = test_container.technical_analyst()(strategy)

    # Mock generate_signal to return unexpected indicator type
    mock_indicator = mocker.Mock()
    mocker.patch.object(strategy, "generate_signal", return_value=(Signal.BUY, mock_indicator))

    with pytest.raises(TypeError, match="Unexpected indicators type"):
        await analyst.analyze("AAPL", sample_ohlcv_data)
