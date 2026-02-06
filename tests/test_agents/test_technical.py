"""Tests for technical analyst agent."""

import pytest

from src.agents.technical import TechnicalAnalysis, TechnicalAnalyst
from src.strategies.momentum import MomentumStrategy
from src.strategies.signal import Signal


def test_technical_analyst_init(mock_llm_client):
    strategy = MomentumStrategy()
    analyst = TechnicalAnalyst(mock_llm_client, strategy)

    assert analyst.llm == mock_llm_client
    assert analyst.strategy == strategy


@pytest.mark.asyncio
async def test_technical_analyst_analyze(mock_llm_client, sample_ohlcv_data):
    strategy = MomentumStrategy()
    analyst = TechnicalAnalyst(mock_llm_client, strategy)

    result = await analyst.analyze("AAPL", sample_ohlcv_data)

    assert isinstance(result, TechnicalAnalysis)
    assert isinstance(result.signal, Signal)
    assert 0.0 <= result.confidence <= 1.0
    assert result.interpretation
    mock_llm_client.acomplete.assert_called_once()


@pytest.mark.asyncio
async def test_technical_analyst_analyze_calls_strategy(mock_llm_client, sample_ohlcv_data):
    strategy = MomentumStrategy()
    analyst = TechnicalAnalyst(mock_llm_client, strategy)

    await analyst.analyze("AAPL", sample_ohlcv_data)

    call_args = mock_llm_client.acomplete.call_args
    assert "AAPL" in call_args.args[0]
    assert "RSI" in call_args.args[0]
    assert "MACD" in call_args.args[0]


def test_repr(mock_llm_client):
    strategy = MomentumStrategy()
    analyst = TechnicalAnalyst(mock_llm_client, strategy)

    repr_str = repr(analyst)

    assert "TechnicalAnalyst" in repr_str
    assert "MomentumStrategy" in repr_str
