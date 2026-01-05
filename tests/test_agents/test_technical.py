"""Tests for technical analyst agent."""

from src.agents.technical import TechnicalAnalysis, TechnicalAnalyst
from src.strategies.momentum import MomentumStrategy, Signal


def test_technical_analyst_init(mock_llm_client):
    strategy = MomentumStrategy()
    analyst = TechnicalAnalyst(mock_llm_client, strategy)

    assert analyst.llm == mock_llm_client
    assert analyst.strategy == strategy


def test_technical_analyst_analyze(mock_llm_client, sample_ohlcv_data):
    strategy = MomentumStrategy()
    analyst = TechnicalAnalyst(mock_llm_client, strategy)

    result = analyst.analyze("AAPL", sample_ohlcv_data)

    assert isinstance(result, TechnicalAnalysis)
    assert isinstance(result.signal, Signal)
    assert 0.0 <= result.confidence <= 1.0
    assert result.interpretation
    mock_llm_client.complete.assert_called_once()


def test_technical_analyst_analyze_calls_strategy(mock_llm_client, sample_ohlcv_data):
    strategy = MomentumStrategy()
    analyst = TechnicalAnalyst(mock_llm_client, strategy)

    analyst.analyze("AAPL", sample_ohlcv_data)

    call_args = mock_llm_client.complete.call_args
    assert "AAPL" in call_args.args[0]
    assert "RSI" in call_args.args[0]
    assert "MACD" in call_args.args[0]


def test_extract_confidence_high(mock_llm_client, sample_ohlcv_data):
    from unittest.mock import Mock

    strategy = MomentumStrategy()
    analyst = TechnicalAnalyst(mock_llm_client, strategy)

    indicators = Mock()
    indicators.rsi_oversold = True
    indicators.macd_bullish = True
    indicators.rsi_overbought = False
    indicators.macd_bearish = False

    confidence = analyst._extract_confidence("High confidence signal", indicators)

    assert confidence >= 0.8


def test_extract_confidence_low(mock_llm_client):
    from unittest.mock import Mock

    strategy = MomentumStrategy()
    analyst = TechnicalAnalyst(mock_llm_client, strategy)

    indicators = Mock()
    indicators.rsi_oversold = False
    indicators.macd_bullish = False
    indicators.rsi_overbought = False
    indicators.macd_bearish = False

    confidence = analyst._extract_confidence("Weak signal", indicators)

    assert confidence == 0.5


def test_repr(mock_llm_client):
    strategy = MomentumStrategy()
    analyst = TechnicalAnalyst(mock_llm_client, strategy)

    repr_str = repr(analyst)

    assert "TechnicalAnalyst" in repr_str
    assert "MomentumStrategy" in repr_str
