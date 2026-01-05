"""Tests for trader agent."""

from src.agents.news import NewsAnalysis
from src.agents.sentiment import SentimentAnalysis
from src.agents.technical import TechnicalAnalysis
from src.agents.trader import TraderAgent, TradingDecision
from src.strategies.momentum import Signal


def test_trader_agent_init(mock_llm_client):
    agent = TraderAgent(mock_llm_client)

    assert agent.llm == mock_llm_client


def test_trader_agent_decide(mock_llm_client):
    agent = TraderAgent(mock_llm_client)

    technical = TechnicalAnalysis(
        signal=Signal.BUY,
        rsi=35.0,
        macd_hist=0.5,
        interpretation="Bullish indicators",
        confidence=0.8,
    )

    sentiment = SentimentAnalysis(
        overall_sentiment="positive",
        sentiment_score=0.6,
        positive_ratio=0.7,
        negative_ratio=0.1,
        neutral_ratio=0.2,
        article_count=10,
        summary="Positive news sentiment",
    )

    news = NewsAnalysis(
        key_themes=["Growth", "Innovation"],
        impact_assessment="Positive outlook",
        recommendation="Consider buying",
    )

    result = agent.decide("AAPL", technical, sentiment, news)

    assert isinstance(result, TradingDecision)
    assert isinstance(result.action, Signal)
    assert 0.0 <= result.confidence <= 1.0
    assert result.risk_level in ["LOW", "MEDIUM", "HIGH"]
    assert result.reasoning
    mock_llm_client.complete.assert_called_once()


def test_extract_action_from_response(mock_llm_client):
    agent = TraderAgent(mock_llm_client)

    response = "Action: BUY\nConfidence: 0.8\nReasoning: Strong signals"

    action = agent._extract_action(response, Signal.HOLD)

    assert action == Signal.BUY


def test_extract_action_fallback(mock_llm_client):
    agent = TraderAgent(mock_llm_client)

    response = "Unclear response without action"

    action = agent._extract_action(response, Signal.HOLD)

    assert action == Signal.HOLD


def test_extract_confidence_from_response(mock_llm_client):
    agent = TraderAgent(mock_llm_client)

    technical = TechnicalAnalysis(
        signal=Signal.BUY,
        rsi=35.0,
        macd_hist=0.5,
        interpretation="Test",
        confidence=0.7,
    )

    sentiment = SentimentAnalysis(
        overall_sentiment="positive",
        sentiment_score=0.5,
        positive_ratio=0.6,
        negative_ratio=0.2,
        neutral_ratio=0.2,
        article_count=5,
        summary="Test",
    )

    response = "Confidence: 0.85\nStrong signals"

    confidence = agent._extract_confidence(response, technical, sentiment)

    assert confidence == 0.85


def test_extract_confidence_fallback(mock_llm_client):
    agent = TraderAgent(mock_llm_client)

    technical = TechnicalAnalysis(
        signal=Signal.BUY,
        rsi=35.0,
        macd_hist=0.5,
        interpretation="Test",
        confidence=0.7,
    )

    sentiment = SentimentAnalysis(
        overall_sentiment="neutral",
        sentiment_score=0.1,
        positive_ratio=0.4,
        negative_ratio=0.3,
        neutral_ratio=0.3,
        article_count=5,
        summary="Test",
    )

    response = "No confidence mentioned"

    confidence = agent._extract_confidence(response, technical, sentiment)

    assert 0.0 <= confidence <= 1.0


def test_extract_risk_level(mock_llm_client):
    agent = TraderAgent(mock_llm_client)

    assert agent._extract_risk_level("Risk: HIGH", 0.5) == "HIGH"
    assert agent._extract_risk_level("Risk: LOW", 0.5) == "LOW"
    assert agent._extract_risk_level("No risk mentioned", 0.8) == "LOW"
    assert agent._extract_risk_level("No risk mentioned", 0.3) == "HIGH"


def test_repr(mock_llm_client):
    agent = TraderAgent(mock_llm_client)

    repr_str = repr(agent)

    assert "TraderAgent" in repr_str
    assert "ollama" in repr_str
