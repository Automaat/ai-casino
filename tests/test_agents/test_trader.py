"""Tests for trader agent."""

from src.agents.fundamental import FundamentalAnalysis
from src.agents.news import NewsAnalysis
from src.agents.sentiment import SentimentAnalysis
from src.agents.technical import TechnicalAnalysis
from src.agents.trader import TraderAgent, TradingDecision
from src.strategies.momentum import Signal


def test_trader_agent_init(mock_llm_client):
    agent = TraderAgent(mock_llm_client)

    assert agent.llm == mock_llm_client


def test_trader_agent_decide(mock_llm_client, sample_bullish_research, sample_bearish_research):
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

    fundamental = FundamentalAnalysis(
        valuation="FAIRLY_VALUED",
        pe_ratio=28.5,
        eps=6.15,
        revenue_growth_yoy=0.062,
        earnings_growth_yoy=0.102,
        debt_to_equity=2.05,
        current_ratio=0.94,
        interpretation="Solid fundamentals",
        confidence=0.75,
    )

    result = agent.decide(
        "AAPL", technical, sentiment, news, fundamental, sample_bullish_research, sample_bearish_research
    )

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


def test_extract_confidence_from_response(mock_llm_client, sample_bullish_research, sample_bearish_research):
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

    confidence = agent._extract_confidence(
        response, technical, sentiment, sample_bullish_research, sample_bearish_research
    )

    assert confidence == 0.85


def test_extract_confidence_fallback(mock_llm_client, sample_bullish_research, sample_bearish_research):
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

    confidence = agent._extract_confidence(
        response, technical, sentiment, sample_bullish_research, sample_bearish_research
    )

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


def test_decide_owns_position_true(mock_llm_client, sample_bullish_research, sample_bearish_research):
    agent = TraderAgent(mock_llm_client)

    technical = TechnicalAnalysis(
        signal=Signal.HOLD,
        rsi=50.0,
        macd_hist=0.1,
        interpretation="Neutral indicators",
        confidence=0.6,
    )

    sentiment = SentimentAnalysis(
        overall_sentiment="neutral",
        sentiment_score=0.0,
        positive_ratio=0.3,
        negative_ratio=0.3,
        neutral_ratio=0.4,
        article_count=5,
        summary="Mixed sentiment",
    )

    news = NewsAnalysis(
        key_themes=["Stable"],
        impact_assessment="Neutral outlook",
        recommendation="Hold position",
    )

    fundamental = FundamentalAnalysis(
        valuation="FAIRLY_VALUED",
        pe_ratio=25.0,
        eps=5.0,
        revenue_growth_yoy=0.05,
        earnings_growth_yoy=0.08,
        debt_to_equity=1.5,
        current_ratio=1.2,
        interpretation="Stable fundamentals",
        confidence=0.7,
    )

    result = agent.decide(
        "AAPL",
        technical,
        sentiment,
        news,
        fundamental,
        sample_bullish_research,
        sample_bearish_research,
        owns_position=True,
        position_qty=100.0,
    )

    assert result.owns_position is True
    assert result.position_qty == 100.0


def test_decide_owns_position_false(mock_llm_client, sample_bullish_research, sample_bearish_research):
    agent = TraderAgent(mock_llm_client)

    technical = TechnicalAnalysis(
        signal=Signal.HOLD,
        rsi=50.0,
        macd_hist=0.1,
        interpretation="Neutral indicators",
        confidence=0.6,
    )

    sentiment = SentimentAnalysis(
        overall_sentiment="neutral",
        sentiment_score=0.0,
        positive_ratio=0.3,
        negative_ratio=0.3,
        neutral_ratio=0.4,
        article_count=5,
        summary="Mixed sentiment",
    )

    news = NewsAnalysis(
        key_themes=["Stable"],
        impact_assessment="Neutral outlook",
        recommendation="Hold position",
    )

    fundamental = FundamentalAnalysis(
        valuation="FAIRLY_VALUED",
        pe_ratio=25.0,
        eps=5.0,
        revenue_growth_yoy=0.05,
        earnings_growth_yoy=0.08,
        debt_to_equity=1.5,
        current_ratio=1.2,
        interpretation="Stable fundamentals",
        confidence=0.7,
    )

    result = agent.decide(
        "TSLA",
        technical,
        sentiment,
        news,
        fundamental,
        sample_bullish_research,
        sample_bearish_research,
        owns_position=False,
        position_qty=None,
    )

    assert result.owns_position is False
    assert result.position_qty is None


def test_prompt_includes_portfolio_context(mock_llm_client, sample_bullish_research, sample_bearish_research):
    agent = TraderAgent(mock_llm_client)

    technical = TechnicalAnalysis(
        signal=Signal.BUY,
        rsi=35.0,
        macd_hist=0.5,
        interpretation="Bullish",
        confidence=0.8,
    )

    sentiment = SentimentAnalysis(
        overall_sentiment="positive",
        sentiment_score=0.6,
        positive_ratio=0.7,
        negative_ratio=0.1,
        neutral_ratio=0.2,
        article_count=10,
        summary="Positive",
    )

    news = NewsAnalysis(
        key_themes=["Growth"],
        impact_assessment="Positive",
        recommendation="Buy",
    )

    fundamental = FundamentalAnalysis(
        valuation="UNDERVALUED",
        pe_ratio=20.0,
        eps=6.0,
        revenue_growth_yoy=0.1,
        earnings_growth_yoy=0.15,
        debt_to_equity=1.0,
        current_ratio=1.5,
        interpretation="Strong fundamentals",
        confidence=0.8,
    )

    agent.decide(
        "AAPL",
        technical,
        sentiment,
        news,
        fundamental,
        sample_bullish_research,
        sample_bearish_research,
        owns_position=True,
        position_qty=50.0,
    )

    call_args = mock_llm_client.complete.call_args
    prompt = call_args[0][0]
    assert "PORTFOLIO STATUS:" in prompt
    assert "currently own 50.0 shares" in prompt


def test_display_action_wait_when_hold_not_owning():
    from src.agents.trader import TradingDecision

    decision = TradingDecision(
        action=Signal.HOLD,
        confidence=0.5,
        reasoning="Mixed signals",
        risk_level="MEDIUM",
        owns_position=False,
        position_qty=None,
    )

    assert decision.display_action == "WAIT"


def test_display_action_hold_when_owning():
    from src.agents.trader import TradingDecision

    decision = TradingDecision(
        action=Signal.HOLD,
        confidence=0.6,
        reasoning="Maintain position",
        risk_level="LOW",
        owns_position=True,
        position_qty=100.0,
    )

    assert decision.display_action == "HOLD"


def test_display_action_buy_unchanged():
    from src.agents.trader import TradingDecision

    decision = TradingDecision(
        action=Signal.BUY,
        confidence=0.8,
        reasoning="Strong signals",
        risk_level="LOW",
        owns_position=False,
    )

    assert decision.display_action == "BUY"
