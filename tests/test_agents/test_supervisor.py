"""Tests for Trading Supervisor agent."""

import pytest

from src.agents.supervisor import (
    AnalysisRoutingDecision,
    AnalysisType,
    PlanningContext,
    SynthesisContext,
)
from src.agents.supervisor.models import TradeApprovalContext, TradeApprovalDecision
from src.models.providers.base import StructuredOutputError
from src.strategies.regime import MarketRegime, RegimeAnalysis, RegimeIndicators
from src.strategies.session import TradingSession
from src.strategies.signal import Signal


def test_supervisor_init(test_container):
    """Test supervisor initialization."""
    supervisor = test_container.supervisor()
    assert supervisor.llm is not None
    assert supervisor._prompts is not None


@pytest.mark.asyncio
async def test_plan_analyses_basic(test_container):
    """Test basic analysis planning."""
    supervisor = test_container.supervisor()

    context = PlanningContext(
        symbol="AAPL",
        regime=None,
        trading_session=TradingSession.REGULAR,
        owns_position=False,
        news_count=10,
        fundamental_available=True,
        social_available=True,
        trump_count=0,
        fundamental_rate_limit=False,
        time_budget_ms=30000,
        market_data_rows=50,
        is_high_volatility=False,
    )

    # Mock falls back to default routing
    decision = await supervisor.plan_analyses(context)

    assert isinstance(decision, AnalysisRoutingDecision)
    assert AnalysisType.TECHNICAL in decision.required_analyses
    assert AnalysisType.SENTIMENT in decision.required_analyses
    assert AnalysisType.NEWS in decision.required_analyses
    assert decision.reasoning is not None


@pytest.mark.asyncio
async def test_plan_analyses_rate_limited(test_container):
    """Test planning with rate limited fundamental API."""
    supervisor = test_container.supervisor()

    context = PlanningContext(
        symbol="TSLA",
        regime=None,
        trading_session=TradingSession.REGULAR,
        owns_position=False,
        news_count=5,
        fundamental_available=True,
        social_available=False,
        trump_count=0,
        fundamental_rate_limit=True,
        time_budget_ms=20000,
        market_data_rows=50,
        is_high_volatility=False,
    )

    # Default routing skips fundamental when rate limited
    decision = await supervisor.plan_analyses(context)

    assert AnalysisType.FUNDAMENTAL in decision.skip_analyses
    assert "rate limit" in decision.skip_analyses[AnalysisType.FUNDAMENTAL].lower()


@pytest.mark.asyncio
async def test_plan_analyses_fallback(test_container):
    """Test fallback to default routing when LLM fails."""
    supervisor = test_container.supervisor()

    # Mock LLM failure
    supervisor.llm.astructured.side_effect = StructuredOutputError("JSON parse error")

    context = PlanningContext(
        symbol="AAPL",
        regime=None,
        trading_session=TradingSession.REGULAR,
        owns_position=False,
        news_count=10,
        fundamental_available=True,
        social_available=True,
        trump_count=2,
        fundamental_rate_limit=False,
        time_budget_ms=30000,
        market_data_rows=50,
        is_high_volatility=False,
    )

    decision = await supervisor.plan_analyses(context)

    assert len(decision.required_analyses) > 0
    assert "fallback" in decision.reasoning.lower()
    assert AnalysisType.TECHNICAL in decision.required_analyses
    assert AnalysisType.TRUMP in decision.optional_analyses


@pytest.mark.asyncio
async def test_synthesize_results_consensus(test_container):
    """Test synthesis returns uniform weights (fallback behavior)."""
    supervisor = test_container.supervisor()

    context = SynthesisContext(
        symbol="AAPL",
        technical_summary="BUY signal, RSI=65, strong momentum",
        sentiment_summary="Positive sentiment, score=0.7",
        news_summary="Bullish product launch news",
    )

    completed = [AnalysisType.TECHNICAL, AnalysisType.SENTIMENT, AnalysisType.NEWS]
    weights = await supervisor.synthesize_results(context, completed)

    # Mock falls back to default weights
    assert all(w == 0.8 for w in weights.weights.values())
    assert weights.confidence_adjustment == 1.0


@pytest.mark.asyncio
async def test_synthesize_results_conflict(test_container):
    """Test synthesis returns uniform weights (fallback behavior)."""
    supervisor = test_container.supervisor()

    context = SynthesisContext(
        symbol="GME",
        technical_summary="SELL signal, RSI=80, overbought",
        sentiment_summary="Extremely positive, score=0.9",
        news_summary="Bearish earnings miss",
    )

    completed = [AnalysisType.TECHNICAL, AnalysisType.SENTIMENT, AnalysisType.NEWS]
    weights = await supervisor.synthesize_results(context, completed)

    # Mock falls back to default weights
    assert all(w == 0.8 for w in weights.weights.values())
    assert weights.confidence_adjustment == 1.0


@pytest.mark.asyncio
async def test_synthesize_results_fallback(test_container):
    """Test fallback to uniform weights when LLM fails."""
    supervisor = test_container.supervisor()

    # Mock LLM failure
    supervisor.llm.astructured.side_effect = StructuredOutputError("JSON parse error")

    context = SynthesisContext(
        symbol="AAPL",
        technical_summary="BUY signal",
        sentiment_summary="Positive",
    )

    completed = [AnalysisType.TECHNICAL, AnalysisType.SENTIMENT]
    weights = await supervisor.synthesize_results(context, completed)

    assert all(w == 0.8 for w in weights.weights.values())
    assert weights.confidence_adjustment == 1.0
    assert "fallback" in weights.reasoning.lower()


def testdefault_routing(test_container):
    """Test default routing logic."""
    supervisor = test_container.supervisor()

    context = PlanningContext(
        symbol="AAPL",
        regime=None,
        trading_session=TradingSession.REGULAR,
        owns_position=False,
        news_count=10,
        fundamental_available=True,
        social_available=True,
        trump_count=0,
        fundamental_rate_limit=False,
        time_budget_ms=30000,
        market_data_rows=50,
        is_high_volatility=False,
    )

    decision = supervisor.default_routing(context)

    # Required analyses
    assert AnalysisType.TECHNICAL in decision.required_analyses
    assert AnalysisType.SENTIMENT in decision.required_analyses
    assert AnalysisType.NEWS in decision.required_analyses
    assert AnalysisType.BULLISH_RESEARCH in decision.required_analyses
    assert AnalysisType.BEARISH_RESEARCH in decision.required_analyses

    # Optional when available
    assert AnalysisType.FUNDAMENTAL in decision.optional_analyses
    assert AnalysisType.SOCIAL_SENTIMENT in decision.optional_analyses


def testdefault_routing_rate_limited(test_container):
    """Test default routing skips fundamental when rate limited."""
    supervisor = test_container.supervisor()

    context = PlanningContext(
        symbol="TSLA",
        regime=None,
        trading_session=TradingSession.REGULAR,
        owns_position=False,
        news_count=5,
        fundamental_available=True,
        social_available=False,
        trump_count=0,
        fundamental_rate_limit=True,
        time_budget_ms=20000,
        market_data_rows=50,
        is_high_volatility=False,
    )

    decision = supervisor.default_routing(context)

    assert AnalysisType.FUNDAMENTAL in decision.skip_analyses
    assert "rate limit" in decision.skip_analyses[AnalysisType.FUNDAMENTAL].lower()


def testdefault_routing_trump_posts(test_container):
    """Test default routing includes trump when posts available."""
    supervisor = test_container.supervisor()

    context = PlanningContext(
        symbol="TSLA",
        regime=None,
        trading_session=TradingSession.REGULAR,
        owns_position=False,
        news_count=5,
        fundamental_available=False,
        social_available=False,
        trump_count=3,
        fundamental_rate_limit=False,
        time_budget_ms=20000,
        market_data_rows=50,
        is_high_volatility=False,
    )

    decision = supervisor.default_routing(context)

    assert AnalysisType.TRUMP in decision.optional_analyses


def test_default_weights(test_container):
    """Test default uniform weights."""
    supervisor = test_container.supervisor()

    completed = [AnalysisType.TECHNICAL, AnalysisType.SENTIMENT, AnalysisType.NEWS]
    weights = supervisor._default_weights(completed)

    assert all(w == 0.8 for w in weights.weights.values())
    assert len(weights.weights) == len(completed)
    assert weights.confidence_adjustment == 1.0


def test_format_analyses_summary(test_container):
    """Test formatting of analyses summary."""
    supervisor = test_container.supervisor()

    context = SynthesisContext(
        symbol="AAPL",
        technical_summary="BUY signal, RSI=65",
        sentiment_summary="Positive sentiment",
        news_summary="Bullish news",
        fundamental_summary=None,  # Not completed
    )

    completed = [AnalysisType.TECHNICAL, AnalysisType.SENTIMENT, AnalysisType.NEWS]
    summary = supervisor._format_analyses_summary(context, completed)

    assert "TECHNICAL: BUY signal" in summary
    assert "SENTIMENT: Positive" in summary
    assert "NEWS: Bullish" in summary
    assert "FUNDAMENTAL" not in summary  # Not completed


def test_format_analyses_summary_empty(test_container):
    """Test formatting with no completed analyses."""
    supervisor = test_container.supervisor()

    context = SynthesisContext(symbol="AAPL")
    completed = []
    summary = supervisor._format_analyses_summary(context, completed)

    assert summary == ""


def test_repr(test_container):
    """Test string representation."""
    supervisor = test_container.supervisor()
    repr_str = repr(supervisor)

    assert "TradingSupervisor" in repr_str
    assert "ollama" in repr_str


@pytest.mark.asyncio
async def test_plan_analyses_with_regime(test_container):
    """Test planning with market regime context."""
    supervisor = test_container.supervisor()

    regime_analysis = RegimeAnalysis(
        regime=MarketRegime.HIGH_VOLATILITY,
        indicators=RegimeIndicators(
            adx=30.0, plus_di=25.0, minus_di=20.0, atr=2.5, atr_ratio=1.8, bb_width=0.15
        ),
        confidence=0.8,
        reasoning="Market showing high volatility",
    )

    context = PlanningContext(
        symbol="AAPL",
        regime=regime_analysis,
        trading_session=TradingSession.REGULAR,
        owns_position=False,
        news_count=10,
        fundamental_available=True,
        social_available=True,
        trump_count=0,
        fundamental_rate_limit=False,
        time_budget_ms=30000,
        market_data_rows=50,
        is_high_volatility=True,
    )

    decision = await supervisor.plan_analyses(context)

    assert isinstance(decision, AnalysisRoutingDecision)
    assert AnalysisType.TECHNICAL in decision.required_analyses


@pytest.mark.asyncio
async def test_plan_analyses_pre_market(test_container):
    """Test planning during pre-market session."""
    supervisor = test_container.supervisor()

    context = PlanningContext(
        symbol="AAPL",
        regime=None,
        trading_session=TradingSession.PRE_MARKET,
        owns_position=False,
        news_count=5,
        fundamental_available=True,
        social_available=False,
        trump_count=0,
        fundamental_rate_limit=False,
        time_budget_ms=15000,
        market_data_rows=50,
        is_high_volatility=False,
    )

    decision = await supervisor.plan_analyses(context)

    # Default routing still applies
    assert isinstance(decision, AnalysisRoutingDecision)
    assert AnalysisType.TECHNICAL in decision.required_analyses


def test_default_routing_no_news(test_container):
    """Sentiment/news optional (not skipped) when news_count=0 — workers handle empty input."""
    supervisor = test_container.supervisor()
    context = PlanningContext(
        symbol="AAPL",
        regime=None,
        trading_session=TradingSession.REGULAR,
        owns_position=False,
        news_count=0,
        fundamental_available=True,
        social_available=False,
        trump_count=0,
        fundamental_rate_limit=False,
        time_budget_ms=30000,
        market_data_rows=50,
        is_high_volatility=False,
    )
    decision = supervisor.default_routing(context)

    assert AnalysisType.SENTIMENT in decision.optional_analyses
    assert AnalysisType.NEWS in decision.optional_analyses
    assert AnalysisType.SENTIMENT not in decision.skip_analyses
    assert AnalysisType.NEWS not in decision.skip_analyses
    assert AnalysisType.TECHNICAL in decision.required_analyses


def test_default_routing_insufficient_data(test_container):
    """Skip technical when market_data_rows < 35."""
    supervisor = test_container.supervisor()
    context = PlanningContext(
        symbol="AAPL",
        regime=None,
        trading_session=TradingSession.REGULAR,
        owns_position=False,
        news_count=10,
        fundamental_available=True,
        social_available=False,
        trump_count=0,
        fundamental_rate_limit=False,
        time_budget_ms=30000,
        market_data_rows=20,
        is_high_volatility=False,
    )
    decision = supervisor.default_routing(context)

    assert AnalysisType.TECHNICAL in decision.skip_analyses
    assert "Insufficient data" in decision.skip_analyses[AnalysisType.TECHNICAL]
    assert AnalysisType.BULLISH_RESEARCH in decision.skip_analyses
    assert AnalysisType.BEARISH_RESEARCH in decision.skip_analyses


def test_default_routing_pre_market_priority(test_container):
    """Pre-market prioritizes news/sentiment."""
    supervisor = test_container.supervisor()
    context = PlanningContext(
        symbol="AAPL",
        regime=None,
        trading_session=TradingSession.PRE_MARKET,
        owns_position=False,
        news_count=5,
        fundamental_available=True,
        social_available=False,
        trump_count=0,
        fundamental_rate_limit=False,
        time_budget_ms=15000,
        market_data_rows=50,
        is_high_volatility=False,
    )
    decision = supervisor.default_routing(context)

    # News and sentiment at start of priority_order
    assert decision.priority_order[0] in [AnalysisType.NEWS, AnalysisType.SENTIMENT]
    assert decision.priority_order[1] in [AnalysisType.NEWS, AnalysisType.SENTIMENT]
    assert AnalysisType.TECHNICAL in decision.required_analyses


def test_default_routing_combined_constraints(test_container):
    """Multiple constraints: no news + insufficient data."""
    supervisor = test_container.supervisor()
    context = PlanningContext(
        symbol="AAPL",
        regime=None,
        trading_session=TradingSession.REGULAR,
        owns_position=False,
        news_count=0,
        fundamental_available=False,
        social_available=False,
        trump_count=0,
        fundamental_rate_limit=False,
        time_budget_ms=30000,
        market_data_rows=15,
        is_high_volatility=True,
    )
    decision = supervisor.default_routing(context)

    # Technical skipped (insufficient data), sentiment/news optional (workers handle empty)
    assert AnalysisType.TECHNICAL in decision.skip_analyses
    assert AnalysisType.SENTIMENT in decision.optional_analyses
    assert AnalysisType.NEWS in decision.optional_analyses
    assert AnalysisType.FUNDAMENTAL in decision.skip_analyses
    assert AnalysisType.BULLISH_RESEARCH in decision.skip_analyses

    # Still should have reasoning
    assert "fallback" in decision.reasoning.lower()


def _make_approval_context(
    *,
    confidence: float = 0.8,
    risk_level: str = "LOW",
    action: Signal = Signal.BUY,
) -> TradeApprovalContext:
    return TradeApprovalContext(
        symbol="AAPL",
        action=action,
        confidence=confidence,
        risk_level=risk_level,
        risk_score=0.3,
        current_price=150.0,
        recommended_shares=10,
        position_value=1500.0,
        stop_loss_price=145.0,
        reward_risk_ratio=2.5,
        decision_reasoning=["Strong momentum", "Positive news"],
        risk_warnings=[],
    )


@pytest.mark.asyncio
async def test_approve_trade_high_confidence_low_risk(test_container):
    """Approve trade with high confidence and low risk."""
    supervisor = test_container.supervisor()
    ctx = _make_approval_context(confidence=0.85, risk_level="LOW")

    decision = await supervisor.approve_trade(ctx, symbol="AAPL")

    assert isinstance(decision, TradeApprovalDecision)
    assert decision.approved is True
    assert decision.reasoning


@pytest.mark.asyncio
async def test_approve_trade_high_confidence_medium_risk(test_container):
    """Approve trade with high confidence and medium risk."""
    supervisor = test_container.supervisor()
    ctx = _make_approval_context(confidence=0.75, risk_level="MEDIUM")

    decision = await supervisor.approve_trade(ctx, symbol="AAPL")

    assert isinstance(decision, TradeApprovalDecision)
    assert decision.approved is True


@pytest.mark.asyncio
async def test_approve_trade_low_confidence_rejected(test_container):
    """Reject trade with low confidence."""
    supervisor = test_container.supervisor()
    ctx = _make_approval_context(confidence=0.5, risk_level="LOW")

    decision = await supervisor.approve_trade(ctx, symbol="AAPL")

    assert isinstance(decision, TradeApprovalDecision)
    assert decision.approved is False
    assert len(decision.key_concerns) > 0


@pytest.mark.asyncio
async def test_approve_trade_high_risk_rejected(test_container):
    """Reject trade with high risk level."""
    supervisor = test_container.supervisor()
    ctx = _make_approval_context(confidence=0.9, risk_level="HIGH")

    decision = await supervisor.approve_trade(ctx, symbol="AAPL")

    assert isinstance(decision, TradeApprovalDecision)
    assert decision.approved is False
    assert len(decision.key_concerns) > 0


@pytest.mark.asyncio
async def test_approve_trade_llm_fallback(test_container):
    """Fall back to heuristic when LLM fails."""
    supervisor = test_container.supervisor()
    supervisor.llm.astructured.side_effect = StructuredOutputError("JSON parse error")
    ctx = _make_approval_context(confidence=0.8, risk_level="LOW")

    decision = await supervisor.approve_trade(ctx, symbol="AAPL")

    assert isinstance(decision, TradeApprovalDecision)
    assert decision.approved is True
    assert "fallback" in decision.reasoning.lower()


@pytest.mark.asyncio
async def test_approve_trade_fallback_logs_rejection(test_container):
    """Fallback heuristic rejects low-confidence/high-risk when LLM fails."""
    supervisor = test_container.supervisor()
    supervisor.llm.astructured.side_effect = StructuredOutputError("JSON parse error")
    ctx = _make_approval_context(confidence=0.4, risk_level="HIGH")

    decision = await supervisor.approve_trade(ctx, symbol="AAPL")

    assert decision.approved is False
    assert len(decision.key_concerns) > 0
