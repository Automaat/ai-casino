"""Integration tests for risk validation in workflow."""

from unittest.mock import AsyncMock, Mock

import pytest

from src.agents.news import NewsAnalysis
from src.agents.sentiment import SentimentAnalysis
from src.agents.technical import TechnicalAnalysis
from src.agents.thesis_researcher import BearishResearchAnalysis, BullishResearchAnalysis
from src.daemon.config.risk_validation import RiskValidationConfig
from src.strategies.signal import Signal
from src.validators.risk import RiskValidator
from src.workflows.models.risk_validation import RiskValidationInput, RiskValidationOutput
from src.workflows.stages.risk_validation import validate_analyses_stage


@pytest.fixture
def risk_validator():
    """Create risk validator with default config."""
    return RiskValidator(RiskValidationConfig(enabled=True))


@pytest.fixture
def weak_analyses():
    """Create weak analyses that should trigger warnings."""
    return {
        "technical": TechnicalAnalysis(
            signal=Signal.HOLD,
            rsi=50.0,
            macd_hist=0.0,
            interpretation="Unclear",
            confidence=0.3,  # Below default threshold (0.4)
        ),
        "sentiment": SentimentAnalysis(
            overall_sentiment="neutral",
            sentiment_score=0.0,
            positive_ratio=0.3,
            negative_ratio=0.3,
            neutral_ratio=0.4,
            article_count=5,
            summary="Neutral",
            confidence=0.35,
        ),
        "news": NewsAnalysis(
            key_themes=["mixed signals"],
            impact_assessment="Mixed signals",
            recommendation="hold",
            confidence=0.5,
        ),
    }


async def test_risk_validation_stage_integration(risk_validator, weak_analyses):
    """Test that validation stage correctly processes analyses and returns warnings."""
    from src.strategies.session import TradingSession

    # Create validation input
    validation_input = RiskValidationInput(
        symbol="AAPL",
        trading_session=TradingSession.REGULAR,
        technical_analysis=weak_analyses["technical"],
        sentiment_analysis=weak_analyses["sentiment"],
        news_analysis=weak_analyses["news"],
        fundamental_analysis=None,
        bullish_research=None,
        bearish_research=None,
        market_data=None,
        degradation_context=None,
    )

    # Run validation stage
    output = validate_analyses_stage(validation_input, risk_validator)

    # Verify output
    assert isinstance(output, RiskValidationOutput)
    assert output.validation_result.approved is True  # Warning-only mode
    assert len(output.validation_result.warnings) > 0  # Should have warnings
    assert output.validation_result.constraints_met["confidence_thresholds"] is False


async def test_validation_warnings_passed_to_trader(risk_validator, weak_analyses):
    """Test that validation warnings are passed to trader via DecisionInput."""
    from src.strategies.session import TradingSession
    from src.workflows.models.decision import DecisionContext, DecisionInput

    # Create validation input
    validation_input = RiskValidationInput(
        symbol="AAPL",
        trading_session=TradingSession.REGULAR,
        technical_analysis=weak_analyses["technical"],
        sentiment_analysis=weak_analyses["sentiment"],
        news_analysis=weak_analyses["news"],
        fundamental_analysis=None,
        bullish_research=None,
        bearish_research=None,
        market_data=None,
        degradation_context=None,
    )

    # Run validation
    validation_output = validate_analyses_stage(validation_input, risk_validator)

    # Create DecisionInput with validation_context
    decision_input = DecisionInput(
        symbol="AAPL",
        technical=weak_analyses["technical"],
        sentiment=weak_analyses["sentiment"],
        news=weak_analyses["news"],
        bullish=None,
        bearish=None,
        fundamental=None,
        comparative=None,
        trump=None,
        account_info=None,
        context=DecisionContext(),
        backtest_validation=None,
        degradation_context=None,
        validation_context=validation_output,  # Validation warnings included
    )

    # Verify validation_context is accessible
    assert decision_input.validation_context is not None
    assert decision_input.validation_context.validation_result.approved is True
    assert len(decision_input.validation_context.validation_result.warnings) > 0


async def test_trader_can_decide_despite_warnings(weak_analyses):
    """Test that trader can still make decision despite validation warnings."""
    from src.agents.trader import TraderAgent
    from src.strategies.session import TradingSession
    from src.workflows.models.decision import DecisionContext, DecisionInput

    # Mock LLM client
    mock_llm = Mock()

    # Create a mock response with actual Signal enum
    mock_response = Mock()
    mock_response.action = Signal.HOLD
    mock_response.confidence = 0.5
    mock_response.risk_level = "MEDIUM"
    mock_response.reasoning = ["Waiting for better entry"]

    mock_llm.astructured = AsyncMock(return_value=mock_response)

    # Create trader
    trader = TraderAgent(mock_llm)

    # Create validator and run validation
    validator = RiskValidator(RiskValidationConfig(enabled=True))
    validation_input = RiskValidationInput(
        symbol="AAPL",
        trading_session=TradingSession.REGULAR,
        technical_analysis=weak_analyses["technical"],
        sentiment_analysis=weak_analyses["sentiment"],
        news_analysis=weak_analyses["news"],
        fundamental_analysis=None,
        bullish_research=None,
        bearish_research=None,
        market_data=None,
        degradation_context=None,
    )
    validation_output = validate_analyses_stage(validation_input, validator)

    # Create DecisionInput with warnings
    decision_input = DecisionInput(
        symbol="AAPL",
        technical=weak_analyses["technical"],
        sentiment=weak_analyses["sentiment"],
        news=weak_analyses["news"],
        bullish=None,
        bearish=None,
        fundamental=None,
        comparative=None,
        trump=None,
        account_info=None,
        context=DecisionContext(),
        backtest_validation=None,
        degradation_context=None,
        validation_context=validation_output,
    )

    # Trader should still be able to make decision
    decision = await trader.decide(
        symbol=decision_input.symbol,
        technical=decision_input.technical,
        sentiment=decision_input.sentiment,
        news=decision_input.news,
        fundamental=decision_input.fundamental,
        bullish=decision_input.bullish or BullishResearchAnalysis(
            thesis="", key_strengths=[], target_upside=0.0, confidence=0.5
        ),
        bearish=decision_input.bearish or BearishResearchAnalysis(
            thesis="", key_weaknesses=[], target_downside=0.0, confidence=0.5
        ),
        comparative=decision_input.comparative,
        backtest_validation=decision_input.backtest_validation,
        degradation_context=decision_input.degradation_context,
    )

    # Verify decision was made
    assert decision is not None
    assert decision.action in {Signal.BUY, Signal.SELL, Signal.HOLD}
