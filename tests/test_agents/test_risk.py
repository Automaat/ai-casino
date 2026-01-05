"""Tests for risk management agent."""

import json

import pandas as pd
import pytest

from src.agents.risk import (
    AccountInfo,
    PositionSizeCalculation,
    RiskAssessment,
    RiskManagementAgent,
    RiskValidation,
    StopLossCalculation,
    TrailingStopConfig,
)
from src.agents.technical import TechnicalAnalysis
from src.strategies.momentum import Signal


@pytest.fixture
def account_info():
    """Sample account info."""
    return AccountInfo(
        balance=100000.0,
        available_cash=80000.0,
        positions={"SPY": 100.0},
        total_exposure=20000.0,
    )


@pytest.fixture
def risk_agent(mock_llm_client):
    """Risk management agent instance."""
    return RiskManagementAgent(mock_llm_client)


@pytest.fixture
def technical_analysis():
    """Sample technical analysis."""
    return TechnicalAnalysis(
        signal=Signal.BUY,
        rsi=25.0,
        macd_hist=0.5,
        interpretation="Bullish momentum",
        confidence=0.8,
    )


def test_risk_agent_init(mock_llm_client):
    """Test risk agent initialization."""
    agent = RiskManagementAgent(mock_llm_client)

    assert agent.llm == mock_llm_client
    assert agent.max_position_risk == 2.0
    assert agent.max_exposure == 80.0
    assert agent.max_single_position == 20.0
    assert agent.enable_trailing_stop is True


def test_risk_agent_custom_limits(mock_llm_client):
    """Test custom risk limits."""
    agent = RiskManagementAgent(
        mock_llm_client,
        max_position_risk=3.0,
        max_exposure=90.0,
        max_single_position=25.0,
        enable_trailing_stop=False,
    )

    assert agent.max_position_risk == 3.0
    assert agent.max_exposure == 90.0
    assert agent.max_single_position == 25.0
    assert agent.enable_trailing_stop is False


def test_assess_hold_action(risk_agent, account_info, sample_ohlcv_data, technical_analysis):
    """Test assessment for HOLD action."""
    technical_analysis.signal = Signal.HOLD

    result = risk_agent.assess(
        symbol="AAPL",
        action=Signal.HOLD,
        current_price=150.0,
        account_info=account_info,
        market_data=sample_ohlcv_data,
        decision_confidence=0.7,
    )

    assert isinstance(result, RiskAssessment)
    assert result.action == Signal.HOLD
    assert result.validation.approved is True
    assert result.position_sizing.recommended_shares == 0
    assert result.confidence == 1.0
    assert result.validation.risk_level == "LOW"


def test_calculate_stop_loss_atr(risk_agent, sample_ohlcv_data):
    """Test ATR-based stop-loss calculation."""
    stop_loss = risk_agent._calculate_stop_loss(150.0, sample_ohlcv_data, Signal.BUY)

    assert isinstance(stop_loss, StopLossCalculation)
    assert stop_loss.stop_loss_price < 150.0
    assert stop_loss.stop_loss_percent > 0
    assert stop_loss.risk_per_share > 0
    assert "ATR" in stop_loss.methodology
    assert isinstance(stop_loss.trailing_stop, TrailingStopConfig)
    assert stop_loss.trailing_stop.enabled is True


def test_calculate_stop_loss_sell_action(risk_agent, sample_ohlcv_data):
    """Test stop-loss for SELL action."""
    stop_loss = risk_agent._calculate_stop_loss(150.0, sample_ohlcv_data, Signal.SELL)

    assert stop_loss.stop_loss_price > 150.0
    assert stop_loss.trailing_stop is None


def test_calculate_stop_loss_no_trailing(mock_llm_client, sample_ohlcv_data):
    """Test stop-loss without trailing stop."""
    agent = RiskManagementAgent(mock_llm_client, enable_trailing_stop=False)
    stop_loss = agent._calculate_stop_loss(150.0, sample_ohlcv_data, Signal.BUY)

    assert stop_loss.trailing_stop is None


def test_calculate_position_size(risk_agent, account_info):
    """Test position size calculation."""
    stop_loss = StopLossCalculation(
        stop_loss_price=147.0,
        stop_loss_percent=2.0,
        risk_per_share=3.0,
        max_loss_amount=0.0,
        methodology="Fixed 2%",
    )

    result = risk_agent._calculate_position_size(150.0, stop_loss, account_info)

    assert isinstance(result, PositionSizeCalculation)
    assert result.recommended_shares > 0
    assert result.risk_percent <= risk_agent.max_position_risk
    assert result.position_value <= account_info.available_cash
    assert result.risk_amount > 0


def test_calculate_position_size_cash_constraint(risk_agent, account_info):
    """Test position sizing with cash constraint."""
    account_info.available_cash = 5000.0

    stop_loss = StopLossCalculation(
        stop_loss_price=147.0,
        stop_loss_percent=2.0,
        risk_per_share=3.0,
        max_loss_amount=0.0,
        methodology="Fixed 2%",
    )

    result = risk_agent._calculate_position_size(150.0, stop_loss, account_info)

    assert result.position_value <= 5000.0


def test_calculate_position_size_max_single_position(risk_agent, account_info):
    """Test position sizing with max single position constraint."""
    stop_loss = StopLossCalculation(
        stop_loss_price=149.0,
        stop_loss_percent=0.67,
        risk_per_share=1.0,
        max_loss_amount=0.0,
        methodology="Fixed 0.67%",
    )

    result = risk_agent._calculate_position_size(150.0, stop_loss, account_info)

    max_allowed = account_info.balance * (risk_agent.max_single_position / 100)
    assert result.position_value <= max_allowed


def test_validate_risk_approved(risk_agent, account_info):
    """Test risk validation for approved trade."""
    position_sizing = PositionSizeCalculation(
        recommended_shares=100,
        position_value=15000.0,
        risk_amount=300.0,
        risk_percent=0.3,
        reasoning="Test",
    )

    validation = risk_agent._validate_risk("AAPL", Signal.BUY, position_sizing, account_info, 0.75)

    assert isinstance(validation, RiskValidation)
    assert validation.approved is True
    assert validation.risk_level in ["LOW", "MEDIUM", "HIGH"]
    assert 0.0 <= validation.risk_score <= 1.0
    assert len(validation.warnings) == 0


def test_validate_risk_insufficient_cash(risk_agent, account_info):
    """Test risk validation with insufficient cash."""
    position_sizing = PositionSizeCalculation(
        recommended_shares=1000,
        position_value=150000.0,
        risk_amount=3000.0,
        risk_percent=3.0,
        reasoning="Test",
    )

    validation = risk_agent._validate_risk("AAPL", Signal.BUY, position_sizing, account_info, 0.75)

    assert validation.approved is False
    assert len(validation.warnings) > 0
    assert validation.constraints_met["cash_available"] is False
    assert any("Insufficient cash" in w for w in validation.warnings)


def test_validate_risk_high_position_risk(risk_agent, account_info):
    """Test risk validation with high position risk."""
    position_sizing = PositionSizeCalculation(
        recommended_shares=100,
        position_value=15000.0,
        risk_amount=5000.0,
        risk_percent=5.0,
        reasoning="Test",
    )

    validation = risk_agent._validate_risk("AAPL", Signal.BUY, position_sizing, account_info, 0.75)

    assert validation.approved is False
    assert validation.constraints_met["position_risk"] is False


def test_validate_risk_high_exposure(risk_agent, account_info):
    """Test risk validation with high total exposure."""
    account_info.total_exposure = 75000.0

    position_sizing = PositionSizeCalculation(
        recommended_shares=100,
        position_value=10000.0,
        risk_amount=200.0,
        risk_percent=0.2,
        reasoning="Test",
    )

    validation = risk_agent._validate_risk("AAPL", Signal.BUY, position_sizing, account_info, 0.75)

    assert validation.approved is False
    assert validation.constraints_met["total_exposure"] is False


def test_validate_risk_duplicate_position(risk_agent, account_info):
    """Test risk validation for duplicate position."""
    position_sizing = PositionSizeCalculation(
        recommended_shares=100,
        position_value=15000.0,
        risk_amount=300.0,
        risk_percent=0.3,
        reasoning="Test",
    )

    validation = risk_agent._validate_risk("SPY", Signal.BUY, position_sizing, account_info, 0.75)

    assert validation.approved is False
    assert validation.constraints_met["no_duplicate"] is False
    assert any("Already have position" in w for w in validation.warnings)


def test_validate_risk_sell_without_position(risk_agent, account_info):
    """Test risk validation for SELL without position."""
    position_sizing = PositionSizeCalculation(
        recommended_shares=100,
        position_value=15000.0,
        risk_amount=300.0,
        risk_percent=0.3,
        reasoning="Test",
    )

    validation = risk_agent._validate_risk("AAPL", Signal.SELL, position_sizing, account_info, 0.75)

    assert validation.approved is False
    assert validation.constraints_met["has_position_to_sell"] is False
    assert any("No position" in w for w in validation.warnings)


def test_validate_risk_sell_with_position(risk_agent, account_info):
    """Test risk validation for SELL with existing position."""
    position_sizing = PositionSizeCalculation(
        recommended_shares=100,
        position_value=15000.0,
        risk_amount=300.0,
        risk_percent=0.3,
        reasoning="Test",
    )

    validation = risk_agent._validate_risk("SPY", Signal.SELL, position_sizing, account_info, 0.75)

    assert "has_position_to_sell" in validation.constraints_met
    assert validation.constraints_met["has_position_to_sell"] is True


def test_validate_risk_low_confidence(risk_agent, account_info):
    """Test risk validation with low confidence."""
    position_sizing = PositionSizeCalculation(
        recommended_shares=100,
        position_value=15000.0,
        risk_amount=300.0,
        risk_percent=0.3,
        reasoning="Test",
    )

    validation = risk_agent._validate_risk("AAPL", Signal.BUY, position_sizing, account_info, 0.5)

    assert validation.constraints_met["confidence"] is False


def test_assess_buy_approved(risk_agent, account_info, sample_ohlcv_data, technical_analysis):
    """Test full assessment for approved BUY."""
    result = risk_agent.assess(
        symbol="AAPL",
        action=Signal.BUY,
        current_price=150.0,
        account_info=account_info,
        market_data=sample_ohlcv_data,
        decision_confidence=0.8,
    )

    assert isinstance(result, RiskAssessment)
    assert result.action == Signal.BUY
    assert result.position_sizing.recommended_shares > 0
    assert result.stop_loss.stop_loss_price < result.current_price
    assert isinstance(result.validation, RiskValidation)
    assert 0.0 <= result.confidence <= 1.0


def test_assess_sell(risk_agent, account_info, sample_ohlcv_data, technical_analysis):
    """Test assessment for SELL action."""
    technical_analysis.signal = Signal.SELL

    result = risk_agent.assess(
        symbol="SPY",
        action=Signal.SELL,
        current_price=150.0,
        account_info=account_info,
        market_data=sample_ohlcv_data,
        decision_confidence=0.7,
    )

    assert result.action == Signal.SELL
    assert result.stop_loss.stop_loss_price > result.current_price


def test_get_atr(risk_agent, sample_ohlcv_data):
    """Test ATR calculation."""
    atr = risk_agent._get_atr(sample_ohlcv_data)

    assert atr is not None
    assert atr > 0


def test_get_atr_failure(risk_agent):
    """Test ATR calculation failure."""
    bad_df = pd.DataFrame({"Close": [100, 101, 102]})
    atr = risk_agent._get_atr(bad_df)

    assert atr is None


def test_calculate_risk_score(risk_agent):
    """Test risk score calculation."""
    score = risk_agent._calculate_risk_score(1.0, 50.0, 0.8)

    assert 0.0 <= score <= 1.0

    high_risk_score = risk_agent._calculate_risk_score(2.0, 80.0, 0.5)
    low_risk_score = risk_agent._calculate_risk_score(0.5, 20.0, 0.9)

    assert low_risk_score > high_risk_score


def test_calculate_risk_confidence_approved(risk_agent):
    """Test risk confidence for approved trade."""
    validation = RiskValidation(
        approved=True,
        risk_score=0.8,
        risk_level="LOW",
        warnings=[],
        constraints_met={},
        reasoning="Test",
    )

    confidence = risk_agent._calculate_risk_confidence(validation, 0.75)

    assert 0.0 <= confidence <= 1.0
    assert confidence > 0.7


def test_calculate_risk_confidence_rejected(risk_agent):
    """Test risk confidence for rejected trade."""
    validation = RiskValidation(
        approved=False,
        risk_score=0.5,
        risk_level="HIGH",
        warnings=["Test warning"],
        constraints_met={},
        reasoning="Test",
    )

    confidence = risk_agent._calculate_risk_confidence(validation, 0.75)

    assert confidence < 0.5


def test_audit_log(risk_agent, account_info, sample_ohlcv_data, technical_analysis, tmp_path):
    """Test audit logging."""
    risk_agent.audit_log_path = tmp_path / "risk_audit.jsonl"

    risk_agent.assess(
        symbol="AAPL",
        action=Signal.BUY,
        current_price=150.0,
        account_info=account_info,
        market_data=sample_ohlcv_data,
        decision_confidence=0.8,
    )

    assert risk_agent.audit_log_path.exists()

    with risk_agent.audit_log_path.open() as f:
        lines = f.readlines()
        assert len(lines) == 1
        log_entry = json.loads(lines[0])
        assert log_entry["symbol"] == "AAPL"
        assert log_entry["action"] == "BUY"
        assert "timestamp" in log_entry


def test_repr(risk_agent):
    """Test string representation."""
    repr_str = repr(risk_agent)

    assert "RiskManagementAgent" in repr_str
    assert "max_risk=2.0%" in repr_str
    assert "trailing=True" in repr_str
