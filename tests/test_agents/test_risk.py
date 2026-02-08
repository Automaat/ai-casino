"""Tests for risk management agent."""

import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.agents.risk import (
    AccountInfo,
    PortfolioRiskReport,
    PortfolioVaRConfig,
    PositionSizeCalculation,
    RiskAssessment,
    RiskManagementAgent,
    RiskValidation,
    StopLossCalculation,
    TrailingStopConfig,
)
from src.agents.technical import TechnicalAnalysis
from src.data.broker import BrokerPosition
from src.metrics.portfolio_var import PortfolioVaRCalculator, PortfolioVaRResult
from src.strategies.signal import Signal


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


# --- Portfolio VaR tests ---


def _make_var_result(
    var_95: float = 0.02, cvar_99: float = 0.04, cdar_95: float = 0.05
) -> PortfolioVaRResult:
    return PortfolioVaRResult(
        var_95=var_95,
        var_99=0.03,
        cvar_95=0.03,
        cvar_99=cvar_99,
        cdar_95=cdar_95,
        max_drawdown=0.08,
        portfolio_volatility=0.15,
        num_positions=2,
        lookback_days=90,
        sufficient_data=True,
    )


def _make_mock_var_calculator(var_result: PortfolioVaRResult) -> MagicMock:
    mock = MagicMock(spec=PortfolioVaRCalculator)
    mock.calculate_with_hypothetical.return_value = var_result
    mock.calculate.return_value = var_result
    return mock


def _make_position(symbol: str, market_value: float) -> BrokerPosition:
    return BrokerPosition(
        symbol=symbol,
        qty=10.0,
        market_value=market_value,
        avg_entry_price=market_value / 10,
        unrealized_pnl=0.0,
        unrealized_pnl_percent=0.0,
    )


class TestPortfolioVaRValidation:
    def test_within_limits(self, mock_llm_client):
        var_result = _make_var_result(var_95=0.02, cvar_99=0.03)
        agent = RiskManagementAgent(
            mock_llm_client,
            portfolio_var_calculator=_make_mock_var_calculator(var_result),
            portfolio_var_config=PortfolioVaRConfig(enabled=True),
        )

        warnings: list[str] = []
        result = agent._validate_portfolio_var(
            "AAPL",
            Signal.BUY,
            15000.0,
            {"SPY": _make_position("SPY", 20000.0)},
            100000.0,
            warnings,
        )

        assert result is True
        assert len(warnings) == 0

    def test_breach(self, mock_llm_client):
        var_result = _make_var_result(var_95=0.05, cvar_99=0.08)
        agent = RiskManagementAgent(
            mock_llm_client,
            portfolio_var_calculator=_make_mock_var_calculator(var_result),
            portfolio_var_config=PortfolioVaRConfig(enabled=True, max_var_95=0.03, max_cvar_99=0.05),
        )

        warnings: list[str] = []
        result = agent._validate_portfolio_var(
            "AAPL",
            Signal.BUY,
            15000.0,
            {"SPY": _make_position("SPY", 20000.0)},
            100000.0,
            warnings,
        )

        assert result is False
        assert any("VaR95" in w for w in warnings)
        assert any("CVaR99" in w for w in warnings)

    def test_disabled(self, mock_llm_client):
        agent = RiskManagementAgent(
            mock_llm_client,
            portfolio_var_config=PortfolioVaRConfig(enabled=False),
        )

        warnings: list[str] = []
        result = agent._validate_portfolio_var(
            "AAPL",
            Signal.BUY,
            15000.0,
            {},
            100000.0,
            warnings,
        )

        assert result is True

    def test_insufficient_data(self, mock_llm_client):
        insufficient_result = PortfolioVaRResult(
            var_95=0.0,
            var_99=0.0,
            cvar_95=0.0,
            cvar_99=0.0,
            cdar_95=0.0,
            max_drawdown=0.0,
            portfolio_volatility=0.0,
            num_positions=0,
            lookback_days=90,
            sufficient_data=False,
        )
        agent = RiskManagementAgent(
            mock_llm_client,
            portfolio_var_calculator=_make_mock_var_calculator(insufficient_result),
            portfolio_var_config=PortfolioVaRConfig(enabled=True),
        )

        warnings: list[str] = []
        result = agent._validate_portfolio_var(
            "AAPL",
            Signal.BUY,
            15000.0,
            {},
            100000.0,
            warnings,
        )

        assert result is True
        assert any("Insufficient" in w for w in warnings)

    def test_sell_always_approved(self, mock_llm_client):
        var_result = _make_var_result(var_95=0.10, cvar_99=0.20)
        agent = RiskManagementAgent(
            mock_llm_client,
            portfolio_var_calculator=_make_mock_var_calculator(var_result),
            portfolio_var_config=PortfolioVaRConfig(enabled=True),
        )

        warnings: list[str] = []
        result = agent._validate_portfolio_var(
            "AAPL",
            Signal.SELL,
            15000.0,
            {"AAPL": _make_position("AAPL", 15000.0)},
            100000.0,
            warnings,
        )

        assert result is True


class TestAdaptiveStopLoss:
    def test_high_cdar(self, mock_llm_client):
        agent = RiskManagementAgent(
            mock_llm_client,
            portfolio_var_config=PortfolioVaRConfig(
                enabled=True,
                adaptive_stop_loss=True,
                cdar_stop_threshold=0.10,
                atr_multiplier_min=1.0,
            ),
        )
        # CDaR at 2x threshold → multiplier should be at min
        agent._portfolio_cdar = 0.20
        multiplier = agent._get_adaptive_atr_multiplier()

        assert multiplier == agent._var_config.atr_multiplier_min

    def test_low_cdar(self, mock_llm_client):
        agent = RiskManagementAgent(
            mock_llm_client,
            portfolio_var_config=PortfolioVaRConfig(
                enabled=True,
                adaptive_stop_loss=True,
                cdar_stop_threshold=0.10,
            ),
        )
        # CDaR below threshold → default multiplier
        agent._portfolio_cdar = 0.05
        multiplier = agent._get_adaptive_atr_multiplier()

        assert multiplier == agent.ATR_MULTIPLIER

    def test_disabled(self, mock_llm_client):
        agent = RiskManagementAgent(
            mock_llm_client,
            portfolio_var_config=PortfolioVaRConfig(
                enabled=True,
                adaptive_stop_loss=False,
            ),
        )
        agent._portfolio_cdar = 0.30
        multiplier = agent._get_adaptive_atr_multiplier()

        assert multiplier == agent.ATR_MULTIPLIER


class TestGenerateRiskReport:
    def test_fields(self, mock_llm_client):
        var_result = _make_var_result(var_95=0.02, cvar_99=0.03)
        agent = RiskManagementAgent(
            mock_llm_client,
            portfolio_var_calculator=_make_mock_var_calculator(var_result),
            portfolio_var_config=PortfolioVaRConfig(enabled=True),
        )

        positions = {"AAPL": _make_position("AAPL", 30000.0)}
        report = agent.generate_risk_report(positions, 100000.0, 30000.0)

        assert isinstance(report, PortfolioRiskReport)
        assert report.var_95 == 0.02
        assert report.cvar_99 == 0.03
        assert report.num_positions == 2
        assert report.current_exposure_percent == 30.0
        assert report.risk_status == "HEALTHY"

    def test_breach_status(self, mock_llm_client):
        var_result = _make_var_result(var_95=0.05, cvar_99=0.08)
        agent = RiskManagementAgent(
            mock_llm_client,
            portfolio_var_calculator=_make_mock_var_calculator(var_result),
            portfolio_var_config=PortfolioVaRConfig(enabled=True, max_var_95=0.03, max_cvar_99=0.05),
        )

        positions = {"AAPL": _make_position("AAPL", 50000.0)}
        report = agent.generate_risk_report(positions, 100000.0, 50000.0)

        assert report.risk_status == "BREACH"
        assert report.var_limit_breached is True
        assert report.cvar_limit_breached is True


class TestWeightBasedPositionSizing:
    def test_weight_based_position_sizing(self, mock_llm_client):
        agent = RiskManagementAgent(mock_llm_client)

        account_info = AccountInfo(balance=100000.0, available_cash=50000.0, positions={}, total_exposure=0.0)

        stop_loss = StopLossCalculation(
            stop_loss_price=95.0,
            stop_loss_percent=5.0,
            risk_per_share=5.0,
            max_loss_amount=0.0,
            methodology="ATR",
        )

        # Test 10% target weight
        result = agent._calculate_weight_based_position(
            current_price=100.0,
            stop_loss=stop_loss,
            account_info=account_info,
            target_weight=0.10,
        )

        expected_shares = 100  # 10% of 100k = 10k / 100 = 100 shares
        assert result.recommended_shares == expected_shares
        assert result.position_value == 10000.0
        assert result.risk_amount == 500.0  # 100 shares * 5.0 risk_per_share
        assert result.risk_percent == 0.5
        assert "Portfolio-weighted position" in result.reasoning
        assert "10.0% target" in result.reasoning

    def test_weight_based_constrained_by_cash(self, mock_llm_client):
        agent = RiskManagementAgent(mock_llm_client)

        account_info = AccountInfo(balance=100000.0, available_cash=5000.0, positions={}, total_exposure=0.0)

        stop_loss = StopLossCalculation(
            stop_loss_price=95.0,
            stop_loss_percent=5.0,
            risk_per_share=5.0,
            max_loss_amount=0.0,
            methodology="ATR",
        )

        # Target 20% but only have 5% cash available
        result = agent._calculate_weight_based_position(
            current_price=100.0,
            stop_loss=stop_loss,
            account_info=account_info,
            target_weight=0.20,
        )

        expected_shares = 50  # Limited by 5k cash / 100 = 50 shares
        assert result.recommended_shares == expected_shares
        assert result.position_value == 5000.0

    def test_weight_based_constrained_by_max_position(self, mock_llm_client):
        agent = RiskManagementAgent(mock_llm_client, max_single_position=10.0)

        account_info = AccountInfo(balance=100000.0, available_cash=50000.0, positions={}, total_exposure=0.0)

        stop_loss = StopLossCalculation(
            stop_loss_price=95.0,
            stop_loss_percent=5.0,
            risk_per_share=5.0,
            max_loss_amount=0.0,
            methodology="ATR",
        )

        # Target 20% but max position is 10%
        result = agent._calculate_weight_based_position(
            current_price=100.0,
            stop_loss=stop_loss,
            account_info=account_info,
            target_weight=0.20,
        )

        expected_shares = 100  # Limited by 10% max = 10k / 100 = 100 shares
        assert result.recommended_shares == expected_shares
        assert result.position_value == 10000.0

    def test_assess_with_target_weight(self, mock_llm_client, sample_ohlcv_data):
        agent = RiskManagementAgent(mock_llm_client)

        account_info = AccountInfo(balance=100000.0, available_cash=50000.0, positions={}, total_exposure=0.0)

        sample_ohlcv_data["Close"] = [100.0] * len(sample_ohlcv_data)
        sample_ohlcv_data["High"] = [105.0] * len(sample_ohlcv_data)
        sample_ohlcv_data["Low"] = [95.0] * len(sample_ohlcv_data)

        result = agent.assess(
            symbol="AAPL",
            action=Signal.BUY,
            current_price=100.0,
            account_info=account_info,
            market_data=sample_ohlcv_data,
            decision_confidence=0.85,
            target_portfolio_weight=0.15,
        )

        assert result.position_sizing.recommended_shares > 0
        assert "Portfolio-weighted position" in result.position_sizing.reasoning
        assert result.position_sizing.position_value <= 15000.0  # 15% target


def test_broker_failure_blocks_approval(mock_llm_client, sample_ohlcv_data):
    """broker_api_failed flag prevents approval."""
    agent = RiskManagementAgent(mock_llm_client)

    account_info = AccountInfo(balance=100000.0, available_cash=50000.0, positions={}, total_exposure=0.0)

    sample_ohlcv_data["Close"] = [150.0] * len(sample_ohlcv_data)
    sample_ohlcv_data["High"] = [155.0] * len(sample_ohlcv_data)
    sample_ohlcv_data["Low"] = [145.0] * len(sample_ohlcv_data)

    assessment = agent.assess(
        symbol="AAPL",
        action=Signal.BUY,
        current_price=150.0,
        account_info=account_info,
        market_data=sample_ohlcv_data,
        decision_confidence=0.8,
        broker_api_failed=True,
    )

    assert not assessment.validation.approved
    assert "broker_available" in assessment.validation.constraints_met
    assert not assessment.validation.constraints_met["broker_available"]
    assert any("Broker API unavailable" in w for w in assessment.validation.warnings)
