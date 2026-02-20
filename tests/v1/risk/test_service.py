"""Tests for RiskService facade."""

from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest

from src.agents.risk.models import (
    AccountInfo,
    PositionSizeCalculation,
    RiskAssessment,
    RiskValidation,
    StopLossCalculation,
    TakeProfitCalculation,
)
from src.strategies.signal import Signal
from src.v1.risk.models import RiskDecision
from src.v1.risk.service import RiskService
from src.v1.trades.brokers.models import BrokerAccountInfo


def _make_market_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [100 + i for i in range(50)],
            "High": [105 + i for i in range(50)],
            "Low": [99 + i for i in range(50)],
            "Close": [104 + i for i in range(50)],
            "Volume": [1000000] * 50,
        }
    )


def _make_broker_account() -> BrokerAccountInfo:
    return BrokerAccountInfo(
        balance=100000.0,
        available_cash=80000.0,
        positions={},
        total_exposure=20000.0,
        portfolio_value=100000.0,
    )


def _make_assessment(approved: bool = True, shares: int = 25) -> RiskAssessment:
    return RiskAssessment(
        symbol="AAPL",
        action=Signal.BUY,
        current_price=150.0,
        account_info=AccountInfo(
            balance=100000.0, available_cash=80000.0, positions={}, total_exposure=20000.0
        ),
        position_sizing=PositionSizeCalculation(
            recommended_shares=shares,
            position_value=shares * 150.0,
            risk_amount=100.0,
            risk_percent=1.5,
            reasoning="Test sizing",
        ),
        stop_loss=StopLossCalculation(
            stop_loss_price=142.0,
            stop_loss_percent=5.3,
            risk_per_share=8.0,
            max_loss_amount=200.0,
            methodology="ATR-based (2.0x)",
        ),
        validation=RiskValidation(
            approved=approved,
            risk_score=0.8,
            risk_level="LOW" if approved else "HIGH",
            warnings=[] if approved else ["Total exposure exceeds max"],
            constraints_met={"position_risk": True, "total_exposure": approved},
            reasoning=f"{'APPROVED' if approved else 'REJECTED'}: test",
        ),
        confidence=0.85,
        take_profit=TakeProfitCalculation(
            take_profit_price=162.0,
            take_profit_percent=8.0,
            potential_profit_per_share=12.0,
            reward_risk_ratio=2.5,
            methodology="R:R-based",
        ),
        reward_risk_ratio=2.5,
    )


def _make_market_result() -> Mock:
    result = Mock()
    result.data = _make_market_data()
    return result


@pytest.fixture
def risk_agent() -> MagicMock:
    agent = MagicMock()
    agent.assess.return_value = _make_assessment()
    return agent


@pytest.fixture
def broker() -> MagicMock:
    b = MagicMock()
    b.get_account_info.return_value = _make_broker_account()
    return b


@pytest.fixture
def market_fetcher() -> MagicMock:
    f = MagicMock()
    f.fetch_daily.return_value = _make_market_result()
    return f


@pytest.fixture
def service(risk_agent: MagicMock, broker: MagicMock, market_fetcher: MagicMock) -> RiskService:
    return RiskService(risk_agent, broker, market_fetcher)


class TestAssessTrade:
    """Tests for RiskService.assess_trade."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_approved_buy(self, service: RiskService, risk_agent: MagicMock) -> None:
        decision = await service.assess_trade("AAPL", Signal.BUY, 0.85)

        assert isinstance(decision, RiskDecision)
        assert decision.approved is True
        assert decision.recommended_shares == 25
        assert decision.stop_loss_price == 142.0
        assert decision.take_profit_price == 162.0
        assert decision.risk_level == "LOW"
        risk_agent.assess.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_rejected_exposure(self, service: RiskService, risk_agent: MagicMock) -> None:
        risk_agent.assess.return_value = _make_assessment(approved=False)

        decision = await service.assess_trade("AAPL", Signal.BUY, 0.85)

        assert decision.approved is False
        assert decision.risk_level == "HIGH"
        assert len(decision.warnings) > 0

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_hold_returns_approved_zero_shares(self, service: RiskService) -> None:
        decision = await service.assess_trade("AAPL", Signal.HOLD, 0.7)

        assert decision.approved is True
        assert decision.recommended_shares == 0
        assert decision.risk_level == "LOW"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_broker_api_failure(self, service: RiskService, broker: MagicMock) -> None:
        broker.get_account_info.side_effect = Exception("Connection refused")

        # Should still work — broker_api_failed=True passed to agent
        decision = await service.assess_trade("AAPL", Signal.BUY, 0.85)

        assert isinstance(decision, RiskDecision)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_market_data_failure(self, risk_agent: MagicMock, broker: MagicMock) -> None:
        market_fetcher = MagicMock()
        market_fetcher.fetch_daily.side_effect = Exception("API error")
        svc = RiskService(risk_agent, broker, market_fetcher)

        decision = await svc.assess_trade("AAPL", Signal.BUY, 0.85)

        assert decision.approved is False
        assert "cannot assess risk" in decision.reasoning

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_current_price_override(self, service: RiskService, risk_agent: MagicMock) -> None:
        await service.assess_trade("AAPL", Signal.BUY, 0.85, current_price=155.0)

        call_kwargs = risk_agent.assess.call_args
        assert call_kwargs.kwargs["current_price"] == 155.0
