"""Tests for trade models."""

import pytest
from pydantic import ValidationError

from src.v1.trades.models import TradeAction, TradeRejection, TradeRejectionReason, TradeRequest, TradeResult


class TestTradeRequest:
    """Tests for TradeRequest validation."""

    @pytest.mark.unit
    def test_valid_request(self) -> None:
        req = TradeRequest(
            symbol="AAPL",
            action=TradeAction.BUY,
            quantity=10,
            confidence=0.8,
            rationale="Strong momentum breakout detected",
        )
        assert req.symbol == "AAPL"
        assert req.action == TradeAction.BUY
        assert req.quantity == 10

    @pytest.mark.unit
    def test_quantity_must_be_positive(self) -> None:
        with pytest.raises(ValidationError, match="greater than 0"):
            TradeRequest(
                symbol="AAPL",
                action=TradeAction.BUY,
                quantity=0,
                confidence=0.8,
                rationale="Strong momentum breakout detected",
            )

    @pytest.mark.unit
    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            TradeRequest(
                symbol="AAPL",
                action=TradeAction.BUY,
                quantity=10,
                confidence=1.5,
                rationale="Strong momentum breakout detected",
            )

    @pytest.mark.unit
    def test_rationale_min_length(self) -> None:
        with pytest.raises(ValidationError, match="String should have at least 10 characters"):
            TradeRequest(
                symbol="AAPL",
                action=TradeAction.BUY,
                quantity=10,
                confidence=0.8,
                rationale="short",
            )

    @pytest.mark.unit
    def test_action_enum(self) -> None:
        assert TradeAction.BUY == "BUY"
        assert TradeAction.SELL == "SELL"

    @pytest.mark.unit
    def test_default_strategy_name(self) -> None:
        req = TradeRequest(
            symbol="AAPL",
            action=TradeAction.BUY,
            quantity=10,
            confidence=0.8,
            rationale="Strong momentum breakout detected",
        )
        assert req.strategy_name == "coordinator"


class TestTradeResult:
    """Tests for TradeResult construction."""

    @pytest.mark.unit
    def test_executed_result(self) -> None:
        result = TradeResult(
            executed=True,
            order_id="order-123",
            symbol="AAPL",
            action=TradeAction.BUY,
            quantity=10,
            status="filled",
            filled_avg_price=150.0,
        )
        assert result.executed is True
        assert result.order_id == "order-123"
        assert result.rejection is None

    @pytest.mark.unit
    def test_rejected_result(self) -> None:
        result = TradeResult(
            executed=False,
            symbol="AAPL",
            action=TradeAction.BUY,
            quantity=10,
            status="rejected",
            rejection=TradeRejection(
                reason=TradeRejectionReason.BELOW_THRESHOLD,
                message="Confidence 50% below threshold 60%",
            ),
        )
        assert result.executed is False
        assert result.rejection is not None
        assert result.rejection.reason == TradeRejectionReason.BELOW_THRESHOLD
