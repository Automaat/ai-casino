"""Tests for take-profit calculator."""

import math

import pytest

from src.agents.risk.models import StopLossCalculation, TakeProfitCalculation
from src.agents.risk.take_profit_calculator import TakeProfitCalculator
from src.strategies.signal import Signal


@pytest.fixture
def calculator() -> TakeProfitCalculator:
    """Default take-profit calculator."""
    return TakeProfitCalculator(min_reward_risk_ratio=2.0, default_take_profit_percent=4.0)


@pytest.fixture
def stop_loss_buy() -> StopLossCalculation:
    """Stop-loss for BUY at $150, stop at $145 (risk=$5/share)."""
    return StopLossCalculation(
        stop_loss_price=145.0,
        stop_loss_percent=3.33,
        risk_per_share=5.0,
        max_loss_amount=500.0,
        methodology="ATR-based (2.0x ATR)",
    )


@pytest.fixture
def stop_loss_sell() -> StopLossCalculation:
    """Stop-loss for SELL at $150, stop at $155 (risk=$5/share)."""
    return StopLossCalculation(
        stop_loss_price=155.0,
        stop_loss_percent=3.33,
        risk_per_share=5.0,
        max_loss_amount=500.0,
        methodology="ATR-based (2.0x ATR)",
    )


class TestTakeProfitCalculator:
    def test_buy_atr_based(
        self, calculator: TakeProfitCalculator, stop_loss_buy: StopLossCalculation
    ) -> None:
        """BUY: take-profit = entry + (risk_per_share * ratio)."""
        result = calculator.calculate(150.0, stop_loss_buy, Signal.BUY)

        assert isinstance(result, TakeProfitCalculation)
        assert result.take_profit_price == 160.0  # 150 + (5 * 2.0)
        assert result.potential_profit_per_share == 10.0
        assert result.reward_risk_ratio == 2.0
        assert "R:R-based" in result.methodology

    def test_sell_direction(
        self, calculator: TakeProfitCalculator, stop_loss_sell: StopLossCalculation
    ) -> None:
        """SELL: take-profit = entry - (risk_per_share * ratio)."""
        result = calculator.calculate(150.0, stop_loss_sell, Signal.SELL)

        assert result.take_profit_price == 140.0  # 150 - (5 * 2.0)
        assert result.potential_profit_per_share == 10.0
        assert result.reward_risk_ratio == 2.0

    def test_zero_risk_fallback(self, calculator: TakeProfitCalculator) -> None:
        """Zero risk_per_share falls back to fixed percentage."""
        zero_risk_sl = StopLossCalculation(
            stop_loss_price=150.0,
            stop_loss_percent=0.0,
            risk_per_share=0.0,
            max_loss_amount=0.0,
            methodology="N/A",
        )

        result = calculator.calculate(150.0, zero_risk_sl, Signal.BUY)

        assert result.take_profit_price == 156.0  # 150 * 1.04
        assert math.isinf(result.reward_risk_ratio)
        assert "Fixed" in result.methodology

    def test_custom_ratio(self, stop_loss_buy: StopLossCalculation) -> None:
        """Custom ratio of 3.0."""
        calc = TakeProfitCalculator(min_reward_risk_ratio=3.0)

        result = calc.calculate(150.0, stop_loss_buy, Signal.BUY)

        assert result.take_profit_price == 165.0  # 150 + (5 * 3.0)
        assert result.reward_risk_ratio == 3.0

    def test_rounding(self) -> None:
        """Prices rounded to 2 decimals."""
        sl = StopLossCalculation(
            stop_loss_price=97.33,
            stop_loss_percent=2.67,
            risk_per_share=2.6700000000000017,
            max_loss_amount=267.0,
            methodology="ATR-based",
        )
        calc = TakeProfitCalculator(min_reward_risk_ratio=2.0)

        result = calc.calculate(100.0, sl, Signal.BUY)

        assert result.take_profit_price == round(result.take_profit_price, 2)
        assert result.potential_profit_per_share == round(result.potential_profit_per_share, 2)

    def test_repr(self, calculator: TakeProfitCalculator) -> None:
        """String representation."""
        repr_str = repr(calculator)

        assert "TakeProfitCalculator" in repr_str
        assert "min_rr=2.0" in repr_str
