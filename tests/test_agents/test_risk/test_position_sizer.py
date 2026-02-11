"""Tests for position sizing component."""

import pytest

from src.agents.risk import AccountInfo, PositionSizeCalculation, StopLossCalculation
from src.agents.risk.position_sizer import PositionSizer


@pytest.fixture
def position_sizer():
    """Position sizer instance with default config."""
    return PositionSizer(
        max_position_risk=2.0,
        max_exposure=80.0,
        max_single_position=20.0,
    )


@pytest.fixture
def account_info():
    """Sample account info."""
    return AccountInfo(
        balance=100000.0,
        available_cash=50000.0,
        positions={"SPY": 100.0},
        total_exposure=20000.0,
    )


def test_calculate_position_size(position_sizer, account_info):
    """Test position size calculation."""
    stop_loss = StopLossCalculation(
        stop_loss_price=147.0,
        stop_loss_percent=2.0,
        risk_per_share=3.0,
        max_loss_amount=0.0,
        methodology="Fixed 2%",
    )

    result = position_sizer.calculate(150.0, stop_loss, account_info)

    assert isinstance(result, PositionSizeCalculation)
    assert result.recommended_shares > 0
    assert result.risk_percent <= position_sizer.max_position_risk
    assert result.position_value <= account_info.available_cash
    assert result.risk_amount > 0


def test_calculate_position_size_cash_constraint(position_sizer, account_info):
    """Test position sizing with cash constraint."""
    account_info.available_cash = 5000.0

    stop_loss = StopLossCalculation(
        stop_loss_price=147.0,
        stop_loss_percent=2.0,
        risk_per_share=3.0,
        max_loss_amount=0.0,
        methodology="Fixed 2%",
    )

    result = position_sizer.calculate(150.0, stop_loss, account_info)

    assert result.position_value <= 5000.0


def test_calculate_position_size_max_single_position(position_sizer, account_info):
    """Test position sizing with max single position constraint."""
    stop_loss = StopLossCalculation(
        stop_loss_price=149.0,
        stop_loss_percent=0.67,
        risk_per_share=1.0,
        max_loss_amount=0.0,
        methodology="Fixed 0.67%",
    )

    result = position_sizer.calculate(150.0, stop_loss, account_info)

    max_allowed = account_info.balance * (position_sizer.max_single_position / 100)
    assert result.position_value <= max_allowed


def test_weight_based_position_sizing():
    """Test weight-based position sizing."""
    sizer = PositionSizer(max_position_risk=2.0, max_exposure=80.0, max_single_position=20.0)

    account_info = AccountInfo(balance=100000.0, available_cash=50000.0, positions={}, total_exposure=0.0)

    stop_loss = StopLossCalculation(
        stop_loss_price=95.0,
        stop_loss_percent=5.0,
        risk_per_share=5.0,
        max_loss_amount=0.0,
        methodology="ATR",
    )

    # Test 10% target weight
    result = sizer.calculate(
        current_price=100.0,
        stop_loss=stop_loss,
        account_info=account_info,
        target_portfolio_weight=0.10,
    )

    expected_shares = 100  # 10% of 100k = 10k / 100 = 100 shares
    assert result.recommended_shares == expected_shares
    assert result.position_value == 10000.0
    assert result.risk_amount == 500.0  # 100 shares * 5.0 risk_per_share
    assert result.risk_percent == 0.5
    assert "Portfolio-weighted position" in result.reasoning
    assert "10.0% target" in result.reasoning


def test_weight_based_constrained_by_cash():
    """Test weight-based sizing constrained by cash."""
    sizer = PositionSizer(max_position_risk=2.0, max_exposure=80.0, max_single_position=20.0)

    account_info = AccountInfo(balance=100000.0, available_cash=5000.0, positions={}, total_exposure=0.0)

    stop_loss = StopLossCalculation(
        stop_loss_price=95.0,
        stop_loss_percent=5.0,
        risk_per_share=5.0,
        max_loss_amount=0.0,
        methodology="ATR",
    )

    # Target 20% but only have 5% cash available
    result = sizer.calculate(
        current_price=100.0,
        stop_loss=stop_loss,
        account_info=account_info,
        target_portfolio_weight=0.20,
    )

    expected_shares = 50  # Limited by 5k cash / 100 = 50 shares
    assert result.recommended_shares == expected_shares
    assert result.position_value == 5000.0


def test_weight_based_constrained_by_max_position():
    """Test weight-based sizing constrained by max position."""
    sizer = PositionSizer(max_position_risk=2.0, max_exposure=80.0, max_single_position=10.0)

    account_info = AccountInfo(balance=100000.0, available_cash=50000.0, positions={}, total_exposure=0.0)

    stop_loss = StopLossCalculation(
        stop_loss_price=95.0,
        stop_loss_percent=5.0,
        risk_per_share=5.0,
        max_loss_amount=0.0,
        methodology="ATR",
    )

    # Target 20% but max position is 10%
    result = sizer.calculate(
        current_price=100.0,
        stop_loss=stop_loss,
        account_info=account_info,
        target_portfolio_weight=0.20,
    )

    expected_shares = 100  # Limited by 10% max = 10k / 100 = 100 shares
    assert result.recommended_shares == expected_shares
    assert result.position_value == 10000.0


def test_position_sizer_repr():
    """Test position sizer string representation."""
    sizer = PositionSizer(max_position_risk=2.0, max_exposure=80.0, max_single_position=20.0)
    repr_str = repr(sizer)

    assert "PositionSizer" in repr_str
    assert "max_risk=2.0%" in repr_str
    assert "max_single=20.0%" in repr_str


def test_invalid_price_raises_error(position_sizer, account_info):
    """Test that invalid price raises ValueError."""
    stop_loss = StopLossCalculation(
        stop_loss_price=147.0,
        stop_loss_percent=2.0,
        risk_per_share=3.0,
        max_loss_amount=0.0,
        methodology="Fixed 2%",
    )

    with pytest.raises(ValueError, match="Invalid current_price"):
        position_sizer.calculate(-10.0, stop_loss, account_info)


def test_zero_risk_per_share_returns_zero_position(position_sizer, account_info):
    """Test that zero risk per share returns zero-sized position."""
    stop_loss = StopLossCalculation(
        stop_loss_price=150.0,  # Same as entry
        stop_loss_percent=0.0,
        risk_per_share=0.0,
        max_loss_amount=0.0,
        methodology="No stop",
    )

    result = position_sizer.calculate(150.0, stop_loss, account_info)

    assert result.recommended_shares == 0
    assert result.position_value == 0.0
    assert result.risk_amount == 0.0
    assert "zero or too small" in result.reasoning


def test_weight_based_zero_shares(position_sizer, account_info):
    """Test weight-based sizing with insufficient capital."""
    account_info.available_cash = 10.0

    stop_loss = StopLossCalculation(
        stop_loss_price=95.0,
        stop_loss_percent=5.0,
        risk_per_share=5.0,
        max_loss_amount=0.0,
        methodology="ATR",
    )

    # Target weight would result in 0 shares
    result = position_sizer.calculate(
        current_price=1000.0,
        stop_loss=stop_loss,
        account_info=account_info,
        target_portfolio_weight=0.10,
    )

    assert result.recommended_shares == 0
    assert result.risk_percent == 100.0
    assert "Insufficient capital" in result.reasoning
