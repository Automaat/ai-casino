"""Tests for risk metrics calculation."""

import pytest

from src.metrics.risk import (
    DrawdownMetrics,
    RiskMetrics,
    RiskMetricsCalculator,
    VaRMetrics,
    calculate_downside_deviation,
    calculate_volatility,
)


@pytest.fixture
def sample_returns():
    """Mixed returns for standard risk testing."""
    return [0.02, -0.01, 0.03, -0.02, 0.01, -0.03, 0.04, -0.01, 0.02, -0.02]


@pytest.fixture
def volatile_returns():
    """High volatility returns for edge case testing."""
    return [0.10, -0.08, 0.12, -0.09, 0.11, -0.10, 0.13, -0.07, 0.09, -0.11]


@pytest.fixture
def risk_calculator():
    """Risk metrics calculator instance."""
    return RiskMetricsCalculator(risk_free_rate=0.02)


def test_volatility_calculation(sample_returns):
    """Test volatility calculation."""
    volatility = calculate_volatility(sample_returns)

    assert isinstance(volatility, float)
    assert volatility > 0.0
    assert 0.0 < volatility < 2.0


def test_volatility_high_vs_low(sample_returns, volatile_returns):
    """Test that volatile returns produce higher volatility."""
    normal_vol = calculate_volatility(sample_returns)
    high_vol = calculate_volatility(volatile_returns)

    assert high_vol > normal_vol


def test_downside_deviation_calculation(sample_returns):
    """Test downside deviation calculation."""
    downside_dev = calculate_downside_deviation(sample_returns)

    assert isinstance(downside_dev, float)
    assert downside_dev > 0.0


def test_downside_deviation_negative_vs_mixed():
    """Test downside deviation with negative-only vs mixed returns."""
    negative_only = [-0.01, -0.02, -0.03, -0.01, -0.02]
    mixed = [0.02, -0.01, 0.03, -0.02, 0.01, -0.03]

    neg_dd = calculate_downside_deviation(negative_only)
    mixed_dd = calculate_downside_deviation(mixed)

    assert neg_dd > 0.0
    assert mixed_dd > 0.0


def test_var_calculation(risk_calculator, sample_returns):
    """Test VaR calculation at multiple confidence levels."""
    var_metrics = risk_calculator.calculate_var(sample_returns)

    assert isinstance(var_metrics, VaRMetrics)
    assert var_metrics.var_95 > 0
    assert var_metrics.var_99 > 0
    assert var_metrics.cvar_95 >= var_metrics.var_95
    assert var_metrics.cvar_99 >= var_metrics.var_99
    assert var_metrics.var_99 >= var_metrics.var_95


def test_cvar_standalone(risk_calculator, sample_returns):
    """Test CVaR calculation at different confidence levels."""
    cvar_95 = risk_calculator.calculate_cvar(sample_returns, confidence=0.95)
    cvar_99 = risk_calculator.calculate_cvar(sample_returns, confidence=0.99)

    assert isinstance(cvar_95, float)
    assert isinstance(cvar_99, float)
    assert cvar_95 > 0
    assert cvar_99 > 0
    assert cvar_99 >= cvar_95


def test_cvar_invalid_confidence(risk_calculator, sample_returns):
    """Test CVaR with invalid confidence level."""
    with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
        risk_calculator.calculate_cvar(sample_returns, confidence=1.5)

    with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
        risk_calculator.calculate_cvar(sample_returns, confidence=-0.1)


def test_drawdown_metrics(risk_calculator, sample_returns):
    """Test drawdown metrics calculation."""
    drawdown = risk_calculator.calculate_cdar(sample_returns)

    assert isinstance(drawdown, DrawdownMetrics)
    assert drawdown.max_drawdown >= 0
    assert drawdown.cdar_95 >= 0
    assert drawdown.avg_drawdown >= 0
    assert drawdown.max_drawdown_duration_days >= 0
    assert drawdown.max_drawdown >= drawdown.avg_drawdown


def test_drawdown_duration_accuracy():
    """Test drawdown duration calculation."""
    returns = [0.05, 0.03, -0.10, -0.05, -0.02, 0.03, 0.04, 0.08]

    calc = RiskMetricsCalculator()
    drawdown = calc.calculate_cdar(returns)

    assert drawdown.max_drawdown_duration_days > 0


def test_comprehensive_metrics(risk_calculator, sample_returns):
    """Test comprehensive risk metrics calculation."""
    metrics = risk_calculator.calculate_all(sample_returns)

    assert isinstance(metrics, RiskMetrics)
    assert isinstance(metrics.var_metrics, VaRMetrics)
    assert isinstance(metrics.drawdown_metrics, DrawdownMetrics)
    assert metrics.volatility_annual > 0.0
    assert metrics.downside_deviation > 0.0

    assert metrics.var_metrics.var_95 > 0
    assert metrics.var_metrics.var_99 > 0
    assert metrics.drawdown_metrics.max_drawdown >= 0


def test_empty_returns(risk_calculator):
    """Test risk metrics with empty returns."""
    empty = []

    var_metrics = risk_calculator.calculate_var(empty)
    assert var_metrics.var_95 == 0.0
    assert var_metrics.var_99 == 0.0
    assert var_metrics.cvar_95 == 0.0
    assert var_metrics.cvar_99 == 0.0

    cvar = risk_calculator.calculate_cvar(empty)
    assert cvar == 0.0

    drawdown = risk_calculator.calculate_cdar(empty)
    assert drawdown.max_drawdown == 0.0
    assert drawdown.cdar_95 == 0.0
    assert drawdown.avg_drawdown == 0.0
    assert drawdown.max_drawdown_duration_days == 0

    volatility = calculate_volatility(empty)
    assert volatility == 0.0

    downside_dev = calculate_downside_deviation(empty)
    assert downside_dev == 0.0


def test_single_return(risk_calculator):
    """Test risk metrics with single return."""
    single = [0.02]

    var_metrics = risk_calculator.calculate_var(single)
    assert var_metrics.var_95 == 0.0

    volatility = calculate_volatility(single)
    assert volatility == 0.0

    downside_dev = calculate_downside_deviation(single)
    assert downside_dev == 0.0


def test_integration_with_performance_module():
    """Test integration with performance module."""
    from datetime import datetime

    from src.metrics.performance import calculate_returns_from_trades
    from src.metrics.tracker import TradeRecord
    from src.strategies.signal import Signal

    trades = [
        TradeRecord(
            symbol="AAPL",
            action=Signal.BUY,
            entry_price=150.0,
            shares=10,
            timestamp=datetime(2024, 1, 1, 10, 0, 0),
            exit_price=155.0,
            stop_loss_price=145.0,
            confidence=0.8,
            risk_level="LOW",
            status="CLOSED",
            pnl=50.0,
            pnl_percent=3.33,
        ),
        TradeRecord(
            symbol="AAPL",
            action=Signal.SELL,
            entry_price=155.0,
            shares=10,
            timestamp=datetime(2024, 1, 2, 10, 0, 0),
            exit_price=153.0,
            stop_loss_price=160.0,
            confidence=0.75,
            risk_level="MEDIUM",
            status="CLOSED",
            pnl=-20.0,
            pnl_percent=-1.29,
        ),
    ]

    returns = calculate_returns_from_trades(trades)

    calc = RiskMetricsCalculator()
    metrics = calc.calculate_all(returns)

    assert isinstance(metrics, RiskMetrics)
    assert metrics.volatility_annual > 0.0


def test_calculator_repr():
    """Test calculator string representation."""
    calc = RiskMetricsCalculator(risk_free_rate=0.03)
    repr_str = repr(calc)

    assert "RiskMetricsCalculator" in repr_str
    assert "0.03" in repr_str


def test_zero_volatility_returns():
    """Test with returns that have zero volatility."""
    constant = [0.02, 0.02, 0.02, 0.02, 0.02]

    volatility = calculate_volatility(constant)
    assert volatility == 0.0


def test_all_positive_returns_downside_deviation():
    """Test downside deviation with all positive returns."""
    positive = [0.01, 0.02, 0.03, 0.04, 0.05]

    downside_dev = calculate_downside_deviation(positive)
    assert downside_dev > 0.0


def test_extreme_drawdown():
    """Test drawdown with extreme losses."""
    extreme_loss = [0.05, 0.03, -0.30, -0.20, 0.01, 0.02]

    calc = RiskMetricsCalculator()
    drawdown = calc.calculate_cdar(extreme_loss)

    assert drawdown.max_drawdown > 0.4
    assert drawdown.max_drawdown_duration_days >= 2
