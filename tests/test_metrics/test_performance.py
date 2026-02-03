"""Unit tests for performance calculation functions."""

from datetime import UTC, datetime

import pytest

from src.metrics.performance import (
    calculate_max_drawdown,
    calculate_returns_from_trades,
    calculate_risk_adjusted_returns,
    calculate_sharpe_ratio,
    calculate_win_rate,
)
from src.metrics.tracker import TradeRecord
from src.strategies.momentum import Signal


@pytest.fixture
def winning_trade():
    """Single winning trade."""
    return TradeRecord(
        timestamp=datetime.now(UTC),
        symbol="AAPL",
        action=Signal.BUY,
        entry_price=100.0,
        exit_price=110.0,
        shares=100,
        stop_loss_price=95.0,
        confidence=0.8,
        risk_level="LOW",
        status="CLOSED",
        pnl=1000.0,
        pnl_percent=10.0,
    )


@pytest.fixture
def losing_trade():
    """Single losing trade."""
    return TradeRecord(
        timestamp=datetime.now(UTC),
        symbol="TSLA",
        action=Signal.SELL,
        entry_price=200.0,
        exit_price=210.0,
        shares=50,
        stop_loss_price=205.0,
        confidence=0.6,
        risk_level="MEDIUM",
        status="CLOSED",
        pnl=-500.0,
        pnl_percent=-5.0,
    )


@pytest.fixture
def open_trade():
    """Open trade (not closed)."""
    return TradeRecord(
        timestamp=datetime.now(UTC),
        symbol="NVDA",
        action=Signal.BUY,
        entry_price=500.0,
        exit_price=None,
        shares=20,
        stop_loss_price=475.0,
        confidence=0.75,
        risk_level="LOW",
        status="OPEN",
        pnl=None,
        pnl_percent=None,
    )


@pytest.fixture
def rejected_trade():
    """Rejected trade."""
    return TradeRecord(
        timestamp=datetime.now(UTC),
        symbol="AMZN",
        action=Signal.BUY,
        entry_price=150.0,
        exit_price=None,
        shares=0,
        stop_loss_price=142.5,
        confidence=0.5,
        risk_level="HIGH",
        status="REJECTED",
        pnl=None,
        pnl_percent=None,
    )


@pytest.fixture
def mixed_trades(winning_trade, losing_trade):
    """Mix of winning and losing trades for testing."""
    trades = [winning_trade, losing_trade]

    trade3 = TradeRecord(
        timestamp=datetime.now(UTC),
        symbol="MSFT",
        action=Signal.BUY,
        entry_price=300.0,
        exit_price=315.0,
        shares=30,
        stop_loss_price=285.0,
        confidence=0.85,
        risk_level="LOW",
        status="CLOSED",
        pnl=450.0,
        pnl_percent=5.0,
    )

    trade4 = TradeRecord(
        timestamp=datetime.now(UTC),
        symbol="GOOGL",
        action=Signal.SELL,
        entry_price=120.0,
        exit_price=125.0,
        shares=80,
        stop_loss_price=122.0,
        confidence=0.7,
        risk_level="MEDIUM",
        status="CLOSED",
        pnl=-400.0,
        pnl_percent=-4.16,
    )

    trades.extend([trade3, trade4])
    return trades


def test_calculate_returns_from_trades_with_closed_trades(mixed_trades):
    """Test returns calculation with closed trades."""
    returns = calculate_returns_from_trades(mixed_trades)

    assert len(returns) == 4
    assert returns[0] == pytest.approx(0.10, abs=0.001)
    assert returns[1] == pytest.approx(-0.05, abs=0.001)
    assert returns[2] == pytest.approx(0.05, abs=0.001)
    assert returns[3] == pytest.approx(-0.0416, abs=0.001)


def test_calculate_returns_from_trades_with_open_trades(open_trade):
    """Test returns calculation excludes open trades."""
    returns = calculate_returns_from_trades([open_trade])

    assert len(returns) == 0


def test_calculate_returns_from_trades_empty_list():
    """Test returns calculation with empty list."""
    returns = calculate_returns_from_trades([])

    assert len(returns) == 0


def test_calculate_win_rate_with_mixed_trades(mixed_trades):
    """Test win rate calculation with mixed trades."""
    win_rate = calculate_win_rate(mixed_trades)

    assert win_rate == 50.0


def test_calculate_win_rate_all_winners(winning_trade):
    """Test win rate with all winning trades."""
    trades = [winning_trade]
    win_rate = calculate_win_rate(trades)

    assert win_rate == 100.0


def test_calculate_win_rate_all_losers(losing_trade):
    """Test win rate with all losing trades."""
    trades = [losing_trade]
    win_rate = calculate_win_rate(trades)

    assert win_rate == 0.0


def test_calculate_win_rate_no_closed_trades(open_trade):
    """Test win rate with no closed trades."""
    win_rate = calculate_win_rate([open_trade])

    assert win_rate == 0.0


def test_calculate_sharpe_ratio_positive_returns():
    """Test Sharpe ratio with positive returns."""
    returns = [0.05, 0.03, 0.07, 0.04, 0.06]
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)

    assert sharpe > 0
    assert isinstance(sharpe, float)


def test_calculate_sharpe_ratio_negative_returns():
    """Test Sharpe ratio with negative returns."""
    returns = [-0.05, -0.03, -0.07, -0.04, -0.06]
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)

    assert sharpe < 0


def test_calculate_sharpe_ratio_mixed_returns():
    """Test Sharpe ratio with mixed returns."""
    returns = [0.10, -0.05, 0.05, -0.0416]
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)

    assert isinstance(sharpe, float)


def test_calculate_sharpe_ratio_zero_std_dev():
    """Test Sharpe ratio with zero standard deviation."""
    returns = [0.05, 0.05, 0.05]
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)

    assert sharpe == 0.0


def test_calculate_sharpe_ratio_empty_returns():
    """Test Sharpe ratio with empty returns."""
    sharpe = calculate_sharpe_ratio([], risk_free_rate=0.02)

    assert sharpe == 0.0


def test_calculate_max_drawdown_with_winning_trades(winning_trade):
    """Test max drawdown with only winning trades."""
    trades = [winning_trade]
    max_dd, max_dd_pct = calculate_max_drawdown(trades)

    assert max_dd == 0.0
    assert max_dd_pct == 0.0


def test_calculate_max_drawdown_with_losing_trades(losing_trade):
    """Test max drawdown with losing trades."""
    trades = [losing_trade]
    max_dd, max_dd_pct = calculate_max_drawdown(trades)

    assert max_dd > 0
    assert max_dd_pct > 0
    assert max_dd == 500.0
    assert max_dd_pct == pytest.approx(0.5, abs=0.01)


def test_calculate_max_drawdown_with_mixed_trades(mixed_trades):
    """Test max drawdown with mixed trades."""
    max_dd, max_dd_pct = calculate_max_drawdown(mixed_trades)

    assert max_dd >= 0
    assert max_dd_pct >= 0
    assert isinstance(max_dd, float)
    assert isinstance(max_dd_pct, float)


def test_calculate_max_drawdown_empty_trades():
    """Test max drawdown with empty list."""
    max_dd, max_dd_pct = calculate_max_drawdown([])

    assert max_dd == 0.0
    assert max_dd_pct == 0.0


def test_calculate_max_drawdown_with_open_trades(open_trade):
    """Test max drawdown excludes open trades."""
    max_dd, max_dd_pct = calculate_max_drawdown([open_trade])

    assert max_dd == 0.0
    assert max_dd_pct == 0.0


def test_calculate_risk_adjusted_returns_positive():
    """Test risk-adjusted returns with positive returns."""
    returns = [0.05, 0.03, 0.07, 0.04, 0.06]
    risk_values = [95.0, 97.0, 93.0, 96.0, 94.0]
    rar = calculate_risk_adjusted_returns(returns, risk_values)

    assert rar > 0
    assert isinstance(rar, float)


def test_calculate_risk_adjusted_returns_negative():
    """Test risk-adjusted returns with negative returns."""
    returns = [-0.05, -0.03, -0.07, -0.04, -0.06]
    risk_values = [105.0, 103.0, 107.0, 104.0, 106.0]
    rar = calculate_risk_adjusted_returns(returns, risk_values)

    assert rar < 0


def test_calculate_risk_adjusted_returns_mixed():
    """Test risk-adjusted returns with mixed returns."""
    returns = [0.10, -0.05, 0.05, -0.0416]
    risk_values = [95.0, 205.0, 285.0, 122.0]
    rar = calculate_risk_adjusted_returns(returns, risk_values)

    assert isinstance(rar, float)


def test_calculate_risk_adjusted_returns_empty():
    """Test risk-adjusted returns with empty inputs."""
    rar = calculate_risk_adjusted_returns([], [])

    assert rar == 0.0


def test_calculate_risk_adjusted_returns_zero_downside():
    """Test risk-adjusted returns with no downside uses all returns for downside deviation."""
    returns = [0.05, 0.03, 0.07]
    risk_values = [95.0, 97.0, 93.0]
    rar = calculate_risk_adjusted_returns(returns, risk_values)

    assert rar > 0
    assert isinstance(rar, float)


def test_trade_record_is_open(open_trade):
    """Test TradeRecord.is_open() method."""
    assert open_trade.is_open() is True
    assert open_trade.is_closed() is False
    assert open_trade.is_rejected() is False


def test_trade_record_is_closed(winning_trade):
    """Test TradeRecord.is_closed() method."""
    assert winning_trade.is_open() is False
    assert winning_trade.is_closed() is True
    assert winning_trade.is_rejected() is False


def test_trade_record_is_rejected(rejected_trade):
    """Test TradeRecord.is_rejected() method."""
    assert rejected_trade.is_open() is False
    assert rejected_trade.is_closed() is False
    assert rejected_trade.is_rejected() is True


def test_trade_record_close_buy_trade():
    """Test closing a BUY trade."""
    trade = TradeRecord(
        timestamp=datetime.now(UTC),
        symbol="AAPL",
        action=Signal.BUY,
        entry_price=100.0,
        exit_price=None,
        shares=100,
        stop_loss_price=95.0,
        confidence=0.8,
        risk_level="LOW",
        status="OPEN",
        pnl=None,
        pnl_percent=None,
    )

    trade.close_trade(110.0)

    assert trade.status == "CLOSED"
    assert trade.exit_price == 110.0
    assert trade.pnl == 1000.0
    assert trade.pnl_percent == 10.0


def test_trade_record_close_sell_trade():
    """Test closing a SELL trade."""
    trade = TradeRecord(
        timestamp=datetime.now(UTC),
        symbol="TSLA",
        action=Signal.SELL,
        entry_price=200.0,
        exit_price=None,
        shares=50,
        stop_loss_price=205.0,
        confidence=0.6,
        risk_level="MEDIUM",
        status="OPEN",
        pnl=None,
        pnl_percent=None,
    )

    trade.close_trade(190.0)

    assert trade.status == "CLOSED"
    assert trade.exit_price == 190.0
    assert trade.pnl == 500.0
    assert trade.pnl_percent == 5.0


def test_trade_record_close_hold_trade():
    """Test closing a HOLD trade."""
    trade = TradeRecord(
        timestamp=datetime.now(UTC),
        symbol="NVDA",
        action=Signal.HOLD,
        entry_price=500.0,
        exit_price=None,
        shares=0,
        stop_loss_price=475.0,
        confidence=0.5,
        risk_level="MEDIUM",
        status="OPEN",
        pnl=None,
        pnl_percent=None,
    )

    trade.close_trade(510.0)

    assert trade.status == "CLOSED"
    assert trade.exit_price == 510.0
    assert trade.pnl == 0.0
    assert trade.pnl_percent == 0.0
