"""Tests for Alpaca broker integration."""

from datetime import UTC, datetime
from unittest.mock import Mock, patch

import pytest
from alpaca.trading.enums import OrderClass
from result import Err

from src.v1.trades.brokers import AlpacaBroker, BrokerAccountInfo, BrokerPosition, OrderStatus


@pytest.fixture
def mock_trading_client():
    """Mock Alpaca TradingClient."""
    with patch("src.v1.trades.brokers.alpaca.TradingClient") as mock_client:
        yield mock_client


@pytest.fixture
def mock_account():
    """Mock account object."""
    account = Mock()
    account.equity = "100000.0"
    account.buying_power = "50000.0"
    account.portfolio_value = "100000.0"
    return account


@pytest.fixture
def mock_positions():
    """Mock positions list."""
    pos1 = Mock()
    pos1.symbol = "AAPL"
    pos1.qty = "10"
    pos1.market_value = "1500.0"
    pos1.avg_entry_price = "150.0"
    pos1.unrealized_pl = "50.0"
    pos1.unrealized_plpc = "0.033"

    pos2 = Mock()
    pos2.symbol = "TSLA"
    pos2.qty = "5"
    pos2.market_value = "1000.0"
    pos2.avg_entry_price = "200.0"
    pos2.unrealized_pl = "-50.0"
    pos2.unrealized_plpc = "-0.05"

    return [pos1, pos2]


@pytest.fixture
def mock_order():
    """Mock order object."""
    order = Mock()
    order.id = "order-123"
    order.symbol = "AAPL"
    order.qty = 10
    order.filled_qty = 10
    order.side = Mock(value="buy")
    order.status = Mock(value="filled")
    order.submitted_at = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
    order.filled_at = datetime(2024, 1, 1, 10, 0, 5, tzinfo=UTC)
    order.filled_avg_price = 150.0
    return order


def test_broker_init_with_env_vars(mock_trading_client, monkeypatch):
    """Test broker initialization with environment variables."""
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")

    broker = AlpacaBroker()

    assert broker.api_key == "test-key"
    assert broker.secret_key == "test-secret"
    assert broker.paper is True
    mock_trading_client.assert_called_once_with(
        api_key="test-key",
        secret_key="test-secret",
        paper=True,
    )


def test_broker_init_with_args(mock_trading_client):
    """Test broker initialization with explicit arguments."""
    broker = AlpacaBroker(
        api_key="key1",
        secret_key="secret1",
        paper=False,
    )

    assert broker.api_key == "key1"
    assert broker.secret_key == "secret1"
    assert broker.paper is False


def test_broker_init_missing_credentials(mock_trading_client, monkeypatch):
    """Test broker initialization fails without credentials."""
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)

    with pytest.raises(ValueError, match="ALPACA_API_KEY and ALPACA_SECRET_KEY"):
        AlpacaBroker()


def test_get_account_info(mock_trading_client, mock_account, mock_positions, monkeypatch):
    """Test fetching account information."""
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")

    client_instance = mock_trading_client.return_value
    client_instance.get_account.return_value = mock_account
    client_instance.get_all_positions.return_value = mock_positions

    broker = AlpacaBroker()
    result = broker.get_account_info()
    account_info = result.ok()

    assert isinstance(account_info, BrokerAccountInfo)
    assert account_info.balance == 100000.0
    assert account_info.available_cash == 50000.0
    assert account_info.portfolio_value == 100000.0
    assert account_info.total_exposure == 2500.0
    assert len(account_info.positions) == 2
    assert "AAPL" in account_info.positions
    assert "TSLA" in account_info.positions


def test_get_account_info_positions(mock_trading_client, mock_account, mock_positions, monkeypatch):
    """Test position details in account info."""
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")

    client_instance = mock_trading_client.return_value
    client_instance.get_account.return_value = mock_account
    client_instance.get_all_positions.return_value = mock_positions

    broker = AlpacaBroker()
    result = broker.get_account_info()
    account_info = result.ok()

    aapl_pos = account_info.positions["AAPL"]
    assert isinstance(aapl_pos, BrokerPosition)
    assert aapl_pos.symbol == "AAPL"
    assert aapl_pos.qty == 10.0
    assert aapl_pos.market_value == 1500.0
    assert aapl_pos.avg_entry_price == 150.0
    assert aapl_pos.unrealized_pnl == 50.0


def test_get_account_info_error(mock_trading_client, monkeypatch):
    """Test error handling in account info fetch."""
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")

    client_instance = mock_trading_client.return_value
    client_instance.get_account.side_effect = Exception("API error")

    broker = AlpacaBroker()

    result = broker.get_account_info()
    assert isinstance(result, Err)
    assert "API error" in str(result.err_value)


def test_submit_order_buy(mock_trading_client, mock_order, monkeypatch):
    """Test submitting buy order."""
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")

    client_instance = mock_trading_client.return_value
    client_instance.submit_order.return_value = mock_order

    broker = AlpacaBroker()
    result = broker.submit_order("AAPL", 10, "buy")
    order_status = result.ok()

    assert isinstance(order_status, OrderStatus)
    assert order_status.order_id == "order-123"
    assert order_status.symbol == "AAPL"
    assert order_status.qty == 10.0
    assert order_status.filled_qty == 10.0
    assert order_status.side == "buy"
    assert order_status.status == "filled"


def test_submit_order_sell(mock_trading_client, mock_order, monkeypatch):
    """Test submitting sell order."""
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")

    mock_order.side = Mock(value="sell")
    client_instance = mock_trading_client.return_value
    client_instance.submit_order.return_value = mock_order

    broker = AlpacaBroker()
    result = broker.submit_order("TSLA", 5, "sell")
    order_status = result.ok()

    assert order_status.side == "sell"


def test_submit_order_with_stop_loss(mock_trading_client, mock_order, monkeypatch):
    """Test submitting order with stop loss."""
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")

    client_instance = mock_trading_client.return_value
    client_instance.submit_order.return_value = mock_order

    broker = AlpacaBroker()
    result = broker.submit_order("AAPL", 10, "buy", stop_loss_price=140.0)
    order_status = result.ok()

    assert order_status.order_id == "order-123"
    client_instance.submit_order.assert_called_once()

    call_args = client_instance.submit_order.call_args
    order_data = call_args.kwargs["order_data"]
    assert order_data.order_class == OrderClass.OTO
    assert order_data.stop_loss.stop_price == 140.0


def test_submit_order_error(mock_trading_client, monkeypatch):
    """Test error handling in order submission."""
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")

    client_instance = mock_trading_client.return_value
    client_instance.submit_order.side_effect = Exception("Order failed")

    broker = AlpacaBroker()

    result = broker.submit_order("AAPL", 10, "buy")
    assert isinstance(result, Err)
    assert "Order failed" in str(result.err_value)


def test_get_order_status(mock_trading_client, mock_order, monkeypatch):
    """Test getting order status."""
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")

    client_instance = mock_trading_client.return_value
    client_instance.get_order_by_id.return_value = mock_order

    broker = AlpacaBroker()
    result = broker.get_order_status("order-123")
    order_status = result.ok()

    assert isinstance(order_status, OrderStatus)
    assert order_status.order_id == "order-123"
    assert order_status.filled_avg_price == 150.0


def test_get_order_status_error(mock_trading_client, monkeypatch):
    """Test error handling in order status fetch."""
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")

    client_instance = mock_trading_client.return_value
    client_instance.get_order_by_id.side_effect = Exception("Order not found")

    broker = AlpacaBroker()

    result = broker.get_order_status("invalid-order")
    assert isinstance(result, Err)
    assert "Order not found" in str(result.err_value)


def test_cancel_order(mock_trading_client, monkeypatch):
    """Test cancelling order."""
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")

    client_instance = mock_trading_client.return_value

    broker = AlpacaBroker()
    broker.cancel_order("order-123")

    client_instance.cancel_order_by_id.assert_called_once_with(order_id="order-123")


def test_cancel_order_error(mock_trading_client, monkeypatch):
    """Test error handling in order cancellation."""
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")

    client_instance = mock_trading_client.return_value
    client_instance.cancel_order_by_id.side_effect = Exception("Cancel failed")

    broker = AlpacaBroker()

    result = broker.cancel_order("order-123")
    assert isinstance(result, Err)
    assert "Cancel failed" in str(result.err_value)


def test_broker_repr(mock_trading_client, monkeypatch):
    """Test broker string representation."""
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")

    broker = AlpacaBroker(paper=True)
    assert repr(broker) == "AlpacaBroker(paper=True)"

    broker = AlpacaBroker(paper=False)
    assert repr(broker) == "AlpacaBroker(paper=False)"


def test_submit_order_invalid_side(mock_trading_client, monkeypatch):
    """Test submitting order with invalid side."""
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")

    broker = AlpacaBroker()

    result = broker.submit_order("AAPL", 10, "invalid")
    assert isinstance(result, Err)
    assert "Invalid order side" in str(result.err_value)


def test_submit_order_invalid_qty(mock_trading_client, monkeypatch):
    """Test submitting order with invalid quantity."""
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")

    broker = AlpacaBroker()

    result = broker.submit_order("AAPL", 0, "buy")
    assert isinstance(result, Err)
    assert "Order quantity must be positive, got 0" in str(result.err_value)

    result = broker.submit_order("AAPL", -5, "buy")
    assert isinstance(result, Err)
    assert "Order quantity must be positive, got -5" in str(result.err_value)


def test_submit_stop_order(mock_trading_client, mock_order, monkeypatch):
    """Test submitting stop order."""
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")

    mock_order.side = Mock(value="sell")
    client_instance = mock_trading_client.return_value
    client_instance.submit_order.return_value = mock_order

    broker = AlpacaBroker()
    result = broker.submit_stop_order("AAPL", 10, 140.0)
    order_status = result.ok()

    assert isinstance(order_status, OrderStatus)
    assert order_status.order_id == "order-123"
    assert order_status.symbol == "AAPL"
    assert order_status.qty == 10.0
    assert order_status.side == "sell"

    client_instance.submit_order.assert_called_once()
    call_args = client_instance.submit_order.call_args
    order_data = call_args.kwargs["order_data"]
    assert order_data.symbol == "AAPL"
    assert order_data.qty == 10
    assert order_data.stop_price == 140.0


def test_submit_stop_order_invalid_qty(mock_trading_client, monkeypatch):
    """Test submitting stop order with invalid quantity."""
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")

    broker = AlpacaBroker()

    result = broker.submit_stop_order("AAPL", 0, 140.0)
    assert isinstance(result, Err)
    assert "Order quantity must be positive, got 0" in str(result.err_value)

    result = broker.submit_stop_order("AAPL", -5, 140.0)
    assert isinstance(result, Err)
    assert "Order quantity must be positive, got -5" in str(result.err_value)


def test_submit_stop_order_invalid_price(mock_trading_client, monkeypatch):
    """Test submitting stop order with invalid stop price."""
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")

    broker = AlpacaBroker()

    result = broker.submit_stop_order("AAPL", 10, 0.0)
    assert isinstance(result, Err)
    assert "Stop price must be positive, got 0" in str(result.err_value)

    result = broker.submit_stop_order("AAPL", 10, -10.5)
    assert isinstance(result, Err)
    assert "Stop price must be positive, got -10.5" in str(result.err_value)


def test_submit_stop_order_error(mock_trading_client, monkeypatch):
    """Test error handling in stop order submission."""
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")

    client_instance = mock_trading_client.return_value
    client_instance.submit_order.side_effect = Exception("Stop order failed")

    broker = AlpacaBroker()

    result = broker.submit_stop_order("AAPL", 10, 140.0)
    assert isinstance(result, Err)
    assert "Stop order failed" in str(result.err_value)


def test_submit_stop_order_rounds_price_above_dollar(mock_trading_client, mock_order, monkeypatch):
    """Test stop order rounds price to 2 decimals for stocks >= $1.00."""
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")

    mock_order.side = Mock(value="sell")
    client_instance = mock_trading_client.return_value
    client_instance.submit_order.return_value = mock_order

    broker = AlpacaBroker()
    broker.submit_stop_order("AAPL", 10, 266.3813952636719)

    call_args = client_instance.submit_order.call_args
    order_data = call_args.kwargs["order_data"]
    assert order_data.stop_price == 266.38  # Rounded to 2 decimals


def test_submit_stop_order_rounds_price_below_dollar(mock_trading_client, mock_order, monkeypatch):
    """Test stop order rounds price to 4 decimals for stocks < $1.00."""
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")

    mock_order.side = Mock(value="sell")
    client_instance = mock_trading_client.return_value
    client_instance.submit_order.return_value = mock_order

    broker = AlpacaBroker()
    broker.submit_stop_order("PENNY", 100, 0.8765432109)

    call_args = client_instance.submit_order.call_args
    order_data = call_args.kwargs["order_data"]
    assert order_data.stop_price == 0.8765  # Rounded to 4 decimals


def test_submit_order_with_stop_loss_rounds_above_dollar(mock_trading_client, mock_order, monkeypatch):
    """Test order with stop_loss_price rounds to 2 decimals for >= $1.00."""
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")

    client_instance = mock_trading_client.return_value
    client_instance.submit_order.return_value = mock_order

    broker = AlpacaBroker()
    broker.submit_order("AAPL", 10, "buy", stop_loss_price=145.6789123)

    call_args = client_instance.submit_order.call_args
    order_data = call_args.kwargs["order_data"]
    assert order_data.stop_loss.stop_price == 145.68  # Rounded to 2 decimals


def test_submit_order_with_stop_loss_rounds_below_dollar(mock_trading_client, mock_order, monkeypatch):
    """Test order with stop_loss_price rounds to 4 decimals for < $1.00."""
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")

    client_instance = mock_trading_client.return_value
    client_instance.submit_order.return_value = mock_order

    broker = AlpacaBroker()
    broker.submit_order("PENNY", 100, "buy", stop_loss_price=0.5432198765)

    call_args = client_instance.submit_order.call_args
    order_data = call_args.kwargs["order_data"]
    assert order_data.stop_loss.stop_price == 0.5432  # Rounded to 4 decimals


def test_submit_order_validates_stop_loss_price(mock_trading_client, monkeypatch):
    """Test order with invalid stop_loss_price raises error."""

    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")

    broker = AlpacaBroker()

    result = broker.submit_order("AAPL", 10, "buy", stop_loss_price=0.0)
    assert isinstance(result, Err)
    assert "Stop loss price must be positive" in str(result.err_value)

    result = broker.submit_order("AAPL", 10, "buy", stop_loss_price=-10.5)
    assert isinstance(result, Err)
    assert "Stop loss price must be positive" in str(result.err_value)
