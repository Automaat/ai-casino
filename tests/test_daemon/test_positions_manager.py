"""Tests for PositionManager._update_stop_loss pending cancel handling."""

from datetime import UTC, datetime
from unittest.mock import Mock, patch

import pytest
from result import Err, Ok

from src.daemon.config import PositionManagementConfig
from src.daemon.positions.manager import PositionManager
from src.daemon.positions.models import PositionRecord
from src.v1.trades.brokers.models import BrokerAccountInfo, BrokerPosition, OrderStatus


@pytest.fixture
def config() -> PositionManagementConfig:
    """Minimal PositionManagementConfig for testing."""
    return PositionManagementConfig(
        enabled=True,
        trailing_stop_enabled=True,
        min_stop_gap_dollars=0.10,
    )


@pytest.fixture
def position() -> PositionRecord:
    """Sample position with an active stop-loss order."""
    return PositionRecord(
        symbol="AAPL",
        entry_timestamp=datetime.now(UTC),
        entry_price=150.0,
        entry_signal="BUY",
        entry_confidence=0.8,
        current_qty=10.0,
        current_stop_loss=145.0,
        initial_stop_loss=140.0,
        stop_loss_order_id="order-123",
        profit_targets=[160.0, 170.0],
        last_updated=datetime.now(UTC),
    )


@pytest.fixture
def mock_broker(position: PositionRecord) -> Mock:
    """Broker mock with a position at $160."""
    broker = Mock()
    broker.get_account_info.return_value = Ok(
        BrokerAccountInfo(
            balance=50000.0,
            available_cash=25000.0,
            total_exposure=10000.0,
            portfolio_value=50000.0,
            positions={
                position.symbol: BrokerPosition(
                    symbol=position.symbol,
                    qty=10.0,
                    market_value=1600.0,
                    avg_entry_price=150.0,
                    unrealized_pnl=100.0,
                    unrealized_pnl_percent=6.67,
                )
            },
        )
    )
    broker.submit_stop_order.return_value = Ok(
        OrderStatus(
            order_id="new-order-456",
            symbol=position.symbol,
            qty=10.0,
            filled_qty=0.0,
            side="sell",
            status="accepted",
            submitted_at=datetime.now(UTC),
            filled_at=None,
            filled_avg_price=None,
        )
    )
    return broker


@pytest.mark.unit
class TestUpdateStopLossPendingCancel:
    """_update_stop_loss proceeds when cancel raises a pending-cancel error."""

    def _make_manager(self, config: PositionManagementConfig, broker: Mock) -> PositionManager:
        manager = PositionManager(broker=None, config=config)
        manager.set_broker(broker)
        return manager

    def test_proceeds_on_order_pending_cancel_message(
        self, config: PositionManagementConfig, position: PositionRecord, mock_broker: Mock
    ) -> None:
        """cancel_order raising 'order pending cancel' should not block the stop update."""
        mock_broker.cancel_order.return_value = Err(Exception("order pending cancel"))
        manager = self._make_manager(config, mock_broker)

        with patch.object(manager._persistence, "persist_action", return_value=None):
            result = manager._update_stop_loss(position, 148.0)

        assert result == "new-order-456"
        mock_broker.submit_stop_order.assert_called_once()

    def test_proceeds_on_alpaca_error_code_42210000(
        self, config: PositionManagementConfig, position: PositionRecord, mock_broker: Mock
    ) -> None:
        """cancel_order raising Alpaca error code 42210000 should not block the stop update."""
        mock_broker.cancel_order.return_value = Err(Exception("42210000: order is in pending_cancel state"))
        manager = self._make_manager(config, mock_broker)

        with patch.object(manager._persistence, "persist_action", return_value=None):
            result = manager._update_stop_loss(position, 148.0)

        assert result == "new-order-456"
        mock_broker.submit_stop_order.assert_called_once()

    def test_returns_none_on_other_cancel_exception(
        self, config: PositionManagementConfig, position: PositionRecord, mock_broker: Mock
    ) -> None:
        """Unrelated cancel_order exceptions should log an error and return None."""
        mock_broker.cancel_order.return_value = Err(Exception("connection timeout"))
        manager = self._make_manager(config, mock_broker)

        with patch.object(manager._persistence, "persist_action", return_value=None):
            result = manager._update_stop_loss(position, 148.0)

        assert result is None
        mock_broker.submit_stop_order.assert_not_called()
