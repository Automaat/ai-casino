"""Broker protocol for trading integrations."""

from typing import Protocol

from result import Result

from src.v1.trades.brokers.models import BrokerAccountInfo, OrderStatus


class Broker(Protocol):
    """Protocol for broker implementations."""

    def get_account_info(self) -> Result[BrokerAccountInfo, Exception]:
        """Fetch current account information."""
        ...

    def submit_order(
        self, symbol: str, qty: int, side: str, stop_loss_price: float | None = None
    ) -> Result[OrderStatus, Exception]:
        """Submit market order with optional stop loss."""
        ...

    def submit_stop_order(self, symbol: str, qty: int, stop_price: float) -> Result[OrderStatus, Exception]:
        """Submit stop order to protect existing position."""
        ...

    def get_order_status(self, order_id: str) -> Result[OrderStatus, Exception]:
        """Get status of an existing order."""
        ...

    def cancel_order(self, order_id: str) -> Result[None, Exception]:
        """Cancel an existing order."""
        ...

    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        ...
