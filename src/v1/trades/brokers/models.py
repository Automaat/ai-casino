"""Broker domain models."""

from datetime import datetime

from pydantic import BaseModel


class BrokerAPIError(Exception):
    """Broker API communication failure."""


class BrokerPosition(BaseModel):
    """Broker position information."""

    symbol: str
    qty: float
    market_value: float
    avg_entry_price: float
    unrealized_pnl: float
    unrealized_pnl_percent: float

    def __repr__(self) -> str:
        """Return string representation."""
        return f"BrokerPosition(symbol={self.symbol}, qty={self.qty})"


class BrokerAccountInfo(BaseModel):
    """Broker account information."""

    balance: float
    available_cash: float
    positions: dict[str, BrokerPosition]
    total_exposure: float
    portfolio_value: float

    def __repr__(self) -> str:
        """Return string representation."""
        return f"BrokerAccountInfo(balance={self.balance}, positions={len(self.positions)})"


class OrderStatus(BaseModel):
    """Order status information."""

    order_id: str
    symbol: str
    qty: float
    filled_qty: float
    side: str
    status: str
    submitted_at: datetime
    filled_at: datetime | None
    filled_avg_price: float | None

    def __repr__(self) -> str:
        """Return string representation."""
        return f"OrderStatus(id={self.order_id}, symbol={self.symbol}, status={self.status})"
