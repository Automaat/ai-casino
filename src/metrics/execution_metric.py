"""Order execution metrics for slippage analysis."""

from datetime import UTC, datetime
from decimal import Decimal

from pydantic import BaseModel, Field

from src.data.broker import OrderStatus


class ExecutionMetric(BaseModel):
    """Order execution metric for tracking slippage and timing."""

    id: str | None = None
    order_id: str = Field(description="Broker order ID")
    symbol: str = Field(description="Stock ticker")
    side: str = Field(description="buy or sell")
    quantity: Decimal = Field(description="Shares ordered")
    requested_price: Decimal = Field(description="Price at order submission (market price)")
    filled_price: Decimal | None = Field(default=None, description="Average fill price")
    submitted_at: datetime = Field(description="Order submission timestamp")
    filled_at: datetime | None = Field(default=None, description="Order fill timestamp")
    execution_time_ms: int | None = Field(default=None, description="Submission to fill latency (ms)")
    slippage_bps: Decimal | None = Field(default=None, description="Slippage in basis points")
    broker: str = Field(default="alpaca", description="Broker name")
    venue: str | None = Field(default=None, description="Execution venue (if available)")
    status: str = Field(description="Order status")
    created_at: datetime | None = None

    @classmethod
    def from_order_status(
        cls,
        order: OrderStatus,
        requested_price: Decimal,
    ) -> ExecutionMetric:
        """Create execution metric from OrderStatus and requested price.

        Args:
            order: OrderStatus from broker API
            requested_price: Market price at order submission (for slippage calculation)

        Returns:
            ExecutionMetric with computed slippage and execution time
        """
        filled_price = Decimal(str(order.filled_avg_price)) if order.filled_avg_price else None
        execution_time_ms = None
        slippage_bps = None

        # Compute execution time if both timestamps available
        if order.filled_at and order.submitted_at:
            execution_time_ms = int((order.filled_at - order.submitted_at).total_seconds() * 1000)

        # Compute slippage: (filled - requested) / requested * 10000 bps
        # Positive = paid more than expected (bad for buys)
        # Negative = paid less than expected (good for buys)
        if filled_price and requested_price and requested_price > 0:
            slippage_pct = ((filled_price - requested_price) / requested_price) * 100
            slippage_bps = Decimal(str(slippage_pct * 100))  # Convert to basis points

        return cls(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=Decimal(str(order.qty)),
            requested_price=requested_price,
            filled_price=filled_price,
            submitted_at=order.submitted_at,
            filled_at=order.filled_at,
            execution_time_ms=execution_time_ms,
            slippage_bps=slippage_bps,
            broker="alpaca",
            status=order.status,
            created_at=datetime.now(UTC),
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ExecutionMetric(order_id={self.order_id}, slippage={self.slippage_bps}bps)"
