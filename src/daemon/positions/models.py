"""Position data models."""

from datetime import datetime

from pydantic import BaseModel, Field


class PositionContext(BaseModel):
    """Position context for trader decisions."""

    entry_price: float = Field(gt=0.0, description="Entry price for the position")
    days_held: int = Field(ge=0, description="Number of days position has been held")
    current_stop_loss: float = Field(gt=0.0, description="Current stop loss price")
    profit_targets: list[float] = Field(default_factory=list, description="Profit target prices")
    trailing_activated: bool = Field(default=False, description="Whether trailing stop is activated")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PositionContext(entry={self.entry_price:.2f}, "
            f"days={self.days_held}, stop={self.current_stop_loss:.2f})"
        )


class MarketEvent(BaseModel):
    """Market event record."""

    timestamp: datetime = Field(description="Event timestamp")
    event_type: str = Field(description="Event type (HALT, NEWS, EARNINGS)")
    symbol: str = Field(description="Stock ticker symbol")
    description: str = Field(description="Event description")

    def __repr__(self) -> str:
        """String representation."""
        return f"MarketEvent(type={self.event_type}, symbol={self.symbol})"


class PositionRecord(BaseModel):
    """Active position requiring management."""

    symbol: str
    entry_timestamp: datetime
    entry_price: float
    entry_signal: str
    entry_confidence: float
    current_qty: float
    current_stop_loss: float
    initial_stop_loss: float
    stop_loss_order_id: str | None = None
    profit_targets: list[float]
    days_held: int = 0
    last_updated: datetime
    trailing_stop_activated: bool = False
    breakeven_activated: bool = False
    high_water_mark: float | None = None
    current_conviction: float | None = None
    last_analysis_timestamp: datetime | None = None
    conviction_history: list[float] = Field(default_factory=list)
    initial_risk_per_share: float | None = None
    r_multiple_targets_hit: list[int] = Field(default_factory=list)
    pending_sell_signal_count: int = 0


class PositionManagementAction(BaseModel):
    """Action taken by position manager."""

    symbol: str
    action_type: str
    timestamp: datetime
    old_stop_loss: float | None = None
    new_stop_loss: float | None = None
    qty_sold: float | None = None
    price: float
    reason: str
    executed: bool
    order_id: str | None = None
