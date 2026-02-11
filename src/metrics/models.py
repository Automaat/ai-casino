"""Typed models for metrics and signal tracking."""

from pydantic import BaseModel, Field


class SignalUpdateRecord(BaseModel):
    """Partial signal record for outcome price updates."""

    id: int = Field(description="Signal record ID")
    symbol: str = Field(description="Stock ticker symbol")
    timestamp: str = Field(description="Signal timestamp (ISO format)")
    signal: str = Field(description="Trading signal (BUY/SELL/HOLD)")
    price_at_signal: float = Field(gt=0.0, description="Stock price when signal was generated")
    actual_exit_price: float | None = Field(default=None, gt=0.0, description="Actual exit price if closed")

    def __repr__(self) -> str:
        """String representation."""
        return f"SignalUpdateRecord(id={self.id}, symbol={self.symbol})"


class SignalRecord(BaseModel):
    """Signal record from historical cache."""

    id: int = Field(description="Signal record ID")
    symbol: str = Field(description="Stock ticker symbol")
    timestamp: str = Field(description="Signal timestamp (ISO format)")
    signal: str = Field(description="Trading signal (BUY/SELL/HOLD)")
    confidence: float = Field(ge=0.0, le=1.0, description="Signal confidence")
    price_at_signal: float = Field(gt=0.0, description="Stock price when signal was generated")
    strategy_used: str | None = Field(default=None, description="Strategy used for signal")
    price_at_1d: float | None = Field(default=None, gt=0.0, description="Price 1 day after signal")
    price_at_5d: float | None = Field(default=None, gt=0.0, description="Price 5 days after signal")
    price_at_20d: float | None = Field(default=None, gt=0.0, description="Price 20 days after signal")
    actual_exit_price: float | None = Field(default=None, gt=0.0, description="Actual exit price if closed")
    regime: str | None = Field(default=None, description="Market regime when signal generated")

    def __repr__(self) -> str:
        """String representation."""
        return f"SignalRecord(id={self.id}, symbol={self.symbol}, signal={self.signal})"
