"""Trade execution stage I/O models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.agents.risk import RiskAssessment
from src.agents.trader import TradingDecision
from src.data.broker import OrderStatus
from src.strategies.session import TradingSession


class TradeExecutionInput(BaseModel):
    """Input for trade execution stage."""

    symbol: str
    final_decision: TradingDecision
    risk_assessment: RiskAssessment
    trading_session: TradingSession

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


class TradeExecutionOutput(BaseModel):
    """Output from trade execution stage."""

    order_status: OrderStatus | None
    warnings: list[str] = Field(default_factory=list)

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
