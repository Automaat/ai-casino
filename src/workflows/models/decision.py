"""Decision stage I/O models."""

from __future__ import annotations

from pydantic import BaseModel

from src.agents.bearish_researcher import BearishResearchAnalysis
from src.agents.bullish_researcher import BullishResearchAnalysis
from src.agents.comparative import ComparativeAnalysis
from src.agents.fundamental import FundamentalAnalysis
from src.agents.news import NewsAnalysis
from src.agents.risk import AccountInfo
from src.agents.sentiment import SentimentAnalysis
from src.agents.technical import TechnicalAnalysis
from src.agents.trader import TradingDecision
from src.agents.trump import TrumpAnalysis
from src.daemon.degradation import DegradationContext
from src.workflows.types import BacktestValidation


class DecisionContext(BaseModel):
    """Context information for decision making."""

    sector_rotation: str | None = None
    earnings: str | None = None
    peer_analysis: str | None = None
    game_plan: str | None = None
    position: dict[str, object] | None = None


class DecisionInput(BaseModel):
    """Input for decision stage."""

    symbol: str
    technical: TechnicalAnalysis | None
    sentiment: SentimentAnalysis | None
    news: NewsAnalysis | None
    bullish: BullishResearchAnalysis | None
    bearish: BearishResearchAnalysis | None
    fundamental: FundamentalAnalysis | None
    comparative: ComparativeAnalysis | None
    trump: TrumpAnalysis | None
    account_info: AccountInfo | None
    context: DecisionContext
    backtest_validation: BacktestValidation | None
    degradation_context: DegradationContext | None

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    @property
    def owns_position(self) -> bool:
        """Check if account owns position in symbol."""
        if self.context.position is None:
            return False
        return bool(self.context.position.get("owns_position", False))

    @property
    def position_qty(self) -> float:
        """Get current position quantity."""
        if self.context.position is None:
            return 0.0
        # Dict.get() returns object type, but we know it's numeric
        return float(self.context.position.get("qty", 0.0))  # type: ignore[arg-type]


class DecisionOutput(BaseModel):
    """Output from decision stage."""

    final_decision: TradingDecision

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
