"""Lightweight workflow type definitions (no heavy dependencies)."""

from __future__ import annotations

from pydantic import BaseModel

from src.agents.bearish_researcher import BearishResearchAnalysis
from src.agents.bullish_researcher import BullishResearchAnalysis
from src.agents.comparative import ComparativeAnalysis
from src.agents.fundamental import FundamentalAnalysis
from src.agents.news import NewsAnalysis
from src.agents.risk import RiskAssessment
from src.agents.sentiment import SentimentAnalysis
from src.agents.technical import TechnicalAnalysis
from src.agents.trader import TradingDecision
from src.agents.trump import TrumpAnalysis
from src.agents.web_researcher import WebResearchAnalysis
from src.data.broker import OrderStatus
from src.metrics.execution import WorkflowExecutionMetrics
from src.strategies.regime import RegimeAnalysis


class TradingWorkflowResult(BaseModel):
    """Complete trading analysis result."""

    symbol: str
    technical: TechnicalAnalysis
    sentiment: SentimentAnalysis
    news: NewsAnalysis
    trump: TrumpAnalysis | None = None
    fundamental: FundamentalAnalysis | None = None
    comparative: ComparativeAnalysis | None = None
    web_research: WebResearchAnalysis | None = None
    bullish: BullishResearchAnalysis
    bearish: BearishResearchAnalysis
    decision: TradingDecision
    risk: RiskAssessment
    order: OrderStatus | None = None
    regime: RegimeAnalysis | None = None
    strategy_used: str | None = None
    warnings: list[str] = []
    execution_metrics: WorkflowExecutionMetrics | None = None

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    @property
    def has_incomplete_data(self) -> bool:
        """Check if analysis was performed with incomplete data."""
        return len(self.warnings) > 0
