"""Lightweight workflow type definitions (no heavy dependencies)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from src.daemon.degradation import DegradationContext

from src.agents.bearish_researcher import BearishResearchAnalysis
from src.agents.bullish_researcher import BullishResearchAnalysis
from src.agents.comparative import ComparativeAnalysis
from src.agents.fundamental import FundamentalAnalysis
from src.agents.news import NewsAnalysis
from src.agents.risk import RiskAssessment
from src.agents.sentiment import SentimentAnalysis
from src.agents.social import SocialSentimentAnalysis
from src.agents.technical import TechnicalAnalysis
from src.agents.trader import TradingDecision
from src.agents.trump import TrumpAnalysis
from src.agents.web_researcher import WebResearchAnalysis
from src.data.broker import OrderStatus
from src.metrics.execution import WorkflowExecutionMetrics
from src.strategies.regime import RegimeAnalysis
from src.strategies.session import TradingSession


class WorkflowExtraContext(TypedDict, total=False):
    """Optional context passed to workflow pipeline."""

    degradation_context: "DegradationContext | None"
    enable_multi_timeframe: bool
    sector_rotation_context: str | None
    earnings_context: str | None
    peer_analysis_context: str | None
    game_plan_context: str | None
    position_context: dict[str, object] | None


class BacktestValidation(BaseModel):
    """Pre-trade backtesting validation result."""

    symbol: str
    strategy_name: str
    passed: bool
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    win_rate: float
    profit_factor: float
    total_trades: int
    lookback_days: int
    failure_reasons: list[str]
    confidence_adjustment: float


class TradingWorkflowResult(BaseModel):
    """Complete trading analysis result."""

    symbol: str
    trading_session: TradingSession = TradingSession.REGULAR
    technical: TechnicalAnalysis
    sentiment: SentimentAnalysis
    news: NewsAnalysis
    trump: TrumpAnalysis | None = None
    fundamental: FundamentalAnalysis | None = None
    comparative: ComparativeAnalysis | None = None
    web_research: WebResearchAnalysis | None = None
    social_sentiment: SocialSentimentAnalysis | None = None
    bullish: BullishResearchAnalysis
    bearish: BearishResearchAnalysis
    decision: TradingDecision
    risk: RiskAssessment
    order: OrderStatus | None = None
    regime: RegimeAnalysis | None = None
    strategy_used: str | None = None
    warnings: list[str] = []
    earnings_context: str | None = None
    peer_analysis_context: str | None = None
    execution_metrics: WorkflowExecutionMetrics | None = None
    backtest_validation: BacktestValidation | None = None
    degradation_tier: str | None = None
    degradation_confidence_penalty: float | None = None

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    @property
    def has_incomplete_data(self) -> bool:
        """Check if analysis was performed with incomplete data."""
        return len(self.warnings) > 0
