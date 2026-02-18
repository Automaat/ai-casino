"""Lightweight workflow type definitions (no heavy dependencies)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.agents.comparative import ComparativeAnalysis
from src.agents.fundamental import FundamentalAnalysis
from src.agents.news import NewsAnalysis
from src.agents.risk import RiskAssessment
from src.agents.sentiment import SentimentAnalysis
from src.agents.social import SocialSentimentAnalysis
from src.agents.supervisor import SupervisorDecision
from src.agents.supervisor.models import AnalysisRoutingDecision
from src.agents.technical import TechnicalAnalysis
from src.agents.thesis_researcher import BearishResearchAnalysis, BullishResearchAnalysis
from src.agents.trader import TradingDecision
from src.agents.trump import TrumpAnalysis
from src.agents.web_researcher import WebResearchAnalysis
from src.daemon.degradation import DegradationContext
from src.data.broker import OrderStatus
from src.metrics.execution import WorkflowExecutionMetrics
from src.strategies.regime import RegimeAnalysis
from src.strategies.session import TradingSession


class WorkflowExtraContext(BaseModel):
    """Optional context passed to workflow pipeline."""

    degradation_context: DegradationContext | None = None
    enable_multi_timeframe: bool = False
    sector_rotation_context: str | None = None
    earnings_context: str | None = None
    peer_analysis_context: str | None = None
    game_plan_context: str | None = None
    position_context: dict[str, object] | None = None
    economic_calendar_context: str | None = None
    options_flow_context: str | None = None
    portfolio_health_context: str | None = None
    social_sentiment_context: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


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
    sentiment: SentimentAnalysis | None = None
    news: NewsAnalysis | None = None
    trump: TrumpAnalysis | None = None
    fundamental: FundamentalAnalysis | None = None
    comparative: ComparativeAnalysis | None = None
    web_research: WebResearchAnalysis | None = None
    social_sentiment: SocialSentimentAnalysis | None = None
    bullish: BullishResearchAnalysis | None = None
    bearish: BearishResearchAnalysis | None = None
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
    supervisor_decision: SupervisorDecision | None = None
    supervisor_routing: AnalysisRoutingDecision | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def has_incomplete_data(self) -> bool:
        """Check if analysis was performed with incomplete data."""
        return len(self.warnings) > 0
