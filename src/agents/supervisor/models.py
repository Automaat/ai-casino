"""Supervisor data models."""

from enum import StrEnum

from pydantic import BaseModel, Field

from src.strategies.regime import RegimeAnalysis
from src.strategies.session import TradingSession


class AnalysisType(StrEnum):
    """Available analyses for routing."""

    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    NEWS = "news"
    FUNDAMENTAL = "fundamental"
    COMPARATIVE = "comparative"
    WEB_RESEARCH = "web_research"
    SOCIAL_SENTIMENT = "social_sentiment"
    BULLISH_RESEARCH = "bullish_research"
    BEARISH_RESEARCH = "bearish_research"
    TRUMP = "trump"


class CandidateRecommendation(StrEnum):
    """Supervisor recommendation for discovery candidates."""

    ADD_WATCHLIST = "ADD_WATCHLIST"
    DEFER = "DEFER"
    SKIP = "SKIP"


class AnalysisRoutingDecision(BaseModel):
    """LLM response for analysis planning."""

    required_analyses: list[AnalysisType] = Field(description="Critical analyses that must run")
    optional_analyses: list[AnalysisType] = Field(description="Valuable but not critical analyses")
    skip_analyses: dict[AnalysisType, str] = Field(
        description="Analyses to skip with reasoning", default_factory=dict
    )
    reasoning: str = Field(description="Overall routing rationale")
    priority_order: list[AnalysisType] = Field(description="Execution priority order")


class AnalysisWeights(BaseModel):
    """LLM response for synthesis."""

    weights: dict[AnalysisType, float] = Field(description="Reliability weights (0.0-1.0) per analysis")
    conflicts: list[str] = Field(description="Identified conflicts between analyses", default_factory=list)
    consensus: list[str] = Field(description="Strong agreement points", default_factory=list)
    confidence_adjustment: float = Field(
        ge=0.5, le=1.5, default=1.0, description="Overall confidence multiplier"
    )
    reasoning: str = Field(description="Synthesis rationale")


class PlanningContext(BaseModel):
    """Input for planning phase."""

    symbol: str
    regime: RegimeAnalysis | None
    trading_session: TradingSession
    owns_position: bool
    news_count: int
    fundamental_available: bool
    social_available: bool
    trump_count: int
    fundamental_rate_limit: bool
    time_budget_ms: int
    market_data_rows: int = 0
    is_high_volatility: bool = False


class SynthesisContext(BaseModel):
    """Input for synthesis phase."""

    symbol: str
    technical_summary: str | None = None
    sentiment_summary: str | None = None
    news_summary: str | None = None
    fundamental_summary: str | None = None
    comparative_summary: str | None = None
    web_research_summary: str | None = None
    social_summary: str | None = None
    bullish_summary: str | None = None
    bearish_summary: str | None = None
    trump_summary: str | None = None


class SupervisorDecision(BaseModel):
    """Final supervisor output."""

    routing_decision: AnalysisRoutingDecision
    analysis_weights: AnalysisWeights | None = None
    final_recommendation: str
    confidence: float = Field(ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)


class CandidateEvaluationContext(BaseModel):
    """Input context for candidate evaluation."""

    candidates: list = Field(description="List of DiscoveryCandidate objects")
    market_regime: RegimeAnalysis | None
    portfolio_symbols: list[str]
    watchlist_symbols: list[str]
    watchlist_capacity: int
    sector_exposure: dict[str, float]
    recent_discovery_outcomes: list[str] | None = None
    time_budget_ms: int
    session: TradingSession


class CandidateEvaluation(BaseModel):
    """LLM evaluation for single candidate."""

    symbol: str
    quality_score: float = Field(ge=0.0, le=1.0)
    momentum_score: float = Field(ge=0.0, le=1.0)
    risk_score: float = Field(ge=0.0, le=1.0)
    portfolio_fit_score: float = Field(ge=0.0, le=1.0)
    recommendation: CandidateRecommendation
    reasoning: str


class CandidateRanking(BaseModel):
    """Supervisor's final ranked candidate list."""

    evaluations: list[CandidateEvaluation]
    add_watchlist: list[str]
    defer: list[str]
    skip: list[str]
    priority_order: list[str]
    overall_reasoning: str
    warnings: list[str] = Field(default_factory=list)
