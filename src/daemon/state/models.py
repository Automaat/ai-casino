"""Pydantic models for daemon state records."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field

from src.discovery.models import DiscoverySource
from src.strategies.session import TradingSession


class AnalysisRecord(BaseModel):
    """Record of a single analysis run."""

    symbol: str
    timestamp: datetime
    signal: str
    confidence: float
    executed_trade: bool = False
    trading_session: TradingSession = TradingSession.REGULAR
    is_paper_trade: bool = True
    rsi: float | None = None
    macd_hist: float | None = None
    reasoning: list[str] = Field(default_factory=list)
    technical_analysis_reasoning: str | None = None
    sentiment_analysis_reasoning: str | None = None
    news_analysis_reasoning: str | None = None


class PortfolioSnapshot(BaseModel):
    """Snapshot of portfolio state at a point in time."""

    balance: float
    available_cash: float
    total_exposure: float
    portfolio_value: float
    positions: dict
    trigger: str


class DiscoveryHistoryRecord(BaseModel):
    """Record of stock discovery outcome for learning."""

    symbol: str
    discovered_at: datetime
    composite_score: float
    sources: list[DiscoverySource]
    added_to_watchlist: bool
    ttl_expires_at: datetime
    first_signal: str | None = None
    first_signal_date: datetime | None = None
    outcome_7d: float | None = None
    outcome_30d: float | None = None
    supervisor_evaluation_score: float | None = None
    supervisor_recommendation: str | None = None
    evaluation_reasoning: str | None = None
    price_at_discovery: float | None = None
    outcome_updated_at: datetime | None = None


class DiscoverySourceMetrics(BaseModel):
    """Discovery source performance metrics."""

    source_type: str
    measurement_date: date
    total_discoveries: int = 0
    watchlist_additions: int = 0
    signal_conversions: int = 0
    discoveries_with_7d_outcome: int = 0
    positive_7d_outcomes: int = 0
    avg_7d_return: float | None = None
    median_7d_return: float | None = None
    discoveries_with_30d_outcome: int = 0
    positive_30d_outcomes: int = 0
    avg_30d_return: float | None = None
    median_30d_return: float | None = None
    precision_score: float | None = None
    recall_score: float | None = None
    f1_score: float | None = None
    false_positives: int = 0
    false_negatives: int = 0


class PortfolioAllocationRecord(BaseModel):
    """Single asset allocation in rebalancing record."""

    symbol: str
    weight: float
    action: str
    delta: float


class PortfolioRebalancingRecord(BaseModel):
    """Record of portfolio rebalancing analysis."""

    timestamp: datetime
    method: str
    allocations: list[PortfolioAllocationRecord]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    rebalances_executed: int
    rebalances_pending: int


class OptimizationRecord(BaseModel):
    """Record of a parameter optimization run."""

    timestamp: datetime
    symbols_optimized: list[str]
    symbols_skipped: list[str]
    total_time_seconds: float


class PrefetchRecord(BaseModel):
    """Record of a data prefetch run."""

    timestamp: datetime
    symbols_prefetched: int
    symbols_failed: int
    finbert_ready: bool
    total_duration_seconds: float


class SectorRotationRecord(BaseModel):
    """Record of a sector rotation analysis run."""

    timestamp: datetime
    leading_sectors: list[str]
    lagging_sectors: list[str]
    sector_strengths: dict[str, float]
    sector_momenta: dict[str, str]
    flagged_positions: list[str]


class SectorAttributionRecord(BaseModel):
    """Record of a sector attribution analysis run."""

    timestamp: datetime
    total_portfolio_value: float
    benchmark_name: str
    contributions: list[dict[str, float | str | int]]


class PeerAnalysisInput(BaseModel):
    """Input parameters for recording peer analysis."""

    symbols_analyzed: list[str]
    rankings: dict[str, int]
    swap_recommendations: list[str]
    total_peers: int
    total_duration_seconds: float
    analyses: list[dict] | None = None


class PeerAnalysisRecord(BaseModel):
    """Record of a deep peer benchmarking analysis run."""

    timestamp: datetime
    symbols_analyzed: list[str]
    rankings: dict[str, int]
    swap_recommendations: list[str]
    analyses: list[dict] = Field(default_factory=list)
    total_peers: int
    total_duration_seconds: float


class CorrelationAuditRecord(BaseModel):
    """Record of a portfolio correlation audit run."""

    timestamp: datetime
    num_positions: int
    num_correlated_pairs: int
    max_correlation: float
    avg_correlation: float
    diversification_ratio: float
    num_substitutions: int
    total_duration_seconds: float


class GamePlanRecord(BaseModel):
    """Record of game plan execution."""

    timestamp: datetime
    priority_symbols: list[str]
    risk_stance: str
    sector_focus: list[str]
    reasoning: str | None = None
    confidence: float | None = None
    overnight_summary: str | None = None
    key_levels: dict[str, float] = Field(default_factory=dict)
    generated_at: datetime | None = None


class RiskReportRecord(BaseModel):
    """Record of a portfolio risk report."""

    timestamp: datetime
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    cdar_95: float
    max_drawdown: float
    portfolio_volatility: float
    current_exposure_percent: float
    num_positions: int
    var_limit_breached: bool
    cvar_limit_breached: bool
    risk_status: str


class PortfolioHealthRecord(BaseModel):
    """Record of a portfolio health check."""

    timestamp: datetime
    total_positions: int
    portfolio_value: float
    cash_percent: float
    max_concentration_percent: float
    max_concentration_symbol: str
    total_pnl_percent: float
    biggest_drawdown_symbol: str | None
    biggest_drawdown_percent: float
    health_status: str  # HEALTHY, WARNING, CRITICAL
    recommendations: list[str]
    constraints: list[str]


class MonteCarloRecord(BaseModel):
    """Record of Monte Carlo stress test execution."""

    timestamp: datetime
    simulation_method: str
    num_simulations: int
    horizon_days: int

    # Key results
    prob_loss_gt_threshold: float
    expected_worst_drawdown: float
    var_95: float
    cvar_95: float
    median_recovery_days: float | None

    # Alert status
    exceeds_risk_tolerance: bool
    alert_message: str | None

    # Portfolio snapshot
    portfolio_symbols: list[str]
    total_market_value: float


class EarningsEventRecord(BaseModel):
    """Record of a single earnings event."""

    symbol: str
    earnings_date: str
    estimate_eps: float | None = None


class EarningsCalendarRecord(BaseModel):
    """Record of an earnings calendar fetch run."""

    timestamp: datetime
    events: list[EarningsEventRecord]
    symbols_fetched: int
    symbols_failed: int


class DegradationRecord(BaseModel):
    """Record of degradation event."""

    timestamp: datetime
    tier: str
    unavailable_services: list[str]
    confidence_adjustment: float
    halt_reason: str | None = None


class ProfilingRecord(BaseModel):
    """Record of cycle profiling metrics."""

    cycle_number: int
    timestamp: datetime
    duration_seconds: float
    profiling_overhead_percent: float
    top_function: str | None = None
    top_function_cumtime: float | None = None


class SignalOutcome(BaseModel):
    """Signal outcome record for persistent learning."""

    symbol: str
    timestamp: datetime
    signal: str
    confidence: float
    price_at_signal: float
    strategy_used: str | None = None
    regime: str | None = None
    trading_session: str = "REGULAR"
    technical_signal: str | None = None
    sentiment_signal: str | None = None
    news_signal: str | None = None
    technical_reasoning: str | None = None
    sentiment_reasoning: str | None = None
    news_reasoning: str | None = None
    price_at_1d: float | None = None
    price_at_5d: float | None = None
    price_at_20d: float | None = None
    actual_exit_price: float | None = None
    actual_exit_date: datetime | None = None
    outcome_updated_at: datetime | None = None


class SignalUpdateRecord(BaseModel):
    """Record representing a signal that needs outcome price update."""

    id: str
    symbol: str
    timestamp: datetime
    target_date: datetime


class HealthReportRecord(BaseModel):
    """Record of a health check execution."""

    id: str | None = None
    timestamp: datetime
    overall_status: str
    service_checks: list[dict]
    cleanup_results: list[dict]
    total_duration_ms: float


class TradeJournalRecord(BaseModel):
    """Record of a daily trade journal."""

    id: str | None = None
    date: date
    outcomes: list[dict]
    winners: list[str]
    losers: list[str]
    lessons: list[str]
    tomorrows_focus: list[str]
    overall_assessment: str
    markdown_content: str | None = None
    total_signals: int
    correct_signals: int
    accuracy_pct: float


class PaperTradingReportRecord(BaseModel):
    """Record of a paper trading validation assessment."""

    id: str | None = None
    assessment_date: datetime
    ready_for_live: bool
    paper_trading_duration_days: int
    total_paper_trades: int
    criteria: list[dict]
    total_pnl: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    simulated_live: dict | None = None
    recommendations: list[str]
