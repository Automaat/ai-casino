"""Response models for FastAPI daemon API."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health endpoint response."""

    status: str = Field(description="Health status (healthy/degraded)")
    uptime_seconds: float = Field(description="Daemon uptime in seconds")
    daemon_running: bool = Field(description="Whether daemon is running")
    last_run: str | None = Field(description="Last analysis run timestamp")


class StateSummaryResponse(BaseModel):
    """State summary endpoint response."""

    total_analyses: int = Field(description="Total analyses performed")
    recent_analyses: list[dict] = Field(description="Recent analysis records", default_factory=list)
    total_trades: int = Field(description="Total trades executed")
    positions_count: int = Field(description="Number of active positions")
    win_rate: float | None = Field(default=None, description="Win rate (0.0-1.0), null if unavailable")
    error_count: int = Field(description="Total errors recorded")
    degradation_tier: str = Field(description="Current degradation tier")
    trading_mode: str = Field(description="Current trading mode (paper/live)")


class ConfigResponse(BaseModel):
    """Config endpoint response."""

    watchlist: list[str] = Field(description="Symbols being monitored")
    interval_minutes: int = Field(description="Analysis interval in minutes")
    market_hours_only: bool = Field(description="Whether restricted to market hours")
    auto_trade: bool = Field(description="Whether auto-trading is enabled")
    trading_mode: str = Field(description="Current trading mode (paper/live)")
    pre_market_enabled: bool = Field(description="Whether pre-market trading is enabled")


class AnalysisRecordResponse(BaseModel):
    """Single analysis for API."""

    symbol: str
    timestamp: datetime
    signal: str
    confidence: float
    executed_trade: bool
    trading_session: str
    is_paper_trade: bool
    rsi: float | None = None
    macd_hist: float | None = None
    reasoning: list[str] = Field(default_factory=list)
    technical_analysis_reasoning: str | None = None
    sentiment_analysis_reasoning: str | None = None
    news_analysis_reasoning: str | None = None


class AnalysesResponse(BaseModel):
    """Analyses endpoint response."""

    analyses: list[AnalysisRecordResponse]
    total_count: int = Field(description="Total in history")
    returned_count: int = Field(description="Returned")


class PositionResponse(BaseModel):
    """Active position (excludes internal fields)."""

    symbol: str
    entry_price: float
    current_qty: float
    current_stop_loss: float
    entry_timestamp: datetime
    entry_signal: str
    entry_confidence: float
    days_held: int
    trailing_stop_activated: bool
    breakeven_activated: bool
    profit_targets: list[float]
    current_price: float


class PositionsResponse(BaseModel):
    """Positions endpoint response."""

    positions: list[PositionResponse]
    count: int


class WatchlistResponse(BaseModel):
    """Watchlist endpoint response."""

    symbols: list[str]
    count: int
    sources: dict[str, int] = Field(description="Breakdown: config/broker/screening")


class SnapshotRecord(BaseModel):
    """Portfolio snapshot record."""

    timestamp: datetime
    portfolio_value: float
    balance: float
    total_exposure: float


class SnapshotsResponse(BaseModel):
    """Snapshots endpoint response."""

    snapshots: list[SnapshotRecord]
    count: int
    database_enabled: bool = Field(description="Whether database persistence is enabled")
    has_trades: bool = Field(default=False, description="Whether any trades have been executed")


class RebalanceAllocation(BaseModel):
    """Rebalance allocation record."""

    symbol: str
    target_weight: float
    current_weight: float
    delta: float
    action: str


class RebalanceResponse(BaseModel):
    """Rebalance endpoint response."""

    enabled: bool = Field(description="Whether rebalancing is enabled")
    timestamp: datetime | None = Field(default=None, description="Last rebalance timestamp")
    method: str | None = Field(default=None, description="Rebalancing method")
    allocations: list[RebalanceAllocation] = Field(default_factory=list, description="Target allocations")
    expected_return: float | None = Field(default=None, description="Expected portfolio return")
    expected_volatility: float | None = Field(default=None, description="Expected volatility")
    sharpe_ratio: float | None = Field(default=None, description="Expected Sharpe ratio")


class RiskReportResponse(BaseModel):
    """Risk report endpoint response."""

    timestamp: datetime
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    cdar_95: float
    max_drawdown: float
    risk_status: str


class DegradationResponse(BaseModel):
    """Degradation endpoint response."""

    tier: str
    unavailable_services: list[str]
    confidence_adjustment: float
    halt_reason: str | None


class ServiceCheck(BaseModel):
    """Individual service health check result."""

    service: str = Field(description="Service name")
    status: str = Field(description="Health status (HEALTHY/DEGRADED/UNHEALTHY/SKIPPED)")
    message: str = Field(description="Health check message")
    duration_ms: float = Field(description="Check duration in milliseconds")
    checked_at: datetime = Field(description="ISO timestamp of check")


class ServiceHealthResponse(BaseModel):
    """Service health endpoint response."""

    overall_status: str = Field(description="Overall health (HEALTHY/DEGRADED/UNHEALTHY)")
    service_checks: list[ServiceCheck] = Field(description="Individual service checks")


class MarketEventsResponse(BaseModel):
    """Market events endpoint response."""

    events: list[dict]
    returned_count: int


class DegradationHistoryResponse(BaseModel):
    """Degradation history endpoint response."""

    records: list[dict]
    count: int


class GamePlanResponse(BaseModel):
    """Game plan endpoint response."""

    date: str
    priority_symbols: list[str]
    risk_stance: str
    sector_focus: list[str]
    reasoning: str
    confidence: float
    generated_at: str


class EventResponse(BaseModel):
    """Events endpoint response."""

    events: list[dict]
    returned_count: int


class RiskHistoryResponse(BaseModel):
    """Risk report history."""

    reports: list[RiskReportResponse]
    count: int


class ExecutionMetricsListResponse(BaseModel):
    """Execution metrics list endpoint response."""

    metrics: list[dict]
    count: int


class ActiveExecutionGraphsResponse(BaseModel):
    """Active execution graphs response."""

    graphs: list[dict] = Field(description="Active execution graph data")
    count: int = Field(description="Number of active graphs")


class ExecutionGraphDetailResponse(BaseModel):
    """Single execution graph detail response."""

    workflow_id: str = Field(description="Workflow ID")
    graph: dict = Field(description="Execution graph data")
    source: str = Field(description="Data source: active | memory | database")


class ExecutionGraphHistoryResponse(BaseModel):
    """Execution graph history response."""

    graphs: list[dict] = Field(description="Historical execution graphs")
    count: int = Field(description="Number of graphs returned")
    database_enabled: bool = Field(description="Whether database persistence is enabled")


class SectorRotationResponse(BaseModel):
    """Sector rotation analysis."""

    timestamp: datetime
    leading_sectors: list[str]
    lagging_sectors: list[str]
    sector_strengths: dict[str, float]
    sector_momenta: dict[str, str]
    flagged_positions: list[str]


class CorrelationMatrixResponse(BaseModel):
    """Correlation matrix."""

    timestamp: datetime
    num_positions: int
    correlation_matrix: dict[str, dict[str, float]]
    symbols: list[str]
    max_correlation: float
    avg_correlation: float


class FullConfigResponse(BaseModel):
    """Full daemon configuration with sensitive fields masked."""

    watchlist: list[str]
    interval_minutes: int
    market_hours_only: bool
    auto_trade: bool
    max_concurrent_analyses: int
    trading_mode: str
    paper_trading: dict[str, Any]
    schedule: dict[str, Any]
    state: dict[str, Any]
    journal: dict[str, Any]
    health: dict[str, Any]
    optimization: dict[str, Any]
    screening: dict[str, Any]
    prefetch: dict[str, Any]
    sector_rotation: dict[str, Any]
    earnings_calendar: dict[str, Any]
    peer_analysis: dict[str, Any]
    correlation_audit: dict[str, Any]
    reporting: dict[str, Any]
    risk_limits: dict[str, Any]
    rebalancing: dict[str, Any]
    signal_tracking: dict[str, Any]
    pre_trade_backtesting: dict[str, Any]
    game_plan: dict[str, Any]
    position_management: dict[str, Any]
    monte_carlo: dict[str, Any]
    notifications: dict[str, Any]
    analysis_orchestration: dict[str, Any]
    news_watcher: dict[str, Any]
    social_watcher: dict[str, Any]
    filings_watcher: dict[str, Any]
    anomaly_watcher: dict[str, Any]
    api: dict[str, Any]
    llm: dict[str, Any]
    api_keys: dict[str, Any]


class SupervisorMetricResponse(BaseModel):
    """Single supervisor metric response."""

    id: str
    created_at: datetime
    workflow_id: str
    symbol: str
    timestamp: datetime
    required_analyses: list[str]
    optional_analyses: list[str]
    skip_analyses: dict[str, str]
    routing_reasoning: str
    total_workers: int
    required_workers: int
    optional_workers: int
    successful_workers: int
    failed_workers: int
    routing_decision_ms: float
    group1_execution_ms: float
    research_execution_ms: float
    total_supervisor_overhead_ms: float
    worker_timings: dict[str, float]
    worker_errors: dict[str, str]
    total_llm_calls: int
    total_cost_usd: float
    planning_fallback_used: bool
    synthesis_fallback_used: bool
    confidence_adjustment: float
    synthesis_reasoning: str
    parallel_efficiency_percent: float
    timeout_triggered: bool


class SupervisorMetricsListResponse(BaseModel):
    """List of supervisor metrics."""

    metrics: list[SupervisorMetricResponse]
    count: int


class WorkerStats(BaseModel):
    """Worker statistics for summary."""

    total_executions: int
    successful_executions: int
    failed_executions: int
    success_rate: float
    avg_duration_ms: float


class WorkerPerformanceResponse(BaseModel):
    """Worker performance by type."""

    worker_stats: dict[str, WorkerStats]
    total_workers: int
    sample_size: int


class SupervisorSummaryResponse(BaseModel):
    """Supervisor metrics summary."""

    avg_efficiency_percent: float
    avg_routing_ms: float
    avg_group1_ms: float
    avg_research_ms: float
    avg_total_ms: float
    timeout_rate_percent: float
    sample_size: int
    symbol: str | None


class ErrorSummaryResponse(BaseModel):
    """Error summary by worker type."""

    error_counts: dict[str, int]
    total_errors: int


class ValidationCriterionResponse(BaseModel):
    """Single validation criterion result."""

    name: str
    passed: bool
    current_value: float
    threshold: float
    message: str


class PaperTradingValidationResponse(BaseModel):
    """Paper trading validation status response."""

    ready_for_live: bool
    assessment_date: datetime
    paper_trading_duration_days: int
    total_paper_trades: int
    criteria: list[ValidationCriterionResponse]
    recommendations: list[str]


class DiscoverySourceBreakdown(BaseModel):
    """Discovery source breakdown for insights."""

    source: str = Field(description="Discovery source name")
    count: int = Field(description="Number of discoveries from this source")
    percentage: float = Field(description="Percentage of total discoveries")


class DiscoveryRecord(BaseModel):
    """Discovery record for insights table."""

    symbol: str
    discovered_at: datetime
    composite_score: float
    sources: list[str]
    added_to_watchlist: bool
    first_signal: str | None
    first_signal_date: datetime | None
    outcome_7d: float | None
    outcome_30d: float | None


class DiscoverySuccessMetrics(BaseModel):
    """Success tracking metrics."""

    total_discovered: int = Field(description="Total symbols discovered")
    added_to_watchlist: int = Field(description="Symbols added to watchlist")
    received_signal: int = Field(description="Discovered symbols that received trading signal")
    signal_rate: float = Field(description="Percentage that received signals")


class DiscoveryInsightsResponse(BaseModel):
    """Discovery insights dashboard data."""

    source_breakdown: list[DiscoverySourceBreakdown] = Field(description="Multi-source breakdown")
    success_metrics: DiscoverySuccessMetrics = Field(description="Success rate tracking")
    recent_discoveries: list[DiscoveryRecord] = Field(description="Recent discoveries with outcomes")
    avg_composite_score: float = Field(description="Average composite score")
    total_discoveries: int = Field(description="Total discoveries in period")


__all__ = [
    "ActiveExecutionGraphsResponse",
    "AnalysesResponse",
    "AnalysisRecordResponse",
    "ConfigResponse",
    "CorrelationMatrixResponse",
    "DegradationHistoryResponse",
    "DegradationResponse",
    "DiscoveryInsightsResponse",
    "DiscoveryRecord",
    "DiscoverySourceBreakdown",
    "DiscoverySuccessMetrics",
    "ErrorSummaryResponse",
    "EventResponse",
    "ExecutionGraphDetailResponse",
    "ExecutionGraphHistoryResponse",
    "ExecutionMetricsListResponse",
    "FullConfigResponse",
    "GamePlanResponse",
    "HealthResponse",
    "MarketEventsResponse",
    "PaperTradingValidationResponse",
    "PositionResponse",
    "PositionsResponse",
    "RebalanceAllocation",
    "RebalanceResponse",
    "RiskHistoryResponse",
    "RiskReportResponse",
    "SectorRotationResponse",
    "ServiceCheck",
    "ServiceHealthResponse",
    "SnapshotRecord",
    "SnapshotsResponse",
    "StateSummaryResponse",
    "SupervisorMetricResponse",
    "SupervisorMetricsListResponse",
    "SupervisorSummaryResponse",
    "ValidationCriterionResponse",
    "WatchlistResponse",
    "WorkerPerformanceResponse",
    "WorkerStats",
]
