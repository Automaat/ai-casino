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
    market_phase: str | None = Field(
        default=None, description="Current market phase (PRE_MARKET/REGULAR/AFTER_HOURS), None if closed"
    )
    phase_end_time: str | None = Field(default=None, description="ISO when current phase ends")


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
    sources: dict[str, int] = Field(description="Breakdown: config/broker/screening/event_watchlist")


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


class MetricsSnapshot(BaseModel):
    """Portfolio metrics snapshot."""

    expected_return: float = Field(description="Expected portfolio return")
    expected_volatility: float = Field(description="Expected volatility")
    sharpe_ratio: float = Field(description="Sharpe ratio")


class RebalanceCalculation(BaseModel):
    """Single rebalancing calculation record."""

    timestamp: datetime = Field(description="Calculation timestamp")
    method: str = Field(description="Rebalancing method used")
    allocations: list[RebalanceAllocation] = Field(description="Full allocation records")
    expected_return: float = Field(description="Expected portfolio return")
    expected_volatility: float = Field(description="Expected volatility")
    sharpe_ratio: float = Field(description="Sharpe ratio")


class RebalanceHistoryEntry(BaseModel):
    """Historical rebalancing record with deviation metrics."""

    timestamp: datetime = Field(description="Record timestamp")
    method: str = Field(description="Rebalancing method")
    avg_deviation_pct: float = Field(description="Average deviation percentage")
    max_deviation_pct: float = Field(description="Maximum deviation percentage")
    metrics: MetricsSnapshot = Field(description="Portfolio metrics at this time")


class RebalancingHistoryResponse(BaseModel):
    """Rebalancing history endpoint response."""

    enabled: bool = Field(description="Whether rebalancing is enabled")
    current_portfolio_value: float = Field(description="Current portfolio value")
    rebalance_threshold: float = Field(description="Rebalance threshold from config")
    current_metrics: MetricsSnapshot | None = Field(default=None, description="Current portfolio metrics")
    latest: RebalanceCalculation | None = Field(default=None, description="Latest calculation")
    history: list[RebalanceHistoryEntry] = Field(default_factory=list, description="Historical records")


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
    database_enabled: bool = False


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


class SectorContributionDetail(BaseModel):
    """Individual sector contribution detail."""

    sector: str
    sector_etf: str
    total_value: float
    portfolio_weight: float
    benchmark_weight: float
    over_under_weight: float
    pnl: float
    return_pct: float
    position_count: int


class SectorAttributionResponse(BaseModel):
    """Sector attribution analysis."""

    timestamp: datetime
    contributions: list[SectorContributionDetail]
    total_portfolio_value: float
    benchmark_name: str


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


class ApiActiveDiscoverySourceDetail(BaseModel):
    """Source detail for active discovery candidate API response."""

    source_type: str = Field(description="Source type")
    weight: float = Field(description="Source weight")


class ApiActiveDiscoveryCandidate(BaseModel):
    """Active discovery candidate API response."""

    symbol: str = Field(description="Stock symbol")
    discovered_at: datetime = Field(description="Discovery timestamp")
    composite_score: float = Field(description="Composite score")
    sources: list[ApiActiveDiscoverySourceDetail] = Field(description="Discovery sources")
    ttl_expires_at: datetime = Field(description="TTL expiration")
    time_remaining_minutes: int = Field(description="Minutes until expiration")


class ActiveDiscoveryResponse(BaseModel):
    """Active discovery candidates response."""

    candidates: list[ApiActiveDiscoveryCandidate] = Field(description="Active candidates")
    count: int = Field(description="Number of active candidates")
    last_discovery: datetime | None = Field(default=None, description="Last discovery run timestamp")


class PositionManagementActionResponse(BaseModel):
    """Single position management action."""

    action_type: str = Field(description="Action type (TRAILING_STOP, BREAKEVEN, PARTIAL_PROFIT, etc)")
    timestamp: datetime = Field(description="When action occurred")
    old_stop_loss: float | None = Field(default=None, description="Previous stop loss price")
    new_stop_loss: float | None = Field(default=None, description="New stop loss price")
    qty_sold: float | None = Field(default=None, description="Quantity sold (for partial exits)")
    price: float = Field(description="Current price at action time")
    reason: str = Field(description="Reason for action")
    executed: bool = Field(description="Whether action was successfully executed")
    order_id: str | None = Field(default=None, description="Broker order ID if executed")


class PositionTimelineResponse(BaseModel):
    """Position timeline with management actions."""

    symbol: str = Field(description="Position symbol")
    entry_price: float = Field(description="Entry price")
    current_price: float = Field(description="Current price")
    current_qty: float = Field(description="Current quantity")
    entry_timestamp: datetime = Field(description="Position entry timestamp")
    days_held: int = Field(description="Days position has been held")
    actions: list[PositionManagementActionResponse] = Field(description="Management actions taken")
    count: int = Field(description="Number of actions")
    database_enabled: bool = Field(description="Whether database persistence is enabled")


class TradeResponse(BaseModel):
    """Single trade record."""

    id: str = Field(description="Trade ID")
    timestamp: datetime = Field(description="Trade timestamp")
    symbol: str = Field(description="Symbol traded")
    action: str = Field(description="Trade action (BUY/SELL)")
    entry_price: float = Field(description="Entry price")
    exit_price: float | None = Field(default=None, description="Exit price (null if OPEN)")
    shares: int = Field(description="Number of shares")
    confidence: float = Field(description="Confidence (0.0-1.0)")
    risk_level: str = Field(description="Risk level (LOW/MEDIUM/HIGH)")
    status: str = Field(description="Trade status (OPEN/CLOSED/REJECTED)")
    pnl: float | None = Field(default=None, description="Profit/loss (null if OPEN)")
    pnl_percent: float | None = Field(default=None, description="P/L percentage (null if OPEN)")
    strategy_name: str | None = Field(default=None, description="Strategy used")
    is_paper_trade: bool = Field(description="Whether paper trade")
    closed_at: datetime | None = Field(default=None, description="Close timestamp (null if OPEN)")


class TradesResponse(BaseModel):
    """Trades endpoint response."""

    trades: list[TradeResponse] = Field(description="List of trades")
    total_count: int = Field(description="Total trades in database")
    returned_count: int = Field(description="Number of trades returned")
    database_enabled: bool = Field(description="Whether database persistence is enabled")


class EnrichedTradeResponse(BaseModel):
    """Trade with analysis reasoning."""

    trade: TradeResponse = Field(description="Trade record")
    analysis: AnalysisRecordResponse | None = Field(
        default=None, description="Matched analysis record with reasoning"
    )


class CostAnalyticsSummaryResponse(BaseModel):
    """Cost analytics summary."""

    total_cost_usd: float = Field(description="Total estimated cost in USD")
    total_tokens: int = Field(description="Total tokens (input + output)")
    total_executions: int = Field(description="Total workflow executions")
    avg_cost_per_execution: float = Field(description="Average cost per execution")
    avg_cost_per_signal: float = Field(description="Average cost per trade signal (BUY/SELL)")
    forecast_30d_usd: float = Field(description="Forecasted 30-day cost based on trend")
    date_range: tuple[str, str] = Field(description="Date range queried (ISO format)")


class CostTrendPointResponse(BaseModel):
    """Cost trend point for time series."""

    timestamp: datetime = Field(description="Bucket timestamp")
    cost_usd: float = Field(description="Total cost for period")
    tokens: int = Field(description="Total tokens for period")
    execution_count: int = Field(description="Execution count for period")


class CostByDimensionResponse(BaseModel):
    """Cost breakdown by dimension."""

    dimension_value: str = Field(description="Dimension value (symbol/agent/model)")
    cost_usd: float = Field(description="Total cost")
    tokens: int = Field(description="Total tokens")
    execution_count: int = Field(description="Execution count")
    percentage: float = Field(description="Percentage of total cost")


class CostTrendsResponse(BaseModel):
    """Cost trends list."""

    trends: list[CostTrendPointResponse] = Field(description="Time series data")
    count: int = Field(description="Number of data points")


class CostByDimensionListResponse(BaseModel):
    """Cost by dimension list."""

    data: list[CostByDimensionResponse] = Field(description="Dimension breakdown")
    count: int = Field(description="Number of dimensions")


class SignalFlowSummaryResponse(BaseModel):
    """Signal flow summary."""

    total_signals: int = Field(description="Total signals generated (BUY/SELL only)")
    total_buy_signals: int = Field(description="Total BUY signals")
    total_sell_signals: int = Field(description="Total SELL signals")
    execution_rate: float = Field(description="Percentage of signals that were executed")
    executed_count: int = Field(description="Number of executed signals")
    not_executed_count: int = Field(description="Number of signals not executed")
    profitable_count: int = Field(description="Number of profitable executed signals (5d)")
    unprofitable_count: int = Field(description="Number of unprofitable executed signals (5d)")
    overall_accuracy: float = Field(description="Overall accuracy of executed signals (5d)")
    avg_confidence: float = Field(description="Average signal confidence")
    date_range: tuple[str, str] = Field(description="Date range queried (ISO format)")


class SankeyNodeResponse(BaseModel):
    """Sankey diagram node."""

    name: str = Field(description="Node name")
    item_style: dict[str, str] = Field(description="Node styling (color)", serialization_alias="itemStyle")


class SankeyLinkResponse(BaseModel):
    """Sankey diagram link."""

    source: str = Field(description="Source node name")
    target: str = Field(description="Target node name")
    value: int = Field(description="Flow value (count)")


class SankeyFlowResponse(BaseModel):
    """Sankey diagram flow data."""

    nodes: list[dict[str, str | dict[str, str]]] = Field(description="Sankey nodes with styling")
    links: list[dict[str, str | int]] = Field(description="Sankey links with flow values")


class AccuracyByTypeResponse(BaseModel):
    """Signal accuracy by type."""

    signal_type: str = Field(description="Signal type (BUY/SELL)")
    horizon: str = Field(description="Time horizon (1d/5d/20d)")
    hit_rate: float = Field(description="Hit rate (0.0-1.0)")
    executed_count: int = Field(description="Number of executed signals")
    total_count: int = Field(description="Total signals with outcome data")


class AccuracyByTypeListResponse(BaseModel):
    """Accuracy by type list."""

    data: list[AccuracyByTypeResponse] = Field(description="Accuracy breakdown by signal type")
    count: int = Field(description="Number of signal types")


class CalibrationBucketResponse(BaseModel):
    """Calibration curve bucket."""

    confidence_bucket: str = Field(description="Confidence bucket range")
    expected_confidence: float = Field(description="Expected confidence (bucket midpoint)")
    actual_accuracy: float = Field(description="Actual accuracy for bucket")
    sample_count: int = Field(description="Number of samples in bucket")


class CalibrationCurveResponse(BaseModel):
    """Calibration curve data."""

    buckets: list[CalibrationBucketResponse] = Field(description="Calibration buckets")


class TimingAnalysisResponse(BaseModel):
    """Signal timing analysis."""

    avg_execution_delay_hours: float = Field(description="Average delay from signal to execution (hours)")
    by_confidence_bucket: dict[str, float] = Field(description="Average delay by confidence bucket (hours)")


class ExecutionRateResponse(BaseModel):
    """Execution rate by confidence bucket."""

    confidence_bucket: str = Field(description="Confidence bucket range")
    execution_rate: float = Field(description="Execution rate (0.0-1.0)")
    executed_count: int = Field(description="Number of executed signals")
    total_count: int = Field(description="Total signals in bucket")


class ExecutionRateListResponse(BaseModel):
    """Execution rate list."""

    data: list[ExecutionRateResponse] = Field(description="Execution rates by confidence bucket")
    count: int = Field(description="Number of confidence buckets")


class ScreeningCandidateResponse(BaseModel):
    """Single screening candidate."""

    symbol: str = Field(description="Stock symbol")
    name: str = Field(description="Company name")
    sector: str = Field(description="Sector")
    score: float = Field(description="Screening score")
    signal: str = Field(description="Signal (BUY/SELL/HOLD)")
    metrics: dict[str, float] = Field(description="Technical metrics")
    reason: str = Field(description="Screening reason")


class ScreeningRecordResponse(BaseModel):
    """Single screening record."""

    id: str = Field(description="Screening record ID")
    timestamp: datetime = Field(description="Screening timestamp")
    criteria: str = Field(description="Screening criteria type")
    universe: str = Field(description="Universe screened")
    top_symbols: list[str] = Field(description="Top candidate symbols")
    candidates: list[ScreeningCandidateResponse] = Field(description="Full candidate details")
    screened_at: datetime = Field(description="When screening was performed")
    candidate_count: int = Field(description="Number of candidates")


class ScreeningHistoryResponse(BaseModel):
    """Screening history endpoint response."""

    records: list[ScreeningRecordResponse] = Field(description="Screening records")
    total_count: int = Field(description="Total records")
    latest_screening: datetime | None = Field(default=None, description="Latest screening date")


class ScreeningInsightsResponse(BaseModel):
    """Screening insights analytics."""

    total_screenings: int = Field(description="Total screenings performed")
    latest_screening_date: datetime | None = Field(default=None, description="Latest screening timestamp")
    criteria_breakdown: dict[str, int] = Field(description="Count by criteria type")
    sector_distribution: dict[str, int] = Field(description="Top sectors from latest screening")
    avg_score: float = Field(description="Average screening score")
    top_signals: dict[str, int] = Field(description="Signal counts (BUY/SELL/HOLD)")


__all__ = [
    "AccuracyByTypeListResponse",
    "AccuracyByTypeResponse",
    "ActiveDiscoveryResponse",
    "ActiveExecutionGraphsResponse",
    "AnalysesResponse",
    "AnalysisRecordResponse",
    "ApiActiveDiscoveryCandidate",
    "ApiActiveDiscoverySourceDetail",
    "CalibrationBucketResponse",
    "CalibrationCurveResponse",
    "ConfigResponse",
    "CorrelationMatrixResponse",
    "CostAnalyticsSummaryResponse",
    "CostByDimensionListResponse",
    "CostByDimensionResponse",
    "CostTrendPointResponse",
    "CostTrendsResponse",
    "DegradationHistoryResponse",
    "DegradationResponse",
    "DiscoveryInsightsResponse",
    "DiscoveryRecord",
    "DiscoverySourceBreakdown",
    "DiscoverySuccessMetrics",
    "EnrichedTradeResponse",
    "ErrorSummaryResponse",
    "EventResponse",
    "ExecutionGraphDetailResponse",
    "ExecutionGraphHistoryResponse",
    "ExecutionMetricsListResponse",
    "ExecutionRateListResponse",
    "ExecutionRateResponse",
    "FullConfigResponse",
    "GamePlanResponse",
    "HealthResponse",
    "MarketEventsResponse",
    "MetricsSnapshot",
    "PaperTradingValidationResponse",
    "PositionManagementActionResponse",
    "PositionResponse",
    "PositionTimelineResponse",
    "PositionsResponse",
    "RebalanceAllocation",
    "RebalanceCalculation",
    "RebalanceHistoryEntry",
    "RebalanceResponse",
    "RebalancingHistoryResponse",
    "RiskHistoryResponse",
    "RiskReportResponse",
    "SankeyFlowResponse",
    "SankeyLinkResponse",
    "SankeyNodeResponse",
    "ScreeningCandidateResponse",
    "ScreeningHistoryResponse",
    "ScreeningInsightsResponse",
    "ScreeningRecordResponse",
    "SectorRotationResponse",
    "ServiceCheck",
    "ServiceHealthResponse",
    "SignalFlowSummaryResponse",
    "SnapshotRecord",
    "SnapshotsResponse",
    "StateSummaryResponse",
    "SupervisorMetricResponse",
    "SupervisorMetricsListResponse",
    "SupervisorSummaryResponse",
    "TimingAnalysisResponse",
    "TradeResponse",
    "TradesResponse",
    "ValidationCriterionResponse",
    "WatchlistResponse",
    "WorkerPerformanceResponse",
    "WorkerStats",
]
