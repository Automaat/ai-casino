"""Response models for FastAPI daemon API."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health endpoint response."""

    status: str = Field(description="Health status (healthy/degraded)")
    uptime_seconds: float = Field(description="Daemon uptime in seconds")
    running: bool = Field(description="Whether daemon is running")
    last_run: str | None = Field(description="Last analysis run timestamp")


class StateSummaryResponse(BaseModel):
    """State summary endpoint response."""

    total_analyses: int = Field(description="Total analyses performed")
    total_trades: int = Field(description="Total trades executed")
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


class RebalanceAllocation(BaseModel):
    """Rebalance allocation record."""

    symbol: str
    target_weight: float
    current_weight: float
    delta: float
    action: str


class RebalanceResponse(BaseModel):
    """Rebalance endpoint response."""

    timestamp: datetime
    method: str
    allocations: list[RebalanceAllocation]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float


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


__all__ = [
    "AnalysesResponse",
    "AnalysisRecordResponse",
    "ConfigResponse",
    "CorrelationMatrixResponse",
    "DegradationHistoryResponse",
    "DegradationResponse",
    "EventResponse",
    "ExecutionMetricsListResponse",
    "FullConfigResponse",
    "GamePlanResponse",
    "HealthResponse",
    "MarketEventsResponse",
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
    "WatchlistResponse",
]
