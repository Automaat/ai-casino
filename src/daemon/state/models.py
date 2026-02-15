"""Pydantic models for daemon state records."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from src.discovery.models import DiscoverySource
from src.screening.screener import ScreeningResult
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


class ScreeningRecord(BaseModel):
    """Record of an after-hours screening run."""

    id: str | None = None
    timestamp: datetime
    criteria: str
    universe: str
    top_symbols: list[str]
    candidates: list[ScreeningResult]
    screened_at: datetime


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


class PeerAnalysisRecord(BaseModel):
    """Record of a deep peer benchmarking analysis run."""

    timestamp: datetime
    symbols_analyzed: list[str]
    rankings: dict[str, int]
    swap_recommendations: list[str]
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
