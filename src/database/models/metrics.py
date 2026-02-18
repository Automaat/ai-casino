"""ORM models for metrics operations."""

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal

from sqlalchemy import DATE, DECIMAL, TIMESTAMP, Boolean, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.database.models.base import Base
from src.database.types import ARRAY, JSONB, UUID


class SupervisorMetricsORM(Base):
    """Supervisor routing and worker execution metrics."""

    __tablename__ = "supervisor_metrics"

    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=lambda: datetime.now(UTC))

    # Identifiers
    workflow_id: Mapped[str] = mapped_column(String(100), nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)

    # Routing decision
    required_analyses: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    optional_analyses: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    skip_analyses: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    routing_reasoning: Mapped[str] = mapped_column(Text, nullable=False)

    # Execution metrics
    total_workers: Mapped[int] = mapped_column(Integer, nullable=False)
    required_workers: Mapped[int] = mapped_column(Integer, nullable=False)
    optional_workers: Mapped[int] = mapped_column(Integer, nullable=False)
    successful_workers: Mapped[int] = mapped_column(Integer, nullable=False)
    failed_workers: Mapped[int] = mapped_column(Integer, nullable=False)

    # Timing metrics in milliseconds
    routing_decision_ms: Mapped[Decimal] = mapped_column(DECIMAL(10, 2), nullable=False)
    group1_execution_ms: Mapped[Decimal] = mapped_column(DECIMAL(10, 2), nullable=False)
    research_execution_ms: Mapped[Decimal] = mapped_column(DECIMAL(10, 2), nullable=False)
    total_supervisor_overhead_ms: Mapped[Decimal] = mapped_column(DECIMAL(10, 2), nullable=False)

    # Worker execution details
    worker_timings: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    worker_errors: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # LLM usage metrics
    total_llm_calls: Mapped[int] = mapped_column(Integer, nullable=False)
    total_cost_usd: Mapped[Decimal] = mapped_column(DECIMAL(10, 4), nullable=False)
    planning_fallback_used: Mapped[bool] = mapped_column(Boolean, nullable=False)
    synthesis_fallback_used: Mapped[bool] = mapped_column(Boolean, nullable=False)

    # Synthesis results
    confidence_adjustment: Mapped[Decimal] = mapped_column(DECIMAL(5, 4), nullable=False)
    synthesis_reasoning: Mapped[str] = mapped_column(Text, nullable=False)

    # Efficiency
    parallel_efficiency_percent: Mapped[Decimal] = mapped_column(DECIMAL(5, 2), nullable=False)
    timeout_triggered: Mapped[bool] = mapped_column(Boolean, nullable=False)

    __table_args__ = (
        Index("idx_supervisor_metrics_symbol", "symbol"),
        Index("idx_supervisor_metrics_timestamp", "timestamp", postgresql_using="btree"),
        Index("idx_supervisor_metrics_workflow_id", "workflow_id"),
        Index("idx_supervisor_metrics_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"SupervisorMetricsORM(id={self.id}, workflow_id={self.workflow_id}, symbol={self.symbol})"


class WorkflowExecutionMetricsORM(Base):
    """Workflow execution metrics ORM model."""

    __tablename__ = "workflow_execution_metrics"

    workflow_id: Mapped[uuid.UUID] = mapped_column(
        UUID,
        primary_key=True,
    )
    symbol: Mapped[str] = mapped_column(String(10), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    total_latency_ms: Mapped[Decimal] = mapped_column(DECIMAL(12, 2), nullable=False)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    total_input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    total_output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    total_estimated_cost_usd: Mapped[Decimal] = mapped_column(DECIMAL(12, 6), nullable=False)
    llm_calls: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    sub_operations: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    agent_timings: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    pipeline_stages: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    __table_args__ = (
        Index("idx_workflow_execution_metrics_symbol", "symbol"),
        Index("idx_workflow_execution_metrics_timestamp", "timestamp"),
        Index("idx_workflow_execution_metrics_symbol_timestamp", "symbol", "timestamp"),
        Index("idx_workflow_execution_metrics_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"WorkflowExecutionMetricsORM(workflow_id={self.workflow_id}, symbol={self.symbol}, "
            f"cost=${self.total_estimated_cost_usd})"
        )


class MonteCarloRecordORM(Base):
    """Monte Carlo record ORM model."""

    __tablename__ = "monte_carlo_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID,
        primary_key=True,
        default=uuid.uuid4,
    )
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    simulation_method: Mapped[str] = mapped_column(String(50), nullable=False)
    num_simulations: Mapped[int] = mapped_column(Integer, nullable=False)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    prob_loss_gt_threshold: Mapped[Decimal] = mapped_column(DECIMAL(5, 4), nullable=False)
    expected_worst_drawdown: Mapped[Decimal] = mapped_column(DECIMAL(8, 4), nullable=False)
    var_95: Mapped[Decimal] = mapped_column(DECIMAL(12, 4), nullable=False)
    cvar_95: Mapped[Decimal] = mapped_column(DECIMAL(12, 4), nullable=False)
    median_recovery_days: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 2), nullable=True)
    exceeds_risk_tolerance: Mapped[bool] = mapped_column(Boolean, nullable=False)
    alert_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    portfolio_symbols: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    total_market_value: Mapped[Decimal] = mapped_column(DECIMAL(16, 4), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    __table_args__ = (
        Index("idx_monte_carlo_records_timestamp", "timestamp"),
        Index("idx_monte_carlo_records_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"MonteCarloRecordORM(id={self.id}, timestamp={self.timestamp})"


class TearSheetORM(Base):
    """TearSheet ORM model."""

    __tablename__ = "tearsheets"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID,
        primary_key=True,
        default=uuid.uuid4,
    )
    symbol: Mapped[str] = mapped_column(String(10), nullable=False)
    start_date: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    end_date: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    cagr: Mapped[Decimal | None] = mapped_column(DECIMAL(8, 4), nullable=True)
    sharpe_ratio: Mapped[Decimal | None] = mapped_column(DECIMAL(8, 4), nullable=True)
    sortino_ratio: Mapped[Decimal | None] = mapped_column(DECIMAL(8, 4), nullable=True)
    calmar_ratio: Mapped[Decimal | None] = mapped_column(DECIMAL(8, 4), nullable=True)
    max_drawdown: Mapped[Decimal | None] = mapped_column(DECIMAL(8, 4), nullable=True)
    max_drawdown_duration_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    volatility_annual: Mapped[Decimal | None] = mapped_column(DECIMAL(8, 4), nullable=True)
    win_rate: Mapped[Decimal | None] = mapped_column(DECIMAL(8, 4), nullable=True)
    profit_factor: Mapped[Decimal | None] = mapped_column(DECIMAL(8, 4), nullable=True)
    avg_win: Mapped[Decimal | None] = mapped_column(DECIMAL(12, 4), nullable=True)
    avg_loss: Mapped[Decimal | None] = mapped_column(DECIMAL(12, 4), nullable=True)
    best_day: Mapped[Decimal | None] = mapped_column(DECIMAL(8, 4), nullable=True)
    worst_day: Mapped[Decimal | None] = mapped_column(DECIMAL(8, 4), nullable=True)
    monthly_returns: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    benchmark_symbol: Mapped[str | None] = mapped_column(String(10), nullable=True)
    benchmark_cagr: Mapped[Decimal | None] = mapped_column(DECIMAL(8, 4), nullable=True)
    benchmark_sharpe: Mapped[Decimal | None] = mapped_column(DECIMAL(8, 4), nullable=True)
    alpha: Mapped[Decimal | None] = mapped_column(DECIMAL(8, 4), nullable=True)
    beta: Mapped[Decimal | None] = mapped_column(DECIMAL(8, 4), nullable=True)
    html_report_path: Mapped[str] = mapped_column(String, nullable=False)
    generated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    __table_args__ = (
        Index("idx_tearsheets_symbol", "symbol"),
        Index("idx_tearsheets_generated_at", "generated_at"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"TearSheetORM(id={self.id}, symbol={self.symbol}, generated_at={self.generated_at})"


class PaperTradingReportORM(Base):
    """Paper trading validation report ORM model."""

    __tablename__ = "paper_trading_reports"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID,
        primary_key=True,
        default=uuid.uuid4,
    )
    assessment_date: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    ready_for_live: Mapped[bool] = mapped_column(Boolean, nullable=False)
    paper_trading_duration_days: Mapped[int] = mapped_column(Integer, nullable=False)
    total_paper_trades: Mapped[int] = mapped_column(Integer, nullable=False)
    criteria: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    total_pnl: Mapped[Decimal] = mapped_column(DECIMAL(16, 4), nullable=False)
    sharpe_ratio: Mapped[Decimal] = mapped_column(DECIMAL(8, 4), nullable=False)
    sortino_ratio: Mapped[Decimal] = mapped_column(DECIMAL(8, 4), nullable=False)
    max_drawdown: Mapped[Decimal] = mapped_column(DECIMAL(8, 4), nullable=False)
    win_rate: Mapped[Decimal] = mapped_column(DECIMAL(5, 4), nullable=False)
    simulated_live: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    recommendations: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    __table_args__ = (
        Index("idx_paper_trading_reports_date", "assessment_date"),
        Index("idx_paper_trading_reports_ready", "ready_for_live"),
        Index("idx_paper_trading_reports_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"PaperTradingReportORM(id={self.id}, date={self.assessment_date}, ready={self.ready_for_live})"
        )


class ExecutionMetricORM(Base):
    """Execution metric ORM model."""

    __tablename__ = "execution_metrics"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID,
        primary_key=True,
        default=uuid.uuid4,
    )
    order_id: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(DECIMAL(12, 4), nullable=False)
    requested_price: Mapped[Decimal] = mapped_column(DECIMAL(12, 4), nullable=False)
    filled_price: Mapped[Decimal | None] = mapped_column(DECIMAL(12, 4), nullable=True)
    submitted_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    filled_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    execution_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    slippage_bps: Mapped[Decimal | None] = mapped_column(DECIMAL(8, 2), nullable=True)
    broker: Mapped[str] = mapped_column(String(50), nullable=False, default="alpaca")
    venue: Mapped[str | None] = mapped_column(String(50), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    __table_args__ = (
        Index("idx_execution_metrics_symbol", "symbol"),
        Index("idx_execution_metrics_submitted_at", "submitted_at"),
        Index("idx_execution_metrics_broker", "broker"),
        Index("idx_execution_metrics_status", "status"),
        Index("idx_execution_metrics_symbol_broker", "symbol", "broker"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"ExecutionMetricORM(order_id={self.order_id}, "
            f"symbol={self.symbol}, slippage_bps={self.slippage_bps})"
        )


class ProfilingRecordORM(Base):
    """Profiling record ORM model."""

    __tablename__ = "profiling_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID,
        primary_key=True,
        default=uuid.uuid4,
    )
    cycle_number: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    duration_seconds: Mapped[Decimal] = mapped_column(DECIMAL(10, 4), nullable=False)
    profiling_overhead_percent: Mapped[Decimal] = mapped_column(DECIMAL(5, 2), nullable=False)
    top_function: Mapped[str | None] = mapped_column(String(200), nullable=True)
    top_function_cumtime: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 4), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    __table_args__ = (
        Index("idx_profiling_records_timestamp", "timestamp"),
        Index("idx_profiling_records_created_at", "created_at"),
        Index("idx_profiling_records_cycle_number", "cycle_number"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ProfilingRecordORM(id={self.id}, cycle={self.cycle_number})"


class TradeJournalORM(Base):
    """Trade journal ORM model."""

    __tablename__ = "trade_journals"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID,
        primary_key=True,
        default=uuid.uuid4,
    )
    date: Mapped[date] = mapped_column(DATE, nullable=False, unique=True)
    outcomes: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    winners: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    losers: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    lessons: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    tomorrows_focus: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    overall_assessment: Mapped[str] = mapped_column(Text, nullable=False)
    markdown_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    total_signals: Mapped[int] = mapped_column(Integer, nullable=False)
    correct_signals: Mapped[int] = mapped_column(Integer, nullable=False)
    accuracy_pct: Mapped[Decimal] = mapped_column(DECIMAL(5, 2), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    __table_args__ = (
        Index("idx_trade_journals_date", "date", unique=True),
        Index("idx_trade_journals_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"TradeJournalORM(id={self.id}, date={self.date}, accuracy={self.accuracy_pct}%)"


class PortfolioHealthRecordORM(Base):
    """Portfolio health check record ORM model."""

    __tablename__ = "portfolio_health_reports"

    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, default=uuid.uuid4)
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    total_positions: Mapped[int] = mapped_column(Integer, nullable=False)
    portfolio_value: Mapped[Decimal] = mapped_column(DECIMAL(16, 4), nullable=False)
    cash_percent: Mapped[Decimal] = mapped_column(DECIMAL(8, 4), nullable=False)
    max_concentration_percent: Mapped[Decimal] = mapped_column(DECIMAL(8, 4), nullable=False)
    max_concentration_symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    total_pnl_percent: Mapped[Decimal] = mapped_column(DECIMAL(8, 4), nullable=False)
    biggest_drawdown_symbol: Mapped[str | None] = mapped_column(String(20), nullable=True)
    biggest_drawdown_percent: Mapped[Decimal] = mapped_column(DECIMAL(8, 4), nullable=False)
    health_status: Mapped[str] = mapped_column(String(20), nullable=False)
    recommendations: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    constraints: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    __table_args__ = (
        Index("idx_portfolio_health_timestamp", "timestamp"),
        Index("idx_portfolio_health_status", "health_status"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"PortfolioHealthRecordORM(id={self.id}, status={self.health_status})"


class RiskReportRecordORM(Base):
    """Risk report record ORM model."""

    __tablename__ = "risk_report_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID,
        primary_key=True,
        default=uuid.uuid4,
    )
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    var_95: Mapped[Decimal] = mapped_column(DECIMAL(12, 4), nullable=False)
    var_99: Mapped[Decimal] = mapped_column(DECIMAL(12, 4), nullable=False)
    cvar_95: Mapped[Decimal] = mapped_column(DECIMAL(12, 4), nullable=False)
    cvar_99: Mapped[Decimal] = mapped_column(DECIMAL(12, 4), nullable=False)
    cdar_95: Mapped[Decimal] = mapped_column(DECIMAL(12, 4), nullable=False)
    max_drawdown: Mapped[Decimal] = mapped_column(DECIMAL(8, 4), nullable=False)
    portfolio_volatility: Mapped[Decimal] = mapped_column(DECIMAL(12, 4), nullable=False)
    current_exposure_percent: Mapped[Decimal] = mapped_column(DECIMAL(8, 4), nullable=False)
    num_positions: Mapped[int] = mapped_column(Integer, nullable=False)
    var_limit_breached: Mapped[bool] = mapped_column(Boolean, nullable=False)
    cvar_limit_breached: Mapped[bool] = mapped_column(Boolean, nullable=False)
    risk_status: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    __table_args__ = (
        Index("idx_risk_report_records_timestamp", "timestamp"),
        Index("idx_risk_report_records_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"RiskReportRecordORM(id={self.id}, timestamp={self.timestamp})"
