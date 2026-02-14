"""SQLAlchemy ORM models for trade history persistence."""

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import DECIMAL, TIMESTAMP, Boolean, Index, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for ORM models."""


class TradeORM(Base):
    """Trade record ORM model."""

    __tablename__ = "trades"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False)
    action: Mapped[str] = mapped_column(String(10), nullable=False)
    entry_price: Mapped[Decimal] = mapped_column(DECIMAL(12, 4), nullable=False)
    exit_price: Mapped[Decimal | None] = mapped_column(DECIMAL(12, 4), nullable=True)
    shares: Mapped[int] = mapped_column(Integer, nullable=False)
    stop_loss_price: Mapped[Decimal] = mapped_column(DECIMAL(12, 4), nullable=False)
    confidence: Mapped[Decimal] = mapped_column(DECIMAL(5, 4), nullable=False)
    risk_level: Mapped[str] = mapped_column(String(10), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default="OPEN")
    pnl: Mapped[Decimal | None] = mapped_column(DECIMAL(12, 4), nullable=True)
    pnl_percent: Mapped[Decimal | None] = mapped_column(DECIMAL(8, 4), nullable=True)
    strategy_name: Mapped[str | None] = mapped_column(String(50), nullable=True)
    broker_order_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    is_paper_trade: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    closed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    __table_args__ = (
        Index("idx_trades_symbol", "symbol"),
        Index("idx_trades_created_at", "created_at"),
        Index("idx_trades_status", "status"),
        Index("idx_trades_is_paper_trade", "is_paper_trade"),
        Index("idx_trades_closed_at", "closed_at"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"TradeORM(id={self.id}, symbol={self.symbol}, action={self.action}, status={self.status})"


class PortfolioSnapshotORM(Base):
    """Portfolio snapshot ORM model."""

    __tablename__ = "portfolio_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    balance: Mapped[Decimal] = mapped_column(DECIMAL(16, 4), nullable=False)
    available_cash: Mapped[Decimal] = mapped_column(DECIMAL(16, 4), nullable=False)
    total_exposure: Mapped[Decimal] = mapped_column(DECIMAL(16, 4), nullable=False)
    portfolio_value: Mapped[Decimal] = mapped_column(DECIMAL(16, 4), nullable=False)
    positions: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    trigger: Mapped[str] = mapped_column(String(50), nullable=False)

    __table_args__ = (Index("idx_portfolio_snapshots_timestamp", "timestamp"),)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"PortfolioSnapshotORM(id={self.id}, timestamp={self.timestamp}, value={self.portfolio_value})"


class AnalysisRecordORM(Base):
    """Analysis record ORM model."""

    __tablename__ = "analysis_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )
    symbol: Mapped[str] = mapped_column(String(10), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    signal: Mapped[str] = mapped_column(String(10), nullable=False)
    confidence: Mapped[Decimal] = mapped_column(DECIMAL(5, 4), nullable=False)
    executed_trade: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    trading_session: Mapped[str] = mapped_column(String(20), nullable=False, server_default="REGULAR")
    is_paper_trade: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    rsi: Mapped[Decimal | None] = mapped_column(DECIMAL(6, 2), nullable=True)
    macd_hist: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 4), nullable=True)
    reasoning: Mapped[list] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    __table_args__ = (
        Index("idx_analysis_records_symbol", "symbol"),
        Index("idx_analysis_records_timestamp", "timestamp"),
        Index("idx_analysis_records_symbol_timestamp", "symbol", "timestamp"),
        Index("idx_analysis_records_signal", "signal"),
        Index("idx_analysis_records_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"AnalysisRecordORM(id={self.id}, symbol={self.symbol}, "
            f"signal={self.signal}, timestamp={self.timestamp})"
        )


class PositionRecordORM(Base):
    """Position record ORM model."""

    __tablename__ = "position_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )
    symbol: Mapped[str] = mapped_column(String(10), nullable=False, unique=True)
    entry_timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    entry_price: Mapped[Decimal] = mapped_column(DECIMAL(12, 4), nullable=False)
    entry_signal: Mapped[str] = mapped_column(String(10), nullable=False)
    entry_confidence: Mapped[Decimal] = mapped_column(DECIMAL(5, 4), nullable=False)
    current_qty: Mapped[Decimal] = mapped_column(DECIMAL(12, 4), nullable=False)
    current_stop_loss: Mapped[Decimal] = mapped_column(DECIMAL(12, 4), nullable=False)
    initial_stop_loss: Mapped[Decimal] = mapped_column(DECIMAL(12, 4), nullable=False)
    stop_loss_order_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    profit_targets: Mapped[list] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    days_held: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    last_updated: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    trailing_stop_activated: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    breakeven_activated: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    high_water_mark: Mapped[Decimal | None] = mapped_column(DECIMAL(12, 4), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    __table_args__ = (
        Index("idx_position_records_symbol", "symbol"),
        Index("idx_position_records_last_updated", "last_updated"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"PositionRecordORM(id={self.id}, symbol={self.symbol}, qty={self.current_qty})"


class PositionManagementActionORM(Base):
    """Position management action ORM model."""

    __tablename__ = "position_management_actions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )
    symbol: Mapped[str] = mapped_column(String(10), nullable=False)
    action_type: Mapped[str] = mapped_column(String(50), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    old_stop_loss: Mapped[Decimal | None] = mapped_column(DECIMAL(12, 4), nullable=True)
    new_stop_loss: Mapped[Decimal | None] = mapped_column(DECIMAL(12, 4), nullable=True)
    qty_sold: Mapped[Decimal | None] = mapped_column(DECIMAL(12, 4), nullable=True)
    price: Mapped[Decimal] = mapped_column(DECIMAL(12, 4), nullable=False)
    reason: Mapped[str] = mapped_column(String, nullable=False)
    executed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    order_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    __table_args__ = (
        Index("idx_position_actions_symbol", "symbol"),
        Index("idx_position_actions_timestamp", "timestamp"),
        Index("idx_position_actions_symbol_timestamp", "symbol", "timestamp"),
        Index("idx_position_actions_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"PositionManagementActionORM(id={self.id}, symbol={self.symbol}, "
            f"action={self.action_type}, timestamp={self.timestamp})"
        )


class DiscoveryHistoryRecordORM(Base):
    """Discovery history record ORM model."""

    __tablename__ = "discovery_history"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )
    symbol: Mapped[str] = mapped_column(String(10), nullable=False)
    discovered_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    composite_score: Mapped[Decimal] = mapped_column(DECIMAL(5, 4), nullable=False)
    sources: Mapped[list] = mapped_column(JSONB, nullable=False)
    added_to_watchlist: Mapped[bool] = mapped_column(Boolean, nullable=False)
    ttl_expires_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    first_signal: Mapped[str | None] = mapped_column(String(10), nullable=True)
    first_signal_date: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    outcome_7d: Mapped[Decimal | None] = mapped_column(DECIMAL(8, 4), nullable=True)
    outcome_30d: Mapped[Decimal | None] = mapped_column(DECIMAL(8, 4), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    __table_args__ = (
        Index("idx_discovery_history_symbol", "symbol"),
        Index("idx_discovery_history_discovered_at", "discovered_at"),
        Index("idx_discovery_history_ttl_expires_at", "ttl_expires_at"),
        Index("idx_discovery_history_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"DiscoveryHistoryRecordORM(id={self.id}, symbol={self.symbol}, score={self.composite_score})"


class TearSheetORM(Base):
    """TearSheet ORM model."""

    __tablename__ = "tearsheets"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
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
    monthly_returns: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    benchmark_symbol: Mapped[str | None] = mapped_column(String(10), nullable=True)
    benchmark_cagr: Mapped[Decimal | None] = mapped_column(DECIMAL(8, 4), nullable=True)
    benchmark_sharpe: Mapped[Decimal | None] = mapped_column(DECIMAL(8, 4), nullable=True)
    alpha: Mapped[Decimal | None] = mapped_column(DECIMAL(8, 4), nullable=True)
    beta: Mapped[Decimal | None] = mapped_column(DECIMAL(8, 4), nullable=True)
    html_report_path: Mapped[str] = mapped_column(String, nullable=False)
    generated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    __table_args__ = (
        Index("idx_tearsheets_symbol", "symbol"),
        Index("idx_tearsheets_generated_at", "generated_at"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"TearSheetORM(id={self.id}, symbol={self.symbol}, generated_at={self.generated_at})"


class SignalOutcomeORM(Base):
    """Signal outcome ORM model for persistent learning."""

    __tablename__ = "signal_outcomes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )
    symbol: Mapped[str] = mapped_column(String(10), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    signal: Mapped[str] = mapped_column(String(10), nullable=False)
    confidence: Mapped[Decimal] = mapped_column(DECIMAL(5, 4), nullable=False)
    price_at_signal: Mapped[Decimal] = mapped_column(DECIMAL(12, 4), nullable=False)
    strategy_used: Mapped[str | None] = mapped_column(String(50), nullable=True)
    regime: Mapped[str | None] = mapped_column(String(30), nullable=True)
    trading_session: Mapped[str] = mapped_column(String(20), nullable=False, server_default="'REGULAR'")
    technical_signal: Mapped[str | None] = mapped_column(String(10), nullable=True)
    sentiment_signal: Mapped[str | None] = mapped_column(String(10), nullable=True)
    news_signal: Mapped[str | None] = mapped_column(String(10), nullable=True)
    technical_reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    sentiment_reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    news_reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    price_at_1d: Mapped[Decimal | None] = mapped_column(DECIMAL(12, 4), nullable=True)
    price_at_5d: Mapped[Decimal | None] = mapped_column(DECIMAL(12, 4), nullable=True)
    price_at_20d: Mapped[Decimal | None] = mapped_column(DECIMAL(12, 4), nullable=True)
    actual_exit_price: Mapped[Decimal | None] = mapped_column(DECIMAL(12, 4), nullable=True)
    actual_exit_date: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    outcome_updated_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    __table_args__ = (
        Index("idx_signal_outcomes_symbol", "symbol"),
        Index("idx_signal_outcomes_timestamp", "timestamp", postgresql_ops={"timestamp": "DESC"}),
        Index("idx_signal_outcomes_regime", "regime", postgresql_where=text("regime IS NOT NULL")),
        Index(
            "idx_signal_outcomes_regime_signal",
            "regime",
            "signal",
            postgresql_where=text("regime IS NOT NULL"),
        ),
        Index(
            "idx_signal_outcomes_needs_update_1d",
            "timestamp",
            postgresql_where=text("price_at_1d IS NULL"),
        ),
        Index(
            "idx_signal_outcomes_needs_update_5d",
            "timestamp",
            postgresql_where=text("price_at_5d IS NULL"),
        ),
        Index(
            "idx_signal_outcomes_needs_update_20d",
            "timestamp",
            postgresql_where=text("price_at_20d IS NULL"),
        ),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"SignalOutcomeORM(id={self.id}, symbol={self.symbol}, "
            f"signal={self.signal}, timestamp={self.timestamp})"
        )


class ExecutionGraphORM(Base):
    """Execution graph ORM model."""

    __tablename__ = "execution_graphs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )
    workflow_id: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    symbol: Mapped[str | None] = mapped_column(String(10), nullable=True)
    graph_jsonb: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    __table_args__ = (
        Index("idx_execution_graphs_workflow_id", "workflow_id"),
        Index("idx_execution_graphs_symbol", "symbol"),
        Index("idx_execution_graphs_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ExecutionGraphORM(id={self.id}, workflow_id={self.workflow_id}, symbol={self.symbol})"


class DaemonMetadataORM(Base):
    """Daemon metadata ORM model."""

    __tablename__ = "daemon_metadata"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[dict] = mapped_column(JSONB, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    __table_args__ = (Index("idx_daemon_metadata_updated_at", "updated_at"),)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"DaemonMetadataORM(key={self.key})"


class OptimizationRecordORM(Base):
    """Optimization record ORM model."""

    __tablename__ = "optimization_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    symbols_optimized: Mapped[list] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    symbols_skipped: Mapped[list] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    total_time_seconds: Mapped[Decimal] = mapped_column(DECIMAL(10, 2), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    __table_args__ = (
        Index("idx_optimization_records_timestamp", "timestamp"),
        Index("idx_optimization_records_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"OptimizationRecordORM(id={self.id}, timestamp={self.timestamp})"


class RebalancingRecordORM(Base):
    """Rebalancing record ORM model."""

    __tablename__ = "rebalancing_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    method: Mapped[str] = mapped_column(String(50), nullable=False)
    allocations: Mapped[dict] = mapped_column(JSONB, nullable=False)
    expected_return: Mapped[Decimal] = mapped_column(DECIMAL(8, 4), nullable=False)
    expected_volatility: Mapped[Decimal] = mapped_column(DECIMAL(8, 4), nullable=False)
    sharpe_ratio: Mapped[Decimal] = mapped_column(DECIMAL(8, 4), nullable=False)
    rebalances_executed: Mapped[int] = mapped_column(Integer, nullable=False)
    rebalances_pending: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    __table_args__ = (
        Index("idx_rebalancing_records_timestamp", "timestamp"),
        Index("idx_rebalancing_records_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"RebalancingRecordORM(id={self.id}, timestamp={self.timestamp})"


class SectorRotationRecordORM(Base):
    """Sector rotation record ORM model."""

    __tablename__ = "sector_rotation_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    leading_sectors: Mapped[list] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    lagging_sectors: Mapped[list] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    sector_strengths: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    sector_momenta: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    flagged_positions: Mapped[list] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    __table_args__ = (
        Index("idx_sector_rotation_records_timestamp", "timestamp"),
        Index("idx_sector_rotation_records_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"SectorRotationRecordORM(id={self.id}, timestamp={self.timestamp})"


class PeerAnalysisRecordORM(Base):
    """Peer analysis record ORM model."""

    __tablename__ = "peer_analysis_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    symbols_analyzed: Mapped[list] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    rankings: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    swap_recommendations: Mapped[list] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    total_peers: Mapped[int] = mapped_column(Integer, nullable=False)
    total_duration_seconds: Mapped[Decimal] = mapped_column(DECIMAL(10, 2), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    __table_args__ = (
        Index("idx_peer_analysis_records_timestamp", "timestamp"),
        Index("idx_peer_analysis_records_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"PeerAnalysisRecordORM(id={self.id}, timestamp={self.timestamp})"


class CorrelationAuditRecordORM(Base):
    """Correlation audit record ORM model."""

    __tablename__ = "correlation_audit_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    num_positions: Mapped[int] = mapped_column(Integer, nullable=False)
    num_correlated_pairs: Mapped[int] = mapped_column(Integer, nullable=False)
    max_correlation: Mapped[Decimal] = mapped_column(DECIMAL(5, 4), nullable=False)
    avg_correlation: Mapped[Decimal] = mapped_column(DECIMAL(5, 4), nullable=False)
    diversification_ratio: Mapped[Decimal] = mapped_column(DECIMAL(8, 4), nullable=False)
    num_substitutions: Mapped[int] = mapped_column(Integer, nullable=False)
    total_duration_seconds: Mapped[Decimal] = mapped_column(DECIMAL(10, 2), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    __table_args__ = (
        Index("idx_correlation_audit_records_timestamp", "timestamp"),
        Index("idx_correlation_audit_records_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"CorrelationAuditRecordORM(id={self.id}, timestamp={self.timestamp})"


class RiskReportRecordORM(Base):
    """Risk report record ORM model."""

    __tablename__ = "risk_report_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
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
        server_default=text("NOW()"),
    )

    __table_args__ = (
        Index("idx_risk_report_records_timestamp", "timestamp"),
        Index("idx_risk_report_records_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"RiskReportRecordORM(id={self.id}, timestamp={self.timestamp})"


class MonteCarloRecordORM(Base):
    """Monte Carlo record ORM model."""

    __tablename__ = "monte_carlo_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
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
    portfolio_symbols: Mapped[list] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    total_market_value: Mapped[Decimal] = mapped_column(DECIMAL(16, 4), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    __table_args__ = (
        Index("idx_monte_carlo_records_timestamp", "timestamp"),
        Index("idx_monte_carlo_records_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"MonteCarloRecordORM(id={self.id}, timestamp={self.timestamp})"


class PrefetchRecordORM(Base):
    """Prefetch record ORM model."""

    __tablename__ = "prefetch_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    symbols_prefetched: Mapped[int] = mapped_column(Integer, nullable=False)
    symbols_failed: Mapped[int] = mapped_column(Integer, nullable=False)
    finbert_ready: Mapped[bool] = mapped_column(Boolean, nullable=False)
    total_duration_seconds: Mapped[Decimal] = mapped_column(DECIMAL(10, 2), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    __table_args__ = (
        Index("idx_prefetch_records_timestamp", "timestamp"),
        Index("idx_prefetch_records_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"PrefetchRecordORM(id={self.id}, timestamp={self.timestamp})"


class ScreeningRecordORM(Base):
    """Screening record ORM model."""

    __tablename__ = "screening_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    criteria: Mapped[str] = mapped_column(String(100), nullable=False)
    universe: Mapped[str] = mapped_column(String(50), nullable=False)
    top_symbols: Mapped[list] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    candidates: Mapped[list] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    screened_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    __table_args__ = (
        Index("idx_screening_records_timestamp", "timestamp"),
        Index("idx_screening_records_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ScreeningRecordORM(id={self.id}, timestamp={self.timestamp})"


class EarningsCalendarRecordORM(Base):
    """Earnings calendar record ORM model."""

    __tablename__ = "earnings_calendar_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    events: Mapped[list] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    symbols_fetched: Mapped[int] = mapped_column(Integer, nullable=False)
    symbols_failed: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    __table_args__ = (
        Index("idx_earnings_calendar_records_timestamp", "timestamp"),
        Index("idx_earnings_calendar_records_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"EarningsCalendarRecordORM(id={self.id}, timestamp={self.timestamp})"


class ProfilingRecordORM(Base):
    """Profiling record ORM model."""

    __tablename__ = "profiling_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
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
        server_default=text("NOW()"),
    )

    __table_args__ = (
        Index("idx_profiling_records_timestamp", "timestamp"),
        Index("idx_profiling_records_created_at", "created_at"),
        Index("idx_profiling_records_cycle_number", "cycle_number"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ProfilingRecordORM(id={self.id}, cycle={self.cycle_number})"


class GamePlanRecordORM(Base):
    """Game plan record ORM model."""

    __tablename__ = "game_plan_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    priority_symbols: Mapped[list] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    risk_stance: Mapped[str] = mapped_column(String(20), nullable=False)
    sector_focus: Mapped[list] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    __table_args__ = (
        Index("idx_game_plan_records_timestamp", "timestamp"),
        Index("idx_game_plan_records_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"GamePlanRecordORM(id={self.id}, timestamp={self.timestamp})"


class DegradationRecordORM(Base):
    """Degradation record ORM model."""

    __tablename__ = "degradation_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    tier: Mapped[str] = mapped_column(String(20), nullable=False)
    unavailable_services: Mapped[list] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    confidence_adjustment: Mapped[Decimal] = mapped_column(DECIMAL(5, 4), nullable=False)
    halt_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    __table_args__ = (
        Index("idx_degradation_records_timestamp", "timestamp"),
        Index("idx_degradation_records_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"DegradationRecordORM(id={self.id}, tier={self.tier})"


class ActiveDiscoveryCandidateORM(Base):
    """Active discovery candidate ORM model."""

    __tablename__ = "active_discovery_candidates"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )
    symbol: Mapped[str] = mapped_column(String(10), nullable=False, unique=True)
    discovered_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    composite_score: Mapped[Decimal] = mapped_column(DECIMAL(5, 4), nullable=False)
    sources: Mapped[list] = mapped_column(JSONB, nullable=False)
    ttl_expires_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    __table_args__ = (
        Index("idx_active_discovery_candidates_symbol", "symbol"),
        Index("idx_active_discovery_candidates_ttl_expires_at", "ttl_expires_at"),
        Index("idx_active_discovery_candidates_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ActiveDiscoveryCandidateORM(id={self.id}, symbol={self.symbol})"


class ExecutionMetricORM(Base):
    """Execution metric ORM model."""

    __tablename__ = "execution_metrics"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
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
    broker: Mapped[str] = mapped_column(String(50), nullable=False, server_default="'alpaca'")
    venue: Mapped[str | None] = mapped_column(String(50), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
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


class RiskAuditORM(Base):
    """Risk audit log entry."""

    __tablename__ = "risk_audit"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False)
    action: Mapped[str] = mapped_column(String(10), nullable=False)
    current_price: Mapped[Decimal] = mapped_column(DECIMAL(12, 4), nullable=False)

    approved: Mapped[bool] = mapped_column(Boolean, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(10), nullable=False)
    risk_score: Mapped[Decimal] = mapped_column(DECIMAL(5, 4), nullable=False)
    confidence: Mapped[Decimal] = mapped_column(DECIMAL(5, 4), nullable=False)

    recommended_shares: Mapped[int] = mapped_column(Integer, nullable=False)
    position_value: Mapped[Decimal] = mapped_column(DECIMAL(12, 2), nullable=False)
    risk_amount: Mapped[Decimal] = mapped_column(DECIMAL(12, 2), nullable=False)
    risk_percent: Mapped[Decimal] = mapped_column(DECIMAL(5, 4), nullable=False)

    stop_loss_price: Mapped[Decimal] = mapped_column(DECIMAL(12, 4), nullable=False)

    warnings: Mapped[list[str]] = mapped_column(
        ARRAY(Text),
        nullable=False,
        server_default=text("'{}'::text[]"),
    )

    portfolio_var_95: Mapped[Decimal | None] = mapped_column(DECIMAL(5, 4))
    portfolio_cvar_99: Mapped[Decimal | None] = mapped_column(DECIMAL(5, 4))
    portfolio_cdar_95: Mapped[Decimal | None] = mapped_column(DECIMAL(5, 4))

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=text("NOW()"),
    )

    __table_args__ = (
        Index(
            "idx_risk_audit_timestamp",
            "timestamp",
            postgresql_using="btree",
            postgresql_ops={"timestamp": "DESC"},
        ),
        Index("idx_risk_audit_symbol", "symbol"),
        Index(
            "idx_risk_audit_symbol_timestamp",
            "symbol",
            "timestamp",
            postgresql_using="btree",
            postgresql_ops={"timestamp": "DESC"},
        ),
        Index("idx_risk_audit_approved", "approved"),
        Index("idx_risk_audit_risk_level", "risk_level"),
        Index(
            "idx_risk_audit_violations",
            "symbol",
            "timestamp",
            postgresql_where=text("approved = false"),
            postgresql_ops={"timestamp": "DESC"},
        ),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"RiskAuditORM(id={self.id}, symbol={self.symbol}, "
            f"action={self.action}, approved={self.approved})"
        )


class CoordinatorMetricsORM(Base):
    """Coordinator decision cycle metrics."""

    __tablename__ = "coordinator_metrics"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=text("NOW()"),
        nullable=False,
    )
    cycle_num: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    symbols_analyzed: Mapped[list[str]] = mapped_column(
        ARRAY(Text),
        nullable=False,
        server_default=text("'{}'::text[]"),
    )
    tool_calls_made: Mapped[int] = mapped_column(Integer, nullable=False)
    trades_proposed: Mapped[int] = mapped_column(Integer, nullable=False)
    trades_executed: Mapped[int] = mapped_column(Integer, nullable=False)
    trades_pending: Mapped[int] = mapped_column(Integer, nullable=False)
    game_plan_generated: Mapped[bool] = mapped_column(Boolean, nullable=False)
    cycle_duration_seconds: Mapped[Decimal] = mapped_column(DECIMAL(12, 4), nullable=False)
    patterns_detected: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        Index(
            "idx_coordinator_metrics_timestamp",
            "timestamp",
            postgresql_using="btree",
            postgresql_ops={"timestamp": "DESC"},
        ),
        Index("idx_coordinator_metrics_cycle_num", "cycle_num"),
        Index(
            "idx_coordinator_metrics_cycle_timestamp",
            "cycle_num",
            "timestamp",
            postgresql_using="btree",
            postgresql_ops={"timestamp": "DESC"},
        ),
        Index("idx_coordinator_metrics_game_plan", "game_plan_generated"),
        Index(
            "idx_coordinator_metrics_trades_executed",
            "trades_executed",
            postgresql_where=text("trades_executed > 0"),
        ),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"CoordinatorMetricsORM(id={self.id}, cycle_num={self.cycle_num}, "
            f"trades_executed={self.trades_executed})"
        )


class SupervisorMetricsORM(Base):
    """Supervisor routing and worker execution metrics."""

    __tablename__ = "supervisor_metrics"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()")
    )
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("NOW()"))

    # Identifiers
    workflow_id: Mapped[str] = mapped_column(String(100), nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)

    # Routing decision
    required_analyses: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    optional_analyses: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    skip_analyses: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
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
    worker_timings: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    worker_errors: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))

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
