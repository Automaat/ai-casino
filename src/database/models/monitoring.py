"""ORM models for monitoring operations."""

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import DECIMAL, TIMESTAMP, Boolean, Index, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.database.models.base import Base


class HealthReportORM(Base):
    """Health report ORM model."""

    __tablename__ = "health_reports"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    overall_status: Mapped[str] = mapped_column(String(20), nullable=False)
    service_checks: Mapped[list] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    cleanup_results: Mapped[list] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    total_duration_ms: Mapped[Decimal] = mapped_column(DECIMAL(10, 2), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    __table_args__ = (
        Index("idx_health_reports_timestamp", "timestamp"),
        Index("idx_health_reports_status", "overall_status"),
        Index("idx_health_reports_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"HealthReportORM(id={self.id}, status={self.overall_status}, timestamp={self.timestamp})"


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
