"""ORM models for portfolio operations."""

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import DECIMAL, TIMESTAMP, Index, Integer, String, text
from sqlalchemy.orm import Mapped, mapped_column

from src.database.models.base import Base
from src.database.types import JSONB, UUID


class OptimizationRecordORM(Base):
    """Optimization record ORM model."""

    __tablename__ = "optimization_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID,
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
        UUID,
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
        UUID,
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


class SectorAttributionRecordORM(Base):
    """Sector attribution record ORM model."""

    __tablename__ = "sector_attribution"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID,
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    total_portfolio_value: Mapped[Decimal] = mapped_column(DECIMAL(16, 4), nullable=False)
    benchmark_name: Mapped[str] = mapped_column(String(20), nullable=False, server_default="'SPY'")
    contributions: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    __table_args__ = (
        Index("idx_sector_attribution_timestamp", "timestamp", postgresql_ops={"timestamp": "DESC"}),
        Index("idx_sector_attribution_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"SectorAttributionRecordORM(id={self.id}, timestamp={self.timestamp})"


class PeerAnalysisRecordORM(Base):
    """Peer analysis record ORM model."""

    __tablename__ = "peer_analysis_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID,
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    symbols_analyzed: Mapped[list] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    rankings: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    swap_recommendations: Mapped[list] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    analyses: Mapped[list] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
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
        UUID,
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
