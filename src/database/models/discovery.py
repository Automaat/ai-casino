"""ORM models for discovery operations."""

import uuid
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import DATE, DECIMAL, TIMESTAMP, Boolean, Index, Integer, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from src.database.models.base import Base
from src.database.types import JSONB, UUID


class DiscoveryHistoryRecordORM(Base):
    """Discovery history record ORM model."""

    __tablename__ = "discovery_history"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID,
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
    supervisor_evaluation_score: Mapped[Decimal | None] = mapped_column(DECIMAL(5, 4), nullable=True)
    supervisor_recommendation: Mapped[str | None] = mapped_column(String(20), nullable=True)
    evaluation_reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    price_at_discovery: Mapped[Decimal | None] = mapped_column(DECIMAL(12, 4), nullable=True)
    outcome_updated_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
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


class DiscoverySourceMetricsORM(Base):
    """Discovery source metrics ORM model."""

    __tablename__ = "discovery_source_metrics"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID,
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    measurement_date: Mapped[date] = mapped_column(DATE, nullable=False)
    total_discoveries: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    watchlist_additions: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    signal_conversions: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    discoveries_with_7d_outcome: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    positive_7d_outcomes: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    avg_7d_return: Mapped[Decimal | None] = mapped_column(DECIMAL(8, 4), nullable=True)
    median_7d_return: Mapped[Decimal | None] = mapped_column(DECIMAL(8, 4), nullable=True)
    discoveries_with_30d_outcome: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    positive_30d_outcomes: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    avg_30d_return: Mapped[Decimal | None] = mapped_column(DECIMAL(8, 4), nullable=True)
    median_30d_return: Mapped[Decimal | None] = mapped_column(DECIMAL(8, 4), nullable=True)
    precision_score: Mapped[Decimal | None] = mapped_column(DECIMAL(5, 4), nullable=True)
    recall_score: Mapped[Decimal | None] = mapped_column(DECIMAL(5, 4), nullable=True)
    f1_score: Mapped[Decimal | None] = mapped_column(DECIMAL(5, 4), nullable=True)
    false_positives: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    false_negatives: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    __table_args__ = (
        Index("idx_discovery_source_metrics_date", "measurement_date"),
        Index("idx_discovery_source_metrics_source", "source_type"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"DiscoverySourceMetricsORM(source={self.source_type}, "
            f"date={self.measurement_date}, total={self.total_discoveries})"
        )


class ActiveDiscoveryCandidateORM(Base):
    """Active discovery candidate ORM model."""

    __tablename__ = "active_discovery_candidates"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID,
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
