"""ORM models for screening operations."""

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import DECIMAL, TIMESTAMP, Boolean, Index, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.database.models.base import Base


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
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[Decimal | None] = mapped_column(DECIMAL(5, 4), nullable=True)
    overnight_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    key_levels: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    generated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    __table_args__ = (
        Index("idx_game_plan_records_timestamp", "timestamp"),
        Index("idx_game_plan_records_created_at", "created_at"),
        Index("idx_game_plan_records_generated_at", "generated_at", postgresql_ops={"generated_at": "DESC"}),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"GamePlanRecordORM(id={self.id}, timestamp={self.timestamp})"


class ScoringWeightsHistoryORM(Base):
    """Scoring weights history ORM model."""

    __tablename__ = "scoring_weights_history"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )
    regime: Mapped[str | None] = mapped_column(String(30), nullable=True)
    technical_weight: Mapped[Decimal] = mapped_column(DECIMAL(5, 4), nullable=False)
    liquidity_weight: Mapped[Decimal] = mapped_column(DECIMAL(5, 4), nullable=False)
    timing_weight: Mapped[Decimal] = mapped_column(DECIMAL(5, 4), nullable=False)
    social_weight: Mapped[Decimal] = mapped_column(DECIMAL(5, 4), nullable=False)
    volatility_weight: Mapped[Decimal] = mapped_column(DECIMAL(5, 4), nullable=False)
    training_window_days: Mapped[int] = mapped_column(Integer, nullable=False)
    discoveries_analyzed: Mapped[int] = mapped_column(Integer, nullable=False)
    avg_performance_improvement: Mapped[Decimal | None] = mapped_column(DECIMAL(8, 4), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    activated_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    __table_args__ = (
        Index("idx_scoring_weights_history_active", "is_active"),
        Index("idx_scoring_weights_history_regime", "regime"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ScoringWeightsHistoryORM(regime={self.regime}, active={self.is_active})"
