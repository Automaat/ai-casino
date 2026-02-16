"""ORM models for analysis operations."""

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import DECIMAL, TIMESTAMP, Boolean, Index, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.database.models.base import Base


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
