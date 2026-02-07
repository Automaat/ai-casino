"""SQLAlchemy ORM models for trade history persistence."""

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import DECIMAL, TIMESTAMP, Boolean, Index, Integer, String, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
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
