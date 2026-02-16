"""ORM models for trading operations."""

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import DECIMAL, TIMESTAMP, Boolean, Index, Integer, String, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.database.models.base import Base


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
