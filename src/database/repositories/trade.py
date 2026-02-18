"""Trade repository for database operations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import func, select, update

from src.database.models import TradeORM
from src.database.repositories.base import BaseRepository
from src.metrics.tracker import TradeRecord
from src.strategies.signal import Signal

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class TradeRepository(BaseRepository[TradeRecord]):
    """Repository for trade record persistence."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(session)

    async def create(self, entity: TradeRecord) -> TradeRecord:
        """Create new trade record.

        Args:
            entity: TradeRecord to persist

        Returns:
            Created TradeRecord with ID
        """
        orm = TradeORM(
            id=uuid.uuid4(),
            timestamp=entity.timestamp,
            symbol=entity.symbol,
            action=entity.action.value,
            entry_price=Decimal(str(entity.entry_price)),
            exit_price=Decimal(str(entity.exit_price)) if entity.exit_price else None,
            shares=entity.shares,
            stop_loss_price=Decimal(str(entity.stop_loss_price)),
            confidence=Decimal(str(entity.confidence)),
            risk_level=entity.risk_level,
            status=entity.status,
            pnl=Decimal(str(entity.pnl)) if entity.pnl else None,
            pnl_percent=Decimal(str(entity.pnl_percent)) if entity.pnl_percent else None,
            strategy_name=entity.strategy_name,
            broker_order_id=entity.broker_order_id,
            is_paper_trade=entity.is_paper_trade,
            closed_at=entity.closed_at,
            created_at=datetime.now(UTC),
        )
        self._session.add(orm)
        await self._session.commit()
        logger.info(f"Created trade: {orm.id} ({entity.symbol} {entity.action.value})")
        entity.id = str(orm.id)
        return entity

    async def get_by_id(self, entity_id: str) -> TradeRecord | None:
        """Get trade by ID.

        Args:
            entity_id: Trade UUID string

        Returns:
            TradeRecord if found, None otherwise
        """
        result = await self._session.execute(select(TradeORM).where(TradeORM.id == uuid.UUID(entity_id)))
        orm = result.scalar_one_or_none()
        return self._to_record(orm) if orm else None

    async def update(self, trade_id: str, **updates: object) -> TradeRecord | None:
        """Update trade record.

        Args:
            trade_id: Trade UUID string
            **updates: Fields to update

        Returns:
            Updated TradeRecord if found, None otherwise
        """
        converted: dict[str, object] = {}
        for key, value in updates.items():
            if isinstance(value, float):
                converted[key] = Decimal(str(value))
            elif isinstance(value, Signal):
                converted[key] = value.value
            else:
                converted[key] = value

        await self._session.execute(
            update(TradeORM).where(TradeORM.id == uuid.UUID(trade_id)).values(**converted)
        )
        await self._session.commit()
        logger.info(f"Updated trade: {trade_id}")
        return await self.get_by_id(trade_id)

    async def get_open_trades(self) -> list[TradeRecord]:
        """Get all open trades.

        Returns:
            List of open TradeRecords
        """
        result = await self._session.execute(select(TradeORM).where(TradeORM.status == "OPEN"))
        return [self._to_record(orm) for orm in result.scalars().all()]

    async def get_by_window(self, window: str) -> list[TradeRecord]:
        """Get trades within time window.

        Args:
            window: Time window ("all", "30d", "7d")

        Returns:
            List of TradeRecords in window
        """
        stmt = select(TradeORM)
        if window != "all":
            days = 30 if window == "30d" else 7 if window == "7d" else 0
            if days > 0:
                cutoff = datetime.now(UTC) - timedelta(days=days)
                stmt = stmt.where(TradeORM.timestamp >= cutoff)
        stmt = stmt.order_by(TradeORM.timestamp.desc())
        result = await self._session.execute(stmt)
        return [self._to_record(orm) for orm in result.scalars().all()]

    async def get_by_symbol(self, symbol: str) -> list[TradeRecord]:
        """Get trades for specific symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            List of TradeRecords for symbol
        """
        result = await self._session.execute(
            select(TradeORM).where(TradeORM.symbol == symbol).order_by(TradeORM.timestamp.desc())
        )
        return [self._to_record(orm) for orm in result.scalars().all()]

    async def get_all(self) -> list[TradeRecord]:
        """Get all trades.

        Returns:
            List of all TradeRecords
        """
        result = await self._session.execute(select(TradeORM).order_by(TradeORM.timestamp.desc()))
        return [self._to_record(orm) for orm in result.scalars().all()]

    async def count_all(self) -> int:
        """Count all trades.

        Returns:
            Total number of trades
        """
        result = await self._session.execute(select(func.count()).select_from(TradeORM))
        return result.scalar_one()

    async def get_closed_since(self, start_date: datetime) -> list[TradeRecord]:
        """Get closed trades since start date.

        Args:
            start_date: Start date for filtering

        Returns:
            List of closed TradeRecords since start_date
        """
        result = await self._session.execute(
            select(TradeORM)
            .where(TradeORM.timestamp >= start_date, TradeORM.status == "CLOSED")
            .order_by(TradeORM.timestamp.desc())
        )
        return [self._to_record(orm) for orm in result.scalars().all()]

    async def get_entry_trade(self, symbol: str) -> TradeRecord | None:
        """Get most recent entry trade for symbol (OPEN BUY).

        Args:
            symbol: Stock ticker symbol

        Returns:
            TradeRecord if found, None otherwise
        """
        result = await self._session.execute(
            select(TradeORM)
            .where(TradeORM.symbol == symbol, TradeORM.status == "OPEN", TradeORM.action == Signal.BUY.value)
            .order_by(TradeORM.timestamp.desc())
            .limit(1)
        )
        orm = result.scalar_one_or_none()
        return self._to_record(orm) if orm else None

    def _to_record(self, orm: TradeORM) -> TradeRecord:
        """Convert ORM model to TradeRecord.

        Args:
            orm: TradeORM instance

        Returns:
            TradeRecord
        """
        return TradeRecord(
            id=str(orm.id),
            timestamp=orm.timestamp,
            symbol=orm.symbol,
            action=Signal(orm.action),
            entry_price=float(orm.entry_price),
            exit_price=float(orm.exit_price) if orm.exit_price else None,
            shares=orm.shares,
            stop_loss_price=float(orm.stop_loss_price),
            confidence=float(orm.confidence),
            risk_level=orm.risk_level,
            status=orm.status,
            pnl=float(orm.pnl) if orm.pnl else None,
            pnl_percent=float(orm.pnl_percent) if orm.pnl_percent else None,
            strategy_name=orm.strategy_name,
            broker_order_id=orm.broker_order_id,
            is_paper_trade=orm.is_paper_trade,
            closed_at=orm.closed_at,
        )

    async def get_recent_closed_by_symbol(self, symbol: str, limit: int = 5) -> list[TradeRecord]:
        """Get recent closed trades for a symbol.

        Args:
            symbol: Stock ticker symbol
            limit: Max trades to return

        Returns:
            List of closed TradeRecords for symbol
        """
        result = await self._session.execute(
            select(TradeORM)
            .where(TradeORM.symbol == symbol, TradeORM.status == "CLOSED")
            .order_by(TradeORM.timestamp.desc())
            .limit(limit)
        )
        return [self._to_record(orm) for orm in result.scalars().all()]

    async def get_aggregate_stats(self, days: int = 30) -> dict[str, float]:
        """Get aggregate trading stats for recent period.

        Args:
            days: Lookback period in days

        Returns:
            Dict with win_rate, avg_gain, total_trades
        """
        cutoff = datetime.now(UTC) - timedelta(days=days)
        result = await self._session.execute(
            select(TradeORM).where(TradeORM.status == "CLOSED", TradeORM.closed_at >= cutoff)
        )
        trades = result.scalars().all()

        if not trades:
            return {"win_rate": 0.0, "avg_gain": 0.0, "total_trades": 0.0}

        winners = sum(1 for t in trades if t.pnl and t.pnl > 0)
        pnl_pcts = [float(t.pnl_percent) for t in trades if t.pnl_percent is not None]
        avg_gain = sum(pnl_pcts) / len(pnl_pcts) if pnl_pcts else 0.0

        return {
            "win_rate": (winners / len(trades)) * 100,
            "avg_gain": avg_gain,
            "total_trades": float(len(trades)),
        }

    def __repr__(self) -> str:
        """Return string representation."""
        return "TradeRepository()"
