"""Trade repository for database operations."""

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from loguru import logger
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import TradeORM
from src.database.repositories.base import BaseRepository
from src.metrics.tracker import TradeRecord
from src.strategies.momentum import Signal


class TradeRepository(BaseRepository[TradeRecord]):
    """Repository for trade record persistence."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(session)
        logger.debug("Initialized TradeRepository")

    async def create(self, trade: TradeRecord) -> TradeRecord:
        """Create new trade record.

        Args:
            trade: TradeRecord to persist

        Returns:
            Created TradeRecord with ID
        """
        orm = TradeORM(
            id=uuid.uuid4(),
            timestamp=trade.timestamp,
            symbol=trade.symbol,
            action=trade.action.value,
            entry_price=Decimal(str(trade.entry_price)),
            exit_price=Decimal(str(trade.exit_price)) if trade.exit_price else None,
            shares=trade.shares,
            stop_loss_price=Decimal(str(trade.stop_loss_price)),
            confidence=Decimal(str(trade.confidence)),
            risk_level=trade.risk_level,
            status=trade.status,
            pnl=Decimal(str(trade.pnl)) if trade.pnl else None,
            pnl_percent=Decimal(str(trade.pnl_percent)) if trade.pnl_percent else None,
            strategy_name=trade.strategy_name,
            created_at=datetime.now(UTC),
        )
        self._session.add(orm)
        await self._session.commit()
        logger.info(f"Created trade: {orm.id} ({trade.symbol} {trade.action.value})")
        return trade

    async def get_by_id(self, trade_id: str) -> TradeRecord | None:
        """Get trade by ID.

        Args:
            trade_id: Trade UUID string

        Returns:
            TradeRecord if found, None otherwise
        """
        result = await self._session.execute(select(TradeORM).where(TradeORM.id == uuid.UUID(trade_id)))
        orm = result.scalar_one_or_none()
        return self._to_record(orm) if orm else None

    async def update(self, trade_id: str, **updates: dict) -> TradeRecord | None:
        """Update trade record.

        Args:
            trade_id: Trade UUID string
            **updates: Fields to update

        Returns:
            Updated TradeRecord if found, None otherwise
        """
        converted = {}
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
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "TradeRepository()"
