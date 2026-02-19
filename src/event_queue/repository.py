"""Repository for market event queue database operations."""

import json
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, or_, select, text, update

from src.database.repositories.base import BaseRepository
from src.event_queue.models import MarketEventQueueORM, QueuedMarketEvent


class MarketEventQueueRepository(BaseRepository[MarketEventQueueORM]):
    """Repository for FIFO market event queue."""

    async def create(self, entity: MarketEventQueueORM) -> MarketEventQueueORM:
        """Insert a queue record (use enqueue for idempotent inserts)."""
        self._session.add(entity)
        await self._session.commit()
        await self._session.refresh(entity)
        return entity

    async def get_by_id(self, entity_id: str) -> MarketEventQueueORM | None:
        """Get queue record by UUID string."""
        result = await self._session.execute(
            select(MarketEventQueueORM).where(MarketEventQueueORM.id == uuid.UUID(entity_id))
        )
        return result.scalar_one_or_none()

    async def enqueue(self, record: MarketEventQueueORM) -> None:
        """Insert record with ON CONFLICT DO NOTHING for idempotency.

        Duplicate event_id is silently ignored.
        """
        stmt = text(
            "INSERT INTO market_event_queue "
            "(id, event_id, event_type, payload, enqueued_at, expires_at, consumed_at, process_after) "
            "VALUES (:id, :event_id, :event_type, CAST(:payload AS JSONB), "
            ":enqueued_at, :expires_at, :consumed_at, :process_after) "
            "ON CONFLICT (event_id) DO NOTHING"
        ).bindparams(
            id=record.id,
            event_id=record.event_id,
            event_type=record.event_type,
            payload=json.dumps(record.payload),
            enqueued_at=record.enqueued_at,
            expires_at=record.expires_at,
            consumed_at=record.consumed_at,
            process_after=record.process_after,
        )
        await self._session.execute(stmt)
        await self._session.commit()

    async def dequeue(self, max_items: int = 1) -> list[QueuedMarketEvent]:
        """Atomically claim up to max_items pending non-expired events (FIFO).

        Marks claimed rows with consumed_at = NOW().
        """
        now = datetime.now(UTC)
        subq = (
            select(MarketEventQueueORM.event_id)
            .where(MarketEventQueueORM.consumed_at.is_(None))
            .where(MarketEventQueueORM.expires_at > now)
            .where(
                or_(
                    MarketEventQueueORM.process_after.is_(None),
                    MarketEventQueueORM.process_after <= now,
                )
            )
            .order_by(MarketEventQueueORM.enqueued_at.asc())
            .limit(max_items)
            .with_for_update(skip_locked=True)
        )
        result = await self._session.execute(subq)
        event_ids = [row[0] for row in result.fetchall()]

        if not event_ids:
            return []

        await self._session.execute(
            update(MarketEventQueueORM)
            .where(MarketEventQueueORM.event_id.in_(event_ids))
            .values(consumed_at=now)
        )
        await self._session.commit()

        rows_result = await self._session.execute(
            select(MarketEventQueueORM)
            .where(MarketEventQueueORM.event_id.in_(event_ids))
            .order_by(MarketEventQueueORM.enqueued_at.asc())
        )
        rows = rows_result.scalars().all()
        return [
            QueuedMarketEvent(
                event_id=row.event_id,
                event_type=row.event_type,
                payload=row.payload,
                enqueued_at=row.enqueued_at,
                process_after=row.process_after,
            )
            for row in rows
        ]

    async def count_pending(self) -> int:
        """Return count of pending non-expired events ready to process."""
        now = datetime.now(UTC)
        result = await self._session.execute(
            select(func.count())
            .select_from(MarketEventQueueORM)
            .where(MarketEventQueueORM.consumed_at.is_(None))
            .where(MarketEventQueueORM.expires_at > now)
            .where(
                or_(
                    MarketEventQueueORM.process_after.is_(None),
                    MarketEventQueueORM.process_after <= now,
                )
            )
        )
        return result.scalar_one()

    async def purge_expired(self) -> int:
        """Delete expired rows. Returns number of rows deleted."""
        now = datetime.now(UTC)
        result = await self._session.execute(
            text("DELETE FROM market_event_queue WHERE expires_at < :now RETURNING id").bindparams(now=now)
        )
        await self._session.commit()
        return len(result.fetchall())

    async def list_events(self, limit: int = 100, status: str = "all") -> list[MarketEventQueueORM]:
        """List events ordered by enqueued_at DESC.

        Args:
            limit: Max rows to return
            status: "all" | "pending" | "consumed" | "expired"
        """
        now = datetime.now(UTC)
        stmt = select(MarketEventQueueORM).order_by(MarketEventQueueORM.enqueued_at.desc()).limit(limit)
        if status == "pending":
            stmt = stmt.where(MarketEventQueueORM.consumed_at.is_(None)).where(
                MarketEventQueueORM.expires_at > now
            )
        elif status == "consumed":
            stmt = stmt.where(MarketEventQueueORM.consumed_at.isnot(None))
        elif status == "expired":
            stmt = stmt.where(MarketEventQueueORM.consumed_at.is_(None)).where(
                MarketEventQueueORM.expires_at <= now
            )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_counts(self) -> dict:
        """Return queue count statistics with one DB query per metric."""
        now = datetime.now(UTC)
        since_24h = now - timedelta(hours=24)

        pending_result = await self._session.execute(
            select(func.count())
            .select_from(MarketEventQueueORM)
            .where(MarketEventQueueORM.consumed_at.is_(None))
            .where(MarketEventQueueORM.expires_at > now)
        )
        pending = pending_result.scalar_one()

        stale_result = await self._session.execute(
            select(func.count())
            .select_from(MarketEventQueueORM)
            .where(MarketEventQueueORM.consumed_at.is_(None))
            .where(MarketEventQueueORM.expires_at <= now)
        )
        stale = stale_result.scalar_one()

        consumed_24h_result = await self._session.execute(
            select(func.count())
            .select_from(MarketEventQueueORM)
            .where(MarketEventQueueORM.consumed_at.isnot(None))
            .where(MarketEventQueueORM.consumed_at >= since_24h)
        )
        consumed_24h = consumed_24h_result.scalar_one()

        total_result = await self._session.execute(select(func.count()).select_from(MarketEventQueueORM))
        total = total_result.scalar_one()

        by_type_result = await self._session.execute(
            select(MarketEventQueueORM.event_type, func.count().label("cnt"))
            .where(MarketEventQueueORM.consumed_at.is_(None))
            .where(MarketEventQueueORM.expires_at > now)
            .group_by(MarketEventQueueORM.event_type)
        )
        by_type = {row.event_type: row.cnt for row in by_type_result}

        return {
            "pending": pending,
            "stale": stale,
            "consumed_24h": consumed_24h,
            "total": total,
            "by_type": by_type,
        }

    def __repr__(self) -> str:
        """Return string representation."""
        return "MarketEventQueueRepository()"
