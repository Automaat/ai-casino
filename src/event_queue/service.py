"""Market event queue service backed by PostgreSQL."""

import uuid
from datetime import UTC, datetime, timedelta
from typing import cast

from loguru import logger
from pydantic import BaseModel

from src.daemon.events import BaseEvent, TriageResult
from src.database.engine import DatabaseEngine
from src.event_queue.models import MarketEventQueueORM, QueuedMarketEvent
from src.event_queue.repository import MarketEventQueueRepository


class MarketEventQueue:
    """PostgreSQL-backed FIFO queue for market event signals."""

    def __init__(self, database_engine: DatabaseEngine) -> None:
        """Initialize queue service.

        Args:
            database_engine: Engine used to create per-call sessions
        """
        self._db = database_engine

    async def enqueue(
        self,
        event: BaseEvent,
        triage: TriageResult,
        ttl_hours: int = 4,
        process_after: datetime | None = None,
    ) -> None:
        """Enqueue an event with its triage result. Idempotent via ON CONFLICT DO NOTHING.

        Args:
            event: Market event implementing BaseEvent protocol
            triage: Triage result from EventTriageAgent
            ttl_hours: Hours until the event expires (default 4)
            process_after: Earliest datetime to dequeue (None = immediately eligible)
        """
        if ttl_hours <= 0:
            msg = f"ttl_hours must be positive, got {ttl_hours}"
            raise ValueError(msg)
        now = datetime.now(UTC)
        event_model = cast("BaseModel", event)
        payload = {
            "event": event_model.model_dump(mode="json"),
            "triage": triage.model_dump(mode="json"),
        }
        record = MarketEventQueueORM(
            id=uuid.uuid4(),
            event_id=event.event_id,
            event_type=event.event_type,
            payload=payload,
            enqueued_at=now,
            expires_at=now + timedelta(hours=ttl_hours),
            consumed_at=None,
            process_after=process_after,
        )
        async with self._db.session() as session:
            repo = MarketEventQueueRepository(session)
            await repo.enqueue(record)

        logger.debug(
            f"Enqueued event event_id={event.event_id!r} type={event.event_type!r}"
            + (f" process_after={process_after.isoformat()}" if process_after else "")
        )

    async def dequeue(self, max_items: int = 1) -> list[QueuedMarketEvent]:
        """Atomically dequeue up to max_items pending events (FIFO).

        Args:
            max_items: Maximum events to claim in one call

        Returns:
            List of QueuedMarketEvent (payload includes raw event + triage dicts)
        """
        if max_items < 1:
            msg = f"max_items must be at least 1, got {max_items}"
            raise ValueError(msg)
        async with self._db.session() as session:
            repo = MarketEventQueueRepository(session)
            return await repo.dequeue(max_items)

    async def size(self) -> int:
        """Return count of pending non-expired events."""
        async with self._db.session() as session:
            repo = MarketEventQueueRepository(session)
            return await repo.count_pending()

    async def purge_expired(self) -> int:
        """Delete all expired events. Returns count of deleted rows."""
        async with self._db.session() as session:
            repo = MarketEventQueueRepository(session)
            deleted = await repo.purge_expired()
        logger.debug(f"Purged {deleted} expired queue events")
        return deleted

    def __repr__(self) -> str:
        """Return string representation."""
        return f"MarketEventQueue(db={self._db!r})"
