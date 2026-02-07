"""Real-time event streaming for dashboard integration."""

import asyncio
import uuid
from collections import deque
from datetime import UTC, datetime
from enum import StrEnum

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field


class EventType(StrEnum):
    """Event types published by daemon."""

    CYCLE_START = "CYCLE_START"
    CYCLE_COMPLETE = "CYCLE_COMPLETE"
    ANALYSIS_START = "ANALYSIS_START"
    ANALYSIS_COMPLETE = "ANALYSIS_COMPLETE"
    ANALYSIS_ERROR = "ANALYSIS_ERROR"
    TRADE_EXECUTED = "TRADE_EXECUTED"
    HEALTH_CHECK = "HEALTH_CHECK"
    DEGRADATION = "DEGRADATION"
    SCHEDULED_TASK = "SCHEDULED_TASK"
    STATE_UPDATE = "STATE_UPDATE"


class DashboardEvent(BaseModel):
    """Event published to dashboard clients."""

    model_config = ConfigDict(frozen=True)

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    data: dict[str, object] = Field(default_factory=dict)


class EventBus:
    """Pub/sub event bus for dashboard real-time updates."""

    def __init__(self, history_size: int = 500, queue_size: int = 100) -> None:
        """Initialize event bus.

        Args:
            history_size: Maximum events to retain in history
            queue_size: Maximum events per subscriber queue
        """
        self._subscribers: dict[str, asyncio.Queue[DashboardEvent]] = {}
        self._history: deque[DashboardEvent] = deque(maxlen=history_size)
        self._lock = asyncio.Lock()
        self._queue_size = queue_size
        logger.info(f"Initialized EventBus (history={history_size}, queue_size={queue_size})")

    async def subscribe(self) -> tuple[str, asyncio.Queue[DashboardEvent]]:
        """Create new subscriber.

        Returns:
            Tuple of (subscriber_id, event_queue)
        """
        subscriber_id = str(uuid.uuid4())
        queue: asyncio.Queue[DashboardEvent] = asyncio.Queue(maxsize=self._queue_size)

        async with self._lock:
            self._subscribers[subscriber_id] = queue

        logger.info(f"Subscriber {subscriber_id} connected (total: {len(self._subscribers)})")
        return subscriber_id, queue

    async def unsubscribe(self, subscriber_id: str) -> None:
        """Remove subscriber.

        Args:
            subscriber_id: Subscriber ID to remove
        """
        async with self._lock:
            if subscriber_id in self._subscribers:
                del self._subscribers[subscriber_id]
                logger.info(f"Subscriber {subscriber_id} disconnected (remaining: {len(self._subscribers)})")
            else:
                logger.warning(f"Attempted to unsubscribe unknown subscriber {subscriber_id}")

    async def publish(self, event: DashboardEvent) -> None:
        """Publish event to all subscribers.

        Args:
            event: Event to publish
        """
        try:
            self._history.append(event)

            async with self._lock:
                subscribers = list(self._subscribers.items())

            dropped_count = 0
            for subscriber_id, queue in subscribers:
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    dropped_count += 1
                    logger.warning(
                        f"Dropped event {event.event_type} for subscriber {subscriber_id} (queue full)"
                    )

            if dropped_count > 0:
                logger.warning(
                    f"Event {event.event_type} dropped for {dropped_count}/{len(subscribers)} subscribers"
                )

        except Exception as e:
            logger.error(f"EventBus publish failed: {e}")

    def get_history(self, limit: int | None = None) -> list[DashboardEvent]:
        """Get event history, newest first.

        Args:
            limit: Maximum events to return (None for all, 0 returns empty list)

        Returns:
            List of events, newest first
        """
        if limit is not None and limit <= 0:
            return []

        events = list(reversed(self._history))
        if limit is not None:
            events = events[:limit]
        return events

    def get_subscriber_count(self) -> int:
        """Get current subscriber count.

        Returns:
            Number of active subscribers
        """
        return len(self._subscribers)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"EventBus(subscribers={len(self._subscribers)}, "
            f"history_size={len(self._history)}/{self._history.maxlen})"
        )
