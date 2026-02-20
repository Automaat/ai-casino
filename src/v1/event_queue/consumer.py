"""Standalone consumer that polls the event queue and spawns coordinator cycles."""

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

from src.v1.event_queue.models import QueuedMarketEvent

if TYPE_CHECKING:
    from src.daemon.market_service import MarketService
    from src.strategies.session import TradingSession
    from src.v1.coordinator.agent import TradingCoordinator
    from src.v1.coordinator.models import CoordinatorConfig
    from src.v1.event_queue.service import MarketEventQueue


class EventQueueConsumer:
    """Polls MarketEventQueue and runs event-driven coordinator cycles."""

    def __init__(
        self,
        queue: MarketEventQueue,
        coordinator: TradingCoordinator,
        market_service: MarketService,
        config: CoordinatorConfig,
    ) -> None:
        """Initialize consumer.

        Args:
            queue: Event queue to poll
            coordinator: Coordinator for running event cycles
            market_service: Market state service for session/hours checks
            config: Coordinator config with polling parameters
        """
        self._queue = queue
        self._coordinator = coordinator
        self._market_service = market_service
        self._config = config
        self._purge_counter = 0
        self._purge_task: asyncio.Task | None = None

    async def run(self) -> None:
        """Main consumer loop — runs as a concurrent task in the daemon event loop.

        Polls queue at config interval, groups events by symbol overlap,
        runs coordinator event cycles. Never crashes the loop.
        """
        logger.info(
            f"EventQueueConsumer started: poll_interval={self._config.event_poll_interval_seconds}s, "
            f"max_dequeue={self._config.event_max_dequeue}"
        )

        while True:
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                logger.info("EventQueueConsumer shutting down")
                raise
            except Exception:
                logger.opt(exception=True).error("EventQueueConsumer iteration failed")

            try:
                await asyncio.sleep(self._config.event_poll_interval_seconds)
            except asyncio.CancelledError:
                logger.info("EventQueueConsumer shutting down")
                raise

    async def _poll_once(self) -> None:
        """Single poll iteration: dequeue, group, run cycles."""
        events = await self._queue.dequeue(max_items=self._config.event_max_dequeue)

        if not events:
            self._maybe_purge()
            return

        logger.info(f"Polled {len(events)} events: {[f'{e.event_type}:{e.event_id}' for e in events]}")

        groups = _group_by_symbol_overlap(events)
        market_open = self._market_service.is_open()
        session = self._get_trading_session()

        for group in groups:
            try:
                result = await self._coordinator.run_event_cycle(
                    events=group,
                    market_open=market_open,
                    trading_session=session,
                )
                logger.info(
                    f"Event cycle result: {result.summary[:100]}, "
                    f"tools={result.tool_calls_made}, events={result.event_ids}"
                )
            except Exception:
                event_ids = [e.event_id for e in group]
                logger.opt(exception=True).error(f"Event cycle failed for events={event_ids}")

        self._maybe_purge()

    def _get_trading_session(self) -> TradingSession:
        """Determine current trading session."""
        from src.strategies.session import TradingSession

        session = self._market_service.current_session()
        if session is not None:
            return session
        return TradingSession.REGULAR

    def _maybe_purge(self) -> None:
        """Purge expired events every ~20 poll iterations."""
        self._purge_counter += 1
        purge_frequency = 20
        if self._purge_counter >= purge_frequency:
            self._purge_counter = 0
            if self._purge_task is None or self._purge_task.done():
                self._purge_task = asyncio.create_task(self._safe_purge())

    async def _safe_purge(self) -> None:
        """Purge expired events with error handling."""
        try:
            deleted = await self._queue.purge_expired()
            if deleted > 0:
                logger.info(f"Purged {deleted} expired events from queue")
        except Exception:
            logger.opt(exception=True).warning("Event queue purge failed")

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"EventQueueConsumer(poll={self._config.event_poll_interval_seconds}s, "
            f"max_dequeue={self._config.event_max_dequeue}, "
            f"market={self._market_service!r})"
        )


def _group_by_symbol_overlap(events: list[QueuedMarketEvent]) -> list[list[QueuedMarketEvent]]:
    """Group events by overlapping affected symbols.

    Events sharing at least one symbol end up in the same group.
    Events with no symbols form individual groups.

    Args:
        events: Flat list of dequeued events

    Returns:
        List of event groups
    """
    from src.v1.coordinator.event_prompt import extract_symbols

    groups: list[tuple[set[str], list[QueuedMarketEvent]]] = []

    for event in events:
        event_symbols = extract_symbols([event])

        if not event_symbols:
            groups.append((set(), [event]))
            continue

        overlapping_indices: list[int] = []
        for idx, (group_symbols, _group_events) in enumerate(groups):
            if group_symbols & event_symbols:
                overlapping_indices.append(idx)

        if not overlapping_indices:
            groups.append((event_symbols.copy(), [event]))
            continue

        target_idx = overlapping_indices[0]
        target_symbols, target_events = groups[target_idx]
        target_symbols.update(event_symbols)
        target_events.append(event)

        for idx in reversed(overlapping_indices[1:]):
            merge_symbols, merge_events = groups[idx]
            target_symbols.update(merge_symbols)
            target_events.extend(merge_events)
            del groups[idx]

    return [group_events for _, group_events in groups]
