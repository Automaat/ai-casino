"""Event triage pipeline: triage → IMMEDIATE to queue, WATCHLIST to discovery."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from loguru import logger

from src.agents.event_triage import EventTriageAgent
from src.daemon.events import BaseEvent, NewsEvent, NewsWatchlistEvent, TriageResult, Urgency
from src.event_queue.service import MarketEventQueue

if TYPE_CHECKING:
    from src.daemon.state.facade import DaemonState


class EventTriagePipeline:
    """Triages events and routes IMMEDIATE to MarketEventQueue, WATCHLIST to discovery."""

    def __init__(
        self,
        triage_agent: EventTriageAgent,
        queue: MarketEventQueue | None,
        state: DaemonState | None = None,
        immediate_ttl_hours: int = 4,
        watchlist_ttl_hours: int = 24,
    ) -> None:
        """Initialize pipeline.

        Args:
            triage_agent: LLM-based event triage agent
            queue: Event queue (None when DB not available)
            state: Daemon state for WATCHLIST discovery routing
            immediate_ttl_hours: TTL for IMMEDIATE events in queue
            watchlist_ttl_hours: TTL for WATCHLIST discovery candidates
        """
        self._triage_agent = triage_agent
        self._queue = queue
        self._state = state
        self._immediate_ttl_hours = immediate_ttl_hours
        self._watchlist_ttl_hours = watchlist_ttl_hours

    async def process(self, events: list[BaseEvent]) -> None:
        """Triage events and route by urgency.

        Args:
            events: Events to triage and route
        """
        triage_results = await self._triage_events(events)
        for event, triage in zip(events, triage_results, strict=True):
            if isinstance(triage, BaseException):
                logger.error(f"Triage failed: {triage}")
                continue
            if triage.urgency == Urgency.IMMEDIATE:
                on_watchlist = await self._get_watchlist_symbols(triage.symbols)
                if on_watchlist:
                    logger.info(
                        f"Urgency escalation {on_watchlist}: WATCHLIST→IMMEDIATE for event {event.event_id}"
                    )
                await self._enqueue(event, triage)
            elif triage.urgency == Urgency.WATCHLIST:
                on_watchlist = await self._get_watchlist_symbols(triage.symbols)
                new_symbols = [s for s in triage.symbols if s not in on_watchlist]
                if not new_symbols:
                    logger.debug(
                        f"Skipping WATCHLIST event {event.event_id}: "
                        f"all symbols {triage.symbols} already on watchlist"
                    )
                    continue
                filtered_triage = (
                    triage.model_copy(update={"symbols": new_symbols}) if on_watchlist else triage
                )
                await self._add_watchlist_candidates(event, filtered_triage)
                if isinstance(event, NewsEvent):
                    await self._enqueue_watchlist(
                        NewsWatchlistEvent(
                            event_id=event.event_id,
                            timestamp=event.timestamp,
                            source=event.source,
                            article=event.article,
                        ),
                        filtered_triage,
                    )
                else:
                    await self._enqueue_watchlist(event, filtered_triage)

    async def _enqueue(self, event: BaseEvent, triage: TriageResult) -> None:
        """Enqueue IMMEDIATE event to MarketEventQueue."""
        if self._queue is None:
            logger.debug(f"Queue unavailable, dropping IMMEDIATE event {event.event_id}")
            return
        logger.info(f"Publishing event {event.event_type} (IMMEDIATE, symbols={triage.symbols})")
        try:
            await self._queue.enqueue(event, triage, ttl_hours=self._immediate_ttl_hours)
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to enqueue event {event.event_id}: {e}")

    async def _enqueue_watchlist(self, event: BaseEvent, triage: TriageResult) -> None:
        """Enqueue WATCHLIST event to MarketEventQueue with watchlist TTL."""
        if self._queue is None:
            logger.debug(f"Queue unavailable, dropping WATCHLIST event {event.event_id}")
            return
        logger.info(f"Publishing event {event.event_type} (WATCHLIST, symbols={triage.symbols})")
        try:
            await self._queue.enqueue(event, triage, ttl_hours=self._watchlist_ttl_hours)
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to enqueue watchlist event {event.event_id}: {e}")

    async def _triage_events(self, events: list[BaseEvent]) -> list[TriageResult | BaseException]:
        """Triage events with LLM in parallel."""

        async def safe_triage(event: BaseEvent) -> TriageResult | BaseException:
            try:
                return await self._triage_agent.analyze(event)
            except BaseException as e:
                return e

        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(safe_triage(e)) for e in events]
        return [t.result() for t in tasks]

    async def _get_watchlist_symbols(self, symbols: list[str]) -> set[str]:
        """Return subset of symbols already in active discovery candidates."""
        if not self._state or not symbols:
            return set()
        try:
            active = await self._state.discovery.get_active_discovery_symbols()
            return {s for s in symbols if s in active}
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to check watchlist symbols: {e}")
            return set()

    async def _add_watchlist_candidates(self, event: BaseEvent, triage: TriageResult) -> None:
        """Add WATCHLIST event to discovery candidates."""
        if not self._state:
            return

        from src.discovery.models import DiscoveryCandidate, DiscoverySource

        now = datetime.now(UTC)
        candidates = []
        for symbol in triage.symbols:
            candidate = DiscoveryCandidate(
                symbol=symbol,
                name="Unknown",
                sector="Unknown",
                sources=[DiscoverySource.EVENT_WATCHLIST],
                composite_score=triage.relevance,
                source_scores={"event_watchlist": triage.relevance},
                discovery_timestamp=now,
                ttl_expires_at=now + timedelta(hours=self._watchlist_ttl_hours),
                metadata={
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "sentiment": triage.sentiment.value,
                    "confidence": triage.confidence,
                    "reasoning": triage.reasoning,
                },
            )
            candidates.append(candidate)

        if not candidates:
            return

        try:
            await self._state.discovery.add_event_candidates(candidates)
            logger.info(f"Added {len(candidates)} WATCHLIST candidates for event {event.event_id}")
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to add WATCHLIST candidates: {e}")

    def __repr__(self) -> str:
        """Return string representation."""
        return f"EventTriagePipeline(queue={self._queue!r})"
