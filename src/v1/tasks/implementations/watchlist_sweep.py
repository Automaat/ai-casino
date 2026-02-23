"""Watchlist sweep task — enqueues stale symbols as WatchlistStaleEvent."""

import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from loguru import logger

from src.daemon.events import Sentiment, StaleSymbolInfo, TriageResult, Urgency, WatchlistStaleEvent
from src.v1.tasks.interface import Task
from src.v1.tasks.models import WEEKDAYS, DedupStrategy, TaskResult, TaskSchedule

if TYPE_CHECKING:
    from src.daemon.broker_manager import BrokerManager
    from src.database.engine import DatabaseEngine
    from src.v1.coordinator.models import SweepPassConfig
    from src.v1.event_queue.service import MarketEventQueue


class WatchlistSweepTask(Task):
    """Identifies stale watchlist symbols and enqueues WatchlistStaleEvent for coordinator."""

    def __init__(
        self,
        queue: MarketEventQueue,
        broker_manager: BrokerManager,
        database_engine: DatabaseEngine,
        config: SweepPassConfig,
    ) -> None:
        """Initialize watchlist sweep task.

        Args:
            queue: Market event queue for enqueuing events
            broker_manager: Broker manager for watchlist retrieval
            database_engine: Database engine for analysis timestamps
            config: Sweep pass configuration
        """
        self._queue = queue
        self._broker_manager = broker_manager
        self._db = database_engine
        self._config = config
        self._last_run: datetime | None = None

    @property
    def name(self) -> str:
        """Task identifier."""
        return "watchlist_sweep"

    @property
    def schedule(self) -> TaskSchedule:
        """Schedule from config."""
        return TaskSchedule(
            days=WEEKDAYS,
            enabled=self._config.enabled,
            dedup=DedupStrategy.INTERVAL,
            dedup_interval_minutes=self._config.interval_minutes,
        )

    async def execute(self) -> TaskResult:
        """Identify stale watchlist symbols and enqueue WatchlistStaleEvent.

        Returns:
            TaskResult with outcome
        """
        start = time.monotonic()

        try:
            watchlist = await self._broker_manager.get_merged_watchlist()
            if not watchlist:
                self._last_run = datetime.now(UTC)
                return TaskResult(
                    task_name=self.name,
                    success=True,
                    duration_seconds=time.monotonic() - start,
                    message="Empty watchlist",
                )

            last_analyzed = await self._fetch_last_analysis_timestamps(watchlist)
            stale_threshold = datetime.now(UTC) - timedelta(hours=self._config.stale_hours)

            never_analyzed = [s for s in watchlist if s not in last_analyzed]
            stale = [
                s
                for s in watchlist
                if s in last_analyzed and last_analyzed[s].replace(tzinfo=UTC) < stale_threshold
            ]

            stale_sorted = sorted(stale, key=lambda s: last_analyzed[s])
            sweep_symbols = (never_analyzed + stale_sorted)[: self._config.max_symbols]

            if not sweep_symbols:
                self._last_run = datetime.now(UTC)
                return TaskResult(
                    task_name=self.name,
                    success=True,
                    duration_seconds=time.monotonic() - start,
                    message="No stale symbols",
                )

            now = datetime.now(UTC)
            stale_infos = [
                StaleSymbolInfo(
                    symbol=s,
                    last_analysis_age_hours=(
                        (now - last_analyzed[s].replace(tzinfo=UTC)).total_seconds() / 3600.0
                        if s in last_analyzed
                        else float(self._config.stale_hours + 1)
                    ),
                )
                for s in sweep_symbols
            ]

            event = WatchlistStaleEvent(stale_symbols=stale_infos)
            triage = TriageResult(
                event_id=event.event_id,
                event_type="watchlist_stale",
                symbols=sweep_symbols,
                urgency=Urgency.IMMEDIATE,
                sentiment=Sentiment.NEUTRAL,
                confidence=1.0,
                reasoning=f"Scheduled sweep: {len(never_analyzed)} never analyzed, {len(stale)} stale",
                relevance=1.0,
            )

            await self._queue.enqueue(event, triage, ttl_hours=self._config.stale_hours)
            self._last_run = datetime.now(UTC)

            msg = (
                f"{len(sweep_symbols)} stale symbols enqueued "
                f"(never={len(never_analyzed)}, stale={len(stale)})"
            )
            logger.info(f"Watchlist sweep enqueued: {msg}")

            return TaskResult(
                task_name=self.name,
                success=True,
                duration_seconds=time.monotonic() - start,
                message=msg,
            )

        except Exception as e:
            logger.opt(exception=True).error(f"Watchlist sweep failed: {e}")
            return TaskResult(
                task_name=self.name,
                success=False,
                duration_seconds=time.monotonic() - start,
                message=f"Failed: {e}",
            )

    async def _fetch_last_analysis_timestamps(self, symbols: list[str]) -> dict:
        """Fetch last analysis timestamps for given symbols from database.

        Args:
            symbols: List of ticker symbols

        Returns:
            Dict mapping symbol to last analysis datetime
        """
        try:
            from src.database.repositories.analysis import AnalysisRecordRepository

            async with self._db.session() as session:
                repo = AnalysisRecordRepository(session)
                return await repo.get_last_analysis_timestamps(symbols)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to fetch analysis timestamps: {e}")
            return {}

    async def last_run_at(self) -> datetime | None:
        """Get last execution timestamp."""
        return self._last_run

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"WatchlistSweepTask(enabled={self._config.enabled}, "
            f"interval={self._config.interval_minutes}m, stale_hours={self._config.stale_hours})"
        )
