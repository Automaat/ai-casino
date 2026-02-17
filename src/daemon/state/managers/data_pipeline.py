"""Data pipeline state manager."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import PrivateAttr

from src.daemon.state.managers.base import StateManager
from src.daemon.state.models import (
    EarningsCalendarRecord,
    EarningsEventRecord,
    PrefetchRecord,
    ProfilingRecord,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class DataPipelineStateManager(StateManager):
    """Background data operations (prefetch, earnings)."""

    _prefetch_cache: list[PrefetchRecord] | None = PrivateAttr(default=None)
    _earnings_cache: list[EarningsCalendarRecord] | None = PrivateAttr(default=None)
    _profiling_cache: list[ProfilingRecord] | None = PrivateAttr(default=None)

    async def get_last_prefetch(self, session: AsyncSession | None = None) -> datetime | None:
        """Get last prefetch timestamp from DB."""
        from src.database.repositories.metadata import MetadataRepository

        if session:
            repo = MetadataRepository(session)
            return await repo.get_datetime("data_pipeline.last_prefetch")

        try:
            from src.database.connection import get_session

            async with get_session() as fresh_session:
                repo = MetadataRepository(fresh_session)
                return await repo.get_datetime("data_pipeline.last_prefetch")
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to get prefetch: {e}")
            return None

    async def set_last_prefetch(self, value: datetime | None) -> None:
        """Set last prefetch timestamp in DB."""
        if value is None:
            return
        try:
            from src.database.connection import get_session
            from src.database.repositories.metadata import MetadataRepository

            async with get_session() as session:
                await MetadataRepository(session).set("data_pipeline.last_prefetch", value)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to set prefetch: {e}")

    async def get_last_pre_market_refresh(self, session: AsyncSession | None = None) -> datetime | None:
        """Get last pre-market refresh timestamp from DB."""
        from src.database.repositories.metadata import MetadataRepository

        if session:
            repo = MetadataRepository(session)
            return await repo.get_datetime("data_pipeline.last_pre_market_refresh")

        try:
            from src.database.connection import get_session

            async with get_session() as fresh_session:
                repo = MetadataRepository(fresh_session)
                return await repo.get_datetime("data_pipeline.last_pre_market_refresh")
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to get pre-market refresh: {e}")
            return None

    async def set_last_pre_market_refresh(self, value: datetime | None) -> None:
        """Set last pre-market refresh timestamp in DB."""
        if value is None:
            return
        try:
            from src.database.connection import get_session
            from src.database.repositories.metadata import MetadataRepository

            async with get_session() as session:
                await MetadataRepository(session).set("data_pipeline.last_pre_market_refresh", value)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to set pre-market refresh: {e}")

    async def get_last_earnings_fetch(self, session: AsyncSession | None = None) -> datetime | None:
        """Get last earnings fetch timestamp from DB."""
        from src.database.repositories.metadata import MetadataRepository

        if session:
            repo = MetadataRepository(session)
            return await repo.get_datetime("data_pipeline.last_earnings_fetch")

        try:
            from src.database.connection import get_session

            async with get_session() as fresh_session:
                repo = MetadataRepository(fresh_session)
                return await repo.get_datetime("data_pipeline.last_earnings_fetch")
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to get earnings fetch: {e}")
            return None

    async def get_prefetch_history(
        self, limit: int = 30, session: AsyncSession | None = None
    ) -> list[PrefetchRecord]:
        """Get prefetch history with lazy loading."""
        from src.database.repositories.prefetch import PrefetchRecordRepository

        if session:
            repo = PrefetchRecordRepository(session)
            return await repo.get_recent(limit)

        if self._prefetch_cache is None:
            try:
                from src.database.connection import get_session

                async with get_session() as fresh_session:
                    repo = PrefetchRecordRepository(fresh_session)
                    self._prefetch_cache = await repo.get_recent(limit)
            except Exception as e:
                logger.opt(exception=True).warning(f"Failed to get prefetch history: {e}")
                return []
        return self._prefetch_cache

    async def get_earnings_calendar_history(
        self, limit: int = 10, session: AsyncSession | None = None
    ) -> list[EarningsCalendarRecord]:
        """Get earnings calendar history with lazy loading."""
        from src.database.repositories.earnings_calendar import EarningsCalendarRecordRepository

        if session:
            repo = EarningsCalendarRecordRepository(session)
            return await repo.get_recent(limit)

        if self._earnings_cache is None:
            try:
                from src.database.connection import get_session

                async with get_session() as fresh_session:
                    repo = EarningsCalendarRecordRepository(fresh_session)
                    self._earnings_cache = await repo.get_recent(limit)
            except Exception as e:
                logger.opt(exception=True).warning(f"Failed to get earnings calendar history: {e}")
                return []
        return self._earnings_cache

    async def get_profiling_history(
        self, limit: int = 100, session: AsyncSession | None = None
    ) -> list[ProfilingRecord]:
        """Get profiling history with lazy loading."""
        from src.database.repositories.profiling import ProfilingRecordRepository

        if session:
            repo = ProfilingRecordRepository(session)
            return await repo.get_recent(limit)

        if self._profiling_cache is None:
            try:
                from src.database.connection import get_session

                async with get_session() as fresh_session:
                    repo = ProfilingRecordRepository(fresh_session)
                    self._profiling_cache = await repo.get_recent(limit)
            except Exception as e:
                logger.opt(exception=True).warning(f"Failed to get profiling history: {e}")
                return []
        return self._profiling_cache

    async def record_prefetch(
        self,
        symbols_prefetched: int,
        symbols_failed: int,
        finbert_ready: bool,
        total_duration_seconds: float,
    ) -> None:
        """Record a data prefetch run."""
        now = datetime.now(UTC)
        record = PrefetchRecord(
            timestamp=now,
            symbols_prefetched=symbols_prefetched,
            symbols_failed=symbols_failed,
            finbert_ready=finbert_ready,
            total_duration_seconds=total_duration_seconds,
        )

        try:
            from src.database.connection import get_session
            from src.database.repositories.metadata import MetadataRepository
            from src.database.repositories.prefetch import PrefetchRecordRepository

            async with get_session() as session:
                await PrefetchRecordRepository(session).create(record)
                await MetadataRepository(session).set("data_pipeline.last_prefetch", now)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to record prefetch: {e}")

        self._prefetch_cache = None

    async def record_earnings_fetch(
        self,
        events: list[EarningsEventRecord],
        symbols_fetched: int,
        symbols_failed: int,
    ) -> None:
        """Record an earnings calendar fetch run."""
        now = datetime.now(UTC)
        record = EarningsCalendarRecord(
            timestamp=now,
            events=events,
            symbols_fetched=symbols_fetched,
            symbols_failed=symbols_failed,
        )

        try:
            from src.database.connection import get_session
            from src.database.repositories.earnings_calendar import EarningsCalendarRecordRepository
            from src.database.repositories.metadata import MetadataRepository

            async with get_session() as session:
                await EarningsCalendarRecordRepository(session).create(record)
                await MetadataRepository(session).set("data_pipeline.last_earnings_fetch", now)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to record earnings fetch: {e}")

        self._earnings_cache = None

    async def record_profiling(self, metrics: object) -> None:
        """Record profiling metrics from cycle."""
        from src.daemon.profiling.metrics import ProfilingMetrics

        if not isinstance(metrics, ProfilingMetrics):
            return

        top_func = None
        top_cumtime = None
        if metrics.top_functions:
            top_func = metrics.top_functions[0].function
            top_cumtime = metrics.top_functions[0].cumtime

        record = ProfilingRecord(
            cycle_number=metrics.cycle_number,
            timestamp=metrics.timestamp,
            duration_seconds=metrics.duration_seconds,
            profiling_overhead_percent=metrics.profiling_overhead_percent,
            top_function=top_func,
            top_function_cumtime=top_cumtime,
        )

        try:
            from src.database.connection import get_session
            from src.database.repositories.profiling import ProfilingRecordRepository

            async with get_session() as session:
                await ProfilingRecordRepository(session).create(record)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to record profiling: {e}")

        self._profiling_cache = None

    def __repr__(self) -> str:
        """Return string representation."""
        return "DataPipelineStateManager()"
