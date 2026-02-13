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
    ScreeningRecord,
)
from src.screening.screener import ScreeningResult

if TYPE_CHECKING:
    from src.database.repositories.earnings_calendar import EarningsCalendarRecordRepository
    from src.database.repositories.metadata import MetadataRepository
    from src.database.repositories.prefetch import PrefetchRecordRepository
    from src.database.repositories.profiling import ProfilingRecordRepository
    from src.database.repositories.screening import ScreeningRecordRepository


class DataPipelineStateManager(StateManager):
    """Background data operations (prefetch, screening, earnings)."""

    _metadata_repository: MetadataRepository | None = PrivateAttr(default=None)
    _prefetch_repository: PrefetchRecordRepository | None = PrivateAttr(default=None)
    _screening_repository: ScreeningRecordRepository | None = PrivateAttr(default=None)
    _earnings_repository: EarningsCalendarRecordRepository | None = PrivateAttr(default=None)
    _profiling_repository: ProfilingRecordRepository | None = PrivateAttr(default=None)

    _prefetch_cache: list[PrefetchRecord] | None = PrivateAttr(default=None)
    _screening_cache: list[ScreeningRecord] | None = PrivateAttr(default=None)
    _earnings_cache: list[EarningsCalendarRecord] | None = PrivateAttr(default=None)
    _profiling_cache: list[ProfilingRecord] | None = PrivateAttr(default=None)

    def set_repositories(
        self,
        metadata_repository: MetadataRepository,
        prefetch_repository: PrefetchRecordRepository,
        screening_repository: ScreeningRecordRepository,
        earnings_repository: EarningsCalendarRecordRepository,
        profiling_repository: ProfilingRecordRepository,
    ) -> None:
        """Inject repositories."""
        self._metadata_repository = metadata_repository
        self._prefetch_repository = prefetch_repository
        self._screening_repository = screening_repository
        self._earnings_repository = earnings_repository
        self._profiling_repository = profiling_repository
        logger.debug("DataPipelineStateManager repositories injected")

    async def get_last_prefetch(self) -> datetime | None:
        """Get last prefetch timestamp from DB."""
        if not self._metadata_repository:
            return None
        return await self._metadata_repository.get("data_pipeline.last_prefetch")

    async def set_last_prefetch(self, value: datetime | None) -> None:
        """Set last prefetch timestamp in DB."""
        if self._metadata_repository:
            await self._metadata_repository.set("data_pipeline.last_prefetch", value)

    async def get_last_pre_market_refresh(self) -> datetime | None:
        """Get last pre-market refresh timestamp from DB."""
        if not self._metadata_repository:
            return None
        return await self._metadata_repository.get("data_pipeline.last_pre_market_refresh")

    async def set_last_pre_market_refresh(self, value: datetime | None) -> None:
        """Set last pre-market refresh timestamp in DB."""
        if self._metadata_repository:
            await self._metadata_repository.set("data_pipeline.last_pre_market_refresh", value)

    async def get_last_after_hours_screening(self) -> datetime | None:
        """Get last after-hours screening timestamp from DB."""
        if not self._metadata_repository:
            return None
        return await self._metadata_repository.get("data_pipeline.last_after_hours_screening")

    async def set_last_after_hours_screening(self, value: datetime | None) -> None:
        """Set last after-hours screening timestamp in DB."""
        if self._metadata_repository:
            await self._metadata_repository.set("data_pipeline.last_after_hours_screening", value)

    async def get_last_earnings_fetch(self) -> datetime | None:
        """Get last earnings fetch timestamp from DB."""
        if not self._metadata_repository:
            return None
        return await self._metadata_repository.get("data_pipeline.last_earnings_fetch")

    async def get_prefetch_history(self, limit: int = 30) -> list[PrefetchRecord]:
        """Get prefetch history with lazy loading."""
        if not self._prefetch_repository:
            return []
        if self._prefetch_cache is None:
            self._prefetch_cache = await self._prefetch_repository.get_recent(limit)
        return self._prefetch_cache

    async def get_screening_history(self, limit: int = 30) -> list[ScreeningRecord]:
        """Get screening history with lazy loading."""
        if not self._screening_repository:
            return []
        if self._screening_cache is None:
            self._screening_cache = await self._screening_repository.get_recent(limit)
        return self._screening_cache

    async def get_earnings_calendar_history(self, limit: int = 10) -> list[EarningsCalendarRecord]:
        """Get earnings calendar history with lazy loading."""
        if not self._earnings_repository:
            return []
        if self._earnings_cache is None:
            self._earnings_cache = await self._earnings_repository.get_recent(limit)
        return self._earnings_cache

    async def get_profiling_history(self, limit: int = 100) -> list[ProfilingRecord]:
        """Get profiling history with lazy loading."""
        if not self._profiling_repository:
            return []
        if self._profiling_cache is None:
            self._profiling_cache = await self._profiling_repository.get_recent(limit)
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

        if self._prefetch_repository:
            await self._prefetch_repository.create(record)
        if self._metadata_repository:
            await self._metadata_repository.set("data_pipeline.last_prefetch", now)

        self._prefetch_cache = None

    async def record_after_hours_screening(
        self,
        criteria: str,
        universe: str,
        candidates: list[ScreeningResult],
        top_n: int = 10,
        screened_at: datetime | None = None,
    ) -> None:
        """Record after-hours screening results."""
        now = datetime.now(UTC)
        top_symbols = [c.symbol for c in candidates[:top_n]]
        record = ScreeningRecord(
            timestamp=now,
            criteria=criteria,
            universe=universe,
            top_symbols=top_symbols,
            candidates=candidates[:top_n],
            screened_at=screened_at or now,
        )

        if self._screening_repository:
            await self._screening_repository.create(record)
        if self._metadata_repository:
            await self._metadata_repository.set("data_pipeline.last_after_hours_screening", now)

        self._screening_cache = None

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

        if self._earnings_repository:
            await self._earnings_repository.create(record)
        if self._metadata_repository:
            await self._metadata_repository.set("data_pipeline.last_earnings_fetch", now)

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

        if self._profiling_repository:
            await self._profiling_repository.create(record)

        self._profiling_cache = None

    def __repr__(self) -> str:
        """Return string representation."""
        return "DataPipelineStateManager()"
