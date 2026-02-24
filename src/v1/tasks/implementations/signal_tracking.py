"""Signal tracking task on v1 task framework."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger

from src.v1.tasks.interface import Task
from src.v1.tasks.models import WEEKDAYS, DedupStrategy, TaskResult, TaskSchedule

if TYPE_CHECKING:
    from src.cache.historical import HistoricalCache
    from src.daemon.config.reporting import SignalTrackingConfig
    from src.daemon.state import DaemonState
    from src.data.market import MarketDataFetcher
    from src.v1.trades.brokers import Broker


class SignalTrackingTask(Task):
    """Batch-updates T+1d/5d/20d outcome prices for tracked signals."""

    def __init__(
        self,
        historical_cache: HistoricalCache,
        market_fetcher: MarketDataFetcher,
        state: DaemonState,
        config: SignalTrackingConfig,
        broker: Broker | None = None,
    ) -> None:
        """Initialize signal tracking task.

        Args:
            historical_cache: Cache for reading/writing signal outcomes
            market_fetcher: Fetcher for OHLCV price data
            state: Daemon state for dedup timestamps
            config: Signal tracking configuration
            broker: Optional broker for early exit detection
        """
        self._cache = historical_cache
        self._market_fetcher = market_fetcher
        self._state = state
        self._config = config
        self._broker = broker

    @property
    def name(self) -> str:
        """Task identifier."""
        return "signal_tracking"

    @property
    def schedule(self) -> TaskSchedule:
        """Schedule from config."""
        return TaskSchedule(
            time=self._config.tracking_time,
            days=WEEKDAYS,
            enabled=self._config.enabled,
            dedup=DedupStrategy.DAILY,
        )

    async def execute(self) -> TaskResult:
        """Update signal outcomes with T+1d/5d/20d prices.

        Returns:
            TaskResult with outcome stats
        """
        from src.daemon.signal_tracker import SignalOutcomeTracker

        start = time.monotonic()
        try:
            tracker = SignalOutcomeTracker(self._cache, self._market_fetcher, self._broker)
            stats = await asyncio.to_thread(tracker.update_outcomes)
            await self._state.set_last_signal_tracking(datetime.now(UTC))
            total = sum(stats.values())
            msg = f"updated={total} {stats}"
            logger.info(f"Signal tracking complete: {msg}")
            return TaskResult(
                task_name=self.name,
                success=True,
                duration_seconds=time.monotonic() - start,
                message=msg,
            )
        except Exception as e:
            logger.opt(exception=True).error(f"Signal tracking failed: {e}")
            return TaskResult(
                task_name=self.name,
                success=False,
                duration_seconds=time.monotonic() - start,
                message=str(e),
            )

    async def last_run_at(self) -> datetime | None:
        """Get last signal tracking timestamp from state."""
        return await self._state.get_last_signal_tracking()

    def __repr__(self) -> str:
        """String representation."""
        return f"SignalTrackingTask(enabled={self._config.enabled}, time={self._config.tracking_time})"
