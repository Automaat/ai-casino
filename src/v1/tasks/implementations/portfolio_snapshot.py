"""Portfolio snapshot task on v1 task framework."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger
from result import Err

from src.database.repositories.snapshot import PortfolioSnapshot, PortfolioSnapshotRepository
from src.v1.tasks.interface import Task
from src.v1.tasks.models import WEEKDAYS, DedupStrategy, TaskResult, TaskSchedule

if TYPE_CHECKING:
    from src.daemon.config.portfolio import PortfolioSnapshotConfig
    from src.database.engine import DatabaseEngine
    from src.v1.trades.brokers import Broker


class PortfolioSnapshotTask(Task):
    """Periodic portfolio snapshot capture."""

    def __init__(
        self,
        broker: Broker,
        database_engine: DatabaseEngine,
        config: PortfolioSnapshotConfig,
    ) -> None:
        """Initialize portfolio snapshot task.

        Args:
            broker: Broker for account info
            database_engine: Database engine for persistence
            config: Portfolio snapshot configuration
        """
        self._broker = broker
        self._database_engine = database_engine
        self._config = config

    @property
    def name(self) -> str:
        """Task identifier."""
        return "portfolio_snapshot"

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
        """Capture portfolio snapshot and persist.

        Returns:
            TaskResult with outcome
        """
        start = time.monotonic()

        _result = await asyncio.to_thread(self._broker.get_account_info)
        if isinstance(_result, Err):
            msg = f"Broker API unavailable: {_result.err_value}"
            logger.opt(exception=_result.err_value).error(msg)
            return TaskResult(
                task_name=self.name, success=False, duration_seconds=time.monotonic() - start, message=msg
            )
        account_info = _result.ok()
        async with self._database_engine.session() as session:
            repo = PortfolioSnapshotRepository(session)
            await repo.create(
                PortfolioSnapshot(
                    timestamp=datetime.now(UTC),
                    balance=account_info.balance,
                    available_cash=account_info.available_cash,
                    total_exposure=account_info.total_exposure,
                    portfolio_value=account_info.portfolio_value,
                    positions={k: v.model_dump() for k, v in account_info.positions.items()},
                    trigger="SCHEDULED",
                )
            )

        duration = time.monotonic() - start
        msg = f"balance={account_info.balance:.2f}, portfolio={account_info.portfolio_value:.2f}"
        logger.info(f"Portfolio snapshot captured: {msg}")

        return TaskResult(
            task_name=self.name,
            success=True,
            duration_seconds=duration,
            message=msg,
        )

    async def last_run_at(self) -> datetime | None:
        """Get last snapshot timestamp from database."""
        async with self._database_engine.session() as session:
            latest = await PortfolioSnapshotRepository(session).get_latest()
            return latest.timestamp if latest else None

    def __repr__(self) -> str:
        """String representation."""
        enabled = self._config.enabled
        interval = self._config.interval_minutes
        return f"PortfolioSnapshotTask(enabled={enabled}, interval={interval}m)"
