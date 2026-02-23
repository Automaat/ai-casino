"""Health check task on v1 task framework."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger

from src.v1.tasks.interface import Task
from src.v1.tasks.models import DayOfWeek, DedupStrategy, TaskResult, TaskSchedule

if TYPE_CHECKING:
    from src.daemon.config import DaemonConfig
    from src.daemon.state.facade import DaemonState
    from src.di.container import AppContainer
    from src.v1.notifications.service import NotificationService


class HealthCheckTask(Task):
    """Periodic API health check and system cleanup."""

    def __init__(
        self,
        config: DaemonConfig,
        state: DaemonState,
        container: AppContainer,
        notification_service: NotificationService | None,
    ) -> None:
        """Initialize health check task.

        Args:
            config: Daemon configuration
            state: Daemon state
            container: DI container
            notification_service: Optional notification service for alerts
        """
        self._config = config
        self._state = state
        self._container = container
        self._notification_service = notification_service

    @property
    def name(self) -> str:
        """Task identifier."""
        return "health_check"

    @property
    def schedule(self) -> TaskSchedule:
        """Schedule from config."""
        return TaskSchedule(
            days=list(DayOfWeek),
            enabled=self._config.health.enabled,
            dedup=DedupStrategy.INTERVAL,
            dedup_interval_minutes=max(1, round(self._config.health.check_interval_seconds / 60)),
        )

    async def execute(self) -> TaskResult:
        """Run health checks and cleanup.

        Returns:
            TaskResult with overall status and service count
        """
        from src.daemon.health import HealthChecker

        start = time.monotonic()
        try:
            checker = HealthChecker(
                self._config,
                self._state,
                container=self._container,
                notification_service=self._notification_service,
            )
            report = await checker.run()

            await self._state.set_last_health_check(datetime.now(UTC))

            duration = time.monotonic() - start
            msg = (
                f"status={report.overall_status}, "
                f"services={len(report.service_checks)}, "
                f"duration={report.total_duration_ms:.0f}ms"
            )
            logger.info(f"Health check complete: {msg}")
            return TaskResult(task_name=self.name, success=True, duration_seconds=duration, message=msg)
        except Exception as e:
            duration = time.monotonic() - start
            msg = f"Health check failed: {e}"
            logger.opt(exception=True).error(msg)
            return TaskResult(task_name=self.name, success=False, duration_seconds=duration, message=msg)

    async def last_run_at(self) -> datetime | None:
        """Get last health check timestamp from state."""
        return await self._state.get_last_health_check()

    def __repr__(self) -> str:
        """String representation."""
        enabled = self._config.health.enabled
        interval = self._config.health.check_interval_seconds
        return f"HealthCheckTask(enabled={enabled}, interval={interval}s)"
