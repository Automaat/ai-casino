"""Monitoring tasks for daemon health, signal tracking, and stress testing."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger
from rich.console import Console

from src.daemon.tasks.base import TaskExecutor

if TYPE_CHECKING:
    from src.daemon.factory import DaemonComponents
    from src.di.container import AppContainer

console = Console()


class HealthCheckTask(TaskExecutor):
    """API health check task with interval-based dedup."""

    def __init__(self, components: DaemonComponents, container: AppContainer) -> None:
        """Initialize health check task.

        Args:
            components: Daemon components
            container: DI container
        """
        super().__init__(components, container)
        self._report = None

    @property
    def task_name(self) -> str:
        """Task display name."""
        return "Health Checks"

    async def execute(self) -> None:
        """Execute health check logic."""
        from src.daemon.health import HealthChecker

        checker = HealthChecker(
            self.components.config,
            self.components.state,
            container=self.container,
            notification_service=self.components.notification_service,
        )
        self._report = await checker.run()

        console.print(
            f"[bold cyan]Health:[/bold cyan] {self._report.overall_status} "
            f"({len(self._report.service_checks)} services, {self._report.total_duration_ms:.0f}ms)"
        )

        # Publish HEALTH_CHECK event
        if self.components.event_bus:
            try:
                from src.daemon.event_bus import DashboardEvent, EventType

                failures = [
                    svc.service_name for svc in self._report.service_checks if svc.status == "UNHEALTHY"
                ]
                await self.components.event_bus.publish(
                    DashboardEvent(
                        event_type=EventType.HEALTH_CHECK,
                        data={
                            "status": self._report.overall_status.value,
                            "failures": failures,
                            "total_duration_ms": self._report.total_duration_ms,
                        },
                    )
                )
            except Exception as e:
                logger.opt(exception=True).error(f"Failed to publish HEALTH_CHECK event: {e}")

    async def get_last_run(self) -> datetime | None:
        """Get last health check timestamp."""
        return await self.components.state.get_last_health_check()

    async def record_success(self, duration: float) -> None:
        """Record health check completion."""
        await self.components.state.set_last_health_check(datetime.now(tz=self.components.scheduler.timezone))

    async def should_skip_today(self) -> bool:
        """Custom dedup: check interval instead of daily.

        Returns:
            True if check interval hasn't elapsed
        """
        last_run = await self.get_last_run()
        if not last_run:
            return False

        now = datetime.now(tz=UTC)
        elapsed = (now - last_run).total_seconds()
        return elapsed < self.components.config.health.check_interval_seconds


class SignalTrackingTask(TaskExecutor):
    """Signal outcome tracking task (T+1d/5d/20d price updates)."""

    @property
    def task_name(self) -> str:
        """Task display name."""
        return "Signal Tracking"

    async def execute(self) -> None:
        """Execute signal tracking logic."""
        from src.daemon.signal_tracker import SignalOutcomeTracker

        market_fetcher = self.container.yfinance_market_fetcher()
        tracker = SignalOutcomeTracker(
            self.components.historical_cache, market_fetcher, self.components.broker
        )
        stats = await asyncio.to_thread(tracker.update_outcomes)
        console.print(f"[dim]Signal tracking: {stats}[/dim]")

    async def get_last_run(self) -> datetime | None:
        """Get last signal tracking timestamp."""
        return await self.components.state.get_last_signal_tracking()

    async def record_success(self, duration: float) -> None:
        """Record signal tracking completion."""
        await self.components.state.set_last_signal_tracking(datetime.now(UTC))


class MonteCarloTask(TaskExecutor):
    """Monte Carlo portfolio stress testing task (6hr dedup)."""

    def __init__(self, components: DaemonComponents, container: AppContainer) -> None:
        """Initialize Monte Carlo task.

        Args:
            components: Daemon components
            container: DI container
        """
        super().__init__(components, container)
        self._record = None

    @property
    def task_name(self) -> str:
        """Task display name."""
        return "Monte Carlo Stress Testing"

    async def execute(self) -> None:
        """Execute Monte Carlo stress testing logic."""
        from src.daemon.stress_testing import DaemonStressTester

        if self.components.broker is None or self.components.market_fetcher is None:
            logger.warning("Skipping: broker or market_fetcher not configured")
            return

        executor = DaemonStressTester(
            broker_client=self.components.broker,
            market_fetcher=self.components.market_fetcher,
            config=self.components.config.monte_carlo,
        )
        self._record = await asyncio.to_thread(executor.execute)

        if self._record.exceeds_risk_tolerance:
            logger.warning(f"ALERT: {self._record.alert_message}")
        else:
            logger.info(
                f"Test passed - P(loss>threshold)={self._record.prob_loss_gt_threshold:.1%}, "
                f"VaR95={self._record.var_95:.1%}"
            )

    async def get_last_run(self) -> datetime | None:
        """Get last Monte Carlo test timestamp."""
        monte_carlo_tests = await self.components.state.get_monte_carlo_tests()
        if not monte_carlo_tests:
            return None
        return monte_carlo_tests[-1].timestamp

    async def record_success(self, duration: float) -> None:
        """Record Monte Carlo test completion."""
        if self._record:
            await self.components.state.record_monte_carlo_test(self._record)

    async def should_skip_today(self) -> bool:
        """Custom dedup: check last run within 6 hours.

        Returns:
            True if ran within last 6 hours
        """
        last_run = await self.get_last_run()
        if not last_run:
            return False

        now = datetime.now(UTC)
        elapsed = (now - last_run).total_seconds()
        return elapsed < 6 * 3600
