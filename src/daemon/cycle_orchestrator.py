"""Daemon cycle orchestration for coordinating full analysis cycles."""

from __future__ import annotations

import time as time_mod
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel
from rich.console import Console

from src.daemon.notification_helper import DaemonNotificationHelper
from src.workflows.types import TradingWorkflowResult

if TYPE_CHECKING:
    from src.daemon.factory import DaemonComponents
    from src.daemon.profiling.profiler import CycleProfiler
    from src.daemon.task_runner import ScheduledTaskRunner

console = Console()


class CycleResult(BaseModel):
    """Result from a single daemon cycle."""

    sleep_seconds: int
    analysis_performed: bool
    halted: bool
    degradation_tier: str
    results_count: int


class DaemonCycleOrchestrator:
    """Orchestrate full daemon cycles (tasks, analysis, health, discovery)."""

    def __init__(
        self,
        components: DaemonComponents,
        task_runner: ScheduledTaskRunner,
        runner: object,  # DaemonRunner (can't import due to circular dependency)
        profiler: CycleProfiler | None = None,
    ) -> None:
        """Initialize cycle orchestrator.

        Args:
            components: Daemon components
            task_runner: Task runner for scheduled tasks
            runner: DaemonRunner for delegation to helper methods
            profiler: Optional cycle profiler
        """
        self.components = components
        self.task_runner = task_runner
        self.runner = runner  # For delegating to runner methods
        self.profiler = profiler
        self._notification_helper = DaemonNotificationHelper()

    def __repr__(self) -> str:
        """Return string representation."""
        return "DaemonCycleOrchestrator()"

    async def run_cycle(self) -> CycleResult:
        """Run single daemon cycle: tasks → health → discovery → degradation → analysis → journal.

        Returns:
            CycleResult with sleep time and cycle info
        """
        from src.daemon.degradation import DegradationTier
        from src.daemon.profiling.profiler import async_nullcontext

        # Start profiling if enabled
        cycle_num = self.components.state.total_analyses
        profiling_context = (
            self.profiler.profile_cycle(cycle_num) if self.profiler else async_nullcontext()
        )

        async with profiling_context as profile_metrics:
            # Phase 1: Run scheduled tasks
            await self.task_runner.run_scheduled_tasks()

            # Phase 2: Run health check (via task service now)
            # Already handled by task_runner if scheduled

            # Phase 3: Run discovery (via task service now)
            # Already handled by task_runner if scheduled

            # Phase 4: Evaluate degradation before analysis
            degradation_context = self.runner._evaluate_degradation()  # type: ignore[attr-defined]  # noqa: SLF001

            if degradation_context.tier == DegradationTier.HALTED:
                logger.warning(f"Analysis HALTED: {degradation_context.halt_reason}")
                console.print(f"[red]HALTED: {degradation_context.halt_reason}[/red]")

                # Notify on every halted cycle
                if self.components.notification_service:
                    await self._notification_helper.notify_degradation(degradation_context, self.components)

                # Record in state
                self.components.state.record_degradation(degradation_context)
                self.components.state.save(self.components.config.state.state_file)

                return CycleResult(
                    sleep_seconds=60,
                    analysis_performed=False,
                    halted=True,
                    degradation_tier=degradation_context.tier.value,
                    results_count=0,
                )

            # Log degradation status if not FULL
            if degradation_context.tier != DegradationTier.FULL:
                logger.warning(
                    f"Degraded mode: {degradation_context.tier}, "
                    f"unavailable: {degradation_context.unavailable_services}"
                )
                console.print(f"[yellow]DEGRADED: {degradation_context.tier}[/yellow]")

                # Notify on every degraded cycle
                if self.components.notification_service:
                    await self._notification_helper.notify_degradation(degradation_context, self.components)

                self.components.state.record_degradation(degradation_context)

            # Phase 5: Check market hours
            if self.components.config.market_hours_only and not self.components.scheduler.is_market_open():
                wait_time = self.components.scheduler.time_until_open()
                if wait_time > 0:
                    logger.info(f"Market closed, waiting {wait_time // 60} minutes until open")
                    return CycleResult(
                        sleep_seconds=min(wait_time, 60),
                        analysis_performed=False,
                        halted=False,
                        degradation_tier=degradation_context.tier.value,
                        results_count=0,
                    )

            # Phase 6: Run watchlist analysis
            watchlist = self.components.broker_manager.get_merged_watchlist()
            logger.info(f"Starting analysis cycle for {len(watchlist)} symbols")
            console.print(f"\n[bold]Running analysis cycle...[/bold] ({datetime.now(tz=UTC):%H:%M:%S})")

            await self._publish_event(
                "CYCLE_START",
                {"watchlist_size": len(watchlist), "degradation_tier": str(degradation_context.tier)},
            )

            cycle_start_time = time_mod.time()
            results = await self.runner._analyze_watchlist(watchlist, degradation_context)  # type: ignore[attr-defined]  # noqa: SLF001
            cycle_duration = time_mod.time() - cycle_start_time

            # Phase 7: Log results
            self._log_results(results)

            # Count results with warnings as potential errors
            error_count = sum(1 for r in results if r.warnings)
            await self._publish_event(
                "CYCLE_COMPLETE",
                {
                    "results_count": len(results),
                    "errors_count": error_count,
                    "duration_seconds": round(cycle_duration, 2),
                },
            )

            # Phase 8: Run journal and paper readiness check
            # These are now handled by task service via scheduled tasks

            # Record profiling metrics to state
            if profile_metrics:
                self.components.state.record_profiling(profile_metrics)

            # Phase 10: Save state
            self.components.state.save(self.components.config.state.state_file)

            return CycleResult(
                sleep_seconds=self.components.config.interval_minutes * 60,
                analysis_performed=True,
                halted=False,
                degradation_tier=degradation_context.tier.value,
                results_count=len(results),
            )

    def _log_results(self, results: list[TradingWorkflowResult]) -> None:
        """Log analysis results to console.

        Args:
            results: List of analysis results
        """
        from src.strategies.session import TradingSession

        console.print(f"\n[bold cyan]Analysis Results ({datetime.now(tz=UTC):%Y-%m-%d %H:%M})[/bold cyan]")
        console.print("-" * 50)

        for result in results:
            signal = result.decision.action.value
            color = {"BUY": "green", "SELL": "red"}.get(signal, "yellow")

            # Add pre-market badge if applicable
            session_badge = ""
            if result.trading_session == TradingSession.PRE_MARKET:
                session_badge = " [dim](PRE-MARKET)[/dim]"

            console.print(
                f"[bold]{result.symbol}[/bold]: "
                f"[{color}]{signal}[/{color}] "
                f"(confidence: {result.decision.confidence:.2f}){session_badge}"
            )

        console.print("-" * 50)
        console.print(f"Total: {len(results)} symbols analyzed\n")

    async def _publish_event(self, event_type: str, data: dict[str, object]) -> None:
        """Publish event to event bus.

        Args:
            event_type: Event type string
            data: Event data dictionary
        """
        if not self.components.event_bus:
            return

        try:
            from src.daemon.event_bus import DashboardEvent, EventType

            await self.components.event_bus.publish(
                DashboardEvent(event_type=EventType[event_type], data=data)
            )
        except Exception as e:
            logger.error(f"Failed to publish {event_type} event: {e}")
