"""Daemon cycle orchestration for coordinating full analysis cycles."""

from __future__ import annotations

import time as time_mod
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel
from rich.console import Console

from src.daemon.notification_helper import DaemonNotificationHelper
from src.strategies.session import TradingSession
from src.workflows.types import TradingWorkflowResult

if TYPE_CHECKING:
    from src.coordinator.models import CoordinatorCycleResult
    from src.daemon.degradation import DegradationContext
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
        from src.daemon.profiling.profiler import async_nullcontext

        # Start profiling if enabled
        cycle_num = self.components.state.total_analyses
        profiling_context = self.profiler.profile_cycle(cycle_num) if self.profiler else async_nullcontext()

        async with profiling_context as profile_metrics:
            # Phase 1: Run scheduled tasks
            await self.task_runner.run_scheduled_tasks()

            # Phase 2: Run health check (via task service now)
            # Already handled by task_runner if scheduled

            # Phase 3: Run discovery (via task service now)
            # Already handled by task_runner if scheduled

            # Phase 4: Evaluate degradation before analysis
            degradation_context = self.runner._evaluate_degradation()  # type: ignore[attr-defined]  # noqa: SLF001

            # Check for halted state (returns early if halted)
            halted_result = await self._handle_degradation_state(degradation_context)
            if halted_result:
                return halted_result

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

            # Route based on coordinator feature flag
            cycle_result = None
            if self.components.config.coordinator.enabled:
                try:
                    cycle_result = await self._run_coordinator_cycle(watchlist, degradation_context)
                except Exception as e:
                    logger.opt(exception=True).error(f"Coordinator cycle failed: {e}, falling back to legacy")
                    # Fall through to legacy cycle

            if cycle_result is None:
                logger.debug("Using legacy cycle (coordinator disabled or failed)")
                cycle_result = await self._run_legacy_cycle(watchlist, degradation_context)

            # Phase 7-8: Handled by coordinator/legacy cycle methods

            # Record profiling metrics to state
            if profile_metrics:
                self.components.state.record_profiling(profile_metrics)

            return cycle_result

    async def _handle_degradation_state(
        self,
        degradation_context: DegradationContext,
    ) -> CycleResult | None:
        """Handle degradation state (halted or degraded).

        Args:
            degradation_context: Degradation context

        Returns:
            CycleResult if halted, None otherwise
        """
        from src.daemon.degradation import DegradationTier

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

        return None

    async def _run_coordinator_cycle(
        self,
        watchlist: list[str],
        degradation_context: DegradationContext,
    ) -> CycleResult:
        """Run coordinator-driven analysis cycle.

        Args:
            watchlist: Symbols to analyze
            degradation_context: Degradation context

        Returns:
            CycleResult with coordinator metrics
        """
        # Get current trading session
        trading_session = self.components.scheduler.get_trading_session()
        if trading_session is None:
            trading_session = TradingSession.REGULAR  # Default fallback

        # Publish cycle start event
        await self._publish_event(
            "CYCLE_START",
            {
                "watchlist_size": len(watchlist),
                "degradation_tier": str(degradation_context.tier),
                "mode": "coordinator",
            },
        )

        cycle_start_time = time_mod.time()

        # Run coordinator cycle
        coordinator_result: CoordinatorCycleResult = await self.runner._run_coordinator_cycle(  # type: ignore[attr-defined]  # noqa: SLF001
            watchlist, degradation_context, trading_session
        )

        cycle_duration = time_mod.time() - cycle_start_time

        # Log coordinator-specific results
        console.print(
            f"\n[bold cyan]Coordinator Cycle Results ({datetime.now(tz=UTC):%Y-%m-%d %H:%M})[/bold cyan]"
        )
        console.print("-" * 50)
        console.print(f"Symbols analyzed: {len(coordinator_result.symbols_analyzed)}")
        console.print(
            f"Trades executed: {coordinator_result.trades_executed}/{coordinator_result.trades_proposed}"
        )
        console.print(f"Tool calls made: {coordinator_result.tool_calls_made}")
        console.print(f"Game plan generated: {coordinator_result.game_plan_generated}")
        console.print(f"Summary: {coordinator_result.summary}")
        console.print("-" * 50 + "\n")

        # Publish cycle complete event with coordinator metrics
        await self._publish_event(
            "CYCLE_COMPLETE",
            {
                "results_count": len(coordinator_result.symbols_analyzed),
                "errors_count": 0,
                "duration_seconds": round(cycle_duration, 2),
                "mode": "coordinator",
                "tool_calls": coordinator_result.tool_calls_made,
                "trades_executed": coordinator_result.trades_executed,
            },
        )

        # Save state (coordinator memory auto-saves observations)
        self.components.state.save(self.components.config.state.state_file)

        # Convert to CycleResult
        return CycleResult(
            sleep_seconds=self.components.config.interval_minutes * 60,
            analysis_performed=True,
            halted=False,
            degradation_tier=degradation_context.tier.value,
            results_count=len(coordinator_result.symbols_analyzed),
        )

    async def _run_legacy_cycle(
        self,
        watchlist: list[str],
        degradation_context: DegradationContext,
    ) -> CycleResult:
        """Run legacy watchlist-driven analysis cycle.

        Args:
            watchlist: Symbols to analyze
            degradation_context: Degradation context

        Returns:
            CycleResult with legacy metrics
        """
        # Publish cycle start event
        await self._publish_event(
            "CYCLE_START",
            {"watchlist_size": len(watchlist), "degradation_tier": str(degradation_context.tier)},
        )

        cycle_start_time = time_mod.time()
        results = await self.runner._analyze_watchlist(watchlist, degradation_context)  # type: ignore[attr-defined]  # noqa: SLF001
        cycle_duration = time_mod.time() - cycle_start_time

        # Log results
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

        # Save state
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
