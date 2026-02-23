"""Daemon cycle orchestration for coordinating full analysis cycles."""

from __future__ import annotations

import json
import time as time_mod
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import BaseModel
from rich.console import Console

from src.daemon.degradation import AgentType
from src.daemon.notification_helper import DaemonNotificationHelper
from src.strategies.session import TradingSession

if TYPE_CHECKING:
    from src.daemon.degradation import DegradationContext
    from src.daemon.factory import DaemonComponents, DaemonFactory
    from src.daemon.profiling.profiler import CycleProfiler
    from src.daemon.task_runner import ScheduledTaskRunner
    from src.v1.coordinator.models import CoordinatorCycleResult

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
        factory: DaemonFactory,
        profiler: CycleProfiler | None = None,
    ) -> None:
        """Initialize cycle orchestrator.

        Args:
            components: Daemon components
            task_runner: Task runner for scheduled tasks
            factory: DaemonFactory for lazy component initialization
            profiler: Optional cycle profiler
        """
        self.components = components
        self.task_runner = task_runner
        self.factory = factory
        self.profiler = profiler
        self._notification_helper = DaemonNotificationHelper()
        self._cycle_counter = 0
        self._event_batch_evaluator: Any = None  # EventBatchEvaluator (Any to avoid circular import)
        self._cached_health_report: Any = None  # HealthReport cached between cycles

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
        cycle_num = await self.components.state.get_total_analyses()
        profiling_context = self.profiler.profile_cycle(cycle_num) if self.profiler else async_nullcontext()

        async with profiling_context as profile_metrics:
            # Phase 1: Run scheduled tasks (may produce new health report)
            await self.task_runner.run_scheduled_tasks()

            # Refresh health report cache after tasks (health check may have written new report)
            self._refresh_health_report_cache()

            # Phase 3.5: Evaluate event-based discovery candidates
            if (
                self.components.config.event_integration.enable_discovery_integration
                and self.components.supervisor
            ):
                try:
                    from src.daemon.event_batch_evaluator import EventBatchEvaluator

                    if self._event_batch_evaluator is None:
                        self._event_batch_evaluator = EventBatchEvaluator(
                            supervisor=self.components.supervisor,
                            state=self.components.state,
                            config=self.components.config,
                            broker_manager=self.components.broker_manager,
                            broker=self.components.broker,
                        )

                    if await self._event_batch_evaluator.should_evaluate_batch():
                        approved_symbols = await self._event_batch_evaluator.evaluate_batch()
                        if approved_symbols:
                            logger.info(
                                f"Event batch: {len(approved_symbols)} candidates approved for watchlist"
                            )
                except Exception as e:
                    logger.opt(exception=True).warning(f"Event batch evaluation failed: {e}")

            # Phase 3.6: Cleanup completed execution trackers (persist to database)
            try:
                await self.components.state.cleanup_completed_trackers()
            except Exception as e:
                logger.opt(exception=True).warning(f"Execution tracker cleanup failed: {e}")

            # Phase 4: Evaluate degradation before analysis
            degradation_context = self._evaluate_degradation()

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
            watchlist = await self.components.broker_manager.get_merged_watchlist()
            logger.info(f"Starting analysis cycle for {len(watchlist)} symbols")
            console.print(f"\n[bold]Running analysis cycle...[/bold] ({datetime.now(tz=UTC):%H:%M:%S})")

            cycle_result = await self._run_coordinator_cycle(watchlist, degradation_context)

            # Phase 7-8: Handled by coordinator cycle method

            # Record profiling metrics to state
            if profile_metrics:
                await self.components.state.record_profiling(profile_metrics)

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
            logger.warning(
                f"Analysis HALTED: {degradation_context.halt_reason} | "
                f"Unavailable services: {', '.join(degradation_context.unavailable_services) or 'none'} | "
                f"Available agents: {len(degradation_context.available_agents)}/{len(AgentType)}"
            )
            console.print(f"[red]HALTED: {degradation_context.halt_reason}[/red]")
            console.print(f"[dim]Unavailable: {', '.join(degradation_context.unavailable_services)}[/dim]")

            # Notify on every halted cycle
            if self.components.notification_service:
                await self._notification_helper.notify_degradation(degradation_context, self.components)

            # Record in state
            await self.components.state.record_degradation(degradation_context)

            return CycleResult(
                sleep_seconds=60,
                analysis_performed=False,
                halted=True,
                degradation_tier=degradation_context.tier.value,
                results_count=0,
            )

        # Log degradation status if not NONE
        if degradation_context.tier != DegradationTier.NONE:
            unavailable_agents = set(AgentType) - degradation_context.available_agents
            logger.warning(
                f"Degraded mode: {degradation_context.tier} | "
                f"Unavailable services: {', '.join(degradation_context.unavailable_services)} | "
                f"Unavailable agents: {', '.join(str(a) for a in sorted(unavailable_agents))} | "
                f"Confidence adjustment: {degradation_context.confidence_adjustment:.2f}"
            )
            console.print(f"[yellow]DEGRADED: {degradation_context.tier}[/yellow]")
            console.print(
                f"[dim]Unavailable: {', '.join(degradation_context.unavailable_services)} | "
                f"Confidence: {degradation_context.confidence_adjustment:.0%}[/dim]"
            )

            # Notify on every degraded cycle
            if self.components.notification_service:
                await self._notification_helper.notify_degradation(degradation_context, self.components)

            await self.components.state.record_degradation(degradation_context)

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

        # Sync positions with broker every cycle
        await self._sync_coordinator_positions()

        # Run coordinator cycle
        coordinator_result: CoordinatorCycleResult = await self._run_coordinator_cycle_impl(
            watchlist, degradation_context, trading_session
        )

        cycle_duration = time_mod.time() - cycle_start_time

        # Pattern detection (every Nth cycle if enabled)
        self._cycle_counter += 1
        patterns_detected = await self._run_pattern_detection()

        # Save metrics
        await self._save_coordinator_metrics(coordinator_result, patterns_detected)

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
        if patterns_detected > 0:
            console.print(f"Patterns detected: {patterns_detected}")
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
                "patterns_detected": patterns_detected,
            },
        )

        # Convert to CycleResult
        return CycleResult(
            sleep_seconds=self.components.config.interval_minutes * 60,
            analysis_performed=True,
            halted=False,
            degradation_tier=degradation_context.tier.value,
            results_count=len(coordinator_result.symbols_analyzed),
        )

    async def _sync_coordinator_positions(self) -> None:
        """Sync daemon state positions with broker (reconciliation)."""
        position_manager = self.components.position_manager
        if not position_manager:
            return

        try:
            all_positions = await self.components.state.get_all_positions()
            state_positions: dict[str, Any] = {pos.symbol: pos for pos in all_positions}

            new_positions, updated_positions, closed_symbols = position_manager.sync_with_broker(
                state_positions
            )
            for pos in new_positions:
                await self.components.state.add_position(pos)
            for pos in updated_positions:
                await self.components.state.update_position(pos)
            for symbol in closed_symbols:
                await self.components.state.remove_position(symbol)

            if new_positions or updated_positions or closed_symbols:
                logger.info(
                    f"Position sync: +{len(new_positions)} ~{len(updated_positions)} -{len(closed_symbols)}"
                )
        except Exception as e:
            logger.opt(exception=True).warning(f"Coordinator position sync failed: {e}")

    async def _save_coordinator_metrics(
        self,
        result: CoordinatorCycleResult,
        patterns_detected: int,
    ) -> None:
        """Save coordinator cycle metrics to PostgreSQL (with JSONL fallback).

        Args:
            result: Coordinator cycle result
            patterns_detected: Number of patterns detected
        """
        try:
            from pathlib import Path

            from src.v1.coordinator.metrics import CoordinatorCycleMetrics, save_metrics_jsonl

            # Create metrics record
            metrics = CoordinatorCycleMetrics(
                cycle_num=self._cycle_counter,
                timestamp=datetime.now(UTC),
                symbols_analyzed=result.symbols_analyzed,
                tool_calls_made=result.tool_calls_made,
                trades_proposed=result.trades_proposed,
                trades_executed=result.trades_executed,
                game_plan_generated=result.game_plan_generated,
                cycle_duration_seconds=result.cycle_duration_seconds,
                patterns_detected=patterns_detected,
            )

            # Try PostgreSQL first
            try:
                if self.components.container:
                    repo = self.components.container.coordinator_metrics_repository()
                    async with repo:
                        await repo.create(metrics)
                    logger.debug(f"Saved coordinator metrics to PostgreSQL: cycle={self._cycle_counter}")
                    return
            except Exception as pg_error:
                logger.opt(exception=True).warning(
                    f"PostgreSQL save failed, using JSONL fallback: {pg_error}"
                )

            # Fallback to JSONL
            metrics_file = Path.home() / ".ai-casino" / "coordinator-metrics.jsonl"
            save_metrics_jsonl(metrics, metrics_file)

        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to save coordinator metrics: {e}")

    async def _run_pattern_detection(self) -> int:
        """Run pattern detection if enabled and on schedule.

        Returns:
            Number of patterns detected
        """
        if not self.components.coordinator:
            return 0

        pattern_config = self.components.config.coordinator.pattern_detection
        if not pattern_config.enabled:
            return 0

        # Check if this cycle should run pattern detection
        if self._cycle_counter % pattern_config.detection_frequency != 0:
            return 0

        try:
            from src.v1.coordinator.pattern_analyzer import PatternAnalyzer

            # Create pattern analyzer with coordinator's memory
            pattern_analyzer = PatternAnalyzer(
                database_engine=self.components.container.database_engine(),
                memory=self.components.coordinator.memory,
                min_sample_size=pattern_config.min_sample_size,
            )

            # Run pattern detection
            insights = await pattern_analyzer.analyze_patterns(lookback_days=pattern_config.lookback_days)

            # Save insights to coordinator memory
            for insight in insights:
                await self.components.coordinator.memory.save(
                    observation=f"{insight.insight_text} (Recommendation: {insight.recommendation})",
                    category="pattern",
                )

            logger.info(f"Pattern detection complete: {len(insights)} insights")
            return len(insights)

        except Exception as e:
            logger.opt(exception=True).warning(f"Pattern detection failed: {e}")
            return 0

    def _evaluate_degradation(self) -> DegradationContext:
        """Evaluate degradation using cached health report (refreshed by health task)."""
        from src.daemon.degradation import DegradationPolicy

        # Use cached report if available, otherwise load from disk once
        if self._cached_health_report is None:
            self._refresh_health_report_cache()

        policy = DegradationPolicy(self.components.config)
        return policy.evaluate_degradation(self._cached_health_report)

    def _refresh_health_report_cache(self) -> None:
        """Refresh in-memory health report cache from disk."""
        from src.daemon.health import HealthReport

        health_dir = Path(self.components.config.health.health_dir).expanduser()
        if not health_dir.exists():
            return

        report_files = sorted(health_dir.glob("health-*.json"), reverse=True)
        if report_files:
            try:
                with report_files[0].open() as f:
                    self._cached_health_report = HealthReport.model_validate(json.load(f))
            except Exception as e:
                logger.opt(exception=True).warning(f"Failed to load health report: {e}")

    async def _run_coordinator_cycle_impl(
        self,
        watchlist: list[str],
        degradation_context: DegradationContext,
        trading_session: TradingSession,
    ) -> CoordinatorCycleResult:
        """Run coordinator-driven cycle.

        Args:
            watchlist: Symbols to analyze
            degradation_context: Degradation context for cycle
            trading_session: Trading session type (REGULAR or PRE_MARKET)

        Returns:
            CoordinatorCycleResult from coordinator
        """
        from src.daemon.degradation import AgentType, DegradationTier

        coordinator = self.factory.init_coordinator(self.components)

        degradation_dict = None
        if degradation_context.tier != DegradationTier.NONE:
            degradation_dict = {
                "tier": degradation_context.tier.value,
                "unavailable_services": degradation_context.unavailable_services,
                "confidence_adjustment": degradation_context.confidence_adjustment,
                "disabled_agents": [
                    str(agent) for agent in AgentType if agent not in degradation_context.available_agents
                ],
            }

        return await coordinator.run_cycle(watchlist, degradation_dict, trading_session)

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
            logger.opt(exception=True).error(f"Failed to publish {event_type} event: {e}")
