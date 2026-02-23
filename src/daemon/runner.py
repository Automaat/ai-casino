"""Main daemon runner for autonomous trading."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from rich.console import Console

from src.agents.game_plan import GamePlanAgent
from src.daemon.config import DaemonConfig
from src.daemon.factory import DaemonFactory
from src.daemon.notification_helper import DaemonNotificationHelper
from src.workflows.types import TradingWorkflowResult

if TYPE_CHECKING:
    from src.daemon.analysis_orchestrator import AnalysisOrchestrator
    from src.daemon.degradation import DegradationContext
    from src.daemon.event_bus import EventBus
    from src.daemon.factory import DaemonComponents
    from src.di.container import AppContainer
    from src.v1.coordinator.agent import TradingCoordinator
    from src.v1.tasks.runner import TaskRunner
    from src.workflows import TradingWorkflow

console = Console()


class DaemonRunner:
    """Main daemon runner for autonomous trading."""

    def __init__(
        self,
        config: DaemonConfig,
        event_bus: EventBus | None = None,
        container: AppContainer | None = None,
    ) -> None:
        """Initialize daemon runner.

        Args:
            config: Daemon configuration
            event_bus: Optional EventBus for real-time event streaming
            container: Optional DI container (auto-created if not provided)
        """
        self.event_bus = event_bus

        # Create factory and components
        factory = DaemonFactory(config, container)
        self._components: DaemonComponents = factory.create_components(event_bus)
        self._factory = factory

        # Expose frequently-used components for backward compatibility
        self.config = self._components.config
        self.state = self._components.state
        self.scheduler = self._components.scheduler
        self.broker = self._components.broker
        self._broker_manager = self._components.broker_manager
        self._container = self._components.container
        self._historical_cache = self._components.historical_cache
        self.running = False

        # Lazy-initialized components
        self._workflow = self._components.workflow
        self._metrics_tracker = self._components.metrics_tracker
        self._analysis_orchestrator = self._components.analysis_orchestrator

        # Optional components
        self.param_store = self._components.param_store
        self._daemon_optimizer = self._components.daemon_optimizer
        self._daemon_rebalancer = self._components.daemon_rebalancer
        self._position_manager = self._components.position_manager
        self.notification_service = self._components.notification_service
        self._tearsheet_generator = self._components.tearsheet_generator
        self._prefetcher = self._components.prefetcher
        self._game_plan_agent = self._components.game_plan_agent
        self.market_fetcher = self._components.market_fetcher

        # Task runner (backward compat - set runner reference)
        self._task_runner = self._components.task_runner
        self._task_runner._runner = self  # type: ignore[assignment]  # noqa: SLF001

        # Wire task service to task runner
        task_service = self._container.task_service(
            components=self._components,
            container=self._container,
        )
        self._task_runner.set_task_service(task_service)

        # Notification helper
        self._notification_helper = DaemonNotificationHelper()

        # Event queue consumer task
        self._consumer_task: asyncio.Task | None = None

        # v1 task runner
        self._v1_task_runner = self._build_v1_task_runner()

        # Backward compatibility
        self._target_allocations_to_apply: dict[str, float] | None = None

        logger.info(f"DaemonRunner initialized with {config}")

    def _build_v1_task_runner(self) -> TaskRunner:
        """Build v1 TaskRunner with registered tasks.

        Returns:
            TaskRunner instance
        """
        from zoneinfo import ZoneInfo

        from src.database.engine import MissingDatabaseURLError
        from src.v1.tasks.implementations.game_plan import GamePlanTask
        from src.v1.tasks.implementations.portfolio_snapshot import PortfolioSnapshotTask
        from src.v1.tasks.implementations.position_review import PositionReviewTask
        from src.v1.tasks.interface import Task
        from src.v1.tasks.runner import TaskRunner

        tasks: list[Task] = []

        # Game plan task
        game_plan_config = self.config.game_plan
        if game_plan_config.enabled:
            agent = self._container.game_plan_agent()
            tasks.append(
                GamePlanTask(
                    agent=agent,
                    state=self.state,
                    broker_manager=self._broker_manager,
                    config=game_plan_config,
                    scheduler=self.scheduler,
                )
            )

        # Position review task
        pr_config = self.config.position_review
        if pr_config.enabled and self.config.coordinator.enabled:
            broker = self._container.alpaca_broker()
            queue = self._container.market_event_queue()
            db_engine = self._container.database_engine() if self.config.database.enable_persistence else None
            tasks.append(
                PositionReviewTask(
                    broker=broker,
                    queue=queue,
                    config=pr_config,
                    scheduler=self.scheduler,
                    database_engine=db_engine,
                )
            )

        # Portfolio snapshot task
        snapshot_config = self.config.portfolio_snapshot
        if snapshot_config.enabled and self._components.broker:
            try:
                database_engine = self._container.database_engine()
                tasks.append(
                    PortfolioSnapshotTask(
                        broker=self._components.broker,
                        database_engine=database_engine,
                        config=snapshot_config,
                    )
                )
            except MissingDatabaseURLError:
                logger.warning("Portfolio snapshot task disabled: database not configured")

        # Risk report task
        risk_config = self.config.risk_limits
        if risk_config.enabled and self._components.broker and self._workflow is not None:
            from src.v1.tasks.implementations.risk_report import RiskReportTask

            queue = self._container.market_event_queue() if self.config.coordinator.enabled else None
            tasks.append(
                RiskReportTask(
                    risk_manager=self._workflow.risk_manager,
                    broker=self._components.broker,
                    queue=queue,
                    state=self.state,
                    scheduler=self.scheduler,
                    config=risk_config,
                    notification_service=self._components.notification_service,
                )
            )

        # Watchlist sweep task
        sweep_config = self.config.coordinator.sweep_pass
        if sweep_config.enabled and self.config.coordinator.enabled:
            from src.v1.tasks.implementations.watchlist_sweep import WatchlistSweepTask

            try:
                database_engine = self._container.database_engine()
                queue = self._container.market_event_queue()
                tasks.append(
                    WatchlistSweepTask(
                        queue=queue,
                        broker_manager=self._broker_manager,
                        database_engine=database_engine,
                        config=sweep_config,
                    )
                )
            except MissingDatabaseURLError:
                logger.warning("Watchlist sweep task disabled: database not configured")

        tz = ZoneInfo(self.config.schedule.timezone)
        return TaskRunner(tasks, tz)

    def _init_game_plan_agent(self) -> GamePlanAgent:
        """Initialize game plan agent (lazy)."""
        if self._components.game_plan_agent is None:
            self._game_plan_agent = self._container.game_plan_agent()
            self._components.game_plan_agent = self._game_plan_agent
        return self._components.game_plan_agent

    def _init_coordinator(self) -> TradingCoordinator:
        """Initialize coordinator (lazy).

        Returns:
            TradingCoordinator instance
        """
        if self._components.coordinator is not None:
            return self._components.coordinator

        return self._factory.init_coordinator(self._components)

    def _init_analysis_orchestrator(self) -> AnalysisOrchestrator:
        """Initialize analysis orchestrator (lazy)."""
        if self._components.analysis_orchestrator is None:
            context_builder = self._container.context_builder(
                components=self._components,
                container=self._container,
            )
            orchestrator = self._factory.init_analysis_orchestrator(
                self._components,
                context_builder=context_builder,
            )
            orchestrator.market_event_queue = self._container.market_event_queue()
            self._analysis_orchestrator = orchestrator
            return orchestrator
        return self._components.analysis_orchestrator

    def _init_workflow(self) -> TradingWorkflow:
        """Initialize trading workflow (lazy)."""
        if self._components.workflow is None:
            self._workflow = self._factory.init_workflow(self._components)
            self._metrics_tracker = self._components.metrics_tracker

        # Apply target allocations if available
        if self._target_allocations_to_apply and self._components.workflow:
            self._components.workflow.set_target_allocations(self._target_allocations_to_apply)

        if self._components.workflow is None:
            msg = "Workflow initialization failed"
            raise RuntimeError(msg)
        return self._components.workflow

    async def get_merged_watchlist(self) -> list[str]:
        """Get watchlist merged with broker positions and screening candidates."""
        return await self._broker_manager.get_merged_watchlist()

    async def _analyze_symbol(
        self,
        symbol: str,
        position_context: dict[str, object] | None = None,
        degradation_context: DegradationContext | None = None,
    ) -> TradingWorkflowResult | None:
        """Analyze a single symbol (delegates to orchestrator)."""
        orchestrator = self._init_analysis_orchestrator()
        # Sync orchestrator with runner's current state (important for testing)
        orchestrator.workflow = self._init_workflow()
        orchestrator.state = self.state
        orchestrator.scheduler = self.scheduler
        orchestrator.notification_service = self.notification_service
        orchestrator.historical_cache = self._historical_cache
        return await orchestrator._analyze_symbol(symbol, position_context, degradation_context)  # noqa: SLF001

    async def _publish_event(self, event_type: str, data: dict[str, object]) -> None:
        """Publish event to EventBus with error handling."""
        if not self.event_bus:
            return

        try:
            from src.daemon.event_bus import DashboardEvent, EventType

            await self.event_bus.publish(DashboardEvent(event_type=EventType[event_type], data=data))
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to publish {event_type} event: {e}")

    async def _run_cycle(self) -> int:
        """Run a single analysis cycle (delegates to cycle orchestrator).

        Returns:
            Seconds to sleep before next cycle
        """
        from src.daemon.cycle_orchestrator import DaemonCycleOrchestrator

        # Create cycle orchestrator and delegate
        from src.daemon.profiling.profiler import CycleProfiler

        profiler = self._components.profiler
        if profiler and not isinstance(profiler, CycleProfiler):
            profiler = None

        cycle_orchestrator = DaemonCycleOrchestrator(
            components=self._components,
            task_runner=self._task_runner,
            factory=self._factory,
            profiler=profiler,
        )

        result = await cycle_orchestrator.run_cycle()
        return result.sleep_seconds

    async def run(self) -> None:
        """Run the daemon main loop."""
        from src.daemon.cycle_orchestrator import DaemonCycleOrchestrator
        from src.daemon.lifecycle import DaemonLifecycle

        # Initialize workflow (needed by scheduled tasks like CorrelationAuditTask and workflow-dependent v1 tasks)
        self._init_workflow()

        # Rebuild v1 task runner with workflow-dependent tasks now available
        self._v1_task_runner = self._build_v1_task_runner()

        # Create lifecycle manager
        lifecycle = DaemonLifecycle(self._components)
        await lifecycle.startup()

        # Sync running state
        self.running = lifecycle.running
        self._components.running = lifecycle.running

        # Create cycle orchestrator
        from src.daemon.profiling.profiler import CycleProfiler

        profiler = self._components.profiler
        if profiler and not isinstance(profiler, CycleProfiler):
            profiler = None

        cycle_orchestrator = DaemonCycleOrchestrator(
            components=self._components,
            task_runner=self._task_runner,
            factory=self._factory,
            profiler=profiler,
        )

        # Spawn event queue consumer if coordinator enabled
        self._consumer_task = self._maybe_start_event_consumer()

        # Spawn v1 task runner as independent background task
        v1_task = asyncio.create_task(self._v1_task_runner.run(), name="v1-task-runner")

        while lifecycle.running:
            try:
                result = await cycle_orchestrator.run_cycle()
                sleep_minutes = result.sleep_seconds // 60
                logger.info(f"Sleeping for {sleep_minutes} minutes")
                await asyncio.sleep(result.sleep_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in daemon loop: {e}")
                await self.state.record_error(str(e))
                await asyncio.sleep(60)

        # Shutdown v1 task runner
        import contextlib

        v1_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await v1_task
        logger.info("v1 task runner stopped")

        # Shutdown consumer task
        await self._stop_event_consumer()

        # Shutdown
        await lifecycle.shutdown()

    def _maybe_start_event_consumer(self) -> asyncio.Task | None:
        """Start event queue consumer task if coordinator is enabled."""
        if not self.config.coordinator.enabled:
            return None

        try:
            from src.daemon.market_service import MarketService
            from src.v1.event_queue.consumer import EventQueueConsumer

            queue = self._container.market_event_queue()
            coordinator = self._init_coordinator()
            market_service = MarketService(self.scheduler)
            consumer = EventQueueConsumer(
                queue=queue,
                coordinator=coordinator,
                market_service=market_service,
                config=self.config.coordinator,
            )
            self._components.event_queue_consumer = consumer
            task = asyncio.create_task(consumer.run(), name="event-queue-consumer")
            logger.info(f"Event queue consumer started: {consumer}")
            return task
        except Exception:
            logger.opt(exception=True).warning("Failed to start event queue consumer")
            return None

    async def _stop_event_consumer(self) -> None:
        """Cancel and await the event consumer task."""
        import contextlib

        if not self._consumer_task:
            return
        self._consumer_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._consumer_task
        logger.info("Event queue consumer stopped")

    @classmethod
    def from_config_file(cls, path: Path) -> DaemonRunner:
        """Create runner from config file.

        Args:
            path: Path to YAML config file

        Returns:
            DaemonRunner instance
        """
        config = DaemonConfig.from_yaml(path)
        return cls(config)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"DaemonRunner(config={self.config})"
