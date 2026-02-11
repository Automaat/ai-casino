"""Daemon component factory for initialization logic extraction."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from src.cache.historical import HistoricalCache
from src.daemon.broker_manager import BrokerManager
from src.daemon.config import DaemonConfig, TradingMode
from src.daemon.prefetch import DataPrefetcher
from src.daemon.scheduler import MarketScheduler
from src.daemon.state import DaemonState
from src.daemon.task_runner import ScheduledTaskRunner
from src.data.broker import AlpacaBroker
from src.data.market import MarketDataFetcher
from src.database.connection import get_db_engine
from src.metrics.tracker import BaseMetricsTracker, create_metrics_tracker
from src.optimization.param_store import OptimizedParamStore
from src.workflows import TradingWorkflow

if TYPE_CHECKING:
    from src.agents.game_plan import GamePlanAgent
    from src.daemon.analysis_orchestrator import AnalysisOrchestrator
    from src.daemon.context_builder import DaemonContextBuilder
    from src.daemon.event_bus import EventBus
    from src.daemon.notifications import NotificationService
    from src.daemon.optimization import DaemonOptimizer
    from src.daemon.positions import PositionManager
    from src.daemon.rebalancing import DaemonRebalancer
    from src.daemon.tearsheet import DaemonTearsheetGenerator
    from src.di.container import AppContainer
    from src.discovery.engine import StockDiscoveryEngine


@dataclass
class DaemonComponents:
    """Container for all daemon components."""

    # Core (always present)
    config: DaemonConfig
    state: DaemonState
    scheduler: MarketScheduler
    broker_manager: BrokerManager
    task_runner: ScheduledTaskRunner
    container: AppContainer
    historical_cache: HistoricalCache
    broker: AlpacaBroker | None

    # Runtime state
    running: bool = False
    event_bus: EventBus | None = None

    # Lazy-initialized (None until needed)
    workflow: TradingWorkflow | None = None
    analysis_orchestrator: AnalysisOrchestrator | None = None
    metrics_tracker: BaseMetricsTracker | None = None

    # Optional (None if disabled)
    daemon_optimizer: DaemonOptimizer | None = None
    daemon_rebalancer: DaemonRebalancer | None = None
    position_manager: PositionManager | None = None
    discovery_engine: StockDiscoveryEngine | None = None
    notification_service: NotificationService | None = None
    tearsheet_generator: DaemonTearsheetGenerator | None = None
    prefetcher: DataPrefetcher | None = None
    game_plan_agent: GamePlanAgent | None = None
    market_fetcher: MarketDataFetcher | None = None
    param_store: OptimizedParamStore | None = None


class DaemonFactory:
    """Factory for creating and initializing daemon components."""

    def __init__(self, config: DaemonConfig, container: AppContainer | None = None) -> None:
        """Initialize factory.

        Args:
            config: Daemon configuration
            container: Optional DI container (auto-created if None)
        """
        self.config = config
        self._container = self._create_or_wire_container(container)

    def __repr__(self) -> str:
        """Return string representation."""
        return "DaemonFactory()"

    def _create_or_wire_container(self, container: AppContainer | None) -> AppContainer:
        """Create container or wire existing one with config.

        Args:
            container: Optional existing container

        Returns:
            Configured container
        """
        if container is not None:
            return container

        from src.di.container import create_container

        container = create_container()
        container.daemon_config.override(self.config)
        return container

    def create_components(self, event_bus: EventBus | None = None) -> DaemonComponents:
        """Create all daemon components based on config.

        Args:
            event_bus: Optional event bus for streaming

        Returns:
            DaemonComponents with all initialized components
        """
        # Phase 1: Core infrastructure
        historical_cache = self._container.historical_cache()
        state = DaemonState.load(self.config.state.state_file)

        # Phase 2: Broker setup
        broker_manager = BrokerManager(self.config, state, historical_cache)
        broker_manager.initialize_broker()
        broker = broker_manager.broker

        # Phase 3: Scheduler
        scheduler = self._create_scheduler()

        # Phase 4: Optimization (if enabled)
        param_store = None
        daemon_optimizer = None
        if self.config.optimization.enabled:
            param_store = OptimizedParamStore(self.config.optimization.params_file)
            from src.daemon.optimization import DaemonOptimizer

            daemon_optimizer = DaemonOptimizer(
                param_store=param_store,
                n_trials=self.config.optimization.n_trials,
                min_trades=self.config.optimization.min_trades,
            )

        # Phase 5: Validate live mode readiness
        metrics_tracker = None
        if self.config.auto_trade and self.config.trading_mode == TradingMode.LIVE:
            metrics_tracker = self._validate_live_mode_and_get_tracker(state)

        # Phase 6: Rebalancing (if enabled)
        daemon_rebalancer = None
        if self.config.rebalancing.enabled:
            daemon_rebalancer = self._create_rebalancer(broker)

        # Phase 7: Position manager (if enabled)
        position_manager = None
        if self.config.position_management.enabled:
            position_manager = self._create_position_manager(broker)

        # Phase 8: Discovery engine (if enabled)
        discovery_engine = None
        market_fetcher = None
        if self.config.discovery.enabled:
            discovery_engine, market_fetcher = self._create_discovery_engine(broker)

        # Phase 9: Notifications (if enabled)
        notification_service = None
        if self.config.notifications.enabled:
            from src.daemon.notifications import NotificationService

            notification_service = NotificationService(self.config.notifications)
            logger.info("Notification service enabled")

        # Phase 10: Tearsheet generator (if enabled)
        tearsheet_generator = None
        if self.config.reporting.enabled:
            from src.daemon.tearsheet import DaemonTearsheetGenerator

            tearsheet_generator = DaemonTearsheetGenerator(
                broker=broker,
                market_fetcher=None,
            )
            logger.info("Tearsheet generator enabled")

        # Phase 11: Assemble components
        components = DaemonComponents(
            config=self.config,
            state=state,
            scheduler=scheduler,
            broker_manager=broker_manager,
            task_runner=None,  # type: ignore[arg-type]  # Set after creation
            container=self._container,
            historical_cache=historical_cache,
            broker=broker,
            running=False,
            event_bus=event_bus,
            workflow=None,
            analysis_orchestrator=None,
            metrics_tracker=metrics_tracker,
            daemon_optimizer=daemon_optimizer,
            daemon_rebalancer=daemon_rebalancer,
            position_manager=position_manager,
            discovery_engine=discovery_engine,
            notification_service=notification_service,
            tearsheet_generator=tearsheet_generator,
            prefetcher=None,
            game_plan_agent=None,
            market_fetcher=market_fetcher,
            param_store=param_store,
        )

        # Phase 12: Create task runner (needs components reference)
        task_runner = ScheduledTaskRunner(self.config, scheduler, daemon_runner=None)  # type: ignore[arg-type]
        components.task_runner = task_runner

        logger.info(f"DaemonComponents created for {self.config}")
        return components

    def _create_scheduler(self) -> MarketScheduler:
        """Create market scheduler from config.

        Returns:
            MarketScheduler instance
        """
        return MarketScheduler(
            start_time=self.config.schedule.start_time,
            end_time=self.config.schedule.end_time,
            timezone=self.config.schedule.timezone,
            enable_pre_market=self.config.schedule.enable_pre_market,
            enable_after_hours=self.config.screening.enabled,
            after_hours_screen_time=self.config.screening.screen_time,
            after_hours_screen_days=self.config.screening.screen_days,
            optimization_time=self.config.optimization.optimization_time,
            optimization_days=self.config.optimization.optimization_days,
            prefetch_time=self.config.prefetch.prefetch_time,
            pre_market_refresh_time=self.config.prefetch.pre_market_refresh_time,
            sector_rotation_time=self.config.sector_rotation.run_time,
            sector_rotation_days=self.config.sector_rotation.run_days,
            enable_sector_rotation=self.config.sector_rotation.enabled,
            earnings_fetch_time=self.config.earnings_calendar.fetch_time,
            earnings_fetch_days=self.config.earnings_calendar.fetch_days,
            enable_earnings_calendar=self.config.earnings_calendar.enabled,
            peer_analysis_time=self.config.peer_analysis.run_time,
            peer_analysis_days=self.config.peer_analysis.run_days,
            enable_peer_analysis=self.config.peer_analysis.enabled,
            correlation_audit_time=self.config.correlation_audit.run_time,
            correlation_audit_days=self.config.correlation_audit.run_days,
            enable_correlation_audit=self.config.correlation_audit.enabled,
            tearsheet_time=self.config.reporting.tearsheet_time,
            enable_reporting=self.config.reporting.enabled,
            rebalancing_time=self.config.rebalancing.run_time,
            rebalancing_days=self.config.rebalancing.run_days,
            enable_rebalancing=self.config.rebalancing.enabled,
            signal_tracking_time=self.config.signal_tracking.tracking_time,
            enable_signal_tracking=self.config.signal_tracking.enabled,
            game_plan_time=self.config.game_plan.generation_time,
            enable_game_plan=self.config.game_plan.enabled,
            monte_carlo_time=self.config.monte_carlo.schedule_time,
            monte_carlo_days=self.config.monte_carlo.schedule_days,
        )

    def _validate_live_mode_and_get_tracker(self, state: DaemonState) -> BaseMetricsTracker | None:
        """Validate live mode readiness and initialize metrics tracker.

        Args:
            state: Daemon state

        Returns:
            Metrics tracker if validation passes

        Raises:
            ValueError: If validation fails and --force-live not used
        """
        logger.warning("LIVE TRADING MODE - real capital at risk")

        force_live = "--force-live" in sys.argv

        if not force_live:
            from src.daemon.paper_trading_validator import PaperTradingValidator

            # Initialize tracker for validation
            trade_repository = None
            if os.getenv("DATABASE_URL"):
                try:
                    from src.database.repositories.trade import TradeRepository

                    db_engine = get_db_engine()
                    trade_repository = TradeRepository(db_engine.session())
                except Exception as e:
                    logger.warning(f"Failed to init DB metrics tracker: {e}, using JSONL")

            metrics_tracker = create_metrics_tracker(trade_repository)

            validator = PaperTradingValidator(
                config=self.config.paper_trading,
                state=state,
                metrics_tracker=metrics_tracker,
            )

            try:
                report = validator.assess_readiness()

                if not report.ready_for_live:
                    failed = [c.name for c in report.criteria if not c.passed]
                    logger.error(f"Paper trading validation failed: {', '.join(failed)}")
                    msg = "Cannot start live trading - use --force-live to bypass"
                    raise ValueError(msg)

                logger.info("Paper trading validation passed")
            except Exception as e:
                logger.error(f"Validation error: {e}")
                raise

            return metrics_tracker
        logger.warning("--force-live flag used, skipping validation")
        return None

    def _create_rebalancer(self, broker: AlpacaBroker | None) -> DaemonRebalancer:
        """Create portfolio rebalancer.

        Args:
            broker: Broker instance

        Returns:
            DaemonRebalancer instance
        """
        from src.daemon.rebalancing import DaemonRebalancer
        from src.optimization.portfolio import PortfolioOptimizer

        market_fetcher = self._container.market_fetcher()
        portfolio_optimizer = PortfolioOptimizer(
            market_fetcher=market_fetcher,
            broker=broker,
            period_days=self.config.rebalancing.lookback_days,
        )
        return DaemonRebalancer(
            optimizer=portfolio_optimizer,
            broker=broker if self.config.auto_trade else None,
            rebalance_threshold=self.config.rebalancing.rebalance_threshold,
        )

    def _create_position_manager(self, broker: AlpacaBroker | None) -> PositionManager:
        """Create position manager.

        Args:
            broker: Broker instance

        Returns:
            PositionManager instance

        Raises:
            ValueError: If auto_trade not enabled
        """
        if not self.config.auto_trade or not broker:
            msg = "position_management requires auto_trade=true"
            raise ValueError(msg)

        from src.daemon.positions import PositionManager

        # Get database engine and trade repository from container if database is available
        database_engine = None
        trade_repository = None

        if os.getenv("DATABASE_URL") or self.config.database.database_url:
            try:
                database_engine = self._container.database_engine()
                trade_repository = self._container.trade_repository()
                logger.debug("Position manager initialized with database engine")
            except Exception as e:
                logger.warning(f"Failed to initialize position manager database: {e}")

        position_manager = PositionManager(
            broker,
            self.config.position_management,
            database_engine=database_engine,
            trade_repository=trade_repository,
        )
        logger.info("Position management enabled")
        return position_manager

    def _create_discovery_engine(
        self, broker: AlpacaBroker | None
    ) -> tuple[StockDiscoveryEngine, MarketDataFetcher]:
        """Create stock discovery engine and market fetcher.

        Args:
            broker: Broker instance for discovery (optional)

        Returns:
            Tuple of (StockDiscoveryEngine, MarketDataFetcher)
        """
        from src.data.universe import StockUniverseFetcher
        from src.discovery.engine import (
            CoreDependencies,
            DiscoveryEngineConfig,
            OptionalServices,
            StockDiscoveryEngine,
        )
        from src.discovery.filters import PortfolioFilterConfig
        from src.discovery.scoring import ScoringWeights
        from src.discovery.triggers import TriggerDetector
        from src.screening.screener import StockScreener

        # Parse portfolio filters from daemon config
        portfolio_filters_data = self.config.discovery.portfolio_filters or {}
        portfolio_filters = PortfolioFilterConfig(
            max_watchlist_size=portfolio_filters_data.get("max_watchlist_size", 50),
            max_sector_concentration=portfolio_filters_data.get("max_sector_concentration", 0.3),
            min_market_cap=portfolio_filters_data.get("min_market_cap", 1_000_000_000),
            min_avg_volume=portfolio_filters_data.get("min_avg_volume", 1_000_000),
            price_range=tuple(portfolio_filters_data.get("price_range", [10.0, 500.0])),
            exclude_sectors=portfolio_filters_data.get("exclude_sectors", []),
        )

        # Create discovery config from daemon config
        discovery_config = DiscoveryEngineConfig(
            enable_technical_screening=self.config.discovery.enable_technical_screening,
            enable_reddit_trending=self.config.discovery.enable_reddit_trending,
            enable_earnings_calendar=self.config.discovery.enable_earnings_calendar,
            enable_sector_rotation=self.config.discovery.enable_sector_rotation,
            enable_volume_spikes=self.config.discovery.enable_volume_spikes,
            enable_price_gaps=self.config.discovery.enable_price_gaps,
            enable_news_trending=self.config.discovery.enable_news_trending,
            screening_criteria=list(self.config.discovery.screening_criteria),  # type: ignore[arg-type]
            screening_universe=self.config.discovery.screening_universe,
            screening_top_n=self.config.discovery.screening_top_n,
            reddit_min_mentions=self.config.discovery.reddit_min_mentions,
            reddit_min_upvote_ratio=self.config.discovery.reddit_min_upvote_ratio,
            earnings_lookahead_days=self.config.discovery.earnings_lookahead_days,
            volume_spike_threshold=self.config.discovery.volume_spike_threshold,
            price_gap_threshold=self.config.discovery.price_gap_threshold,
            scoring_weights=ScoringWeights(**self.config.discovery.scoring_weights),
            max_discovered_per_cycle=self.config.discovery.max_discovered_per_cycle,
            min_composite_score=self.config.discovery.min_composite_score,
            max_watchlist_size=self.config.discovery.max_watchlist_size,
            candidate_ttl_days=self.config.discovery.candidate_ttl_days,
            auto_remove_on_signal=self.config.discovery.auto_remove_on_signal,
            track_outcomes=self.config.discovery.track_outcomes,
            outcome_lookback_days=self.config.discovery.outcome_lookback_days,
            portfolio_filters=portfolio_filters,
        )

        # Create screener
        screener = StockScreener(
            universe_fetcher=StockUniverseFetcher(),
            liquidity_filters=self.config.liquidity_filters,
            cache_dir="data/cache/screening",
        )

        # Market fetcher
        market_fetcher = self._container.market_fetcher()

        # Trigger detector
        trigger_detector = TriggerDetector(
            market_fetcher=market_fetcher,
            volume_spike_threshold=self.config.discovery.volume_spike_threshold,
            price_gap_threshold=self.config.discovery.price_gap_threshold,
        )

        # Core dependencies
        deps = CoreDependencies(
            screener=screener,
            market_fetcher=market_fetcher,
            universe_fetcher=StockUniverseFetcher(),
            trigger_detector=trigger_detector,
        )

        # Optional services (broker passed from create_components)
        services = OptionalServices(
            reddit_fetcher=None,
            earnings_fetcher=None,
            news_fetcher=None,
            broker=broker,
        )

        engine = StockDiscoveryEngine(
            deps=deps,
            config=discovery_config,
            services=services,
        )

        logger.info("Stock discovery engine initialized")
        return engine, market_fetcher

    def init_workflow(self, components: DaemonComponents) -> TradingWorkflow:
        """Initialize trading workflow (lazy).

        Args:
            components: Daemon components

        Returns:
            TradingWorkflow instance
        """
        if components.workflow is not None:
            return components.workflow

        # Initialize metrics tracker if not already done
        if components.metrics_tracker is None:
            trade_repository = None
            if os.getenv("DATABASE_URL"):
                try:
                    from src.database.repositories.trade import TradeRepository

                    db_engine = get_db_engine()
                    trade_repository = TradeRepository(db_engine.session())
                except Exception as e:
                    logger.warning(f"Failed to init DB metrics tracker: {e}, using JSONL")

            components.metrics_tracker = create_metrics_tracker(trade_repository)

        workflow = self._container.workflow_meta(
            broker=components.broker,
            metrics_tracker=components.metrics_tracker,
            param_store=components.param_store,
            notification_service=components.notification_service,
            historical_cache=components.historical_cache,
            container=self._container,
        )
        logger.info("Trading workflow initialized")

        components.workflow = workflow
        return workflow

    def init_analysis_orchestrator(
        self,
        components: DaemonComponents,
        event_bus: EventBus | None = None,
        context_builder: DaemonContextBuilder | None = None,
    ) -> AnalysisOrchestrator:
        """Initialize analysis orchestrator (lazy).

        Args:
            components: Daemon components
            event_bus: Optional event bus for publishing
            context_builder: Optional context builder

        Returns:
            AnalysisOrchestrator instance
        """
        if components.analysis_orchestrator is not None:
            return components.analysis_orchestrator

        from src.daemon.analysis_orchestrator import AnalysisOrchestrator
        from src.daemon.positions import PositionManager

        workflow = self.init_workflow(components)

        # Type-narrow position_manager
        position_manager: PositionManager | None = None
        if isinstance(components.position_manager, PositionManager):
            position_manager = components.position_manager

        orchestrator = AnalysisOrchestrator(
            workflow=workflow,
            state=components.state,
            scheduler=components.scheduler,
            config=self.config.analysis_orchestration,
            trading_mode=self.config.trading_mode.value,
            broker=components.broker,
            position_manager=position_manager,
            event_bus=event_bus,
            historical_cache=components.historical_cache,
            notification_service=components.notification_service,
            context_builder=context_builder,
            components=components,  # type: ignore[arg-type]  # pyrefly module resolution issue
        )
        logger.info("Analysis orchestrator initialized")

        components.analysis_orchestrator = orchestrator
        return orchestrator
