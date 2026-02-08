"""Main daemon runner for autonomous trading."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sys
import time as time_mod
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import uvicorn
from loguru import logger
from rich.console import Console

from src.agents.game_plan import GamePlan, GamePlanAgent
from src.agents.news import NewsAnalysis
from src.agents.sentiment import SentimentAnalysis

if TYPE_CHECKING:
    from src.agents.risk import PortfolioRiskReport
    from src.daemon.degradation import DegradationContext
    from src.daemon.event_bus import EventBus
    from src.daemon.health import HealthReport
    from src.daemon.tearsheet import DaemonTearsheetGenerator
    from src.metrics.correlation import CorrelationAuditResult
from src.cache.historical import HistoricalCache
from src.daemon.analysis_orchestrator import AnalysisOrchestrator
from src.daemon.broker_manager import BrokerManager
from src.daemon.config import DaemonConfig, TradingMode
from src.daemon.prefetch import DataPrefetcher
from src.daemon.scheduler import MarketScheduler
from src.daemon.state import DaemonState, EarningsEventRecord, RiskReportRecord, SectorRotationRecord
from src.daemon.task_runner import ScheduledTaskRunner
from src.data.fundamental import FundamentalDataFetcher
from src.data.market import MarketDataFetcher
from src.data.news import NewsFetcher
from src.metrics.sector_rotation import SectorRotationAnalysis
from src.metrics.tracker import BaseMetricsTracker, create_metrics_tracker
from src.models.llm import LLMClient
from src.models.sentiment import get_finbert_sentiment
from src.optimization.param_store import OptimizedParamStore
from src.workflows.trading import TradingWorkflow
from src.workflows.types import TradingWorkflowResult

console = Console()


class DaemonRunner:
    """Main daemon runner for autonomous trading."""

    def __init__(self, config: DaemonConfig, event_bus: EventBus | None = None) -> None:  # noqa: PLR0915, C901, PLR0912
        """Initialize daemon runner.

        Args:
            config: Daemon configuration
            event_bus: Optional EventBus for real-time event streaming
        """
        self.config = config
        self.event_bus = event_bus
        self._historical_cache = HistoricalCache()
        self.state = DaemonState.load(config.state.state_file)
        self._broker_manager = BrokerManager(config, self.state, self._historical_cache)
        self.scheduler = MarketScheduler(
            start_time=config.schedule.start_time,
            end_time=config.schedule.end_time,
            timezone=config.schedule.timezone,
            enable_pre_market=config.schedule.enable_pre_market,
            enable_after_hours=config.screening.enabled,
            after_hours_screen_time=config.screening.screen_time,
            after_hours_screen_days=config.screening.screen_days,
            optimization_time=config.optimization.optimization_time,
            optimization_days=config.optimization.optimization_days,
            health_check_time=config.health.run_time,
            prefetch_time=config.prefetch.prefetch_time,
            pre_market_refresh_time=config.prefetch.pre_market_refresh_time,
            sector_rotation_time=config.sector_rotation.run_time,
            sector_rotation_days=config.sector_rotation.run_days,
            enable_sector_rotation=config.sector_rotation.enabled,
            earnings_fetch_time=config.earnings_calendar.fetch_time,
            earnings_fetch_days=config.earnings_calendar.fetch_days,
            enable_earnings_calendar=config.earnings_calendar.enabled,
            peer_analysis_time=config.peer_analysis.run_time,
            peer_analysis_days=config.peer_analysis.run_days,
            enable_peer_analysis=config.peer_analysis.enabled,
            correlation_audit_time=config.correlation_audit.run_time,
            correlation_audit_days=config.correlation_audit.run_days,
            enable_correlation_audit=config.correlation_audit.enabled,
            tearsheet_time=config.reporting.tearsheet_time,
            enable_reporting=config.reporting.enabled,
            rebalancing_time=config.rebalancing.run_time,
            rebalancing_days=config.rebalancing.run_days,
            enable_rebalancing=config.rebalancing.enabled,
            signal_tracking_time=config.signal_tracking.tracking_time,
            enable_signal_tracking=config.signal_tracking.enabled,
            game_plan_time=config.game_plan.generation_time,
            enable_game_plan=config.game_plan.enabled,
            monte_carlo_time=config.monte_carlo.schedule_time,
            monte_carlo_days=config.monte_carlo.schedule_days,
        )
        self.running = False
        self._workflow: TradingWorkflow | None = None
        self._metrics_tracker: BaseMetricsTracker | None = None
        self.param_store: OptimizedParamStore | None = None
        self._daemon_optimizer = None
        if config.optimization.enabled:
            self.param_store = OptimizedParamStore(config.optimization.params_file)
            from src.daemon.optimization import DaemonOptimizer

            self._daemon_optimizer = DaemonOptimizer(
                param_store=self.param_store,
                n_trials=config.optimization.n_trials,
                min_trades=config.optimization.min_trades,
            )
        # Initialize broker via manager
        self._broker_manager.initialize_broker()
        self.broker = self._broker_manager.broker

        # Validate live mode readiness
        if config.auto_trade and config.trading_mode == TradingMode.LIVE:
            logger.warning("LIVE TRADING MODE - real capital at risk")

            force_live = "--force-live" in sys.argv

            if not force_live:
                from src.daemon.paper_trading_validator import PaperTradingValidator

                # Initialize tracker early for validation
                if self._metrics_tracker is None:
                    trade_repository = None
                    if os.getenv("DATABASE_URL"):
                        try:
                            from src.database.repositories.trade import TradeRepository
                            from src.database.session import get_session_factory

                            session_factory = get_session_factory()
                            trade_repository = TradeRepository(session_factory())
                        except Exception as e:
                            logger.warning(f"Failed to init DB metrics tracker: {e}, using JSONL")

                    self._metrics_tracker = create_metrics_tracker(trade_repository)

                validator = PaperTradingValidator(
                    config=config.paper_trading,
                    state=self.state,
                    metrics_tracker=self._metrics_tracker,
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
            else:
                logger.warning("--force-live flag used, skipping validation")

        self._daemon_rebalancer = None
        if config.rebalancing.enabled:
            from src.daemon.rebalancing import DaemonRebalancer
            from src.optimization.portfolio import PortfolioOptimizer

            market_fetcher = MarketDataFetcher(
                use_alpha_vantage=False,
                api_key=self._resolve_config_or_env(
                    self.config.api_keys.alpha_vantage_api_key, "ALPHA_VANTAGE_API_KEY"
                ),
                historical_cache=self._historical_cache,
            )
            portfolio_optimizer = PortfolioOptimizer(
                market_fetcher=market_fetcher,
                broker=self.broker,
                period_days=config.rebalancing.lookback_days,
            )
            self._daemon_rebalancer = DaemonRebalancer(
                optimizer=portfolio_optimizer,
                broker=self.broker if config.auto_trade else None,
                rebalance_threshold=config.rebalancing.rebalance_threshold,
            )
        self._prefetcher: DataPrefetcher | None = None
        self._target_allocations_to_apply: dict[str, float] | None = None
        self._game_plan_agent: GamePlanAgent | None = None

        # Position manager
        self._position_manager = None
        if config.position_management.enabled:
            if not config.auto_trade or not self.broker:
                msg = "position_management requires auto_trade=true"
                raise ValueError(msg)
            from src.daemon.positions import PositionManager

            self._position_manager = PositionManager(self.broker, config.position_management)
            logger.info("Position management enabled")

        # Notification service
        self.notification_service = None
        if config.notifications.enabled:
            from src.daemon.notifications import NotificationService

            self.notification_service = NotificationService(config.notifications)
            logger.info("Notification service enabled")

        # Tearsheet generator (for performance reporting)
        self._tearsheet_generator: DaemonTearsheetGenerator | None = None
        if config.reporting.enabled:
            # Import at runtime (also in TYPE_CHECKING for type hints)
            from src.daemon.tearsheet import DaemonTearsheetGenerator

            self._tearsheet_generator: DaemonTearsheetGenerator = DaemonTearsheetGenerator(
                broker=self.broker,
                market_fetcher=None,  # Will be set later if needed
            )
            logger.info("Tearsheet generator enabled")

        # Market data fetcher (shared instance for various features)
        self.market_fetcher: MarketDataFetcher | None = None

        # Analysis orchestrator (initialized after workflow is ready)
        self._analysis_orchestrator: AnalysisOrchestrator | None = None

        # Scheduled task runner
        self._task_runner = ScheduledTaskRunner(config, self.scheduler, daemon_runner=self)

        # API server components
        self._api_server: uvicorn.Server | None = None
        self._api_task: asyncio.Task | None = None

        logger.info(f"DaemonRunner initialized with {config}")

    def _resolve_config_or_env(self, config_value: str | None, env_var: str) -> str | None:
        """Resolve config value from daemon config or env var.

        Config takes priority over environment variable.

        Args:
            config_value: Value from daemon config (priority)
            env_var: Environment variable name (fallback)

        Returns:
            Resolved config value or None
        """
        return config_value or os.getenv(env_var)

    def _create_llm_client(self) -> LLMClient:
        """Create LLM client with config/env resolution.

        Returns:
            Configured LLMClient instance
        """
        provider = self.config.llm.provider or os.getenv("LLM_PROVIDER", "ollama")
        if provider == "anthropic":
            api_key = self._resolve_config_or_env(self.config.api_keys.anthropic_api_key, "ANTHROPIC_API_KEY")
        elif provider == "openai":
            api_key = self._resolve_config_or_env(self.config.api_keys.openai_api_key, "OPENAI_API_KEY")
        else:
            api_key = None

        return LLMClient(
            provider=self.config.llm.provider,
            model=self.config.llm.model,
            api_key=api_key,
            openai_base_url=self._resolve_config_or_env(
                self.config.api_keys.openai_api_base, "OPENAI_API_BASE"
            ),
        )

    def _init_prefetcher(self) -> DataPrefetcher | None:
        """Initialize data prefetcher (lazy initialization).

        Returns:
            DataPrefetcher instance or None if API key missing
        """
        if self._prefetcher is None:
            try:
                market_fetcher = MarketDataFetcher(
                    use_alpha_vantage=False,
                    api_key=self._resolve_config_or_env(
                        self.config.api_keys.alpha_vantage_api_key, "ALPHA_VANTAGE_API_KEY"
                    ),
                    historical_cache=self._historical_cache,
                )
                news_fetcher = NewsFetcher(
                    api_key=self._resolve_config_or_env(
                        self.config.api_keys.marketaux_api_key, "MARKETAUX_API_KEY"
                    ),
                    historical_cache=self._historical_cache,
                )
                fundamental_fetcher = FundamentalDataFetcher(
                    api_key=self._resolve_config_or_env(
                        self.config.api_keys.alpha_vantage_api_key, "ALPHA_VANTAGE_API_KEY"
                    ),
                    historical_cache=self._historical_cache,
                )

                self._prefetcher = DataPrefetcher(
                    market_fetcher=market_fetcher,
                    news_fetcher=news_fetcher,
                    fundamental_fetcher=fundamental_fetcher,
                    cache_dir=self.config.prefetch.cache_dir,
                )
                logger.info("DataPrefetcher initialized")
            except ValueError as e:
                logger.warning(f"Failed to initialize prefetcher: {e}")
                return None
        return self._prefetcher

    def _init_game_plan_agent(self) -> GamePlanAgent:
        """Initialize game plan agent (lazy).

        Returns:
            GamePlanAgent instance
        """
        if self._game_plan_agent is None:
            llm_client = self._create_llm_client()
            market_fetcher = MarketDataFetcher(
                use_alpha_vantage=False,
                api_key=self._resolve_config_or_env(
                    self.config.api_keys.alpha_vantage_api_key, "ALPHA_VANTAGE_API_KEY"
                ),
                historical_cache=self._historical_cache,
            )
            self._game_plan_agent = GamePlanAgent(llm_client, market_fetcher)
        return self._game_plan_agent

    def _init_analysis_orchestrator(self) -> AnalysisOrchestrator:
        """Initialize analysis orchestrator (lazy).

        Returns:
            AnalysisOrchestrator instance
        """
        if self._analysis_orchestrator is None:
            workflow = self._init_workflow()
            self._analysis_orchestrator = AnalysisOrchestrator(
                workflow=workflow,
                state=self.state,
                scheduler=self.scheduler,
                config=self.config.analysis_orchestration,
                trading_mode=self.config.trading_mode.value,
                broker=self.broker,
                position_manager=self._position_manager,
                event_bus=self.event_bus,
                historical_cache=self._historical_cache,
                notification_service=self.notification_service,
                context_builder=self,
            )
            logger.info("Analysis orchestrator initialized")
        return self._analysis_orchestrator

    def _init_workflow(self) -> TradingWorkflow:
        """Initialize trading workflow (lazy initialization)."""
        if self._workflow is None:
            llm_client = self._create_llm_client()
            market_fetcher = MarketDataFetcher(
                use_alpha_vantage=False,
                api_key=self._resolve_config_or_env(
                    self.config.api_keys.alpha_vantage_api_key, "ALPHA_VANTAGE_API_KEY"
                ),
                historical_cache=self._historical_cache,
            )
            news_fetcher = NewsFetcher(
                api_key=self._resolve_config_or_env(
                    self.config.api_keys.marketaux_api_key, "MARKETAUX_API_KEY"
                ),
                historical_cache=self._historical_cache,
            )
            finbert = get_finbert_sentiment()
            fundamental_fetcher = FundamentalDataFetcher(
                api_key=self._resolve_config_or_env(
                    self.config.api_keys.alpha_vantage_api_key, "ALPHA_VANTAGE_API_KEY"
                ),
                historical_cache=self._historical_cache,
            )

            # Initialize metrics tracker (DB or JSONL based on DATABASE_URL)
            if self._metrics_tracker is None:
                trade_repository = None
                if os.getenv("DATABASE_URL"):
                    try:
                        from src.database.repositories.trade import TradeRepository
                        from src.database.session import get_session_factory

                        session_factory = get_session_factory()
                        trade_repository = TradeRepository(session_factory())
                    except Exception as e:
                        logger.warning(f"Failed to init DB metrics tracker: {e}, using JSONL")

                self._metrics_tracker = create_metrics_tracker(trade_repository)

            # Portfolio VaR calculator (if risk limits enabled)
            portfolio_var_calculator = None
            portfolio_var_config = None
            if self.config.risk_limits.enabled:
                from src.agents.risk import PortfolioVaRConfig
                from src.metrics.portfolio_var import PortfolioVaRCalculator
                from src.metrics.risk import RiskMetricsCalculator

                portfolio_var_calculator = PortfolioVaRCalculator(RiskMetricsCalculator(), market_fetcher)
                portfolio_var_config = PortfolioVaRConfig(
                    enabled=self.config.risk_limits.enabled,
                    max_var_95=self.config.risk_limits.max_var_95,
                    max_cvar_99=self.config.risk_limits.max_cvar_99,
                    lookback_days=self.config.risk_limits.lookback_days,
                    adaptive_stop_loss=self.config.risk_limits.adaptive_stop_loss,
                    cdar_stop_threshold=self.config.risk_limits.cdar_stop_threshold,
                    atr_multiplier_min=self.config.risk_limits.atr_multiplier_min,
                )

            self._workflow = TradingWorkflow(
                llm_client,
                market_fetcher,
                news_fetcher,
                finbert,
                fundamental_fetcher,
                broker=self.broker,
                metrics_tracker=self._metrics_tracker,
                use_meta_agent=True,
                param_store=self.param_store,
                historical_cache=self._historical_cache,
                portfolio_var_calculator=portfolio_var_calculator,
                portfolio_var_config=portfolio_var_config,
                pre_trade_backtest_config=self.config.pre_trade_backtesting,
                notification_service=self.notification_service,
            )
            logger.info("Trading workflow initialized")

        # Apply target allocations if available
        if hasattr(self, "_target_allocations_to_apply") and self._target_allocations_to_apply:
            self._workflow.set_target_allocations(self._target_allocations_to_apply)

        return self._workflow

    def get_merged_watchlist(self) -> list[str]:
        """Get watchlist merged with broker positions and screening candidates.

        Returns:
            Deduplicated list combining config watchlist, broker positions,
            and latest screening candidates.
        """
        return self._broker_manager.get_merged_watchlist()

    async def _analyze_symbol(
        self,
        symbol: str,
        position_context: dict[str, object] | None = None,
        degradation_context: DegradationContext | None = None,
    ) -> TradingWorkflowResult | None:
        """Analyze a single symbol.

        Args:
            symbol: Stock ticker symbol
            position_context: Position context (entry price, P&L, days held) (optional)
            degradation_context: Optional degradation context

        Returns:
            TradingWorkflowResult or None on error
        """
        from src.strategies.session import TradingSession

        try:
            session = self.scheduler.get_trading_session() or TradingSession.REGULAR
            await self._publish_event("ANALYSIS_START", {"symbol": symbol, "trading_session": session.value})

            workflow = self._init_workflow()
            sector_ctx, earnings_ctx, peer_ctx, game_plan_ctx = self.build_analysis_contexts(symbol)

            result = await workflow.analyze(
                symbol,
                period_days=90,
                trading_session=session,
                position_context=position_context,
                sector_context=sector_ctx,
                earnings_context=earnings_ctx,
                peer_analysis_context=peer_ctx,
                game_plan_context=game_plan_ctx,
                degradation_context=degradation_context,
            )

            if self.notification_service:
                await self._maybe_notify_signal(result)

            rsi = result.technical.rsi if result.technical else None
            macd_hist = result.technical.macd_hist if result.technical else None

            self.state.record_analysis(
                symbol=symbol,
                signal=result.decision.action.value,
                confidence=result.decision.confidence,
                executed=result.order is not None,
                trading_session=result.trading_session.value,
                is_paper_trade=self.config.trading_mode.value == "paper",
                rsi=rsi,
                macd_hist=macd_hist,
            )

            try:
                self._historical_cache.record_signal_outcome(
                    symbol=symbol,
                    timestamp=datetime.now(UTC),
                    signal=result.decision.action.value,
                    confidence=result.decision.confidence,
                    price_at_signal=result.risk.current_price,
                    strategy_used=result.strategy_used,
                    regime=result.regime.regime.value if result.regime else None,
                    trading_session=result.trading_session.value,
                    technical_signal=result.technical.signal.value,
                    sentiment_signal=self._extract_sentiment_signal(result.sentiment),
                    news_signal=self._extract_news_signal(result.news),
                )
            except Exception as e:
                logger.warning(f"Failed to record signal outcome for accuracy tracking: {e}")

            await self._publish_event(
                "ANALYSIS_COMPLETE",
                {
                    "symbol": symbol,
                    "signal": result.decision.action.value,
                    "confidence": result.decision.confidence,
                    "executed": result.order is not None,
                },
            )

            return result
        except Exception as e:
            error_msg = f"Failed to analyze {symbol}: {e}"
            logger.error(error_msg)
            self.state.record_error(error_msg)
            await self._publish_event("ANALYSIS_ERROR", {"symbol": symbol, "error": str(e)})
            return None

    async def _publish_event(self, event_type: str, data: dict[str, object]) -> None:
        """Publish event to EventBus with error handling (async).

        Args:
            event_type: Event type string
            data: Event data dictionary
        """
        if not self.event_bus:
            return

        try:
            from src.daemon.event_bus import DashboardEvent, EventType

            await self.event_bus.publish(DashboardEvent(event_type=EventType[event_type], data=data))
        except Exception as e:
            logger.error(f"Failed to publish {event_type} event: {e}")

    def _publish_event_sync(self, event_type: str, data: dict[str, object]) -> None:
        """Publish event to EventBus with error handling (sync).

        Args:
            event_type: Event type string
            data: Event data dictionary
        """
        if not self.event_bus:
            return

        try:
            from src.daemon.event_bus import DashboardEvent, EventType

            asyncio.run(self.event_bus.publish(DashboardEvent(event_type=EventType[event_type], data=data)))
        except Exception as e:
            logger.error(f"Failed to publish {event_type} event: {e}")

    def build_analysis_contexts(self, symbol: str) -> tuple[str | None, str | None, str | None, str | None]:
        """Build all analysis contexts (sector, earnings, peer, game_plan).

        Args:
            symbol: Stock ticker symbol

        Returns:
            Tuple of (sector_context, earnings_context, peer_context, game_plan_context)
        """
        sector_context: str | None = None
        if self.config.sector_rotation.enabled and self.state.sector_rotation_history:
            try:
                latest_record = self.state.sector_rotation_history[-1]
                sector_context = self._format_sector_context(latest_record)
            except Exception as e:
                logger.warning(f"Failed to build sector context: {e}")

        earnings_context: str | None = None
        if self.config.earnings_calendar.enabled and self.state.earnings_calendar_history:
            try:
                earnings_context = self._build_earnings_context(symbol)
            except Exception as e:
                logger.warning(f"Failed to build earnings context: {e}")

        peer_context: str | None = None
        if self.config.peer_analysis.enabled:
            try:
                peer_context = self._build_peer_context(symbol)
            except Exception as e:
                logger.warning(f"Failed to build peer context: {e}")

        game_plan_context: str | None = None
        if self.config.game_plan.enabled:
            try:
                game_plan_context = self._load_game_plan_context()
            except Exception as e:
                logger.warning(f"Failed to load game plan context: {e}")

        return sector_context, earnings_context, peer_context, game_plan_context

    def _extract_sentiment_signal(self, sentiment: SentimentAnalysis) -> str:
        """Extract signal from sentiment analysis.

        Args:
            sentiment: Sentiment analysis result

        Returns:
            Signal string (BUY/SELL/NEUTRAL)
        """
        if sentiment.sentiment_score > 0.2:
            return "BUY"
        if sentiment.sentiment_score < -0.2:
            return "SELL"
        return "NEUTRAL"

    def _extract_news_signal(self, news: NewsAnalysis) -> str:
        """Extract signal from news analysis.

        Args:
            news: News analysis result

        Returns:
            Signal string (BUY/SELL/NEUTRAL)
        """
        recommendation = news.recommendation.upper()
        if "BUY" in recommendation:
            return "BUY"
        if "SELL" in recommendation:
            return "SELL"
        return "NEUTRAL"

    async def _maybe_notify_signal(self, result: TradingWorkflowResult) -> None:
        """Send signal notification if conditions met.

        Args:
            result: Trading workflow result
        """
        if not self.notification_service:
            return

        if result.decision.action.value == "HOLD":
            return

        if result.decision.confidence < self.config.notifications.min_confidence:
            return

        from src.daemon.config import NotificationTrigger
        from src.daemon.notifications import NotificationMessage

        message = NotificationMessage(
            trigger=NotificationTrigger.SIGNAL,
            title=f"{result.decision.action.value} Signal: {result.symbol}",
            body=" | ".join(result.decision.reasoning),
            metadata={
                "symbol": result.symbol,
                "signal": result.decision.action.value,
                "confidence": result.decision.confidence,
                "price": result.risk.current_price,
                "risk_level": result.risk.validation.risk_level,
                "rsi": result.technical.rsi if result.technical.rsi is not None else "N/A",
                "macd": result.technical.macd_hist if result.technical.macd_hist is not None else "N/A",
                "reasoning": " | ".join(result.decision.reasoning),
                "session": result.trading_session.value,
            },
            timestamp=datetime.now(UTC),
        )

        await self.notification_service.notify(NotificationTrigger.SIGNAL, message)

    async def _notify_var_breach(self, report: PortfolioRiskReport) -> None:
        """Send VaR breach notification.

        Args:
            report: Portfolio risk report
        """
        from src.daemon.config import NotificationTrigger
        from src.daemon.notifications import NotificationMessage

        message = NotificationMessage(
            trigger=NotificationTrigger.PORTFOLIO_VAR_BREACH,
            title="Portfolio VaR Limit Breached",
            body=f"VaR95: {report.var_95:.1%} | CVaR99: {report.cvar_99:.1%}",
            metadata={
                "symbol": "PORTFOLIO",
                "var_95": report.var_95,
                "cvar_99": report.cvar_99,
                "var_breached": report.var_limit_breached,
                "cvar_breached": report.cvar_limit_breached,
                "num_positions": report.num_positions,
            },
            timestamp=datetime.now(UTC),
        )

        if self.notification_service:
            await self.notification_service.notify(NotificationTrigger.PORTFOLIO_VAR_BREACH, message)

    def _evaluate_degradation(self) -> DegradationContext:
        """Load latest health report and evaluate degradation tier.

        Returns:
            DegradationContext with tier and available agents
        """
        from src.daemon.degradation import DegradationPolicy

        policy = DegradationPolicy(self.config)
        health_report = self._load_latest_health_report()
        return policy.evaluate_degradation(health_report)

    def _load_latest_health_report(self) -> HealthReport | None:
        """Load most recent health report from disk.

        Returns:
            HealthReport if available, None otherwise
        """
        from src.daemon.health import HealthReport

        health_dir = Path(self.config.health.health_dir).expanduser()
        if not health_dir.exists():
            return None

        report_files = sorted(health_dir.glob("health-*.json"), reverse=True)
        if not report_files:
            return None

        try:
            with report_files[0].open() as f:
                return HealthReport.model_validate(json.load(f))
        except Exception as e:
            logger.warning(f"Failed to load health report: {e}")
            return None

    async def _notify_degradation(self, context: DegradationContext) -> None:
        """Send degradation notification.

        Args:
            context: Degradation context
        """
        from src.daemon.config import NotificationTrigger
        from src.daemon.degradation import DegradationTier
        from src.daemon.notifications import NotificationMessage

        # Determine title and body
        if context.tier == DegradationTier.HALTED:
            title = "Trading System HALTED"
            body = context.halt_reason or "Critical services unavailable"
        else:
            title = f"Trading System {context.tier.value}"
            services = ", ".join(context.unavailable_services) if context.unavailable_services else "Unknown"
            body = f"APIs down: {services}"

        message = NotificationMessage(
            trigger=NotificationTrigger.HEALTH_FAILURE,
            title=title,
            body=body,
            metadata={
                "tier": context.tier.value,
                "unavailable_services": context.unavailable_services,
                "confidence_adjustment": context.confidence_adjustment,
            },
            timestamp=datetime.now(UTC),
        )

        if self.notification_service:
            await self.notification_service.notify(NotificationTrigger.HEALTH_FAILURE, message)

        # Publish DEGRADATION event
        if self.event_bus:
            try:
                from src.daemon.event_bus import DashboardEvent, EventType

                await self.event_bus.publish(
                    DashboardEvent(
                        event_type=EventType.DEGRADATION,
                        data={
                            "tier": context.tier.value,
                            "unavailable_services": context.unavailable_services,
                            "confidence_adjustment": context.confidence_adjustment,
                        },
                    )
                )
            except Exception as e:
                logger.error(f"Failed to publish DEGRADATION event: {e}")

    def _format_sector_context(self, record: SectorRotationRecord) -> str:
        """Format sector rotation record as text for trader prompt.

        Args:
            record: Sector rotation state record

        Returns:
            Formatted context string
        """
        lines = [
            f"Leading Sectors: {', '.join(record.leading_sectors)}",
            f"Lagging Sectors: {', '.join(record.lagging_sectors)}",
            "",
        ]

        # Sort by strength descending
        sorted_sectors = sorted(record.sector_strengths.items(), key=lambda x: x[1], reverse=True)

        for rank, (sector, strength) in enumerate(sorted_sectors, 1):
            momentum = record.sector_momenta.get(sector, "NEUTRAL")
            lines.append(f"  {rank}. {sector}: strength={strength:+.2f} [{momentum}]")

        return "\n".join(lines)

    def _build_sector_context(self) -> str | None:
        """Build sector rotation context from latest record.

        Returns:
            Formatted sector context string or None if not available
        """
        if not self.config.sector_rotation.enabled or not self.state.sector_rotation_history:
            return None

        try:
            latest_record = self.state.sector_rotation_history[-1]
            return self._format_sector_context(latest_record)
        except Exception as e:
            logger.warning(f"Failed to build sector context: {e}")
            return None

    def _reconstruct_rotation_analysis(self, record: SectorRotationRecord) -> SectorRotationAnalysis:
        """Reconstruct SectorRotationAnalysis from state record.

        Args:
            record: Sector rotation state record

        Returns:
            Full SectorRotationAnalysis pydantic model
        """
        from src.data.comparative import Sector
        from src.metrics.sector_rotation import Momentum, SectorStrength

        # Reconstruct SectorStrength list from record data
        sectors = []
        sorted_sectors = sorted(record.sector_strengths.items(), key=lambda x: x[1], reverse=True)

        for rank, (sector_name, strength) in enumerate(sorted_sectors, 1):
            momentum_str = record.sector_momenta.get(sector_name, "NEUTRAL")

            # Find ETF for sector (map back from Sector enum)
            try:
                sector_enum = Sector[sector_name]
                etf = sector_enum.value
            except KeyError:
                logger.warning(f"Unknown sector {sector_name}, skipping")
                continue

            sectors.append(
                SectorStrength(
                    sector=sector_name,
                    etf=etf,
                    return_1w=0.0,  # Not stored in record
                    return_1m=0.0,
                    return_3m=0.0,
                    relative_strength=strength,
                    momentum=Momentum(momentum_str),
                    rank=rank,
                )
            )

        return SectorRotationAnalysis(
            sectors=sectors,
            leading_sectors=record.leading_sectors,
            lagging_sectors=record.lagging_sectors,
            spy_return_1w=0.0,  # Not stored, not needed for weighting
            spy_return_1m=0.0,
            spy_return_3m=0.0,
            timestamp=record.timestamp,
        )

    async def _analyze_watchlist(
        self,
        watchlist: list[str],
        degradation_context: DegradationContext | None = None,
    ) -> list[TradingWorkflowResult]:
        """Analyze all symbols in watchlist (delegates to orchestrator).

        Args:
            watchlist: List of symbols to analyze
            degradation_context: Optional degradation context

        Returns:
            List of analysis results
        """
        # Build target allocations from last rebalancing (if recent)
        target_allocations = None
        if self.state.active_target_allocations and self.state.last_portfolio_rebalancing:
            days_old = (datetime.now(UTC) - self.state.last_portfolio_rebalancing).days
            if days_old < self.config.analysis_orchestration.target_allocation_ttl_days:
                target_allocations = self.state.active_target_allocations
                logger.info(f"Using target allocations from {days_old} days ago")

        # Delegate to orchestrator
        orchestrator = self._init_analysis_orchestrator()
        result = await orchestrator.orchestrate(watchlist, target_allocations, degradation_context)

        logger.info(
            f"Orchestration complete: {result.successful}/{result.total_symbols} successful, "
            f"{result.failed} failed, {result.position_actions} position actions, "
            f"{result.duration_seconds:.2f}s"
        )

        return result.results

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

    async def _maybe_run_journal(self) -> None:
        """Run after-hours journal if conditions are met."""
        if not self.config.journal.enabled:
            return

        if not self.scheduler.is_journal_window(self.config.journal.run_offset_minutes):
            return

        today = datetime.now(self.scheduler.timezone).date()
        if self.state.last_journal_date == today.isoformat():
            return

        # Filter today's analysis records
        today_records = [r for r in self.state.analyses if r.timestamp.date() == today]
        if not today_records:
            logger.info("No analyses today, skipping journal")
            return

        logger.info(f"Generating trade journal for {today} ({len(today_records)} records)")
        console.print(f"\n[bold magenta]Generating trade journal for {today}...[/bold magenta]")

        try:
            from src.agents.journal import TradeJournalAgent

            workflow = self._init_workflow()
            market_fetcher = MarketDataFetcher(
                use_alpha_vantage=False,
                api_key=self._resolve_config_or_env(
                    self.config.api_keys.alpha_vantage_api_key, "ALPHA_VANTAGE_API_KEY"
                ),
            )
            journal_agent = TradeJournalAgent(workflow.llm_client, market_fetcher)

            journal = await journal_agent.generate(today, today_records)
            file_path = journal_agent.persist(journal, self.config.journal.journal_dir)

            self.state.last_journal_date = today.isoformat()
            self.state.save(self.config.state.state_file)

            correct = sum(1 for o in journal.outcomes if o.signal_correct)
            total = len(journal.outcomes)
            console.print(f"[bold magenta]Journal saved:[/bold magenta] {file_path}")
            if total > 0:
                console.print(f"[bold magenta]Signal accuracy:[/bold magenta] {correct}/{total}")
        except Exception as e:
            logger.error(f"Journal generation failed: {e}")
            self.state.record_error(f"Journal failed: {e}")
            self.state.save(self.config.state.state_file)

    async def _maybe_check_paper_readiness(self) -> None:
        """Check paper trading readiness and notify if ready (once per day)."""
        if self.config.trading_mode != TradingMode.PAPER:
            return

        if not self.notification_service:
            return

        if not hasattr(self, "_last_readiness_check"):
            self._last_readiness_check = None
            self._notified_paper_ready = False

        now = datetime.now(UTC)

        # Check once per day
        if self._last_readiness_check is not None:
            elapsed_days = (now - self._last_readiness_check).days
            if elapsed_days < 1:
                return

        self._last_readiness_check = now

        try:
            from src.daemon.notification_formatter import NotificationTrigger
            from src.daemon.paper_trading_validator import PaperTradingValidator

            validator = PaperTradingValidator(
                config=self.config.paper_trading,
                state=self.state,
                metrics_tracker=self._metrics_tracker,
            )
            report = validator.assess_readiness()

            if report.ready_for_live and not self._notified_paper_ready:
                await self.notification_service.notify(
                    symbol="SYSTEM",
                    trigger=NotificationTrigger.PAPER_TRADING_READY,
                    message_data={
                        "duration_days": report.paper_trading_duration_days,
                        "total_trades": report.total_paper_trades,
                        "sharpe": report.metrics.sharpe_ratio,
                        "max_dd": report.metrics.max_drawdown_percent,
                    },
                )
                self._notified_paper_ready = True
                logger.info("Sent paper trading readiness notification")
        except Exception as e:
            logger.debug(f"Paper readiness check failed: {e}")

    def _run_optimization(self) -> None:
        """Run after-hours strategy parameter optimization."""
        if not self._daemon_optimizer:
            return

        now = datetime.now(self.scheduler.timezone)
        if self.state.last_optimization:
            last_date = self.state.last_optimization.astimezone(self.scheduler.timezone).date()
            if last_date == now.date():
                logger.debug("Optimization already completed today")
                return

        logger.info("Starting after-hours parameter optimization")
        console.print(f"\n[bold cyan]Parameter Optimization ({now:%H:%M})[/bold cyan]")
        console.print("-" * 50)

        self._publish_event_sync("SCHEDULED_TASK", {"task_name": "optimization", "status": "started"})

        try:
            import time as time_mod

            start_time = time_mod.time()
            watchlist = self.get_merged_watchlist()

            optimized, skipped, failed = self._daemon_optimizer.optimize_watchlist(
                watchlist=watchlist,
                strategies=self.config.optimization.strategies,
                refresh_days=self.config.optimization.refresh_days,
            )

            total_time = time_mod.time() - start_time

            self.state.record_optimization(
                symbols_optimized=optimized,
                symbols_skipped=skipped,
                total_time_seconds=total_time,
            )
            self.state.save(self.config.state.state_file)

            if failed:
                for symbol, strategies_str in failed:
                    logger.warning(f"Failed to optimize {symbol}: {strategies_str}")

            console.print(
                f"\n[dim]Optimization complete: {len(optimized)} symbols optimized, "
                f"{len(skipped)} skipped ({total_time:.0f}s)[/dim]\n"
            )
            logger.info(f"Parameter optimization completed in {total_time:.0f}s")

            self._publish_event_sync("SCHEDULED_TASK", {"task_name": "optimization", "status": "completed"})

        except Exception as e:
            error_msg = f"Parameter optimization failed: {e}"
            logger.error(error_msg)
            self.state.record_error(error_msg)

    def _run_portfolio_rebalancing(self) -> None:
        """Run portfolio rebalancing optimization."""
        if not self._daemon_rebalancer:
            return

        # Check if already rebalanced today
        now = datetime.now(self.scheduler.timezone)
        if self.state.last_portfolio_rebalancing:
            last_date = self.state.last_portfolio_rebalancing.astimezone(self.scheduler.timezone).date()
            if last_date == now.date():
                logger.debug("Portfolio rebalancing already completed today")
                return

        logger.info("Starting portfolio rebalancing")
        console.print(f"\n[bold cyan]Portfolio Rebalancing ({now:%H:%M})[/bold cyan]")
        console.print("-" * 50)

        try:
            watchlist = self.get_merged_watchlist()
            method = self.config.rebalancing.method
            auto_execute = self.config.auto_trade

            console.print(f"[dim]Method: {method}, Universe: {len(watchlist)} symbols[/dim]")

            result = self._daemon_rebalancer.run(watchlist, method, auto_execute)

            # Convert to state records
            from src.daemon.state import PortfolioAllocationRecord

            allocations = [
                PortfolioAllocationRecord(symbol=alloc.symbol, weight=alloc.weight, action="HOLD", delta=0.0)
                for alloc in result.optimized_portfolio.allocations
            ]

            # Update allocations with rebalance actions
            rebalance_map = {r.symbol: r for r in result.rebalance_instructions}
            for alloc in allocations:
                if alloc.symbol in rebalance_map:
                    rebalance = rebalance_map[alloc.symbol]
                    alloc.action = rebalance.action
                    alloc.delta = rebalance.delta

            self.state.record_portfolio_rebalancing(
                method=method,
                allocations=allocations,
                expected_return=result.optimized_portfolio.expected_return,
                expected_volatility=result.optimized_portfolio.expected_volatility,
                sharpe_ratio=result.optimized_portfolio.sharpe_ratio,
                rebalances_executed=result.executed_count,
                rebalances_pending=result.pending_count,
            )
            self.state.save(self.config.state.state_file)

            # Display summary
            console.print("\n[bold]Portfolio Metrics:[/bold]")
            console.print(f"  Expected Return: {result.optimized_portfolio.expected_return:.2%}")
            console.print(f"  Volatility: {result.optimized_portfolio.expected_volatility:.2%}")
            console.print(f"  Sharpe Ratio: {result.optimized_portfolio.sharpe_ratio:.2f}")

            if result.rebalance_instructions:
                console.print("\n[bold]Rebalancing Instructions:[/bold]")
                for rebalance in result.rebalance_instructions[:10]:
                    action_color = (
                        "green"
                        if rebalance.action == "BUY"
                        else "red"
                        if rebalance.action == "SELL"
                        else "dim"
                    )
                    console.print(
                        f"  [{action_color}]{rebalance.action:4}[/{action_color}] "
                        f"{rebalance.symbol:6} "
                        f"{rebalance.target_weight:6.2%} "
                        f"({rebalance.delta:+.2%})"
                    )

                if len(result.rebalance_instructions) > 10:
                    console.print(f"  [dim]... and {len(result.rebalance_instructions) - 10} more[/dim]")

            console.print(
                f"\n[dim]Rebalancing complete: {result.executed_count} executed, "
                f"{result.pending_count} pending[/dim]\n"
            )
            logger.info(
                f"Portfolio rebalancing completed: {result.executed_count}/"
                f"{len(result.rebalance_instructions)} executed"
            )

        except Exception as e:
            error_msg = f"Portfolio rebalancing failed: {e}"
            logger.error(error_msg)
            self.state.record_error(error_msg)

    def _run_prefetch(self) -> None:
        """Run after-hours data prefetching for watchlist symbols."""
        if not self.config.prefetch.enabled:
            return

        # Dedup check
        now = datetime.now(self.scheduler.timezone)
        if self.state.last_prefetch:
            last_date = self.state.last_prefetch.astimezone(self.scheduler.timezone).date()
            if last_date == now.date():
                logger.debug("Prefetch already completed today")
                return

        logger.info("Starting after-hours data prefetching")
        console.print(f"\n[bold cyan]Data Prefetch ({now:%H:%M})[/bold cyan]")
        console.print("-" * 50)

        try:
            prefetcher = self._init_prefetcher()
            if prefetcher is None:
                logger.warning("Prefetcher unavailable (missing ALPHA_VANTAGE_API_KEY), skipping")
                return

            watchlist = self.get_merged_watchlist()

            console.print(f"[dim]Prefetching {len(watchlist)} symbols...[/dim]")
            report = prefetcher.prefetch_watchlist(watchlist)

            # Warm FinBERT if configured
            finbert_ready = False
            if self.config.prefetch.warm_finbert:
                console.print("[dim]Warming FinBERT model...[/dim]")
                finbert_ready = prefetcher.warm_finbert()
            report.finbert_ready = finbert_ready

            # Check API connectivity if configured
            if self.config.prefetch.check_connectivity:
                report.api_connectivity = prefetcher.check_api_key_presence()

            # Count successes/failures
            succeeded = sum(1 for r in report.results if r.market_data or r.news or r.fundamentals)
            failed = len(report.results) - succeeded

            self.state.record_prefetch(
                symbols_prefetched=succeeded,
                symbols_failed=failed,
                finbert_ready=finbert_ready,
                total_duration_seconds=report.total_duration_seconds,
            )
            self.state.save(self.config.state.state_file)

            console.print(
                f"\n[dim]Prefetch complete: {succeeded} symbols cached, "
                f"{failed} failed ({report.total_duration_seconds:.0f}s)[/dim]\n"
            )
            logger.info(
                f"Data prefetch completed: {succeeded} cached, {failed} failed "
                f"in {report.total_duration_seconds:.0f}s"
            )

        except Exception as e:
            error_msg = f"Data prefetch failed: {e}"
            logger.error(error_msg)
            self.state.record_error(error_msg)

    def _run_pre_market_refresh(self) -> None:
        """Run pre-market data refresh to update stale cache."""
        if not self.config.prefetch.enabled or not self.config.prefetch.enable_pre_market_refresh:
            return

        # Dedup check
        now = datetime.now(self.scheduler.timezone)
        if self.state.last_pre_market_refresh:
            last_date = self.state.last_pre_market_refresh.astimezone(self.scheduler.timezone).date()
            if last_date == now.date():
                logger.debug("Pre-market refresh already completed today")
                return

        logger.info("Starting pre-market data refresh")
        console.print(f"\n[bold cyan]Pre-Market Data Refresh ({now:%H:%M})[/bold cyan]")
        console.print("-" * 50)

        try:
            prefetcher = self._init_prefetcher()
            if prefetcher is None:
                logger.warning("Prefetcher unavailable (missing ALPHA_VANTAGE_API_KEY), skipping")
                return

            watchlist = self.get_merged_watchlist()

            console.print(f"[dim]Refreshing {len(watchlist)} symbols...[/dim]")
            report = prefetcher.prefetch_watchlist(watchlist)

            succeeded = sum(1 for r in report.results if r.market_data or r.news or r.fundamentals)

            self.state.last_pre_market_refresh = datetime.now(self.scheduler.timezone)
            self.state.save(self.config.state.state_file)

            console.print(
                f"\n[dim]Pre-market refresh complete: {succeeded} symbols updated "
                f"({report.total_duration_seconds:.0f}s)[/dim]\n"
            )
            logger.info(
                f"Pre-market refresh completed: {succeeded} symbols in {report.total_duration_seconds:.0f}s"
            )

        except Exception as e:
            error_msg = f"Pre-market refresh failed: {e}"
            logger.error(error_msg)
            self.state.record_error(error_msg)

    def _run_after_hours_screening(self) -> None:
        """Run after-hours screening for watchlist candidates."""
        from src.data.universe import StockUniverseFetcher
        from src.screening.exporter import ScreeningExporter
        from src.screening.screener import ScreeningCriteria, StockScreener

        # Check if already screened today
        now = datetime.now(self.scheduler.timezone)
        if self.state.last_after_hours_screening:
            last_date = self.state.last_after_hours_screening.astimezone(self.scheduler.timezone).date()
            if last_date == now.date():
                logger.debug("After-hours screening already completed today")
                return

        logger.info("Starting after-hours watchlist screening")
        console.print(f"\n[bold cyan]After-Hours Screening ({now:%H:%M})[/bold cyan]")
        console.print("-" * 50)

        try:
            # Initialize screener
            universe_fetcher = StockUniverseFetcher()
            screener = StockScreener(universe_fetcher)

            # Parse criteria
            criteria_map = {
                "momentum": ScreeningCriteria.MOMENTUM,
                "value": ScreeningCriteria.VALUE,
                "breakout": ScreeningCriteria.BREAKOUT,
            }
            criteria = criteria_map.get(self.config.screening.criteria.lower(), ScreeningCriteria.MOMENTUM)

            # Run screening
            console.print(
                f"[dim]{criteria.value.title()} Screening[/dim]\n"
                f"[dim]Universe: {self.config.screening.universe}[/dim]"
            )
            output = screener.screen(
                criteria=criteria,
                universe=self.config.screening.universe,
                top_n=self.config.screening.top_n,
            )

            # Apply sector rotation weighting if available
            results_to_save = output.results
            if self.config.sector_rotation.enabled and self.state.sector_rotation_history:
                try:
                    from src.daemon.sector_rotation import DaemonSectorRotation

                    # Reconstruct analysis from latest state record
                    latest_record = self.state.sector_rotation_history[-1]
                    rotation_analysis = self._reconstruct_rotation_analysis(latest_record)

                    daemon_rotation = DaemonSectorRotation()
                    results_to_save = daemon_rotation.weight_candidates(
                        output.results,
                        rotation_analysis,
                        self.config.sector_rotation.boost_factor,
                    )
                    logger.info("Applied sector rotation weighting to screening candidates")
                except Exception as e:
                    logger.warning(f"Failed to apply sector weighting: {e}")

            # Log top 5 to console
            self._log_screening_results(results_to_save[:5])

            # Save to watchlist file
            exporter = ScreeningExporter()
            exporter.save_to_watchlist(
                results=results_to_save[: self.config.screening.top_n],
                criteria=criteria,
                watchlist_name=self.config.screening.watchlist_name,
            )

            # Record in state
            self.state.record_after_hours_screening(
                criteria=criteria.value,
                universe=self.config.screening.universe,
                candidates=results_to_save,
                top_n=self.config.screening.top_n,
                screened_at=output.screened_at,
            )
            self.state.save(self.config.state.state_file)

            console.print(
                f"\n[dim]Top {self.config.screening.top_n} candidates saved to daemon state "
                f"({len(output.results)} total screened)[/dim]\n"
            )
            logger.info(f"After-hours screening completed: {len(output.results)} candidates")

        except Exception as e:
            error_msg = f"After-hours screening failed: {e}"
            logger.error(error_msg)
            self.state.record_error(error_msg)

    async def _run_game_plan(self) -> None:
        """Generate daily game plan from overnight market action."""
        now = datetime.now(self.scheduler.timezone)
        if self.state.last_game_plan:
            last_date = self.state.last_game_plan.astimezone(self.scheduler.timezone).date()
            if last_date == now.date():
                logger.debug("Game plan already generated today")
                return

        logger.info("Generating daily game plan")
        console.print(f"\n[bold cyan]Game Plan Generation ({now:%H:%M})[/bold cyan]")
        console.print("-" * 50)

        try:
            agent = self._init_game_plan_agent()
            watchlist = self.get_merged_watchlist()

            sector_context = self._build_sector_context()
            earnings_context = self._build_earnings_context_for_watchlist(watchlist)

            plan = await agent.generate(
                watchlist,
                futures_symbols=self.config.game_plan.futures_symbols,
                sector_context=sector_context,
                earnings_context=earnings_context,
                timezone=self.scheduler.timezone,
            )

            plan_path = agent.persist(plan, self.config.game_plan.plan_dir)

            self.state.record_game_plan(
                priority_symbols=plan.priority_symbols,
                risk_stance=plan.risk_stance,
                sector_focus=plan.sector_focus,
            )
            self.state.save(self.config.state.state_file)

            console.print("[bold green]✓ Game Plan Generated[/bold green]")
            console.print(f"  Risk Stance: {plan.risk_stance}")
            console.print(f"  Priority: {', '.join(plan.priority_symbols)}")
            console.print(f"  Sectors: {', '.join(plan.sector_focus)}")
            console.print(f"  Saved: {plan_path}\n")

        except Exception as e:
            error_msg = f"Game plan generation failed: {e}"
            logger.error(error_msg)
            self.state.record_error(error_msg)
            console.print(f"[red]✗ {error_msg}[/red]\n")

    def _load_game_plan_context(self) -> str | None:
        """Load today's game plan and format as context string.

        Returns:
            Formatted game plan context or None
        """
        plan_dir = Path(self.config.game_plan.plan_dir).expanduser()
        today = datetime.now(self.scheduler.timezone).date()
        plan_file = plan_dir / f"{today}.json"

        if not plan_file.exists():
            return None

        try:
            with plan_file.open() as f:
                data = json.load(f)
                plan = GamePlan.model_validate(data)

            key_levels_str = ", ".join(f"{sym}: ${lvl:.2f}" for sym, lvl in plan.key_levels.items())
            return (
                f"Risk Stance: {plan.risk_stance}\n"
                f"Priority Symbols: {', '.join(plan.priority_symbols)}\n"
                f"Sector Focus: {', '.join(plan.sector_focus)}\n"
                f"Key Levels: {key_levels_str}\n"
                f"Reasoning: {plan.reasoning}"
            )
        except Exception as e:
            logger.warning(f"Failed to load game plan context: {e}")
            return None

    def _build_earnings_context_for_watchlist(self, watchlist: list[str]) -> str | None:
        """Build earnings context for all watchlist symbols.

        Args:
            watchlist: List of symbols

        Returns:
            Combined earnings context or None
        """
        if not watchlist:
            return None

        contexts = []
        for symbol in watchlist:
            ctx = self._build_earnings_context(symbol)
            if ctx:
                contexts.append(ctx)

        return "\n".join(contexts) if contexts else None

    def _run_sector_rotation(self) -> None:
        """Run sector rotation analysis."""
        from src.daemon.sector_rotation import DaemonSectorRotation

        now = datetime.now(self.scheduler.timezone)
        if self.state.last_sector_rotation:
            last_date = self.state.last_sector_rotation.astimezone(self.scheduler.timezone).date()
            if last_date == now.date():
                logger.debug("Sector rotation already completed today")
                return

        logger.info("Starting sector rotation analysis")
        console.print(f"\n[bold cyan]Sector Rotation Analysis ({now:%H:%M})[/bold cyan]")
        console.print("-" * 50)

        self._publish_event_sync("SCHEDULED_TASK", {"task_name": "sector_rotation", "status": "started"})

        try:
            daemon_rotation = DaemonSectorRotation()
            analysis = daemon_rotation.run()

            flagged: list[str] = []
            if self.broker:
                try:
                    account_info = self.broker.get_account_info()
                    position_symbols = list(account_info.positions.keys())
                    flagged = daemon_rotation.flag_weak_positions(position_symbols, analysis)
                except Exception as e:
                    logger.warning(f"Failed to flag positions: {e}")

            sector_strengths = {s.sector: s.relative_strength for s in analysis.sectors}
            sector_momenta = {s.sector: s.momentum.value for s in analysis.sectors}

            self.state.record_sector_rotation(
                leading_sectors=analysis.leading_sectors,
                lagging_sectors=analysis.lagging_sectors,
                sector_strengths=sector_strengths,
                sector_momenta=sector_momenta,
                flagged_positions=flagged,
            )
            self.state.save(self.config.state.state_file)

            console.print(f"[dim]Leading: {', '.join(analysis.leading_sectors)}[/dim]")
            console.print(f"[dim]Lagging: {', '.join(analysis.lagging_sectors)}[/dim]")
            if flagged:
                console.print(f"[bold yellow]Flagged positions: {', '.join(flagged)}[/bold yellow]")
            console.print(
                f"\n[dim]Sector rotation complete: {len(analysis.sectors)} sectors analyzed[/dim]\n"
            )
            logger.info("Sector rotation analysis completed")

            self._publish_event_sync(
                "SCHEDULED_TASK", {"task_name": "sector_rotation", "status": "completed"}
            )

        except Exception as e:
            error_msg = f"Sector rotation failed: {e}"
            logger.error(error_msg)
            self.state.record_error(error_msg)

    def _build_earnings_context(self, symbol: str) -> str | None:
        """Build earnings context string from latest calendar state.

        Args:
            symbol: Stock ticker to build context for

        Returns:
            Formatted earnings context or None
        """
        if not self.state.earnings_calendar_history:
            return None

        from datetime import date

        from src.daemon.earnings import DaemonEarningsCalendar
        from src.data.earnings import EarningsEvent

        latest = self.state.earnings_calendar_history[-1]
        events = [
            EarningsEvent(
                symbol=e.symbol,
                earnings_date=date.fromisoformat(e.earnings_date),
                estimate_eps=e.estimate_eps,
            )
            for e in latest.events
        ]

        daemon_earnings = DaemonEarningsCalendar()
        upcoming = daemon_earnings.get_upcoming(
            events, days_ahead=self.config.earnings_calendar.lookahead_days
        )
        if not upcoming:
            return None

        # Filter to current symbol + overall context
        symbol_events = [e for e in upcoming if e.symbol == symbol]
        other_events = [e for e in upcoming if e.symbol != symbol]

        lines: list[str] = []
        if symbol_events:
            lines.append(daemon_earnings.format_context(symbol_events))
        if other_events:
            lines.append(f"Other watchlist earnings upcoming: {', '.join(e.symbol for e in other_events)}")

        return "\n".join(lines) if lines else None

    def _run_earnings_fetch(self) -> None:
        """Run earnings calendar fetch for watchlist symbols."""
        from src.daemon.earnings import DaemonEarningsCalendar

        # Weekly dedup: check if already fetched this week on a configured day
        now = datetime.now(self.scheduler.timezone)
        if self.state.last_earnings_fetch:
            last_date = self.state.last_earnings_fetch.astimezone(self.scheduler.timezone).date()
            # Skip if already fetched today
            if last_date == now.date():
                logger.debug("Earnings calendar already fetched today")
                return

        # Check calendar-aware weekly schedule
        if not self.scheduler.is_earnings_fetch_time():
            logger.debug("Not earnings fetch time, skipping")
            return

        logger.info("Starting earnings calendar fetch")
        console.print(f"\n[bold cyan]Earnings Calendar Fetch ({now:%H:%M})[/bold cyan]")
        console.print("-" * 50)

        try:
            daemon_earnings = DaemonEarningsCalendar()
            watchlist = self.get_merged_watchlist()

            console.print(f"[dim]Fetching earnings for {len(watchlist)} symbols...[/dim]")
            calendar = daemon_earnings.fetch(watchlist)

            # Build event records
            event_records = [
                EarningsEventRecord(
                    symbol=e.symbol,
                    earnings_date=e.earnings_date.isoformat(),
                    estimate_eps=e.estimate_eps,
                )
                for e in calendar.events
            ]

            symbols_with_earnings = len(calendar.events)
            symbols_without_earnings = max(0, len(watchlist) - symbols_with_earnings)
            if symbols_without_earnings:
                logger.info(
                    "Earnings calendar: %d symbols with earnings data, %d symbols with no earnings data",
                    symbols_with_earnings,
                    symbols_without_earnings,
                )

            # NOTE: Missing earnings data is normal, not a failure
            self.state.record_earnings_fetch(
                events=event_records,
                symbols_fetched=symbols_with_earnings,
                symbols_failed=0,  # Only track known fetch failures
            )
            self.state.save(self.config.state.state_file)

            # Show upcoming earnings
            upcoming = daemon_earnings.get_upcoming(
                calendar.events, days_ahead=self.config.earnings_calendar.lookahead_days
            )
            if upcoming:
                console.print("[bold yellow]Upcoming earnings:[/bold yellow]")
                for event in upcoming:
                    days_until = (event.earnings_date - now.date()).days
                    console.print(f"  {event.symbol}: {event.earnings_date} ({days_until}d away)")
            else:
                console.print("[dim]No upcoming earnings within lookahead window[/dim]")

            console.print(
                f"\n[dim]Earnings fetch complete: {len(calendar.events)} symbols with earnings data[/dim]\n"
            )
            logger.info(f"Earnings calendar fetch completed: {len(calendar.events)} events")

        except Exception as e:
            error_msg = f"Earnings calendar fetch failed: {e}"
            logger.error(error_msg)
            self.state.record_error(error_msg)

    def _run_peer_analysis(self) -> None:
        """Run weekly deep peer benchmarking analysis."""
        from src.daemon.peer_analysis import DeepPeerAnalyzer
        from src.data.universe import StockUniverseFetcher

        # Dedup check
        now = datetime.now(self.scheduler.timezone)
        if self.state.last_peer_analysis:
            last_date = self.state.last_peer_analysis.astimezone(self.scheduler.timezone).date()
            if last_date == now.date():
                logger.debug("Peer analysis already completed today")
                return

        logger.info("Starting deep peer benchmarking analysis")
        console.print(f"\n[bold cyan]Peer Benchmarking Analysis ({now:%H:%M})[/bold cyan]")
        console.print("-" * 50)

        try:
            fundamental_fetcher = FundamentalDataFetcher(
                api_key=self._resolve_config_or_env(
                    self.config.api_keys.alpha_vantage_api_key, "ALPHA_VANTAGE_API_KEY"
                ),
                historical_cache=self._historical_cache,
            )
            universe_fetcher = StockUniverseFetcher()
            analyzer = DeepPeerAnalyzer(
                fundamental_fetcher=fundamental_fetcher,
                universe_fetcher=universe_fetcher,
                output_dir=self.config.peer_analysis.output_dir,
                max_peers=self.config.peer_analysis.max_peers,
                rate_limit_sleep=self.config.peer_analysis.rate_limit_sleep,
                historical_cache=self._historical_cache,
            )

            watchlist = self.get_merged_watchlist()
            console.print(f"[dim]Analyzing {len(watchlist)} positions against peers...[/dim]")

            result = analyzer.analyze_positions(watchlist)

            # Build state record
            rankings = {a.symbol: a.rank for a in result.analyses}
            swaps = [a.swap_recommendation for a in result.analyses if a.swap_recommendation]

            self.state.record_peer_analysis(
                symbols_analyzed=[a.symbol for a in result.analyses],
                rankings=rankings,
                swap_recommendations=swaps,
                total_peers=result.total_peers_analyzed,
                total_duration_seconds=result.total_duration_seconds,
            )
            self.state.save(self.config.state.state_file)

            # Console output
            for analysis in result.analyses:
                rank_color = "green" if analysis.rank <= 3 else "yellow" if analysis.rank <= 5 else "red"
                console.print(
                    f"  [bold]{analysis.symbol}[/bold]: "
                    f"[{rank_color}]#{analysis.rank}[/{rank_color}] of {analysis.peer_count} "
                    f"in {analysis.sector}"
                )
            if swaps:
                console.print(f"[bold yellow]Swap recommendations: {len(swaps)}[/bold yellow]")
                for swap in swaps:
                    console.print(f"  {swap}")

            console.print(
                f"\n[dim]Peer analysis complete: {len(result.analyses)} positions, "
                f"{result.total_peers_analyzed} peers ({result.total_duration_seconds:.0f}s)[/dim]\n"
            )
            logger.info("Deep peer benchmarking analysis completed")

        except Exception as e:
            error_msg = f"Peer benchmarking analysis failed: {e}"
            logger.error(error_msg)
            self.state.record_error(error_msg)

    def _should_skip_correlation_audit(self, now: datetime) -> bool:
        """Check if correlation audit should be skipped (already ran today)."""
        if not self.state.last_correlation_audit:
            return False
        last_date = self.state.last_correlation_audit.astimezone(self.scheduler.timezone).date()
        return last_date == now.date()

    def _print_correlation_audit_results(self, result: CorrelationAuditResult, duration: float) -> None:
        """Print correlation audit results to console."""
        console.print(f"[dim]Positions: {result.num_positions}[/dim]")
        console.print(f"[dim]Diversification ratio: {result.diversification_ratio:.3f}[/dim]")

        if result.highly_correlated_pairs:
            count = len(result.highly_correlated_pairs)
            console.print(f"\n[bold yellow]Correlated Pairs ({count}):[/bold yellow]")
            for pair in result.highly_correlated_pairs[:5]:
                console.print(f"  {pair.symbol_a} ↔ {pair.symbol_b}: {pair.correlation:.3f}")

        if result.substitution_suggestions:
            count = len(result.substitution_suggestions)
            console.print(f"\n[bold yellow]Substitutions ({count}):[/bold yellow]")
            for suggestion in result.substitution_suggestions[:3]:
                alts = ", ".join(suggestion.alternatives)
                console.print(f"  Replace {suggestion.symbol_to_replace}: {suggestion.reason}")
                console.print(f"    → {alts}")

        if result.warnings:
            console.print(f"\n[dim]Warnings: {', '.join(result.warnings)}[/dim]")

        console.print(f"\n[dim]Complete in {duration:.1f}s[/dim]\n")

    def _run_correlation_audit(self) -> None:
        """Run portfolio correlation audit."""
        from src.metrics.correlation import CorrelationAuditor

        now = datetime.now(self.scheduler.timezone)
        if self._should_skip_correlation_audit(now):
            logger.debug("Correlation audit already run today")
            return

        logger.info("Starting portfolio correlation audit")
        console.print(f"\n[bold cyan]Portfolio Correlation Audit ({now:%H:%M})[/bold cyan]")
        console.print("-" * 50)

        try:
            if not self.broker:
                logger.warning("No broker configured")
                return

            account_info = self.broker.get_account_info()
            positions = account_info.positions

            if len(positions) < 2:
                logger.info(f"Insufficient positions ({len(positions)}), need ≥2")
                console.print("[dim]Insufficient positions[/dim]\n")
                self.state.last_correlation_audit = now
                return

            screening_results = (
                self.state.screening_history[-1].candidates if self.state.screening_history else None
            )

            workflow = self._init_workflow()
            auditor = CorrelationAuditor(
                market_fetcher=workflow.market_fetcher,
                correlation_threshold=self.config.correlation_audit.correlation_threshold,
                lookback_days=self.config.correlation_audit.lookback_days,
                output_dir=self.config.correlation_audit.output_dir,
            )

            start = time_mod.time()
            result = auditor.audit(positions, screening_results)
            duration = time_mod.time() - start

            self.state.record_correlation_audit(
                num_positions=result.num_positions,
                num_correlated_pairs=len(result.highly_correlated_pairs),
                max_correlation=result.max_correlation,
                avg_correlation=result.avg_correlation,
                diversification_ratio=result.diversification_ratio,
                num_substitutions=len(result.substitution_suggestions),
                total_duration_seconds=duration,
            )
            self.state.save(self.config.state.state_file)

            self._print_correlation_audit_results(result, duration)

        except Exception as e:
            error_msg = f"Correlation audit failed: {e}"
            logger.error(error_msg)
            self.state.record_error(error_msg)

    def _build_peer_context(self, symbol: str) -> str | None:
        """Build peer analysis context string from persisted data.

        Args:
            symbol: Stock ticker to build context for

        Returns:
            Formatted peer analysis context or None
        """
        try:
            from src.daemon.peer_analysis import DeepPeerAnalyzer

            analyzer = DeepPeerAnalyzer(output_dir=self.config.peer_analysis.output_dir)
            return analyzer.format_context(symbol)
        except Exception as e:
            logger.warning(f"Failed to build peer context for {symbol}: {e}")
            return None

    def _run_tearsheet_generation(self) -> None:
        """Generate performance tearsheet from analysis history."""
        if not self._tearsheet_generator:
            return

        # Check if already generated today
        now = datetime.now(self.scheduler.timezone)
        if self.state.last_tearsheet:
            last_date = self.state.last_tearsheet.astimezone(self.scheduler.timezone).date()
            if last_date == now.date():
                logger.debug("Tearsheet already generated today")
                return

        logger.info("Starting tearsheet generation")
        console.print(f"\n[bold cyan]Performance Tearsheet Generation ({now:%H:%M})[/bold cyan]")
        console.print("-" * 50)

        try:
            today = now.date()
            today_analyses = [
                r
                for r in self.state.analyses
                if r.timestamp.astimezone(self.scheduler.timezone).date() == today
            ]

            if not today_analyses:
                logger.info("No analyses today, skipping tearsheet")
                return

            console.print(f"[dim]Generating tearsheet from {len(today_analyses)} analyses...[/dim]")

            tearsheet = self._tearsheet_generator.generate_portfolio_tearsheet(
                analyses=today_analyses,
                benchmark_symbol=self.config.reporting.benchmark,
            )

            if tearsheet:
                self._tearsheet_generator.cleanup_old_tearsheets(
                    retention_days=self.config.reporting.retention_days
                )

                self.state.record_tearsheet(
                    symbol="PORTFOLIO",
                    html_path=tearsheet.html_report_path,
                )
                self.state.save(self.config.state.state_file)

                console.print(f"[bold cyan]Tearsheet saved:[/bold cyan] {tearsheet.html_report_path}")
                if tearsheet.sharpe_ratio is not None:
                    console.print(f"[bold cyan]Sharpe Ratio:[/bold cyan] {tearsheet.sharpe_ratio:.2f}")
                if tearsheet.cagr is not None:
                    console.print(f"[bold cyan]CAGR:[/bold cyan] {tearsheet.cagr:.2%}")
            else:
                logger.info("Insufficient data for tearsheet generation")

            console.print("\n[dim]Tearsheet generation complete[/dim]\n")

        except Exception as e:
            error_msg = f"Tearsheet generation failed: {e}"
            logger.error(error_msg)
            self.state.record_error(error_msg)

    def _log_screening_results(self, results: list) -> None:
        """Log screening results to console.

        Args:
            results: List of ScreeningResult objects (top 5)
        """
        for i, result in enumerate(results, 1):
            console.print(
                f"[bold]{i}. {result.symbol}[/bold] ({result.name}) - Score: {result.score:.2f}\n"
                f"   {result.reason}"
            )

    async def _maybe_run_health_check(self) -> None:
        """Run health check if conditions are met."""
        if not self.config.health.enabled:
            return

        if not self.scheduler.is_health_check_time(self.config.health.run_time):
            return

        today = datetime.now(self.scheduler.timezone).date()
        if self.state.last_health_check and self.state.last_health_check.date() == today:
            return

        logger.info("Starting API health checks")
        console.print(f"\n[bold cyan]Running Health Checks ({datetime.now(tz=UTC):%H:%M})[/bold cyan]")

        try:
            from src.daemon.health import HealthChecker

            checker = HealthChecker(self.config, self.state, notification_service=self.notification_service)
            report = await checker.run()

            self.state.last_health_check = datetime.now(tz=self.scheduler.timezone)
            self.state.save(self.config.state.state_file)

            console.print(
                f"[bold cyan]Health:[/bold cyan] {report.overall_status} "
                f"({len(report.service_checks)} services, {report.total_duration_ms:.0f}ms)"
            )
            logger.info(f"Health check complete: {report.overall_status}")

            # Publish HEALTH_CHECK event
            if self.event_bus:
                try:
                    from src.daemon.event_bus import DashboardEvent, EventType

                    failures = [
                        svc.service_name for svc in report.service_checks if svc.status == "UNHEALTHY"
                    ]
                    await self.event_bus.publish(
                        DashboardEvent(
                            event_type=EventType.HEALTH_CHECK,
                            data={
                                "status": report.overall_status.value,
                                "failures": failures,
                            },
                        )
                    )
                except Exception as ex:
                    logger.error(f"Failed to publish HEALTH_CHECK event: {ex}")

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self.state.record_error(f"Health check failed: {e}")
            self.state.save(self.config.state.state_file)

    def run_daily_risk_report(self) -> None:
        """Generate and persist daily portfolio risk report."""
        if not self.config.risk_limits.enabled or not self.broker:
            return

        # Dedup: only run once per day
        now = datetime.now(self.scheduler.timezone)
        if self.state.last_risk_report:
            last_date = self.state.last_risk_report.astimezone(self.scheduler.timezone).date()
            if last_date == now.date():
                logger.debug("Risk report already generated today")
                return

        logger.info("Generating daily portfolio risk report")
        console.print(f"\n[bold cyan]Portfolio Risk Report ({now:%H:%M})[/bold cyan]")
        console.print("-" * 50)

        try:
            account_info = self.broker.get_account_info()
            workflow = self._init_workflow()

            report = workflow.risk_manager.generate_risk_report(
                broker_positions=account_info.positions,
                portfolio_value=account_info.portfolio_value,
                total_exposure=account_info.total_exposure,
                lookback_days=self.config.risk_limits.lookback_days,
            )

            # Persist to JSON file
            report_dir = Path(self.config.risk_limits.report_dir).expanduser()
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / f"risk-report-{report.date}.json"
            with report_path.open("w") as f:
                json.dump(report.model_dump(), f, indent=2)

            # Record in state
            from datetime import UTC

            self.state.record_risk_report(
                RiskReportRecord(
                    timestamp=datetime.now(UTC),
                    var_95=report.var_95,
                    var_99=report.var_99,
                    cvar_95=report.cvar_95,
                    cvar_99=report.cvar_99,
                    cdar_95=report.cdar_95,
                    max_drawdown=report.max_drawdown,
                    risk_status=report.risk_status,
                )
            )
            self.state.save(self.config.state.state_file)

            status_color = {"HEALTHY": "green", "WARNING": "yellow", "BREACH": "red"}.get(
                report.risk_status, "white"
            )
            console.print(f"[{status_color}]Risk status: {report.risk_status}[/{status_color}]")
            console.print(f"[dim]VaR95={report.var_95:.4f}, CVaR99={report.cvar_99:.4f}[/dim]")
            console.print(f"[dim]Report saved: {report_path}[/dim]\n")
            logger.info(f"Risk report generated: {report.risk_status}")

            # Send notification if VaR limits breached
            if (report.var_limit_breached or report.cvar_limit_breached) and self.notification_service:
                task = asyncio.create_task(self._notify_var_breach(report))
                _ = task  # Suppress RUF006

        except Exception as e:
            error_msg = f"Risk report generation failed: {e}"
            logger.error(error_msg)
            self.state.record_error(error_msg)

    def _run_signal_tracking(self) -> None:
        """Update signal outcomes with T+1d/5d/20d prices."""
        if not self.config.signal_tracking.enabled:
            return

        # Dedup: check if already ran today
        now = datetime.now(self.scheduler.timezone)
        if self.state.last_signal_tracking:
            last_date = self.state.last_signal_tracking.astimezone(self.scheduler.timezone).date()
            if last_date == now.date():
                logger.debug("Signal tracking already completed today")
                return

        console.print(f"\n[bold cyan]Running Signal Tracking ({now:%H:%M})[/bold cyan]")

        try:
            from src.daemon.signal_tracker import SignalOutcomeTracker

            tracker = SignalOutcomeTracker(self._historical_cache, self.broker)
            stats = tracker.update_outcomes()

            self.state.last_signal_tracking = datetime.now(UTC)
            self.state.save(self.config.state.state_file)

            console.print(f"[dim]Signal tracking: {stats}[/dim]\n")
            logger.info(f"Signal tracking completed: {stats}")
        except Exception as e:
            error_msg = f"Signal tracking failed: {e}"
            logger.error(error_msg)
            self.state.record_error(error_msg)

    def _run_monte_carlo_stress_testing(self) -> None:
        """Execute Monte Carlo portfolio stress testing (weekly/daily task)."""
        logger.info("[MONTE CARLO] Starting stress test")

        # Deduplication (check last run within 6 hours)
        if self.state.monte_carlo_tests:
            last_run = self.state.monte_carlo_tests[-1].timestamp
            now = datetime.now(UTC)
            if (now - last_run).total_seconds() < 6 * 3600:
                logger.info("[MONTE CARLO] Already ran recently, skipping")
                return

        try:
            from src.daemon.stress_testing import DaemonStressTester

            executor = DaemonStressTester(
                broker_client=self.broker,
                market_fetcher=self.market_fetcher,
                config=self.config.monte_carlo,
            )
            record = executor.execute()

            self.state.record_monte_carlo_test(record, self.config.monte_carlo.max_history_records)
            self.state.save(self.config.state.state_file)

            if record.exceeds_risk_tolerance:
                logger.warning(f"[MONTE CARLO] ALERT: {record.alert_message}")
            else:
                logger.info(
                    f"[MONTE CARLO] Test passed - P(loss>threshold)={record.prob_loss_gt_threshold:.1%}, "
                    f"VaR95={record.var_95:.1%}"
                )
        except Exception as e:
            logger.error(f"[MONTE CARLO] Stress test failed: {e}")
            self.state.record_error(f"Monte Carlo stress test error: {e}")

    async def _run_cycle(self) -> int:
        """Run a single analysis cycle.

        Returns:
            Seconds to sleep before next cycle
        """
        from src.daemon.degradation import DegradationTier

        await self._task_runner.run_scheduled_tasks()
        await self._maybe_run_health_check()

        # Evaluate degradation before analysis
        degradation_context = self._evaluate_degradation()

        if degradation_context.tier == DegradationTier.HALTED:
            logger.warning(f"Analysis HALTED: {degradation_context.halt_reason}")
            console.print(f"[red]HALTED: {degradation_context.halt_reason}[/red]")

            # Notify on every halted cycle
            if self.notification_service:
                await self._notify_degradation(degradation_context)

            # Record in state
            self.state.record_degradation(degradation_context)
            self.state.save(self.config.state.state_file)

            return 60  # Retry in 1 minute

        # Log degradation status if not FULL
        if degradation_context.tier != DegradationTier.FULL:
            logger.warning(
                f"Degraded mode: {degradation_context.tier}, "
                f"unavailable: {degradation_context.unavailable_services}"
            )
            console.print(f"[yellow]DEGRADED: {degradation_context.tier}[/yellow]")

            # Notify on every degraded cycle
            if self.notification_service:
                await self._notify_degradation(degradation_context)

            self.state.record_degradation(degradation_context)

        if self.config.market_hours_only and not self.scheduler.is_market_open():
            await self._maybe_run_journal()
            wait_time = self.scheduler.time_until_open()
            if wait_time > 0:
                logger.info(f"Market closed, waiting {wait_time // 60} minutes until open")
                return min(wait_time, 60)

        watchlist = self.get_merged_watchlist()
        logger.info(f"Starting analysis cycle for {len(watchlist)} symbols")
        console.print(f"\n[bold]Running analysis cycle...[/bold] ({datetime.now(tz=UTC):%H:%M:%S})")

        await self._publish_event(
            "CYCLE_START",
            {"watchlist_size": len(watchlist), "degradation_tier": str(degradation_context.tier)},
        )

        cycle_start_time = time_mod.time()
        results = await self._analyze_watchlist(watchlist, degradation_context)
        cycle_duration = time_mod.time() - cycle_start_time

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

        # Check journal regardless of market_hours_only setting
        await self._maybe_run_journal()

        # Check paper trading readiness (once per day)
        await self._maybe_check_paper_readiness()

        self.state.save(self.config.state.state_file)
        return self.config.interval_minutes * 60

    async def run(self) -> None:
        """Run the daemon main loop."""
        self.running = True

        def shutdown_handler(sig: int, _frame: object) -> None:
            logger.info(f"Received signal {sig}, shutting down...")
            self.running = False
            if self._api_server:
                self._api_server.should_exit = True

        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)

        console.print("\n[bold green]Daemon started[/bold green]")
        console.print(f"Watchlist: {', '.join(self.config.watchlist)}")
        console.print(f"Interval: {self.config.interval_minutes} minutes")
        console.print(f"Market hours only: {self.config.market_hours_only}")
        console.print(f"Auto trade: {self.config.auto_trade}")
        console.print()

        # Start API server if enabled
        if self.config.api.enabled:
            self._start_api_server()

        while self.running:
            try:
                sleep_seconds = await self._run_cycle()
                logger.info(f"Sleeping for {sleep_seconds // 60} minutes")
                await asyncio.sleep(sleep_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in daemon loop: {e}")
                self.state.record_error(str(e))
                await asyncio.sleep(60)

        # Stop API server before saving state
        await self._stop_api_server()

        self.state.save(self.config.state.state_file)
        console.print("\n[bold yellow]Daemon stopped[/bold yellow]")
        logger.info("Daemon shutdown complete")

    def _start_api_server(self) -> None:
        """Start embedded API server as background task."""
        try:
            from src.daemon.api import create_api_app

            app = create_api_app(self)
            config = uvicorn.Config(
                app,
                host=self.config.api.host,
                port=self.config.api.port,
                log_level="info",
                access_log=False,
            )
            self._api_server = uvicorn.Server(config)
            self._api_task = asyncio.create_task(self._api_server.serve())

            logger.info(f"API server started at http://{self.config.api.host}:{self.config.api.port}")
            console.print(
                f"[bold cyan]API server: http://{self.config.api.host}:{self.config.api.port}[/bold cyan]"
            )
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            self._api_server = None
            self._api_task = None

    async def _stop_api_server(self) -> None:
        """Stop embedded API server gracefully."""
        if self._api_server and self._api_task:
            try:
                logger.info("Stopping API server...")
                self._api_server.should_exit = True
                await asyncio.wait_for(self._api_task, timeout=5.0)
                logger.info("API server stopped")
            except TimeoutError:
                logger.warning("API server shutdown timed out, cancelling task")
                self._api_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._api_task
            except Exception as e:
                logger.error(f"Error stopping API server: {e}")

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
