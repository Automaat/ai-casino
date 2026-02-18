"""Trading workflow orchestrator coordinating all stages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from src.daemon.degradation import DegradationContext
    from src.di.container import AppContainer

from src.backtesting import VectorBTRunner
from src.data.truth_social import TruthSocialFetcher
from src.metrics.execution import ExecutionMetricsCollector, current_collector
from src.strategies.ensemble import EnsembleStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.session import TradingSession
from src.workflows.config import WorkflowComponents, WorkflowConfig
from src.workflows.models.decision import DecisionContext, DecisionInput
from src.workflows.stages import data_fetch, decision, execution
from src.workflows.types import TradingWorkflowResult, WorkflowExtraContext


class TradingWorkflow:
    """Orchestrate multi-agent trading analysis."""

    def __init__(self, config: WorkflowConfig, components: WorkflowComponents) -> None:
        """Initialize trading workflow.

        Args:
            config: Workflow behavior configuration
            components: Injected dependencies
        """
        # Extract config
        self.use_ensemble = config.use_ensemble
        self.use_meta_agent = config.use_meta_agent
        self.trump_mode = config.trump_mode
        self.snapshot_on_trade = config.snapshot_on_trade or False
        self.execution_metrics_enabled = config.execution_metrics_enabled
        self.pre_trade_backtest_config = config.pre_trade_backtest_config

        # Extract components
        self.llm_client = components.llm_client
        self.market_fetcher = components.market_fetcher
        self.news_fetcher = components.news_fetcher
        self.finbert = components.finbert
        self.fundamental_fetcher = components.fundamental_fetcher
        self.broker = components.broker
        self.metrics_tracker = components.metrics_tracker
        self.snapshot_repository = components.snapshot_repository
        self.execution_metric_repository = components.execution_metric_repository
        self.notification_service = components.notification_service
        self._container = components.container
        self.analysis_orchestrator_config = components.analysis_orchestrator_config
        self.web_search_fetcher = components.web_search_fetcher
        self.event_bus = components.event_bus
        self._original_components = components  # Store for supervisor mode

        # Initialize components
        self._initialize_trump_components(components)
        self._initialize_meta_agent(components)
        self._initialize_default_strategy()
        self._initialize_agents(components)
        self._initialize_risk_validation(components)
        self._initialize_backtest_runner()
        self._initialize_supervisor()

        mode = "meta-agent" if self.use_meta_agent else ("ensemble" if self.use_ensemble else "momentum")
        trump_str = "+trump" if self.trump_mode else ""
        logger.info(f"Initialized TradingWorkflow (mode={mode}{trump_str})")

    def _initialize_trump_components(self, components: WorkflowComponents) -> None:
        """Initialize Trump mode components."""
        self.trump_fetcher: TruthSocialFetcher | None = None
        if self.trump_mode:
            self.trump_fetcher = TruthSocialFetcher(historical_cache=components.historical_cache)

    def _initialize_meta_agent(self, components: WorkflowComponents) -> None:
        """Initialize meta-agent for dynamic strategy selection."""
        from src.agents.meta import MetaAgent

        self.meta_agent: MetaAgent | None = None
        if self.use_meta_agent:
            self.meta_agent = self._container.meta_agent()
            if components.metrics_tracker is not None:
                self.meta_agent.metrics_tracker = components.metrics_tracker
            if components.param_store is not None:
                self.meta_agent.param_store = components.param_store

    def _initialize_default_strategy(self) -> None:
        """Initialize default strategy based on config."""
        self._default_strategy: MomentumStrategy | EnsembleStrategy = (
            EnsembleStrategy() if self.use_ensemble else MomentumStrategy()
        )

    def _initialize_agents(self, components: WorkflowComponents) -> None:
        """Initialize trading and risk agents."""
        from src.agents.risk import RiskManagementAgent

        self.trader = self._container.trader_agent()
        self.risk_manager = RiskManagementAgent(
            components.llm_client,
            portfolio_var_calculator=components.portfolio_var_calculator,
            portfolio_var_config=components.portfolio_var_config,
            position_sizing_config=components.position_sizing_config,
        )

    def _initialize_risk_validation(self, components: WorkflowComponents) -> None:
        """Initialize risk validation components."""
        from src.daemon.config import RiskValidationConfig
        from src.validators.risk import RiskValidator

        self.risk_validation_config = components.risk_validation_config or RiskValidationConfig()
        self.risk_validator = components.risk_validator or RiskValidator(self.risk_validation_config)

    def _initialize_backtest_runner(self) -> None:
        """Initialize VectorBT runner for pre-trade validation."""
        self._target_allocations: dict[str, float] | None = None
        self.vectorbt_runner: VectorBTRunner | None = None
        if self.pre_trade_backtest_config and self.pre_trade_backtest_config.enabled:
            self.vectorbt_runner = VectorBTRunner()
            logger.info("VectorBTRunner initialized for pre-trade validation")

    def _initialize_supervisor(self) -> None:
        """Initialize trading supervisor for conditional analysis routing."""
        from src.agents.supervisor import TradingSupervisor

        self.supervisor = TradingSupervisor(self.llm_client)
        logger.debug("TradingSupervisor initialized")

    async def analyze(
        self,
        symbol: str,
        period_days: int = 90,
        trading_session: TradingSession = TradingSession.REGULAR,
        extra_context: WorkflowExtraContext | None = None,
        **deprecated_kwargs: dict[str, object] | bool | DegradationContext | str | None,
    ) -> TradingWorkflowResult:
        """Run complete trading analysis.

        Args:
            symbol: Stock ticker symbol
            period_days: Days of historical data to fetch
            trading_session: Trading session type (REGULAR or PRE_MARKET)
            extra_context: Optional workflow extra context (preferred)
            **deprecated_kwargs: Deprecated params (position_context, enable_multi_timeframe,
                                degradation_context, context_kwargs). Use extra_context.

        Returns:
            TradingWorkflowResult with all analyses and final decision
        """
        logger.info(f"Starting trading workflow for {symbol} (session={trading_session.value})")

        # Set up execution metrics collector if enabled
        collector: ExecutionMetricsCollector | None = None
        collector_token = None
        if self.execution_metrics_enabled:
            collector = ExecutionMetricsCollector(symbol, self.llm_client.provider, self.llm_client.model)
            self.llm_client.set_metrics_collector(collector)
            collector_token = current_collector.set(collector)

        try:
            # Extract deprecated kwargs for backward compatibility
            position_context = deprecated_kwargs.get("position_context")
            enable_multi_timeframe = deprecated_kwargs.get("enable_multi_timeframe", False)
            degradation_context = deprecated_kwargs.get("degradation_context")

            # Backward compat: construct extra_context from individual params if needed
            if extra_context is None and (
                position_context is not None
                or enable_multi_timeframe
                or degradation_context is not None
                or any(
                    k not in {"position_context", "enable_multi_timeframe", "degradation_context"}
                    for k in deprecated_kwargs
                )
            ):
                from src.daemon.degradation import DegradationContext

                # Extract context values with type narrowing
                sector_ctx = deprecated_kwargs.get("sector_context")
                earnings_ctx = deprecated_kwargs.get("earnings_context")
                peer_ctx = deprecated_kwargs.get("peer_analysis_context")
                game_plan_ctx = deprecated_kwargs.get("game_plan_context")

                # Narrow position_context type
                pos_ctx = position_context if isinstance(position_context, dict) else None

                # Narrow degradation_context type
                deg_ctx = degradation_context if isinstance(degradation_context, DegradationContext) else None

                extra_context = WorkflowExtraContext(
                    sector_rotation_context=sector_ctx if isinstance(sector_ctx, str) else None,
                    earnings_context=earnings_ctx if isinstance(earnings_ctx, str) else None,
                    peer_analysis_context=peer_ctx if isinstance(peer_ctx, str) else None,
                    game_plan_context=game_plan_ctx if isinstance(game_plan_ctx, str) else None,
                    position_context=pos_ctx,
                    enable_multi_timeframe=bool(enable_multi_timeframe),
                    degradation_context=deg_ctx,
                )

            return await self._analyze_instrumented(
                symbol, period_days, trading_session, collector, extra_context
            )
        finally:
            if collector_token is not None:
                current_collector.reset(collector_token)
            self.llm_client.set_metrics_collector(None)

    async def _analyze_instrumented(
        self,
        symbol: str,
        period_days: int,
        trading_session: TradingSession,
        collector: ExecutionMetricsCollector | None,
        extra_context: WorkflowExtraContext | None = None,
    ) -> TradingWorkflowResult:
        """Run analysis pipeline with optional metrics instrumentation.

        Args:
            symbol: Stock ticker symbol
            period_days: Days of historical data
            trading_session: Trading session type (REGULAR or PRE_MARKET)
            collector: Optional metrics collector
            extra_context: Optional context with degradation_context, enable_multi_timeframe, etc
        """
        return await self._analyze_supervisor(symbol, period_days, trading_session, collector, extra_context)

    async def _analyze_supervisor(
        self,
        symbol: str,
        period_days: int,
        trading_session: TradingSession,
        collector: ExecutionMetricsCollector | None,
        extra_context: WorkflowExtraContext | None = None,
    ) -> TradingWorkflowResult:
        """Run supervisor-driven workflow using workers.

        Args:
            symbol: Stock ticker symbol
            period_days: Days of historical data
            trading_session: Trading session type
            collector: Optional metrics collector
            extra_context: Optional workflow context

        Returns:
            TradingWorkflowResult from supervisor coordination
        """
        if not self.supervisor:
            msg = "Supervisor not initialized - cannot use supervisor mode"
            raise ValueError(msg)

        # Build components bundle
        components = WorkflowComponents(
            llm_client=self.llm_client,
            market_fetcher=self.market_fetcher,
            news_fetcher=self.news_fetcher,
            finbert=self.finbert,
            fundamental_fetcher=self.fundamental_fetcher,
            container=self._container,
            broker=self.broker,
            metrics_tracker=self.metrics_tracker,
            snapshot_repository=self.snapshot_repository,
            execution_metric_repository=self.execution_metric_repository,
            param_store=self._original_components.param_store,
            historical_cache=self._original_components.historical_cache,
            portfolio_var_calculator=self.risk_manager.portfolio_var_calculator,
            portfolio_var_config=self.risk_manager.portfolio_var_config,
            notification_service=self.notification_service,
            position_sizing_config=self.risk_manager.position_sizing_config,
            risk_validation_config=self.risk_validation_config,
            risk_validator=self.risk_validator,
            analysis_orchestrator_config=self.analysis_orchestrator_config,
        )

        # Build config from self
        from src.workflows.config import WorkflowConfig

        config = WorkflowConfig(
            use_ensemble=self.use_ensemble,
            use_meta_agent=self.use_meta_agent,
            trump_mode=self.trump_mode,
            snapshot_on_trade=self.snapshot_on_trade,
            execution_metrics_enabled=self.execution_metrics_enabled,
            pre_trade_backtest_config=self.pre_trade_backtest_config,
        )

        # Delegate to supervisor
        return await self.supervisor.coordinate(
            symbol=symbol,
            period_days=period_days,
            components=components,
            config=config,
            trading_session=trading_session,
            collector=collector,
            target_allocations=self._target_allocations,
            extra_context=extra_context,
        )

    def set_target_allocations(self, allocations: dict[str, float] | None) -> None:
        """Set target portfolio allocations for position sizing.

        Args:
            allocations: Dict of {symbol: weight} for target portfolio
        """
        self._target_allocations = allocations
        if allocations:
            logger.info(f"Set target allocations for {len(allocations)} symbols")

    def get_target_allocation(self, symbol: str) -> float | None:
        """Get target allocation for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Target portfolio weight or None
        """
        return self._target_allocations.get(symbol) if self._target_allocations else None

    def get_default_strategy(self) -> MomentumStrategy | EnsembleStrategy:
        """Get default strategy instance.

        Returns:
            Default strategy (momentum or ensemble)
        """
        return self._default_strategy

    def get_container(self) -> AppContainer:
        """Get DI container instance.

        Returns:
            Dependency injection container
        """
        return self._container

    # Backward compatibility methods for tests
    async def _fetch_data(
        self, symbol: str, period_days: int, trading_session: TradingSession
    ) -> dict[str, Any]:
        """Fetch data stage (backward compatibility for tests).

        Args:
            symbol: Stock ticker
            period_days: Historical data period
            trading_session: Trading session type

        Returns:
            State dict with market and news data
        """
        from src.workflows.stages.data_fetch import DataFetchConfig

        data_fetch_config = DataFetchConfig(
            market_fetcher=self.market_fetcher,
            news_fetcher=self.news_fetcher,
            enable_multi_timeframe=False,
            trump_mode=self.trump_mode,
            trump_fetcher=self.trump_fetcher if self.trump_mode else None,
            web_search_fetcher=self.web_search_fetcher,
        )
        data_output = await data_fetch.fetch_data(
            symbol=symbol,
            period_days=period_days,
            trading_session=trading_session,
            config=data_fetch_config,
        )

        return {
            "symbol": data_output.symbol,
            "trading_session": data_output.trading_session,
            "market_data": data_output.market_data,
            "news_articles": data_output.news_articles,
            "trump_posts": data_output.trump_posts,
            "enable_multi_timeframe": data_output.enable_multi_timeframe,
            "warnings": data_output.warnings,
        }

    async def make_decision(self, state: dict[str, Any]) -> dict[str, Any]:
        """Make trading decision stage (backward compatibility for tests).

        Args:
            state: State dict from previous stages

        Returns:
            Updated state dict with trading decision
        """
        # Derive position_context from account_info if not explicitly provided
        position_context = state.get("position_context")
        if position_context is None and state.get("account_info"):
            account_info = state["account_info"]
            symbol = state["symbol"]
            position_qty = account_info.positions.get(symbol)
            if position_qty is not None:
                position_context = {"owns_position": True, "qty": position_qty}
            else:
                position_context = {"owns_position": False, "qty": 0.0}

        # Build context from state dict fields
        context = DecisionContext(
            sector_rotation=state.get("sector_rotation_context"),
            earnings=state.get("earnings_context"),
            peer_analysis=state.get("peer_analysis_context"),
            game_plan=state.get("game_plan_context"),
            position=position_context,
        )

        decision_input = DecisionInput(
            symbol=state["symbol"],
            technical=state.get("technical_analysis"),
            sentiment=state.get("sentiment_analysis"),
            news=state.get("news_analysis"),
            bullish=state.get("bullish_research"),
            bearish=state.get("bearish_research"),
            fundamental=state.get("fundamental_analysis"),
            comparative=state.get("comparative_analysis"),
            trump=state.get("trump_analysis"),
            account_info=state.get("account_info"),
            context=context,
            backtest_validation=state.get("backtest_validation"),
            degradation_context=state.get("degradation_context"),
        )
        decision_output = await decision.make_decision(decision_input, self.trader, None)

        return {**state, "final_decision": decision_output.final_decision}

    async def _execute_trade(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute trade stage (backward compatibility for tests).

        Args:
            state: State dict from previous stages

        Returns:
            Updated state dict with order status
        """
        from src.workflows.models.execution import TradeExecutionInput

        if not self.broker:
            return {**state, "order_status": None}

        execution_input = TradeExecutionInput(
            symbol=state["symbol"],
            final_decision=state["final_decision"],
            risk_assessment=state["risk_assessment"],
            trading_session=state.get("trading_session", TradingSession.REGULAR),
        )
        execution_output = await execution.execute_trade(
            execution_input,
            self.broker,
            self.market_fetcher,
            self.execution_metric_repository,
        )

        return {**state, "order_status": execution_output.order_status}

    def __repr__(self) -> str:
        """String representation."""
        mode = "meta-agent" if self.use_meta_agent else ("ensemble" if self.use_ensemble else "momentum")
        trump_str = "+trump" if self.trump_mode else ""
        return f"TradingWorkflow(mode={mode}{trump_str})"
