"""Trading Supervisor Agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from src.agents.supervisor.models import (
    AnalysisRoutingDecision,
    AnalysisType,
    AnalysisWeights,
    PlanningContext,
    SynthesisContext,
)
from src.execution_tracking import track_agent
from src.models.llm import LLMClient
from src.models.providers.base import StructuredOutputError
from src.prompts import PromptLoader

if TYPE_CHECKING:
    from src.di.container import AppContainer
    from src.metrics.execution import ExecutionMetricsCollector
    from src.strategies.ensemble import EnsembleStrategy
    from src.strategies.momentum import MomentumStrategy
    from src.strategies.session import TradingSession
    from src.workflows.config import WorkflowComponents, WorkflowConfig
    from src.workflows.types import TradingWorkflowResult, WorkflowExtraContext


class TradingSupervisor:
    """Intelligent analysis orchestrator with adaptive routing and result synthesis."""

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize trading supervisor.

        Args:
            llm_client: LLM client for planning and synthesis
        """
        self.llm = llm_client
        self._prompts = PromptLoader("supervisor")
        logger.info("Initialized TradingSupervisor")

    @track_agent
    async def plan_analyses(
        self, context: PlanningContext, *, symbol: str | None = None
    ) -> AnalysisRoutingDecision:
        """Phase 1: Determine which analyses to run.

        Args:
            context: Planning context with market state and constraints
            symbol: Trading symbol for execution tracking; defaults to context.symbol

        Returns:
            AnalysisRoutingDecision with required/optional/skip lists
        """
        if symbol is None:
            symbol = context.symbol
        elif symbol != context.symbol:
            logger.warning(
                "plan_analyses called with mismatched symbol (%s) and context.symbol (%s)",
                symbol,
                context.symbol,
            )

        from src.strategies.regime import MIN_ROWS_FOR_REGIME

        prompt = self._prompts.load(
            "plan",
            symbol=symbol,
            regime=context.regime.regime.value if context.regime else "unknown",
            session=context.trading_session.value,
            owns_position=context.owns_position,
            news_count=context.news_count,
            market_data_rows=context.market_data_rows,
            min_rows_required=MIN_ROWS_FOR_REGIME,
            fundamental_status="available" if context.fundamental_available else "unavailable",
            social_status="available" if context.social_available else "unavailable",
            trump_count=context.trump_count,
            is_high_volatility=context.is_high_volatility,
            fundamental_api_status="rate limited" if context.fundamental_rate_limit else "available",
            time_budget_ms=context.time_budget_ms,
        )
        system = self._prompts.load("system")

        try:
            decision = await self.llm.astructured(
                prompt, AnalysisRoutingDecision, system=system, temperature=0.4
            )
        except StructuredOutputError as e:
            logger.opt(exception=True).warning(f"Structured output failed, using default: {e}")
            decision = self.default_routing(context)

        # Log routing summary
        required_types = (
            ", ".join([a.value for a in decision.required_analyses]) if decision.required_analyses else "none"
        )
        optional_types = (
            ", ".join([a.value for a in decision.optional_analyses]) if decision.optional_analyses else "none"
        )
        logger.info(
            f"Routing: {len(decision.required_analyses)} required ({required_types}), "
            f"{len(decision.optional_analyses)} optional ({optional_types}), "
            f"{len(decision.skip_analyses)} skipped"
        )

        # Log skip reasons for each skipped analysis
        for analysis_type, reason in decision.skip_analyses.items():
            logger.info(f"Skip {analysis_type.value}: {reason}")

        # Debug log: full routing reasoning
        logger.debug(f"Routing reasoning: {decision.reasoning}")

        return decision

    @track_agent
    async def synthesize_results(
        self, context: SynthesisContext, completed: list[AnalysisType], *, symbol: str | None = None
    ) -> AnalysisWeights:
        """Phase 2: Synthesize completed analyses.

        Args:
            context: Synthesis context with completed analysis summaries
            completed: List of completed analysis types
            symbol: Trading symbol for execution tracking; defaults to context.symbol

        Returns:
            AnalysisWeights with reliability scores and confidence adjustment
        """
        if symbol is None:
            symbol = context.symbol
        elif symbol != context.symbol:
            logger.warning(
                "synthesize_results called with mismatched symbol (%s) and context.symbol (%s)",
                symbol,
                context.symbol,
            )

        # Short-circuit when no analyses completed (avoid wasting LLM tokens)
        if not completed:
            logger.info("No analyses completed, returning default weights")
            return self._default_weights(completed)

        analyses_summary = self._format_analyses_summary(context, completed)

        prompt = self._prompts.load("synthesize", symbol=context.symbol, analyses_summary=analyses_summary)
        system = self._prompts.load("system")

        try:
            weights = await self.llm.astructured(prompt, AnalysisWeights, system=system, temperature=0.4)
        except StructuredOutputError as e:
            logger.opt(exception=True).warning(f"Structured output failed, uniform weights: {e}")
            weights = self._default_weights(completed)

        # Log synthesis summary
        weighted_analyses = ", ".join([f"{t.value}={w:.2f}" for t, w in weights.weights.items()])
        conflict_pairs = ", ".join(weights.conflicts) if weights.conflicts else "none"
        consensus_items = ", ".join(weights.consensus) if weights.consensus else "none"
        logger.info(
            f"Synthesis: {len(completed)} analyses weighted ({weighted_analyses}), "
            f"{len(weights.conflicts)} conflicts ({conflict_pairs}), "
            f"{len(weights.consensus)} consensus ({consensus_items}), "
            f"confidence_adj={weights.confidence_adjustment:.2f}"
        )

        # Debug log: full synthesis reasoning
        logger.debug(f"Synthesis reasoning: {weights.reasoning}")

        return weights

    def default_routing(self, context: PlanningContext) -> AnalysisRoutingDecision:
        """Fallback routing when LLM unavailable - intelligent data-driven decisions.

        Args:
            context: Planning context

        Returns:
            Default routing decision
        """
        from src.strategies.regime import MIN_ROWS_FOR_REGIME

        required: list[AnalysisType] = []
        optional: list[AnalysisType] = []
        skip: dict[AnalysisType, str] = {}

        # Technical: skip if insufficient data
        if context.market_data_rows < MIN_ROWS_FOR_REGIME:
            skip[AnalysisType.TECHNICAL] = (
                f"Insufficient data ({context.market_data_rows} < {MIN_ROWS_FOR_REGIME} required)"
            )
        else:
            required.append(AnalysisType.TECHNICAL)

        # Sentiment + News: skip if no articles
        if context.news_count == 0:
            skip[AnalysisType.SENTIMENT] = "No news articles available"
            skip[AnalysisType.NEWS] = "No news articles available"
        else:
            required.extend([AnalysisType.SENTIMENT, AnalysisType.NEWS])

        # Fundamental: optional unless rate-limited
        if context.fundamental_rate_limit:
            skip[AnalysisType.FUNDAMENTAL] = "API rate limited"
        elif not context.fundamental_available:
            skip[AnalysisType.FUNDAMENTAL] = "Fundamental data unavailable"
        else:
            optional.append(AnalysisType.FUNDAMENTAL)

        # Social sentiment: optional if available
        if context.social_available:
            optional.append(AnalysisType.SOCIAL_SENTIMENT)

        # Trump: optional if posts exist
        if context.trump_count > 0:
            optional.append(AnalysisType.TRUMP)

        # Research: required only if technical not skipped
        if AnalysisType.TECHNICAL not in skip:
            required.extend([AnalysisType.BULLISH_RESEARCH, AnalysisType.BEARISH_RESEARCH])
        else:
            skip[AnalysisType.BULLISH_RESEARCH] = "Technical skipped (dependency)"
            skip[AnalysisType.BEARISH_RESEARCH] = "Technical skipped (dependency)"

        # Build priority order based on context
        priority_order = self._build_priority_order(context, required, optional)

        return AnalysisRoutingDecision(
            required_analyses=required,
            optional_analyses=optional,
            skip_analyses=skip,
            reasoning="Intelligent fallback routing based on data availability",
            priority_order=priority_order,
        )

    def _build_priority_order(
        self,
        context: PlanningContext,
        required: list[AnalysisType],
        optional: list[AnalysisType],
    ) -> list[AnalysisType]:
        """Build execution priority order based on session and conditions.

        Args:
            context: Planning context
            required: Required analyses
            optional: Optional analyses

        Returns:
            Priority-ordered list of analyses
        """
        from src.strategies.session import TradingSession

        if context.trading_session == TradingSession.PRE_MARKET:
            # Pre-market: prioritize news/sentiment for breaking developments
            priority = []
            for analysis_type in [AnalysisType.NEWS, AnalysisType.SENTIMENT]:
                if analysis_type in required:
                    priority.append(analysis_type)
            # Add remaining required analyses
            for analysis_type in required:
                if analysis_type not in priority:
                    priority.append(analysis_type)
            return priority + optional

        # Regular session: standard order (technical → sentiment → news → research)
        return required + optional

    def _default_weights(self, completed: list[AnalysisType]) -> AnalysisWeights:
        """Fallback uniform weights.

        Args:
            completed: List of completed analyses

        Returns:
            Uniform weights for all completed analyses
        """
        weights = dict.fromkeys(completed, 0.8)
        return AnalysisWeights(
            weights=weights,
            conflicts=[],
            consensus=[],
            confidence_adjustment=1.0,
            reasoning="Uniform weights (LLM fallback)",
        )

    def _format_analyses_summary(self, context: SynthesisContext, completed: list[AnalysisType]) -> str:
        """Format completed analyses for synthesis prompt.

        Args:
            context: Synthesis context with analysis summaries
            completed: List of completed analysis types

        Returns:
            Formatted summary string
        """
        summary_map = {
            AnalysisType.TECHNICAL: context.technical_summary,
            AnalysisType.SENTIMENT: context.sentiment_summary,
            AnalysisType.NEWS: context.news_summary,
            AnalysisType.FUNDAMENTAL: context.fundamental_summary,
            AnalysisType.COMPARATIVE: context.comparative_summary,
            AnalysisType.WEB_RESEARCH: context.web_research_summary,
            AnalysisType.SOCIAL_SENTIMENT: context.social_summary,
            AnalysisType.BULLISH_RESEARCH: context.bullish_summary,
            AnalysisType.BEARISH_RESEARCH: context.bearish_summary,
            AnalysisType.TRUMP: context.trump_summary,
        }

        lines = []
        for analysis_type in completed:
            summary = summary_map.get(analysis_type)
            if summary:
                lines.append(f"{analysis_type.value.upper()}: {summary}")

        return "\n".join(lines)

    async def coordinate(
        self,
        symbol: str,
        period_days: int,
        components: WorkflowComponents,
        config: WorkflowConfig,
        trading_session: TradingSession | None = None,
        collector: ExecutionMetricsCollector | None = None,
        target_allocations: dict[str, float] | None = None,
        extra_context: WorkflowExtraContext | None = None,
    ) -> TradingWorkflowResult:
        """Coordinate full trading workflow with adaptive stage execution.

        This method orchestrates all 8 workflow stages using workers instead of agents.

        Args:
            symbol: Stock ticker symbol
            period_days: Days of historical data
            components: Workflow components (fetchers, broker, etc.)
            config: Workflow configuration
            trading_session: Trading session type (defaults to REGULAR)
            collector: Optional metrics collector
            target_allocations: Optional target portfolio allocations
            extra_context: Optional workflow context

        Returns:
            TradingWorkflowResult with all analyses and final decision

        Note:
            This method delegates to existing stage functions from workflows/stages/
            for most stages, using workers for the analysis stage (Stage 5).
        """
        from src.strategies.session import TradingSession
        from src.workflows.stages.instrumented_analysis import (
            AnalysisRequest,
            AnalysisRequestParams,
            run_instrumented_analysis,
        )

        logger.info(
            f"Supervisor coordinating workflow for {symbol} (supervisor mode - using existing pipeline)"
        )

        # Create minimal workflow and delegate to existing pipeline
        workflow = _MinimalWorkflow(components, config, self, target_allocations)
        session = trading_session or TradingSession.REGULAR
        params = AnalysisRequestParams(period_days, session, extra_context)
        # _MinimalWorkflow duck-types as TradingWorkflow (structural compatibility)
        request = AnalysisRequest(workflow, symbol, params, collector)  # pyrefly: ignore[bad-argument-type]

        return await run_instrumented_analysis(request)

    def __repr__(self) -> str:
        """String representation."""
        return f"TradingSupervisor(llm={self.llm.provider})"


class _MinimalWorkflow:
    """Minimal workflow object for instrumented analysis delegation."""

    def __init__(
        self,
        components: WorkflowComponents,
        config: WorkflowConfig,
        supervisor: TradingSupervisor,
        target_allocations: dict[str, float] | None = None,
    ) -> None:
        self._init_components(components)
        self._init_config(components, config)
        self._init_conditional_components(components)
        self._init_agents(components)
        self.supervisor = supervisor
        self._target_allocations = target_allocations

    def _init_components(self, components: WorkflowComponents) -> None:
        """Initialize core component references."""
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

    def _init_config(self, components: WorkflowComponents, config: WorkflowConfig) -> None:
        """Initialize configuration attributes."""
        self.use_ensemble = config.use_ensemble
        self.use_meta_agent = config.use_meta_agent
        self.trump_mode = config.trump_mode
        self.snapshot_on_trade = config.snapshot_on_trade or False
        self.execution_metrics_enabled = config.execution_metrics_enabled
        self.pre_trade_backtest_config = config.pre_trade_backtest_config
        self.risk_validation_config = components.risk_validation_config
        self.risk_validator = components.risk_validator

    def _init_conditional_components(self, components: WorkflowComponents) -> None:
        """Initialize conditional components (Trump, meta-agent, backtest)."""
        # Trump components
        if self.trump_mode:
            from src.data.truth_social import TruthSocialFetcher

            self.trump_fetcher: TruthSocialFetcher | None = TruthSocialFetcher(
                historical_cache=components.historical_cache
            )
            self.trump_analyst = self._container.trump_analyst()
        else:
            self.trump_fetcher: TruthSocialFetcher | None = None
            self.trump_analyst = None  # pyrefly: ignore[bad-assignment]

        # Meta-agent
        if self.use_meta_agent:
            self.meta_agent = self._container.meta_agent()
            if components.metrics_tracker:
                self.meta_agent.metrics_tracker = components.metrics_tracker
            if components.param_store:
                self.meta_agent.param_store = components.param_store
        else:
            self.meta_agent = None  # pyrefly: ignore[bad-assignment]

        # Default strategy
        from src.strategies.ensemble import EnsembleStrategy
        from src.strategies.momentum import MomentumStrategy

        self._default_strategy = EnsembleStrategy() if self.use_ensemble else MomentumStrategy()

        # Backtest runner
        if self.pre_trade_backtest_config and self.pre_trade_backtest_config.enabled:
            from src.backtesting import VectorBTRunner

            self.vectorbt_runner: VectorBTRunner | None = VectorBTRunner()
        else:
            self.vectorbt_runner: VectorBTRunner | None = None

    def _init_agents(self, components: WorkflowComponents) -> None:
        """Initialize analysis and risk agents."""
        from src.agents.fundamental import FundamentalAnalyst
        from src.agents.risk import RiskManagementAgent
        from src.agents.sentiment import SentimentAnalyst

        self.sentiment_analyst = SentimentAnalyst(components.finbert)
        self.news_analyst = self._container.news_analyst()
        self.fundamental_analyst = FundamentalAnalyst(components.llm_client, components.fundamental_fetcher)
        self.comparative_analyst = self._container.comparative_analyst()
        self.web_researcher = self._container.web_research_agent()
        self.social_analyst = self._container.social_sentiment_analyst()
        self.bullish_researcher = self._container.bullish_researcher()
        self.bearish_researcher = self._container.bearish_researcher()
        self.trader = self._container.trader_agent()
        self.risk_manager = RiskManagementAgent(
            components.llm_client,
            portfolio_var_calculator=components.portfolio_var_calculator,
            portfolio_var_config=components.portfolio_var_config,
            position_sizing_config=components.position_sizing_config,
        )

    def get_default_strategy(self) -> EnsembleStrategy | MomentumStrategy:
        return self._default_strategy

    def get_container(self) -> AppContainer:
        return self._container

    def get_target_allocation(self, symbol: str) -> float | None:
        return self._target_allocations.get(symbol) if self._target_allocations else None
