"""Trading workflow orchestrator coordinating all stages."""

import time
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from src.daemon.config import PositionSizingConfig
    from src.daemon.degradation import DegradationContext
    from src.daemon.notifications import NotificationService
    from src.data.finnhub import FinnhubFetcher
    from src.database.repositories.snapshot import PortfolioSnapshotRepository
    from src.di.container import AppContainer
    from src.metrics.portfolio_var import PortfolioVaRCalculator
    from src.optimization.param_store import OptimizedParamStore

from src.agents.fundamental import FundamentalAnalyst
from src.agents.risk import PortfolioVaRConfig
from src.agents.sentiment import SentimentAnalyst
from src.agents.trump import TrumpAnalyst
from src.backtesting import VectorBTRunner
from src.cache.historical import HistoricalCache
from src.daemon.config import PreTradeBacktestingConfig
from src.data.broker import AlpacaBroker
from src.data.fundamental import FundamentalDataFetcher
from src.data.market import MarketDataFetcher
from src.data.news import NewsFetcher
from src.data.truth_social import TruthSocialFetcher
from src.metrics.execution import (
    ExecutionMetricsCollector,
    current_collector,
    is_metrics_enabled,
    persist_jsonl,
)
from src.metrics.tracker import BaseMetricsTracker, DatabaseMetricsTracker
from src.models.llm import LLMClient
from src.models.sentiment import FinBERTSentiment
from src.strategies.ensemble import EnsembleStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.session import TradingSession
from src.strategies.signal import Signal
from src.workflows.models.account import AccountInfoOutput
from src.workflows.models.analysis import AnalysisInput, AnalysisOutput
from src.workflows.models.backtest import BacktestValidationOutput
from src.workflows.models.data_fetch import FetchDataOutput
from src.workflows.models.decision import DecisionContext, DecisionInput, DecisionOutput
from src.workflows.models.execution import TradeExecutionInput, TradeExecutionOutput
from src.workflows.models.risk import RiskAssessmentInput, RiskAssessmentOutput
from src.workflows.models.strategy import StrategySelectionInput, StrategySelectionOutput
from src.workflows.stages import analysis, data_fetch, decision, execution, risk, strategy_selection
from src.workflows.types import TradingWorkflowResult, WorkflowExtraContext


class TradingWorkflow:
    """Orchestrate multi-agent trading analysis."""

    def __init__(  # noqa: PLR0913, PLR0915, C901, PLR0912
        self,
        llm_client: LLMClient,
        market_fetcher: MarketDataFetcher,
        news_fetcher: NewsFetcher,
        finbert: FinBERTSentiment,
        fundamental_fetcher: FundamentalDataFetcher,
        broker: AlpacaBroker | None = None,
        metrics_tracker: BaseMetricsTracker | None = None,
        use_ensemble: bool = False,
        use_meta_agent: bool = True,
        trump_mode: bool = False,
        snapshot_on_trade: bool | None = None,
        snapshot_repository: PortfolioSnapshotRepository | None = None,
        param_store: OptimizedParamStore | None = None,
        historical_cache: HistoricalCache | None = None,
        portfolio_var_calculator: PortfolioVaRCalculator | None = None,
        portfolio_var_config: PortfolioVaRConfig | None = None,
        finnhub_fetcher: FinnhubFetcher | None = None,
        pre_trade_backtest_config: PreTradeBacktestingConfig | None = None,
        notification_service: NotificationService | None = None,
        position_sizing_config: PositionSizingConfig | None = None,
        container: AppContainer | None = None,
    ) -> None:
        """Initialize trading workflow.

        Args:
            llm_client: LLM client for agents
            market_fetcher: Market data fetcher
            news_fetcher: News data fetcher
            finbert: FinBERT sentiment model
            fundamental_fetcher: Fundamental data fetcher
            broker: Optional Alpaca broker for trade execution
            metrics_tracker: Optional metrics tracker for performance monitoring
            use_ensemble: Use ensemble strategy instead of momentum only (ignored if use_meta_agent=True)
            use_meta_agent: Use meta-agent for dynamic strategy selection (default True)
            trump_mode: Enable Trump social media analysis
            snapshot_on_trade: Capture portfolio snapshot after trades (env: PORTFOLIO_SNAPSHOT_ON_TRADE)
            snapshot_repository: Repository for portfolio snapshots (required if snapshot_on_trade)
            param_store: Optional optimized parameter store for strategy tuning
            historical_cache: Optional permanent cache for historical data
            portfolio_var_calculator: Optional VaR calculator for portfolio-level risk limits
            portfolio_var_config: Optional VaR limit configuration
            finnhub_fetcher: Optional Finnhub fetcher for fundamental data
            pre_trade_backtest_config: Optional pre-trade backtesting configuration
            notification_service: Optional notification service for risk rejection alerts
            position_sizing_config: Optional position sizing configuration
            container: Optional DI container for agent instantiation (preferred over manual)
        """
        import os

        if snapshot_on_trade is None:
            snapshot_on_trade = os.getenv("PORTFOLIO_SNAPSHOT_ON_TRADE", "false").lower() == "true"
        self.snapshot_on_trade = snapshot_on_trade
        self.snapshot_repository = snapshot_repository
        self.notification_service = notification_service
        self.llm_client = llm_client
        self.market_fetcher = market_fetcher
        self.news_fetcher = news_fetcher
        self.finbert = finbert
        self.fundamental_fetcher = fundamental_fetcher
        self.broker = broker
        self.metrics_tracker = metrics_tracker
        self.use_ensemble = use_ensemble
        self.use_meta_agent = use_meta_agent
        self.trump_mode = trump_mode
        self._container = container

        # Trump mode components
        self.trump_fetcher: TruthSocialFetcher | None = None
        self.trump_analyst: TrumpAnalyst | None = None
        if trump_mode:
            self.trump_fetcher = TruthSocialFetcher(historical_cache=historical_cache)
            if container:
                self.trump_analyst = container.trump_analyst()
            else:
                self.trump_analyst = TrumpAnalyst(llm_client)

        # Meta-agent for dynamic strategy selection
        from src.agents.meta import MetaAgent

        self.meta_agent: MetaAgent | None = None
        if use_meta_agent:
            if container:
                self.meta_agent = container.meta_agent()
                # Override metrics_tracker and param_store if provided
                if metrics_tracker is not None:
                    self.meta_agent.metrics_tracker = metrics_tracker
                if param_store is not None:
                    self.meta_agent.param_store = param_store
            else:
                from src.strategies.regime import MarketRegimeDetector

                regime_detector = MarketRegimeDetector()
                self.meta_agent = MetaAgent(
                    llm_client, regime_detector, metrics_tracker, param_store=param_store
                )

        # Default strategy (used if meta-agent disabled)
        self._default_strategy: MomentumStrategy | EnsembleStrategy = (
            EnsembleStrategy() if use_ensemble else MomentumStrategy()
        )

        # Agents (use container if available, fallback to manual instantiation)
        if container:
            self.sentiment_analyst = SentimentAnalyst(finbert)
            self.news_analyst = container.news_analyst()
            self.fundamental_analyst = FundamentalAnalyst(llm_client, fundamental_fetcher)
            self.comparative_analyst = container.comparative_analyst()
            self.web_researcher = container.web_research_agent()
            self.social_analyst = container.social_sentiment_analyst()
            self.bullish_researcher = container.bullish_researcher()
            self.bearish_researcher = container.bearish_researcher()
            self.trader = container.trader_agent()
            # Note: we instantiate RiskManagementAgent manually here using the provided config;
            # container.risk_management_agent() is not used, so container-based config is ignored.
            from src.agents.risk import RiskManagementAgent

            self.risk_manager = RiskManagementAgent(
                llm_client,
                portfolio_var_calculator=portfolio_var_calculator,
                portfolio_var_config=portfolio_var_config,
                position_sizing_config=position_sizing_config,
            )
        else:
            from src.agents.base_researcher import ResearchDirection
            from src.agents.comparative import ComparativeAnalyst
            from src.agents.news import NewsAnalyst
            from src.agents.risk import RiskManagementAgent
            from src.agents.social import SocialSentimentAnalyst
            from src.agents.thesis_researcher import ThesisResearcher
            from src.agents.trader import TraderAgent
            from src.agents.web_researcher import WebResearchAgent
            from src.data.comparative import ComparativeDataFetcher
            from src.data.finnhub import FinnhubFetcher
            from src.data.reddit import RedditFetcher

            self.sentiment_analyst = SentimentAnalyst(finbert)
            self.news_analyst = NewsAnalyst(llm_client)
            self.fundamental_analyst = FundamentalAnalyst(llm_client, fundamental_fetcher)
            self.comparative_analyst = ComparativeAnalyst(llm_client, ComparativeDataFetcher())
            self.web_researcher = WebResearchAgent(llm_client)
            # Get Finnhub fetcher from explicit parameter or fall back to env-var-based configuration
            finnhub = finnhub_fetcher
            if finnhub is None:
                # Last resort: create without DI container (will read from env var)
                logger.warning("Creating FinnhubFetcher without DI - falling back to env var")
                finnhub = FinnhubFetcher()

            self.social_analyst = SocialSentimentAnalyst(
                llm_client,
                finnhub,
                RedditFetcher(historical_cache=historical_cache),
                finbert,
            )
            self.bullish_researcher = ThesisResearcher(llm_client, ResearchDirection.BULLISH)
            self.bearish_researcher = ThesisResearcher(llm_client, ResearchDirection.BEARISH)
            self.trader = TraderAgent(llm_client)
            self.risk_manager = RiskManagementAgent(
                llm_client,
                portfolio_var_calculator=portfolio_var_calculator,
                portfolio_var_config=portfolio_var_config,
                position_sizing_config=position_sizing_config,
            )

        mode = "meta-agent" if use_meta_agent else ("ensemble" if use_ensemble else "momentum")
        trump_str = "+trump" if trump_mode else ""
        logger.info(f"Initialized TradingWorkflow (mode={mode}{trump_str})")

        self._target_allocations: dict[str, float] | None = None
        self.pre_trade_backtest_config = pre_trade_backtest_config
        self.vectorbt_runner: VectorBTRunner | None = None
        if pre_trade_backtest_config and pre_trade_backtest_config.enabled:
            self.vectorbt_runner = VectorBTRunner()
            logger.info("VectorBTRunner initialized for pre-trade validation")

    async def analyze(  # noqa: PLR0913
        self,
        symbol: str,
        period_days: int = 90,
        trading_session: TradingSession = TradingSession.REGULAR,
        position_context: dict[str, object] | None = None,
        enable_multi_timeframe: bool = False,
        degradation_context: DegradationContext | None = None,
        **context_kwargs: str | None,
    ) -> TradingWorkflowResult:
        """Run complete trading analysis.

        Args:
            symbol: Stock ticker symbol
            period_days: Days of historical data to fetch
            trading_session: Trading session type (REGULAR or PRE_MARKET)
            position_context: Optional position context (entry price, P&L, days held)
            enable_multi_timeframe: Enable multi-timeframe analysis (requires market hours)
            degradation_context: Optional degradation context
            **context_kwargs: Optional context keys: sector_context, earnings_context,
                peer_analysis_context, game_plan_context

        Returns:
            TradingWorkflowResult with all analyses and final decision
        """
        logger.info(f"Starting trading workflow for {symbol} (session={trading_session.value})")

        # Set up execution metrics collector if enabled
        collector: ExecutionMetricsCollector | None = None
        collector_token = None
        if is_metrics_enabled():
            collector = ExecutionMetricsCollector(symbol, self.llm_client.provider, self.llm_client.model)
            self.llm_client.set_metrics_collector(collector)
            collector_token = current_collector.set(collector)

        try:
            extra_context: WorkflowExtraContext = {
                "sector_rotation_context": context_kwargs.get("sector_context"),
                "earnings_context": context_kwargs.get("earnings_context"),
                "peer_analysis_context": context_kwargs.get("peer_analysis_context"),
                "game_plan_context": context_kwargs.get("game_plan_context"),
                "position_context": position_context,
                "enable_multi_timeframe": enable_multi_timeframe,
                "degradation_context": degradation_context,
            }
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
        from src.daemon.degradation import DegradationTier

        ctx = extra_context or {}
        degradation_context: DegradationContext | None = ctx.get("degradation_context")

        # Check if halted
        if degradation_context and degradation_context.tier == DegradationTier.HALTED:
            msg = f"Analysis halted: {degradation_context.halt_reason}"
            raise RuntimeError(msg)

        enable_multi_timeframe = bool(ctx.get("enable_multi_timeframe", False))

        # Stage 1: Fetch data
        start = time.perf_counter()
        data_output = await data_fetch.fetch_data(
            symbol,
            period_days,
            trading_session,
            self.market_fetcher,
            self.news_fetcher,
            enable_multi_timeframe=enable_multi_timeframe,
            trump_mode=self.trump_mode,
            trump_fetcher=self.trump_fetcher,
        )
        self._record_stage(collector, "fetch_data", start)

        # Stage 2: Fetch account info
        start = time.perf_counter()
        account_output = await data_fetch.fetch_account_info(self.broker)
        self._record_stage(collector, "fetch_account_info", start)

        # Stage 3: Select strategy
        start = time.perf_counter()
        strategy_input = StrategySelectionInput(symbol=symbol, market_data=data_output.market_data)
        strategy_output = await strategy_selection.select_strategy(
            strategy_input,
            self.meta_agent,
            self._default_strategy,
            self.use_ensemble,
            collector,
        )
        self._record_stage(collector, "strategy_selection", start)

        # Stage 4: Validate strategy with backtest
        start = time.perf_counter()
        backtest_output = await strategy_selection.validate_strategy_with_backtest(
            symbol,
            strategy_output.strategy_instance,
            strategy_output.strategy_name,
            strategy_input,
            self.pre_trade_backtest_config,
            self.vectorbt_runner,
            collector,
        )
        self._record_stage(collector, "backtest_validation", start)

        # Create TechnicalAnalyst with selected strategy
        if self._container:
            technical_analyst = self._container.technical_analyst()(strategy_output.strategy_instance)
        else:
            from src.agents.technical import TechnicalAnalyst

            technical_analyst = TechnicalAnalyst(self.llm_client, strategy_output.strategy_instance)

        # Stage 5: Run analyses
        start = time.perf_counter()
        analysis_input = AnalysisInput(
            symbol=symbol,
            market_data=data_output.market_data,
            news_articles=data_output.news_articles,
            trump_posts=data_output.trump_posts,
            enable_multi_timeframe=enable_multi_timeframe,
        )
        analysis_output = await analysis.run_analyses(
            analysis_input,
            technical_analyst,
            self.sentiment_analyst,
            self.news_analyst,
            self.fundamental_analyst,
            self.comparative_analyst,
            self.web_researcher,
            self.social_analyst,
            self.bullish_researcher,
            self.bearish_researcher,
            self.trump_mode,
            self.trump_analyst,
            collector,
        )
        self._record_stage(collector, "analyses", start)

        # Stage 6: Make decision
        start = time.perf_counter()
        decision_context = DecisionContext(
            sector_rotation=ctx.get("sector_rotation_context"),
            earnings=ctx.get("earnings_context"),
            peer_analysis=ctx.get("peer_analysis_context"),
            game_plan=ctx.get("game_plan_context"),
            position=ctx.get("position_context"),
        )
        decision_input = DecisionInput(
            symbol=symbol,
            technical=analysis_output.technical_analysis,
            sentiment=analysis_output.sentiment_analysis,
            news=analysis_output.news_analysis,
            bullish=analysis_output.bullish_research,
            bearish=analysis_output.bearish_research,
            fundamental=analysis_output.fundamental_analysis,
            comparative=analysis_output.comparative_analysis,
            trump=analysis_output.trump_analysis,
            account_info=account_output.account_info,
            context=decision_context,
            backtest_validation=backtest_output.backtest_validation,
            degradation_context=degradation_context,
        )
        decision_output = await decision.make_decision(decision_input, self.trader, collector)
        self._record_stage(collector, "decision", start)

        # Stage 7: Assess risk
        start = time.perf_counter()
        # Get target weight from allocations if available
        target_weight = self._target_allocations.get(symbol) if self._target_allocations else None
        risk_input = RiskAssessmentInput(
            symbol=symbol,
            market_data=data_output.market_data,
            final_decision=decision_output.final_decision,
            account_info=account_output.account_info,
            broker_positions=account_output.broker_positions,
            portfolio_value=account_output.portfolio_value,
            target_portfolio_weight=target_weight,
            backtest_validation=backtest_output.backtest_validation,
            degradation_context=degradation_context,
            broker_api_failed=account_output.broker_api_failed,
        )
        risk_output = await risk.assess_risk(risk_input, self.risk_manager)
        self._record_stage(collector, "risk_assessment", start)

        # Notify if trade rejected by risk gate (only during regular hours when trades can execute)
        if (
            risk_output.risk_assessment
            and decision_output.final_decision
            and not risk_output.risk_assessment.validation.approved
            and decision_output.final_decision.action != Signal.HOLD
            and self.notification_service
            and trading_session == TradingSession.REGULAR
        ):
            await execution.notify_trade_execution(
                symbol,
                decision_output.final_decision,
                risk_output.risk_assessment,
                self.notification_service,
            )

        # Stage 8: Execute trade
        execution_output = None
        if (
            self.broker
            and risk_output.risk_assessment
            and decision_output.final_decision
            and risk_output.risk_assessment.validation.approved
            and decision_output.final_decision.action != Signal.HOLD
        ):
            execution_input = TradeExecutionInput(
                symbol=symbol,
                final_decision=decision_output.final_decision,
                risk_assessment=risk_output.risk_assessment,
                trading_session=trading_session,
            )
            execution_output = await execution.execute_trade(execution_input, self.broker)

        # Log final result
        logger.info(
            f"Workflow complete: {decision_output.final_decision.action.value} "
            f"(confidence={decision_output.final_decision.confidence:.2f}, "
            f"risk_approved={risk_output.risk_assessment.validation.approved})"
        )

        # Build result
        return await self._build_and_persist_result(
            symbol,
            data_output,
            account_output,
            strategy_output,
            backtest_output,
            analysis_output,
            decision_output,
            risk_output,
            execution_output,
            decision_context,
            degradation_context,
            target_weight,
            trading_session,
            collector,
        )

    async def _build_and_persist_result(  # noqa: PLR0913
        self,
        symbol: str,
        data_output: FetchDataOutput,
        account_output: AccountInfoOutput,
        strategy_output: StrategySelectionOutput,
        backtest_output: BacktestValidationOutput,
        analysis_output: AnalysisOutput,
        decision_output: DecisionOutput,
        risk_output: RiskAssessmentOutput,
        execution_output: TradeExecutionOutput | None,
        decision_context: DecisionContext,
        degradation_context: DegradationContext | None,
        target_weight: float | None,  # noqa: ARG002
        trading_session: TradingSession,
        collector: ExecutionMetricsCollector | None,
    ) -> TradingWorkflowResult:
        """Build workflow result and persist metrics/snapshots.

        Args:
            symbol: Stock ticker
            data_output: Data fetch output
            account_output: Account info output
            strategy_output: Strategy selection output
            backtest_output: Backtest validation output
            analysis_output: Analysis output
            decision_output: Decision output
            risk_output: Risk assessment output
            execution_output: Execution output
            decision_context: Decision context
            degradation_context: Degradation context
            target_weight: Target portfolio weight
            trading_session: Trading session type
            collector: Optional metrics collector
        """
        execution_metrics = collector.finalize() if collector else None

        # Extract degradation fields
        degradation_tier = degradation_context.tier.value if degradation_context else None
        degradation_confidence_penalty = (
            (1 - degradation_context.confidence_adjustment) if degradation_context else None
        )

        # Aggregate warnings
        all_warnings = []
        all_warnings.extend(data_output.warnings)
        all_warnings.extend(account_output.warnings)
        all_warnings.extend(backtest_output.warnings)
        all_warnings.extend(analysis_output.warnings)
        if execution_output:
            all_warnings.extend(execution_output.warnings)

        result = TradingWorkflowResult(
            symbol=symbol,
            trading_session=trading_session,
            technical=analysis_output.technical_analysis,
            sentiment=analysis_output.sentiment_analysis,
            news=analysis_output.news_analysis,
            trump=analysis_output.trump_analysis,
            fundamental=analysis_output.fundamental_analysis,
            comparative=analysis_output.comparative_analysis,
            web_research=analysis_output.web_research,
            social_sentiment=analysis_output.social_sentiment_analysis,
            bullish=analysis_output.bullish_research,
            bearish=analysis_output.bearish_research,
            decision=decision_output.final_decision,
            risk=risk_output.risk_assessment,
            order=execution_output.order_status if execution_output else None,
            regime=strategy_output.regime_analysis,
            strategy_used=strategy_output.strategy_name,
            warnings=all_warnings,
            earnings_context=decision_context.earnings,
            peer_analysis_context=decision_context.peer_analysis,
            execution_metrics=execution_metrics,
            backtest_validation=backtest_output.backtest_validation,
            degradation_tier=degradation_tier,
            degradation_confidence_penalty=degradation_confidence_penalty,
        )

        if execution_metrics:
            try:
                persist_jsonl(execution_metrics)
            except Exception as e:
                logger.error(f"Failed to persist execution metrics (continuing): {e}")

        if self.metrics_tracker:
            try:
                is_paper = self.broker.paper if self.broker else True
                if isinstance(self.metrics_tracker, DatabaseMetricsTracker):
                    await self.metrics_tracker.record_decision_async(
                        result, strategy_name=strategy_output.strategy_name, is_paper_trade=is_paper
                    )
                else:
                    self.metrics_tracker.record_decision(
                        result, strategy_name=strategy_output.strategy_name, is_paper_trade=is_paper
                    )
            except Exception as e:
                logger.error(f"Failed to record metrics (continuing): {e}")

        if (
            self.snapshot_on_trade
            and self.snapshot_repository
            and risk_output.risk_assessment
            and decision_output.final_decision
            and risk_output.risk_assessment.validation.approved
            and decision_output.final_decision.action != Signal.HOLD
        ):
            await execution.create_portfolio_snapshot(
                symbol,
                account_output.account_info,
                self.snapshot_repository,
            )

        return result

    def set_target_allocations(self, allocations: dict[str, float] | None) -> None:
        """Set target portfolio allocations for position sizing.

        Args:
            allocations: Dict of {symbol: weight} for target portfolio
        """
        self._target_allocations = allocations
        if allocations:
            logger.info(f"Set target allocations for {len(allocations)} symbols")

    def _record_stage(
        self,
        collector: ExecutionMetricsCollector | None,
        stage: str,
        start: float,
    ) -> None:
        """Record pipeline stage timing if collector is active.

        Args:
            collector: Optional metrics collector
            stage: Stage name
            start: perf_counter start time
        """
        if collector:
            collector.record_pipeline_stage(stage, (time.perf_counter() - start) * 1000)

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
        from src.workflows.stages import data_fetch

        data_output = await data_fetch.fetch_data(
            symbol=symbol,
            period_days=period_days,
            trading_session=trading_session,
            market_fetcher=self.market_fetcher,
            news_fetcher=self.news_fetcher,
            enable_multi_timeframe=False,
            trump_mode=self.trump_mode,
            trump_fetcher=self.trump_fetcher if self.trump_mode else None,
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
        from src.workflows.models.decision import DecisionContext, DecisionInput
        from src.workflows.stages import decision

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
        from src.workflows.stages import execution

        if not self.broker:
            return {**state, "order_status": None}

        execution_input = TradeExecutionInput(
            symbol=state["symbol"],
            final_decision=state["final_decision"],
            risk_assessment=state["risk_assessment"],
            trading_session=state.get("trading_session", TradingSession.REGULAR),
        )
        execution_output = await execution.execute_trade(execution_input, self.broker)

        return {**state, "order_status": execution_output.order_status}

    def __repr__(self) -> str:
        """String representation."""
        mode = "meta-agent" if self.use_meta_agent else ("ensemble" if self.use_ensemble else "momentum")
        trump_str = "+trump" if self.trump_mode else ""
        return f"TradingWorkflow(agents=9, mode={mode}{trump_str})"
