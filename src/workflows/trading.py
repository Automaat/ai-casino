"""Trading workflow orchestrating all agents."""

import asyncio
import time
import zoneinfo
from collections.abc import Coroutine
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypeVar, cast

import pandas as pd
from loguru import logger
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from src.daemon.degradation import DegradationContext
    from src.daemon.notifications import NotificationService
    from src.database.repositories.snapshot import PortfolioSnapshotRepository
    from src.metrics.portfolio_var import PortfolioVaRCalculator
    from src.optimization.param_store import OptimizedParamStore

from src.agents.bearish_researcher import BearishResearchAnalysis, BearishResearcher
from src.agents.bullish_researcher import BullishResearchAnalysis, BullishResearcher
from src.agents.comparative import ComparativeAnalysis, ComparativeAnalyst
from src.agents.fundamental import FundamentalAnalysis, FundamentalAnalyst
from src.agents.meta import MetaAgent, StrategySelection, StrategyType
from src.agents.news import NewsAnalysis, NewsAnalyst
from src.agents.risk import AccountInfo, PortfolioVaRConfig, RiskAssessment, RiskManagementAgent
from src.agents.sentiment import SentimentAnalysis, SentimentAnalyst
from src.agents.social import SocialSentimentAnalysis, SocialSentimentAnalyst
from src.agents.technical import TechnicalAnalysis, TechnicalAnalyst
from src.agents.trader import TraderAgent, TradingDecision
from src.agents.trump import TrumpAnalysis, TrumpAnalyst
from src.agents.web_researcher import WebResearchAgent, WebResearchAnalysis
from src.backtesting import VectorBTRunner
from src.cache.historical import HistoricalCache
from src.daemon.config import PreTradeBacktestingConfig
from src.data.broker import AlpacaBroker, BrokerAccountInfo, BrokerAPIError, BrokerPosition, OrderStatus
from src.data.comparative import ComparativeDataFetcher
from src.data.finnhub import FinnhubFetcher
from src.data.fundamental import FundamentalDataFetcher
from src.data.market import MarketDataFetcher
from src.data.news import NewsArticle, NewsFetcher
from src.data.reddit import RedditFetcher
from src.data.truth_social import TruthPost, TruthSocialFetcher
from src.metrics.execution import (
    ExecutionMetricsCollector,
    current_agent,
    current_collector,
    is_metrics_enabled,
    persist_jsonl,
)
from src.metrics.tracker import BaseMetricsTracker, DatabaseMetricsTracker
from src.models.llm import LLMClient
from src.models.sentiment import FinBERTSentiment
from src.strategies.ensemble import EnsembleStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.regime import MarketRegimeDetector, RegimeAnalysis
from src.strategies.session import TradingSession
from src.strategies.signal import Signal
from src.strategies.timeframe import MultiTimeframeData, Timeframe
from src.workflows.types import BacktestValidation, TradingWorkflowResult

T = TypeVar("T")

ET_TIMEZONE = zoneinfo.ZoneInfo("America/New_York")
MARKET_HOURS_START = 4
MARKET_HOURS_END = 20


class TradingState(TypedDict):
    """State for trading workflow."""

    symbol: str
    market_data: pd.DataFrame | MultiTimeframeData | None
    enable_multi_timeframe: bool
    news_articles: list[NewsArticle] | None
    trump_posts: list[TruthPost] | None
    technical_analysis: TechnicalAnalysis | None
    sentiment_analysis: SentimentAnalysis | None
    news_analysis: NewsAnalysis | None
    trump_analysis: TrumpAnalysis | None
    fundamental_analysis: FundamentalAnalysis | None
    comparative_analysis: ComparativeAnalysis | None
    web_research: WebResearchAnalysis | None
    social_sentiment_analysis: SocialSentimentAnalysis | None
    bullish_research: BullishResearchAnalysis | None
    bearish_research: BearishResearchAnalysis | None
    final_decision: TradingDecision | None
    risk_assessment: RiskAssessment | None
    account_info: AccountInfo | None
    order_status: OrderStatus | None
    regime_analysis: RegimeAnalysis | None
    strategy_selection: StrategySelection | None
    sector_rotation_context: str | None
    earnings_context: str | None
    peer_analysis_context: str | None
    game_plan_context: str | None
    position_context: dict[str, object] | None
    broker_positions: dict[str, BrokerPosition] | None
    portfolio_value: float | None
    backtest_validation: BacktestValidation | None
    degradation_context: "DegradationContext | None"
    warnings: list[str]
    broker_api_failed: bool


class WorkflowExtraContext(TypedDict, total=False):
    """Optional context passed to workflow pipeline."""

    degradation_context: "DegradationContext | None"
    enable_multi_timeframe: bool
    sector_rotation_context: str | None
    earnings_context: str | None
    peer_analysis_context: str | None
    game_plan_context: str | None
    position_context: dict[str, object] | None


class TradingWorkflow:
    """Orchestrate multi-agent trading analysis."""

    def __init__(  # noqa: PLR0913
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
        snapshot_repository: "PortfolioSnapshotRepository | None" = None,
        param_store: "OptimizedParamStore | None" = None,
        historical_cache: HistoricalCache | None = None,
        portfolio_var_calculator: "PortfolioVaRCalculator | None" = None,
        portfolio_var_config: PortfolioVaRConfig | None = None,
        pre_trade_backtest_config: PreTradeBacktestingConfig | None = None,
        notification_service: "NotificationService | None" = None,
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
            pre_trade_backtest_config: Optional pre-trade backtesting configuration
            notification_service: Optional notification service for risk rejection alerts
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

        # Trump mode components
        self.trump_fetcher: TruthSocialFetcher | None = None
        self.trump_analyst: TrumpAnalyst | None = None
        if trump_mode:
            self.trump_fetcher = TruthSocialFetcher(historical_cache=historical_cache)
            self.trump_analyst = TrumpAnalyst(llm_client)

        # Meta-agent for dynamic strategy selection
        self.meta_agent: MetaAgent | None = None
        if use_meta_agent:
            regime_detector = MarketRegimeDetector()
            self.meta_agent = MetaAgent(llm_client, regime_detector, metrics_tracker, param_store=param_store)

        # Default strategy (used if meta-agent disabled)
        self._default_strategy: MomentumStrategy | EnsembleStrategy = (
            EnsembleStrategy() if use_ensemble else MomentumStrategy()
        )

        # Non-technical agents (always same)
        self.sentiment_analyst = SentimentAnalyst(finbert)
        self.news_analyst = NewsAnalyst(llm_client)
        self.fundamental_analyst = FundamentalAnalyst(llm_client, fundamental_fetcher)
        self.comparative_analyst = ComparativeAnalyst(llm_client, ComparativeDataFetcher())
        self.web_researcher = WebResearchAgent(llm_client)
        self.social_analyst = SocialSentimentAnalyst(
            llm_client, FinnhubFetcher(), RedditFetcher(historical_cache=historical_cache), finbert
        )
        self.bullish_researcher = BullishResearcher(llm_client)
        self.bearish_researcher = BearishResearcher(llm_client)
        self.trader = TraderAgent(llm_client)
        self.risk_manager = RiskManagementAgent(
            llm_client,
            portfolio_var_calculator=portfolio_var_calculator,
            portfolio_var_config=portfolio_var_config,
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

    def set_target_allocations(self, allocations: dict[str, float] | None) -> None:
        """Set target portfolio allocations for position sizing.

        Args:
            allocations: Dict of {symbol: weight} for target portfolio
        """
        self._target_allocations = allocations
        if allocations:
            logger.info(f"Set target allocations for {len(allocations)} symbols")

    def _is_rate_limit_error(self, e: Exception) -> bool:
        """Check if exception is related to API rate limiting."""
        msg = str(e).lower()
        return any(
            pattern in msg
            for pattern in [
                "rate limit",
                "call frequency",
                "premium endpoint",
                "5 api calls per minute",
            ]
        )

    def _handle_fundamental_result(self, result: object, state: TradingState) -> FundamentalAnalysis | None:
        """Handle fundamental analysis result with rate-limit awareness."""
        if isinstance(result, Exception):
            if self._is_rate_limit_error(result):
                warning = f"Fundamental analysis unavailable: {result}"
                logger.warning(warning)
                state["warnings"].append(warning)
                return None
            raise result
        assert isinstance(result, FundamentalAnalysis)  # noqa: S101
        return result

    def _handle_optional_result(self, result: T | Exception, name: str, state: TradingState) -> T | None:
        """Handle optional analysis result, logging failures as warnings."""
        if isinstance(result, Exception):
            warning = f"{name} analysis failed: {result}"
            logger.warning(warning)
            state["warnings"].append(warning)
            return None
        return result

    async def analyze(  # noqa: PLR0913
        self,
        symbol: str,
        period_days: int = 90,
        trading_session: TradingSession = TradingSession.REGULAR,
        position_context: dict[str, object] | None = None,
        enable_multi_timeframe: bool = False,
        degradation_context: "DegradationContext | None" = None,
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
            extra_context = {
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

    async def _select_strategy(
        self,
        symbol: str,
        state: TradingState,
        collector: ExecutionMetricsCollector | None,
    ) -> tuple[StrategyType, str]:
        """Select trading strategy via meta-agent or fallback.

        Args:
            symbol: Stock ticker
            state: Current workflow state
            collector: Optional metrics collector

        Returns:
            Tuple of (strategy_instance, strategy_name)
        """
        if self.meta_agent:
            # Extract daily DataFrame for meta-agent
            market_data = state["market_data"]
            if market_data is None:
                msg = "market_data is None, cannot select strategy"
                raise ValueError(msg)
            if isinstance(market_data, MultiTimeframeData):
                daily_data = market_data.timeframes[Timeframe.DAILY]
            else:
                daily_data = market_data

            selection = await self._timed_agent_call(
                "meta_agent",
                self.meta_agent.select_strategy(symbol, daily_data),
                collector,
            )
            state["regime_analysis"] = selection.regime_analysis
            state["strategy_selection"] = selection
            return selection.strategy_instance, selection.strategy_name

        state["regime_analysis"] = None
        state["strategy_selection"] = None
        name = "ensemble" if self.use_ensemble else "momentum"
        return self._default_strategy, name

    async def _validate_strategy_with_backtest(
        self,
        symbol: str,
        strategy: StrategyType,
        strategy_name: str,
        state: TradingState,
        collector: ExecutionMetricsCollector | None,  # noqa: ARG002
    ) -> TradingState:
        """Run pre-trade backtesting validation on selected strategy.

        Args:
            symbol: Stock ticker
            strategy: Strategy instance
            strategy_name: Strategy name for logging
            state: Current workflow state
            collector: Optional metrics collector

        Returns:
            Updated state with backtest_validation field
        """
        if not self.pre_trade_backtest_config or not self.pre_trade_backtest_config.enabled:
            state["backtest_validation"] = None
            return state

        if not self.vectorbt_runner:
            logger.warning("Backtesting enabled but VectorBTRunner not initialized")
            state["backtest_validation"] = None
            return state

        logger.info(f"Running pre-trade backtest for {symbol} ({strategy_name})")

        try:
            from datetime import UTC, datetime, timedelta

            end_date = datetime.now(UTC)
            start_date = end_date - timedelta(days=self.pre_trade_backtest_config.lookback_days)

            backtest_result = await asyncio.to_thread(
                self.vectorbt_runner.run_backtest,
                symbol,
                start_date,
                end_date,
                strategy,
            )

            failure_reasons = []
            if backtest_result.sharpe_ratio < self.pre_trade_backtest_config.min_sharpe_threshold:
                min_sharpe = self.pre_trade_backtest_config.min_sharpe_threshold
                failure_reasons.append(f"Sharpe {backtest_result.sharpe_ratio:.2f} < {min_sharpe}")
            if abs(backtest_result.max_drawdown) > self.pre_trade_backtest_config.max_drawdown_threshold:
                failure_reasons.append(
                    f"Max drawdown {abs(backtest_result.max_drawdown):.1%} > "
                    f"{self.pre_trade_backtest_config.max_drawdown_threshold:.1%}"
                )

            passed = len(failure_reasons) == 0
            confidence_adjustment = (
                1.0 if passed else self.pre_trade_backtest_config.confidence_penalty_multiplier
            )

            validation = BacktestValidation(
                symbol=symbol,
                strategy_name=strategy_name,
                passed=passed,
                sharpe_ratio=backtest_result.sharpe_ratio,
                max_drawdown=backtest_result.max_drawdown,
                total_return=backtest_result.total_return,
                win_rate=backtest_result.win_rate,
                profit_factor=backtest_result.profit_factor,
                total_trades=backtest_result.total_trades,
                lookback_days=self.pre_trade_backtest_config.lookback_days,
                failure_reasons=failure_reasons,
                confidence_adjustment=confidence_adjustment,
            )

            state["backtest_validation"] = validation

            if not passed:
                warning = f"Backtest FAILED ({strategy_name}): {'; '.join(failure_reasons)}"
                logger.warning(warning)
                state["warnings"].append(warning)

            logger.info(
                f"Backtest {'PASSED' if passed else 'FAILED'}: "
                f"Sharpe={backtest_result.sharpe_ratio:.2f}, MaxDD={abs(backtest_result.max_drawdown):.1%}"
            )

        except Exception as e:
            logger.warning(f"Backtest validation error: {e}, continuing without validation")
            state["backtest_validation"] = None
            state["warnings"].append(f"Backtest error: {e}")

        return state

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
        from src.daemon.degradation import DegradationContext, DegradationTier

        ctx = extra_context or {}
        degradation_context: DegradationContext | None = ctx.get("degradation_context")

        # Check if halted
        if degradation_context and degradation_context.tier == DegradationTier.HALTED:
            msg = f"Analysis halted: {degradation_context.halt_reason}"
            raise RuntimeError(msg)

        enable_multi_timeframe = bool(ctx.get("enable_multi_timeframe", False))
        start = time.perf_counter()
        state = await self._fetch_data(symbol, period_days, enable_multi_timeframe)
        state["sector_rotation_context"] = ctx.get("sector_rotation_context")
        state["earnings_context"] = ctx.get("earnings_context")
        state["peer_analysis_context"] = ctx.get("peer_analysis_context")
        state["game_plan_context"] = ctx.get("game_plan_context")
        state["position_context"] = ctx.get("position_context")
        state["degradation_context"] = degradation_context
        self._record_stage(collector, "fetch_data", start)

        start = time.perf_counter()
        state = self._fetch_account_info(state)
        self._record_stage(collector, "fetch_account_info", start)

        start = time.perf_counter()
        strategy, strategy_name = await self._select_strategy(symbol, state, collector)
        self._record_stage(collector, "strategy_selection", start)

        start = time.perf_counter()
        state = await self._validate_strategy_with_backtest(symbol, strategy, strategy_name, state, collector)
        self._record_stage(collector, "backtest_validation", start)

        technical_analyst = TechnicalAnalyst(self.llm_client, strategy)

        start = time.perf_counter()
        state = await self.run_analyses(state, technical_analyst, collector)
        self._record_stage(collector, "analyses", start)

        start = time.perf_counter()
        state = await self._timed_agent_call("trader", self.make_decision(state), collector)
        self._record_stage(collector, "decision", start)

        start = time.perf_counter()
        state = self._assess_risk(state)
        self._record_stage(collector, "risk_assessment", start)

        # Notify if trade rejected by risk gate
        if (
            state["risk_assessment"]
            and state["final_decision"]
            and not state["risk_assessment"].validation.approved
            and state["final_decision"].action != Signal.HOLD
            and self.notification_service
        ):
            await self._notify_risk_rejection(symbol, state)

        if (
            self.broker
            and state["risk_assessment"]
            and state["final_decision"]
            and state["risk_assessment"].validation.approved
            and state["final_decision"].action != Signal.HOLD
        ):
            state = self._execute_trade(state)

        final_decision = state["final_decision"]
        risk_assessment = state["risk_assessment"]
        logger.info(
            f"Workflow complete: {final_decision.action.value if final_decision else 'NONE'} "
            f"(confidence={final_decision.confidence if final_decision else 0.0:.2f}, "
            f"risk_approved={risk_assessment.validation.approved if risk_assessment else False})"
        )

        return await self._build_and_persist_result(symbol, state, strategy_name, trading_session, collector)

    async def _build_and_persist_result(
        self,
        symbol: str,
        state: TradingState,
        strategy_name: str,
        trading_session: TradingSession,
        collector: ExecutionMetricsCollector | None,
    ) -> TradingWorkflowResult:
        """Build workflow result and persist metrics/snapshots.

        Args:
            symbol: Stock ticker
            state: Final workflow state
            strategy_name: Selected strategy name
            trading_session: Trading session type
            collector: Optional metrics collector
        """
        execution_metrics = collector.finalize() if collector else None

        # Extract degradation fields
        degradation_context = state.get("degradation_context")
        degradation_tier = degradation_context.tier.value if degradation_context else None
        degradation_confidence_penalty = (
            (1 - degradation_context.confidence_adjustment) if degradation_context else None
        )

        result = TradingWorkflowResult(
            symbol=symbol,
            trading_session=trading_session,
            technical=state["technical_analysis"],
            sentiment=state["sentiment_analysis"],
            news=state["news_analysis"],
            trump=state.get("trump_analysis"),
            fundamental=state["fundamental_analysis"],
            comparative=state["comparative_analysis"],
            web_research=state["web_research"],
            social_sentiment=state.get("social_sentiment_analysis"),
            bullish=state["bullish_research"],
            bearish=state["bearish_research"],
            decision=state["final_decision"],
            risk=state["risk_assessment"],
            order=state.get("order_status"),
            regime=state.get("regime_analysis"),
            strategy_used=strategy_name,
            warnings=state.get("warnings", []),
            earnings_context=state.get("earnings_context"),
            peer_analysis_context=state.get("peer_analysis_context"),
            execution_metrics=execution_metrics,
            backtest_validation=state.get("backtest_validation"),
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
                        result, strategy_name=strategy_name, is_paper_trade=is_paper
                    )
                else:
                    self.metrics_tracker.record_decision(
                        result, strategy_name=strategy_name, is_paper_trade=is_paper
                    )
            except Exception as e:
                logger.error(f"Failed to record metrics (continuing): {e}")

        if (
            self.snapshot_on_trade
            and self.snapshot_repository
            and state["risk_assessment"]
            and state["final_decision"]
            and state["risk_assessment"].validation.approved
            and state["final_decision"].action != Signal.HOLD
        ):
            await self._capture_portfolio_snapshot(state)

        return result

    async def _timed_agent_call(
        self,
        agent_name: str,
        coro: Coroutine[Any, Any, T],
        collector: ExecutionMetricsCollector | None,
    ) -> T:
        """Wrap an agent coroutine with timing and context var tracking.

        Args:
            agent_name: Agent name for metrics
            coro: Coroutine to execute
            collector: Optional metrics collector
        """
        if collector is None:
            return await coro
        token = current_agent.set(agent_name)
        start = time.perf_counter()
        try:
            return await coro
        finally:
            collector.record_agent_timing(agent_name, (time.perf_counter() - start) * 1000)
            current_agent.reset(token)

    async def run_analyses(  # noqa: PLR0915
        self,
        state: TradingState,
        technical_analyst: TechnicalAnalyst,
        collector: ExecutionMetricsCollector | None = None,
    ) -> TradingState:
        """Run all analysis agents in parallel groups.

        Args:
            state: Current workflow state
            technical_analyst: Technical analyst with selected strategy
            collector: Optional metrics collector

        Returns:
            Updated state with all analyses
        """
        market_data = state["market_data"]
        if market_data is None:
            msg = "market_data is None, cannot run analyses"
            raise ValueError(msg)

        if isinstance(market_data, MultiTimeframeData):
            daily_data = market_data.timeframes[Timeframe.DAILY]
        else:
            daily_data = market_data

        current_price = float(daily_data["Close"].iloc[-1])

        # Validate news_articles
        news_articles = state["news_articles"]
        if news_articles is None:
            msg = "news_articles is None, cannot run sentiment/news analyses"
            raise ValueError(msg)

        # Parallel Group 1: independent analyses (comparative, web_research, social, trump are optional)
        technical_task = self._timed_agent_call(
            "technical",
            technical_analyst.analyze(
                state["symbol"], market_data, enable_multi_timeframe=state["enable_multi_timeframe"]
            ),
            collector,
        )
        sentiment_task = self._timed_agent_call(
            "sentiment", self.sentiment_analyst.analyze(state["symbol"], news_articles), collector
        )
        news_task = self._timed_agent_call(
            "news", self.news_analyst.analyze(state["symbol"], news_articles), collector
        )
        fundamental_task = self._timed_agent_call(
            "fundamental", self.fundamental_analyst.analyze(state["symbol"], current_price), collector
        )
        comparative_task = self._timed_agent_call(
            "comparative", self.comparative_analyst.analyze(state["symbol"]), collector
        )
        web_research_task = self._timed_agent_call(
            "web_research", self.web_researcher.research(state["symbol"]), collector
        )
        social_task = self._timed_agent_call(
            "social", self.social_analyst.analyze(state["symbol"]), collector
        )

        # Include trump analysis if enabled
        tasks: list[Coroutine[Any, Any, Any]] = [
            technical_task,
            sentiment_task,
            news_task,
            fundamental_task,
            comparative_task,
            web_research_task,
            social_task,
        ]

        if self.trump_mode and self.trump_analyst and state["trump_posts"]:
            trump_task = self._timed_agent_call(
                "trump", self.trump_analyst.analyze(state["trump_posts"]), collector
            )
            tasks.append(trump_task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Core tasks count (before optional trump task)
        core_task_count = 7
        (
            technical,
            sentiment,
            news,
            fundamental_result,
            comparative_result,
            web_research_result,
            social_result,
        ) = results[:core_task_count]
        trump_result = results[core_task_count] if len(results) > core_task_count else None

        # Re-raise if core analyses failed
        if isinstance(technical, Exception):
            raise technical
        if isinstance(sentiment, Exception):
            raise sentiment
        if isinstance(news, Exception):
            raise news

        # Type narrowing - after exception checks, these must be their proper types
        assert isinstance(technical, TechnicalAnalysis)  # noqa: S101
        assert isinstance(sentiment, SentimentAnalysis)  # noqa: S101
        assert isinstance(news, NewsAnalysis)  # noqa: S101

        # Process optional analyses
        fundamental = self._handle_fundamental_result(fundamental_result, state)
        comparative = cast(
            "ComparativeAnalysis | None",
            self._handle_optional_result(comparative_result, "Comparative", state),
        )
        web_research = cast(
            "WebResearchAnalysis | None",
            self._handle_optional_result(web_research_result, "Web research", state),
        )
        social_sentiment = cast(
            "SocialSentimentAnalysis | None",
            self._handle_optional_result(social_result, "Social sentiment", state),
        )
        trump_analysis_processed = (
            cast("TrumpAnalysis | None", self._handle_optional_result(trump_result, "Trump", state))
            if trump_result
            else None
        )

        state["technical_analysis"] = technical
        state["sentiment_analysis"] = sentiment
        state["news_analysis"] = news
        state["trump_analysis"] = trump_analysis_processed
        state["fundamental_analysis"] = fundamental
        state["comparative_analysis"] = comparative
        state["social_sentiment_analysis"] = social_sentiment
        state["web_research"] = web_research

        # Parallel Group 2: research (depends on Group 1)
        # Ensure required analyses are present
        technical = state["technical_analysis"]
        sentiment = state["sentiment_analysis"]
        news = state["news_analysis"]
        assert technical is not None, "technical_analysis must be present"  # noqa: S101
        assert sentiment is not None, "sentiment_analysis must be present"  # noqa: S101
        assert news is not None, "news_analysis must be present"  # noqa: S101

        bullish_task = self._timed_agent_call(
            "bullish_researcher",
            self.bullish_researcher.analyze(
                state["symbol"],
                technical,
                sentiment,
                news,
                state["fundamental_analysis"],
                state["comparative_analysis"],
                state["trump_analysis"],
            ),
            collector,
        )
        bearish_task = self._timed_agent_call(
            "bearish_researcher",
            self.bearish_researcher.analyze(
                state["symbol"],
                technical,
                sentiment,
                news,
                state["fundamental_analysis"],
                state["comparative_analysis"],
                state["trump_analysis"],
            ),
            collector,
        )

        bullish, bearish = await asyncio.gather(bullish_task, bearish_task)
        state["bullish_research"] = bullish
        state["bearish_research"] = bearish

        return state

    @staticmethod
    def _is_market_hours() -> bool:
        """Check if currently within market hours (4am-8pm ET)."""
        now = datetime.now(ET_TIMEZONE)
        return MARKET_HOURS_START <= now.hour < MARKET_HOURS_END

    async def _fetch_data(
        self, symbol: str, period_days: int, enable_multi_timeframe: bool = False
    ) -> TradingState:
        """Fetch market and news data (async, parallel execution).

        Args:
            symbol: Stock ticker
            period_days: Historical data period
            enable_multi_timeframe: Enable multi-timeframe data fetching

        Returns:
            Updated state with data
        """
        logger.info("Fetching market and news data")

        # Prepare parallel tasks
        if enable_multi_timeframe and self._is_market_hours():
            logger.info("Multi-timeframe mode enabled (market hours)")
            market_task = self.market_fetcher.fetch_multi_timeframe(
                symbol, [Timeframe.DAILY, Timeframe.HOURLY], period_days
            )
        else:
            if enable_multi_timeframe and not self._is_market_hours():
                logger.info("Multi-timeframe requested but outside market hours, using daily only")
            market_task = asyncio.to_thread(self.market_fetcher.fetch_daily, symbol, period_days)

        news_task = asyncio.to_thread(self.news_fetcher.fetch_company_news, symbol, limit=10)
        tasks: list[Coroutine[Any, Any, Any]] = [market_task, news_task]

        if self.trump_mode and self.trump_fetcher:
            tasks.append(asyncio.to_thread(self.trump_fetcher.fetch_recent, hours=24))

        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Extract market data
        if isinstance(results[0], Exception):
            logger.error(f"Market data fetch failed: {results[0]}")
            raise results[0]
        market_result = results[0]
        if enable_multi_timeframe and self._is_market_hours():
            # fetch_multi_timeframe returns MultiTimeframeData
            assert isinstance(market_result, MultiTimeframeData)  # noqa: S101
            market_data = market_result
        else:
            # fetch_daily returns MarketData with .data attribute
            from src.data.market import MarketData

            assert isinstance(market_result, MarketData)  # noqa: S101
            market_data = market_result.data

        # Extract news data
        if isinstance(results[1], Exception):
            logger.error(f"News fetch failed: {results[1]}")
            raise results[1]
        news_result = results[1]
        assert isinstance(news_result, list)  # noqa: S101

        # Extract trump data
        trump_posts: list[TruthPost] | None = None
        if self.trump_mode and self.trump_fetcher:
            trump_result = results[2]
            if isinstance(trump_result, Exception):
                logger.warning(f"Failed to fetch Trump posts: {trump_result}")
            else:
                from src.data.truth_social import TrumpPostData

                assert isinstance(trump_result, TrumpPostData)  # noqa: S101
                trump_posts = trump_result.posts
                logger.info(f"Fetched {len(trump_posts)} Trump posts")

        return TradingState(
            symbol=symbol,
            market_data=market_data,
            enable_multi_timeframe=enable_multi_timeframe,
            news_articles=news_result,
            trump_posts=trump_posts,
            technical_analysis=None,
            sentiment_analysis=None,
            news_analysis=None,
            trump_analysis=None,
            fundamental_analysis=None,
            comparative_analysis=None,
            web_research=None,
            social_sentiment_analysis=None,
            bullish_research=None,
            bearish_research=None,
            final_decision=None,
            risk_assessment=None,
            account_info=None,
            order_status=None,
            regime_analysis=None,
            strategy_selection=None,
            sector_rotation_context=None,
            earnings_context=None,
            peer_analysis_context=None,
            game_plan_context=None,
            position_context=None,
            broker_positions=None,
            portfolio_value=None,
            backtest_validation=None,
            degradation_context=None,
            warnings=[],
            broker_api_failed=False,
        )

    def _fetch_account_info(self, state: TradingState) -> TradingState:
        """Fetch account info for portfolio-aware decisions.

        Args:
            state: Current workflow state

        Returns:
            Updated state with account info and broker positions
        """
        logger.info("Fetching account info")
        account_info, broker_info, account_info_valid = self._get_account_info()
        state["account_info"] = account_info

        # Track broker availability for risk assessment
        if not account_info_valid:
            warning = (
                "Broker API unavailable - using mock account data. "
                "Trade execution will be blocked to prevent incorrect position sizing."
            )
            state["warnings"].append(warning)
            state["broker_api_failed"] = True

        # Set VaR fields from broker_info (if available)
        if broker_info:
            state["broker_positions"] = broker_info.positions
            state["portfolio_value"] = broker_info.portfolio_value

        return state

    async def make_decision(self, state: TradingState) -> TradingState:
        """Make final trading decision.

        Args:
            state: Current workflow state

        Returns:
            Updated state with final decision
        """
        logger.info("Making final trading decision")

        account_info = state["account_info"]
        positions = account_info.positions if account_info else {}
        symbol = state["symbol"]
        owns_position = symbol in positions
        position_qty = positions.get(symbol)

        # Ensure critical analyses are present
        technical = state["technical_analysis"]
        sentiment = state["sentiment_analysis"]
        news = state["news_analysis"]
        bullish = state["bullish_research"]
        bearish = state["bearish_research"]

        if not technical or not sentiment or not news:
            msg = "Missing critical analyses (technical, sentiment, news)"
            raise ValueError(msg)
        if not bullish or not bearish:
            msg = "Missing research analyses (bullish, bearish)"
            raise ValueError(msg)

        # Optional analyses can be None
        decision = await self.trader.decide(
            symbol,
            technical,
            sentiment,
            news,
            state["fundamental_analysis"],
            bullish,
            bearish,
            comparative=state["comparative_analysis"],
            owns_position=owns_position,
            position_qty=position_qty,
            sector_context=state.get("sector_rotation_context"),
            earnings_context=state.get("earnings_context"),
            peer_analysis_context=state.get("peer_analysis_context"),
            backtest_validation=state.get("backtest_validation"),
            game_plan_context=state.get("game_plan_context"),
            position_context=state.get("position_context"),
            degradation_context=state.get("degradation_context"),
        )

        state["final_decision"] = decision
        return state

    def _assess_risk(self, state: TradingState) -> TradingState:
        """Assess risk for trading decision.

        Args:
            state: Current workflow state

        Returns:
            Updated state with risk assessment
        """
        logger.info("Assessing risk for trading decision")

        if not state["final_decision"]:
            msg = "Cannot assess risk without final decision"
            raise ValueError(msg)

        # Extract daily DataFrame for price lookup
        market_data = state["market_data"]
        if market_data is None:
            msg = "market_data is None, cannot assess risk"
            raise ValueError(msg)
        if isinstance(market_data, MultiTimeframeData):
            daily_data = market_data.timeframes[Timeframe.DAILY]
        else:
            daily_data = market_data

        current_price = float(daily_data["Close"].iloc[-1])

        # Get target weight from allocations if available
        target_weight = self._target_allocations.get(state["symbol"]) if self._target_allocations else None

        # Ensure account_info is present
        account_info = state["account_info"]
        if account_info is None:
            msg = "account_info is None, cannot assess risk"
            raise ValueError(msg)

        risk_assessment = self.risk_manager.assess(
            symbol=state["symbol"],
            action=state["final_decision"].action,
            current_price=current_price,
            account_info=account_info,
            market_data=daily_data,
            decision_confidence=state["final_decision"].confidence,
            broker_positions=state.get("broker_positions"),
            portfolio_value=state.get("portfolio_value"),
            target_portfolio_weight=target_weight,
            backtest_validation=state.get("backtest_validation"),
            degradation_context=state.get("degradation_context"),
            broker_api_failed=state.get("broker_api_failed", False),
        )

        state["risk_assessment"] = risk_assessment
        return state

    def _get_account_info(self) -> tuple[AccountInfo, BrokerAccountInfo | None, bool]:
        """Get account information.

        Returns:
            Tuple of (AccountInfo, BrokerAccountInfo | None, account_info_valid: bool)
        """
        # Safe case: intentional paper trading
        if not self.broker:
            return (
                AccountInfo(
                    balance=100000.0,
                    available_cash=100000.0,
                    positions={},
                    total_exposure=0.0,
                ),
                None,
                True,
            )  # No broker = safe mock mode

        # Dangerous case: broker configured but API fails
        try:
            broker_info = self.broker.get_account_info()
            return (
                AccountInfo(
                    balance=broker_info.balance,
                    available_cash=broker_info.available_cash,
                    positions={sym: pos.qty for sym, pos in broker_info.positions.items()},
                    total_exposure=broker_info.total_exposure,
                ),
                broker_info,
                True,
            )
        except BrokerAPIError:
            logger.critical(
                "BROKER API FAILURE: Account info unavailable but auto_trade configured. "
                "This would cause incorrect position sizing. Trade execution disabled for this symbol."
            )
            return (
                AccountInfo(
                    balance=100000.0,  # Mock data - DO NOT USE FOR REAL TRADES
                    available_cash=100000.0,
                    positions={},
                    total_exposure=0.0,
                ),
                None,
                False,
            )  # Signal broker failure

    def _execute_trade(self, state: TradingState) -> TradingState:
        """Execute trade via broker.

        Args:
            state: Current workflow state

        Returns:
            Updated state with order status
        """
        risk = state["risk_assessment"]
        final_decision = state["final_decision"]

        if not final_decision or not risk or not self.broker:
            msg = "Cannot execute trade without decision, risk assessment, and broker"
            raise ValueError(msg)

        action = final_decision.action

        order: OrderStatus | None = None
        try:
            stop_loss_price = risk.stop_loss.stop_loss_price if risk.stop_loss else None
            order = self.broker.submit_order(
                symbol=state["symbol"],
                qty=int(risk.position_sizing.recommended_shares) if risk.position_sizing else 0,
                side=action.value.lower(),
                stop_loss_price=stop_loss_price,
            )
            stop_loss_str = f"{stop_loss_price:.2f}" if stop_loss_price is not None else "None"
            logger.info(
                f"Executed {action.value}: {state['symbol']} x{order.qty} (stop-loss={stop_loss_str})"
            )
        except BrokerAPIError as e:
            logger.critical(
                f"BROKER API FAILURE during order submission for {state['symbol']} "
                f"with action {action.value}: {e}"
            )
            if "warnings" in state:
                state["warnings"].append(f"Order submission failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error submitting order for {state['symbol']}: {e}")
            if "warnings" in state:
                state["warnings"].append(f"Order submission error: {e}")

        state["order_status"] = order
        return state

    async def _capture_portfolio_snapshot(self, state: TradingState) -> None:
        """Capture portfolio snapshot after trade execution.

        Args:
            state: Current workflow state
        """
        from datetime import UTC, datetime

        from src.database.repositories.snapshot import PortfolioSnapshot

        if not self.snapshot_repository or not state["account_info"]:
            return

        try:
            account = state["account_info"]
            snapshot = PortfolioSnapshot(
                timestamp=datetime.now(UTC),
                balance=account.balance,
                available_cash=account.available_cash,
                total_exposure=account.total_exposure,
                portfolio_value=account.balance,
                positions={k: float(v) for k, v in account.positions.items()},
                trigger="TRADE",
            )
            await self.snapshot_repository.create(snapshot)
            logger.info("Captured portfolio snapshot (trigger=TRADE)")
        except Exception as e:
            logger.error(f"Failed to capture portfolio snapshot: {e}")

    async def _notify_risk_rejection(self, symbol: str, state: TradingState) -> None:
        """Send risk rejection notification.

        Args:
            symbol: Stock symbol
            state: Current workflow state
        """
        from datetime import UTC

        from src.daemon.config import NotificationTrigger
        from src.daemon.notifications import NotificationMessage

        risk = state["risk_assessment"]
        decision = state["final_decision"]

        if not risk or not decision:
            return  # Nothing to notify if missing data

        message = NotificationMessage(
            trigger=NotificationTrigger.RISK_REJECTION,
            title=f"Trade Blocked: {symbol}",
            body=risk.validation.reasoning,
            metadata={
                "symbol": symbol,
                "signal": decision.action.value,
                "price": risk.current_price,
                "confidence": decision.confidence,
                "rejection_reason": risk.validation.reasoning,
                "risk_score": risk.validation.risk_score,
            },
            timestamp=datetime.now(UTC),
        )

        if self.notification_service:
            await self.notification_service.notify(NotificationTrigger.RISK_REJECTION, message)

    def __repr__(self) -> str:
        """String representation."""
        mode = "meta-agent" if self.use_meta_agent else ("ensemble" if self.use_ensemble else "momentum")
        trump_str = "+trump" if self.trump_mode else ""
        return f"TradingWorkflow(agents=9, mode={mode}{trump_str})"
