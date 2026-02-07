"""Trading workflow orchestrating all agents."""

import asyncio
import time
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any, TypeVar

import pandas as pd
from loguru import logger
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from src.database.repositories.snapshot import PortfolioSnapshotRepository
    from src.optimization.param_store import OptimizedParamStore

from src.agents.bearish_researcher import BearishResearchAnalysis, BearishResearcher
from src.agents.bullish_researcher import BullishResearchAnalysis, BullishResearcher
from src.agents.comparative import ComparativeAnalysis, ComparativeAnalyst
from src.agents.fundamental import FundamentalAnalysis, FundamentalAnalyst
from src.agents.meta import MetaAgent, StrategySelection
from src.agents.news import NewsAnalysis, NewsAnalyst
from src.agents.risk import AccountInfo, RiskAssessment, RiskManagementAgent
from src.agents.sentiment import SentimentAnalysis, SentimentAnalyst
from src.agents.social import SocialSentimentAnalysis, SocialSentimentAnalyst
from src.agents.technical import TechnicalAnalysis, TechnicalAnalyst
from src.agents.trader import TraderAgent, TradingDecision
from src.agents.trump import TrumpAnalysis, TrumpAnalyst
from src.agents.web_researcher import WebResearchAgent, WebResearchAnalysis
from src.cache.historical import HistoricalCache
from src.data.broker import AlpacaBroker, OrderStatus
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
from src.workflows.types import TradingWorkflowResult

T = TypeVar("T")


class TradingState(TypedDict):
    """State for trading workflow."""

    symbol: str
    market_data: pd.DataFrame | None
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
    warnings: list[str]


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
        """
        import os

        if snapshot_on_trade is None:
            snapshot_on_trade = os.getenv("PORTFOLIO_SNAPSHOT_ON_TRADE", "false").lower() == "true"
        self.snapshot_on_trade = snapshot_on_trade
        self.snapshot_repository = snapshot_repository
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
        self.risk_manager = RiskManagementAgent(llm_client)

        mode = "meta-agent" if use_meta_agent else ("ensemble" if use_ensemble else "momentum")
        trump_str = "+trump" if trump_mode else ""
        logger.info(f"Initialized TradingWorkflow (mode={mode}{trump_str})")

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
        return result

    def _handle_optional_result(self, result: object, name: str, state: TradingState) -> object | None:
        """Handle optional analysis result, logging failures as warnings."""
        if isinstance(result, Exception):
            warning = f"{name} analysis failed: {result}"
            logger.warning(warning)
            state["warnings"].append(warning)
            return None
        return result

    async def analyze(
        self,
        symbol: str,
        period_days: int = 90,
        trading_session: TradingSession = TradingSession.REGULAR,
    ) -> TradingWorkflowResult:
        """Run complete trading analysis.

        Args:
            symbol: Stock ticker symbol
            period_days: Days of historical data to fetch
            trading_session: Trading session type (REGULAR or PRE_MARKET)

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
            return await self._analyze_instrumented(symbol, period_days, trading_session, collector)
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
    ) -> tuple[object, str]:
        """Select trading strategy via meta-agent or fallback.

        Args:
            symbol: Stock ticker
            state: Current workflow state
            collector: Optional metrics collector

        Returns:
            Tuple of (strategy_instance, strategy_name)
        """
        if self.meta_agent:
            selection = await self._timed_agent_call(
                "meta_agent",
                self.meta_agent.select_strategy(symbol, state["market_data"]),
                collector,
            )
            state["regime_analysis"] = selection.regime_analysis
            state["strategy_selection"] = selection
            return selection.strategy_instance, selection.strategy_name

        state["regime_analysis"] = None
        state["strategy_selection"] = None
        name = "ensemble" if self.use_ensemble else "momentum"
        return self._default_strategy, name

    async def _analyze_instrumented(
        self,
        symbol: str,
        period_days: int,
        trading_session: TradingSession,
        collector: ExecutionMetricsCollector | None,
    ) -> TradingWorkflowResult:
        """Run analysis pipeline with optional metrics instrumentation.

        Args:
            symbol: Stock ticker symbol
            period_days: Days of historical data
            trading_session: Trading session type (REGULAR or PRE_MARKET)
            collector: Optional metrics collector
        """
        start = time.perf_counter()
        state = await self._fetch_data(symbol, period_days)
        self._record_stage(collector, "fetch_data", start)

        start = time.perf_counter()
        state = self._fetch_account_info(state)
        self._record_stage(collector, "fetch_account_info", start)

        start = time.perf_counter()
        strategy, strategy_name = await self._select_strategy(symbol, state, collector)
        self._record_stage(collector, "strategy_selection", start)

        technical_analyst = TechnicalAnalyst(self.llm_client, strategy)

        start = time.perf_counter()
        state = await self._run_analyses(state, technical_analyst, collector)
        self._record_stage(collector, "analyses", start)

        start = time.perf_counter()
        state = await self._timed_agent_call("trader", self._make_decision(state), collector)
        self._record_stage(collector, "decision", start)

        start = time.perf_counter()
        state = self._assess_risk(state)
        self._record_stage(collector, "risk_assessment", start)

        if (
            self.broker
            and state["risk_assessment"].validation.approved
            and state["final_decision"].action != Signal.HOLD
        ):
            state = self._execute_trade(state)

        logger.info(
            f"Workflow complete: {state['final_decision'].action.value} "
            f"(confidence={state['final_decision'].confidence:.2f}, "
            f"risk_approved={state['risk_assessment'].validation.approved})"
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
            execution_metrics=execution_metrics,
        )

        if execution_metrics:
            try:
                persist_jsonl(execution_metrics)
            except Exception as e:
                logger.error(f"Failed to persist execution metrics (continuing): {e}")

        if self.metrics_tracker:
            try:
                if isinstance(self.metrics_tracker, DatabaseMetricsTracker):
                    await self.metrics_tracker.record_decision_async(result, strategy_name=strategy_name)
                else:
                    self.metrics_tracker.record_decision(result, strategy_name=strategy_name)
            except Exception as e:
                logger.error(f"Failed to record metrics (continuing): {e}")

        if (
            self.snapshot_on_trade
            and self.snapshot_repository
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

    async def _run_analyses(
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
        current_price = float(state["market_data"]["Close"].iloc[-1])

        # Parallel Group 1: independent analyses (comparative, web_research, social, trump are optional)
        technical_task = self._timed_agent_call(
            "technical", technical_analyst.analyze(state["symbol"], state["market_data"]), collector
        )
        sentiment_task = self._timed_agent_call(
            "sentiment", self.sentiment_analyst.analyze(state["symbol"], state["news_articles"]), collector
        )
        news_task = self._timed_agent_call(
            "news", self.news_analyst.analyze(state["symbol"], state["news_articles"]), collector
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
        tasks = [
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
        for result in (technical, sentiment, news):
            if isinstance(result, Exception):
                raise result

        # Process optional analyses
        fundamental = self._handle_fundamental_result(fundamental_result, state)
        comparative = self._handle_optional_result(comparative_result, "Comparative", state)
        web_research = self._handle_optional_result(web_research_result, "Web research", state)
        social_sentiment = self._handle_optional_result(social_result, "Social sentiment", state)
        trump_analysis = self._handle_optional_result(trump_result, "Trump", state) if trump_result else None

        state["technical_analysis"] = technical
        state["sentiment_analysis"] = sentiment
        state["news_analysis"] = news
        state["trump_analysis"] = trump_analysis
        state["fundamental_analysis"] = fundamental
        state["comparative_analysis"] = comparative
        state["social_sentiment_analysis"] = social_sentiment
        state["web_research"] = web_research

        # Parallel Group 2: research (depends on Group 1)
        bullish_task = self._timed_agent_call(
            "bullish_researcher",
            self.bullish_researcher.analyze(
                state["symbol"],
                state["technical_analysis"],
                state["sentiment_analysis"],
                state["news_analysis"],
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
                state["technical_analysis"],
                state["sentiment_analysis"],
                state["news_analysis"],
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

    async def _fetch_data(self, symbol: str, period_days: int) -> TradingState:
        """Fetch market and news data (async, parallel execution).

        Args:
            symbol: Stock ticker
            period_days: Historical data period

        Returns:
            Updated state with data
        """
        logger.info("Fetching market and news data")

        # Create parallel tasks
        tasks = [
            asyncio.to_thread(self.market_fetcher.fetch_daily, symbol, period_days),
            asyncio.to_thread(self.news_fetcher.fetch_company_news, symbol, limit=10),
        ]

        # Add trump task if enabled
        trump_index = None
        if self.trump_mode and self.trump_fetcher:
            trump_index = len(tasks)
            tasks.append(asyncio.to_thread(self.trump_fetcher.fetch_recent, hours=24))

        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Extract results
        market_result, news_result = results[0], results[1]
        trump_result = results[trump_index] if trump_index is not None else None

        # Handle critical fetches (re-raise on failure)
        if isinstance(market_result, Exception):
            logger.error(f"Market data fetch failed: {market_result}")
            raise market_result
        if isinstance(news_result, Exception):
            logger.error(f"News fetch failed: {news_result}")
            raise news_result

        market_data = market_result
        news_articles = news_result

        # Handle optional trump fetch
        trump_posts: list[TruthPost] | None = None
        if trump_result is not None:
            if isinstance(trump_result, Exception):
                logger.warning(f"Failed to fetch Trump posts: {trump_result}")
            else:
                trump_posts = trump_result.posts
                logger.info(f"Fetched {len(trump_posts)} Trump posts")

        return TradingState(
            symbol=symbol,
            market_data=market_data.data,
            news_articles=news_articles,
            trump_posts=trump_posts,
            technical_analysis=None,
            sentiment_analysis=None,
            news_analysis=None,
            trump_analysis=None,
            fundamental_analysis=None,
            comparative_analysis=None,
            web_research=None,
            bullish_research=None,
            bearish_research=None,
            final_decision=None,
            risk_assessment=None,
            account_info=None,
            order_status=None,
            regime_analysis=None,
            strategy_selection=None,
            warnings=[],
        )

    def _fetch_account_info(self, state: TradingState) -> TradingState:
        """Fetch account info for portfolio-aware decisions.

        Args:
            state: Current workflow state

        Returns:
            Updated state with account info
        """
        logger.info("Fetching account info")
        state["account_info"] = self._get_account_info()
        return state

    async def _make_decision(self, state: TradingState) -> TradingState:
        """Make final trading decision.

        Args:
            state: Current workflow state

        Returns:
            Updated state with final decision
        """
        logger.info("Making final trading decision")

        positions = state["account_info"].positions
        symbol = state["symbol"]
        owns_position = symbol in positions
        position_qty = positions.get(symbol)

        decision = await self.trader.decide(
            symbol,
            state["technical_analysis"],
            state["sentiment_analysis"],
            state["news_analysis"],
            state["fundamental_analysis"],
            state["bullish_research"],
            state["bearish_research"],
            comparative=state["comparative_analysis"],
            owns_position=owns_position,
            position_qty=position_qty,
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

        current_price = float(state["market_data"]["Close"].iloc[-1])

        risk_assessment = self.risk_manager.assess(
            symbol=state["symbol"],
            action=state["final_decision"].action,
            current_price=current_price,
            account_info=state["account_info"],
            market_data=state["market_data"],
            decision_confidence=state["final_decision"].confidence,
        )

        state["risk_assessment"] = risk_assessment
        return state

    def _get_account_info(self) -> AccountInfo:
        """Get account information.

        Returns:
            AccountInfo from broker or mocked data
        """
        if not self.broker:
            return AccountInfo(
                balance=100000.0,
                available_cash=100000.0,
                positions={},
                total_exposure=0.0,
            )

        try:
            broker_info = self.broker.get_account_info()
            return AccountInfo(
                balance=broker_info.balance,
                available_cash=broker_info.available_cash,
                positions={sym: pos.qty for sym, pos in broker_info.positions.items()},
                total_exposure=broker_info.total_exposure,
            )
        except Exception:
            logger.exception("Failed to fetch account info from broker, using mock data")
            return AccountInfo(
                balance=100000.0,
                available_cash=100000.0,
                positions={},
                total_exposure=0.0,
            )

    def _execute_trade(self, state: TradingState) -> TradingState:
        """Execute trade via broker.

        Args:
            state: Current workflow state

        Returns:
            Updated state with order status
        """
        risk = state["risk_assessment"]
        action = state["final_decision"].action

        order: OrderStatus | None = None
        try:
            order = self.broker.submit_order(
                symbol=state["symbol"],
                qty=int(risk.position_sizing.recommended_shares),
                side=action.value.lower(),
                stop_loss_price=risk.stop_loss.stop_loss_price,
            )
            logger.info(
                f"Executed {action.value}: {state['symbol']} "
                f"x{order.qty} (stop-loss={risk.stop_loss.stop_loss_price:.2f})"
            )
        except Exception:
            logger.exception(f"Failed to submit order for {state['symbol']} with action {action.value}")

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

    def __repr__(self) -> str:
        """String representation."""
        mode = "meta-agent" if self.use_meta_agent else ("ensemble" if self.use_ensemble else "momentum")
        trump_str = "+trump" if self.trump_mode else ""
        return f"TradingWorkflow(agents=9, mode={mode}{trump_str})"
