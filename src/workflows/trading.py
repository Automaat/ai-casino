"""Trading workflow orchestrating all agents."""

import asyncio
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger
from pydantic import BaseModel
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from src.database.repositories.snapshot import PortfolioSnapshotRepository

from src.agents.bearish_researcher import BearishResearchAnalysis, BearishResearcher
from src.agents.bullish_researcher import BullishResearchAnalysis, BullishResearcher
from src.agents.comparative import ComparativeAnalysis, ComparativeAnalyst
from src.agents.fundamental import FundamentalAnalysis, FundamentalAnalyst
from src.agents.meta import MetaAgent, StrategySelection
from src.agents.news import NewsAnalysis, NewsAnalyst
from src.agents.risk import AccountInfo, RiskAssessment, RiskManagementAgent
from src.agents.sentiment import SentimentAnalysis, SentimentAnalyst
from src.agents.technical import TechnicalAnalysis, TechnicalAnalyst
from src.agents.trader import TraderAgent, TradingDecision
from src.agents.web_researcher import WebResearchAgent, WebResearchAnalysis
from src.data.broker import AlpacaBroker, OrderStatus
from src.data.comparative import ComparativeDataFetcher
from src.data.fundamental import FundamentalDataFetcher
from src.data.market import MarketDataFetcher
from src.data.news import NewsArticle, NewsFetcher
from src.metrics.tracker import BaseMetricsTracker, DatabaseMetricsTracker
from src.models.llm import LLMClient
from src.models.sentiment import FinBERTSentiment
from src.strategies.ensemble import EnsembleStrategy
from src.strategies.momentum import MomentumStrategy, Signal
from src.strategies.regime import MarketRegimeDetector, RegimeAnalysis


class TradingState(TypedDict):
    """State for trading workflow."""

    symbol: str
    market_data: pd.DataFrame | None
    news_articles: list[NewsArticle] | None
    technical_analysis: TechnicalAnalysis | None
    sentiment_analysis: SentimentAnalysis | None
    news_analysis: NewsAnalysis | None
    fundamental_analysis: FundamentalAnalysis | None
    comparative_analysis: ComparativeAnalysis | None
    web_research: WebResearchAnalysis | None
    bullish_research: BullishResearchAnalysis | None
    bearish_research: BearishResearchAnalysis | None
    final_decision: TradingDecision | None
    risk_assessment: RiskAssessment | None
    account_info: AccountInfo | None
    order_status: OrderStatus | None
    regime_analysis: RegimeAnalysis | None
    strategy_selection: StrategySelection | None
    warnings: list[str]


class TradingWorkflowResult(BaseModel):
    """Complete trading analysis result."""

    symbol: str
    technical: TechnicalAnalysis
    sentiment: SentimentAnalysis
    news: NewsAnalysis
    fundamental: FundamentalAnalysis | None = None
    comparative: ComparativeAnalysis | None = None
    web_research: WebResearchAnalysis | None = None
    bullish: BullishResearchAnalysis
    bearish: BearishResearchAnalysis
    decision: TradingDecision
    risk: RiskAssessment
    order: OrderStatus | None = None
    regime: RegimeAnalysis | None = None
    strategy_used: str | None = None
    warnings: list[str] = []

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    @property
    def has_incomplete_data(self) -> bool:
        """Check if analysis was performed with incomplete data."""
        return len(self.warnings) > 0


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
        snapshot_on_trade: bool | None = None,
        snapshot_repository: "PortfolioSnapshotRepository | None" = None,
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
            snapshot_on_trade: Capture portfolio snapshot after trades (env: PORTFOLIO_SNAPSHOT_ON_TRADE)
            snapshot_repository: Repository for portfolio snapshots (required if snapshot_on_trade)
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

        # Meta-agent for dynamic strategy selection
        self.meta_agent: MetaAgent | None = None
        if use_meta_agent:
            regime_detector = MarketRegimeDetector()
            self.meta_agent = MetaAgent(llm_client, regime_detector, metrics_tracker)

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
        self.bullish_researcher = BullishResearcher(llm_client)
        self.bearish_researcher = BearishResearcher(llm_client)
        self.trader = TraderAgent(llm_client)
        self.risk_manager = RiskManagementAgent(llm_client)

        mode = "meta-agent" if use_meta_agent else ("ensemble" if use_ensemble else "momentum")
        logger.info(f"Initialized TradingWorkflow (mode={mode})")

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

    async def analyze(self, symbol: str, period_days: int = 90) -> TradingWorkflowResult:
        """Run complete trading analysis.

        Args:
            symbol: Stock ticker symbol
            period_days: Days of historical data to fetch

        Returns:
            TradingWorkflowResult with all analyses and final decision
        """
        logger.info(f"Starting trading workflow for {symbol}")

        state = self._fetch_data(symbol, period_days)
        state = self._fetch_account_info(state)

        # Strategy selection via meta-agent or fallback to default
        strategy_name: str | None = None
        if self.meta_agent:
            selection = await self.meta_agent.select_strategy(symbol, state["market_data"])
            strategy = selection.strategy_instance
            strategy_name = selection.strategy_name
            state["regime_analysis"] = selection.regime_analysis
            state["strategy_selection"] = selection
        else:
            strategy = self._default_strategy
            strategy_name = "ensemble" if self.use_ensemble else "momentum"
            state["regime_analysis"] = None
            state["strategy_selection"] = None

        # Create TechnicalAnalyst with selected strategy
        technical_analyst = TechnicalAnalyst(self.llm_client, strategy)

        # Run all analyses (parallel where possible)
        state = await self._run_analyses(state, technical_analyst)

        state = await self._make_decision(state)
        state = self._assess_risk(state)

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

        result = TradingWorkflowResult(
            symbol=symbol,
            technical=state["technical_analysis"],
            sentiment=state["sentiment_analysis"],
            news=state["news_analysis"],
            fundamental=state["fundamental_analysis"],
            comparative=state["comparative_analysis"],
            web_research=state["web_research"],
            bullish=state["bullish_research"],
            bearish=state["bearish_research"],
            decision=state["final_decision"],
            risk=state["risk_assessment"],
            order=state.get("order_status"),
            regime=state.get("regime_analysis"),
            strategy_used=strategy_name,
            warnings=state.get("warnings", []),
        )

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

    async def _run_analyses(self, state: TradingState, technical_analyst: TechnicalAnalyst) -> TradingState:
        """Run all analysis agents in parallel groups.

        Args:
            state: Current workflow state
            technical_analyst: Technical analyst with selected strategy

        Returns:
            Updated state with all analyses
        """
        current_price = float(state["market_data"]["Close"].iloc[-1])

        # Parallel Group 1: independent analyses (comparative and web_research are optional)
        technical_task = technical_analyst.analyze(state["symbol"], state["market_data"])
        sentiment_task = self.sentiment_analyst.analyze(state["symbol"], state["news_articles"])
        news_task = self.news_analyst.analyze(state["symbol"], state["news_articles"])
        fundamental_task = self.fundamental_analyst.analyze(state["symbol"], current_price)
        comparative_task = self.comparative_analyst.analyze(state["symbol"])
        web_research_task = self.web_researcher.research(state["symbol"])

        results = await asyncio.gather(
            technical_task,
            sentiment_task,
            news_task,
            fundamental_task,
            comparative_task,
            web_research_task,
            return_exceptions=True,
        )
        technical, sentiment, news, fundamental_result, comparative_result, web_research_result = results

        # Re-raise if core analyses failed (fundamental is optional due to Alpha Vantage rate limits)
        for result in (technical, sentiment, news):
            if isinstance(result, Exception):
                raise result

        # Fundamental is optional - but only swallow rate-limit errors
        fundamental: FundamentalAnalysis | None = None
        if isinstance(fundamental_result, Exception):
            if self._is_rate_limit_error(fundamental_result):
                warning = f"Fundamental analysis unavailable: {fundamental_result}"
                logger.warning(warning)
                state["warnings"].append(warning)
            else:
                raise fundamental_result
        else:
            fundamental = fundamental_result

        # Comparative is optional - log warning and continue if it failed
        comparative = comparative_result if not isinstance(comparative_result, Exception) else None
        if isinstance(comparative_result, Exception):
            warning = f"Comparative analysis failed: {comparative_result}"
            logger.warning(warning)
            state["warnings"].append(warning)

        # Web research is optional - log warning and continue if it failed
        web_research = web_research_result if not isinstance(web_research_result, Exception) else None
        if isinstance(web_research_result, Exception):
            warning = f"Web research failed: {web_research_result}"
            logger.warning(warning)
            state["warnings"].append(warning)

        state["technical_analysis"] = technical
        state["sentiment_analysis"] = sentiment
        state["news_analysis"] = news
        state["fundamental_analysis"] = fundamental
        state["comparative_analysis"] = comparative
        # Web research stored for final result but not passed to downstream agents (informational only)
        state["web_research"] = web_research

        # Parallel Group 2: research (depends on Group 1)
        bullish_task = self.bullish_researcher.analyze(
            state["symbol"],
            state["technical_analysis"],
            state["sentiment_analysis"],
            state["news_analysis"],
            state["fundamental_analysis"],
            state["comparative_analysis"],
        )
        bearish_task = self.bearish_researcher.analyze(
            state["symbol"],
            state["technical_analysis"],
            state["sentiment_analysis"],
            state["news_analysis"],
            state["fundamental_analysis"],
            state["comparative_analysis"],
        )

        bullish, bearish = await asyncio.gather(bullish_task, bearish_task)
        state["bullish_research"] = bullish
        state["bearish_research"] = bearish

        return state

    def _fetch_data(self, symbol: str, period_days: int) -> TradingState:
        """Fetch market and news data.

        Args:
            symbol: Stock ticker
            period_days: Historical data period

        Returns:
            Updated state with data
        """
        logger.info("Fetching market and news data")

        market_data = self.market_fetcher.fetch_daily(symbol, period_days)

        news_articles = self.news_fetcher.fetch_company_news(symbol, limit=10)

        return TradingState(
            symbol=symbol,
            market_data=market_data.data,
            news_articles=news_articles,
            technical_analysis=None,
            sentiment_analysis=None,
            news_analysis=None,
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
            logger.info(f"Executed {action.value}: {state['symbol']} x{order.qty}")
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
        return f"TradingWorkflow(agents=9, mode={mode})"
