"""Trading workflow orchestrating all agents."""

import pandas as pd
from loguru import logger
from pydantic import BaseModel
from typing_extensions import TypedDict

from src.agents.fundamental import FundamentalAnalysis, FundamentalAnalyst
from src.agents.news import NewsAnalysis, NewsAnalyst
from src.agents.risk import AccountInfo, RiskAssessment, RiskManagementAgent
from src.agents.sentiment import SentimentAnalysis, SentimentAnalyst
from src.agents.technical import TechnicalAnalysis, TechnicalAnalyst
from src.agents.trader import TraderAgent, TradingDecision
from src.data.broker import AlpacaBroker, OrderStatus
from src.data.fundamental import FundamentalDataFetcher
from src.data.market import MarketDataFetcher
from src.data.news import NewsArticle, NewsFetcher
from src.metrics.tracker import MetricsTracker
from src.models.llm import LLMClient
from src.models.sentiment import FinBERTSentiment
from src.strategies.momentum import MomentumStrategy, Signal


class TradingState(TypedDict):
    """State for trading workflow."""

    symbol: str
    market_data: pd.DataFrame | None
    news_articles: list[NewsArticle] | None
    technical_analysis: TechnicalAnalysis | None
    sentiment_analysis: SentimentAnalysis | None
    news_analysis: NewsAnalysis | None
    fundamental_analysis: FundamentalAnalysis | None
    final_decision: TradingDecision | None
    risk_assessment: RiskAssessment | None
    account_info: AccountInfo | None
    order_status: OrderStatus | None


class TradingWorkflowResult(BaseModel):
    """Complete trading analysis result."""

    symbol: str
    technical: TechnicalAnalysis
    sentiment: SentimentAnalysis
    news: NewsAnalysis
    fundamental: FundamentalAnalysis
    decision: TradingDecision
    risk: RiskAssessment
    order: OrderStatus | None = None

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


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
        metrics_tracker: MetricsTracker | None = None,
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
        """
        self.market_fetcher = market_fetcher
        self.news_fetcher = news_fetcher
        self.broker = broker
        self.metrics_tracker = metrics_tracker

        strategy = MomentumStrategy()

        self.technical_analyst = TechnicalAnalyst(llm_client, strategy)
        self.sentiment_analyst = SentimentAnalyst(finbert)
        self.news_analyst = NewsAnalyst(llm_client)
        self.fundamental_analyst = FundamentalAnalyst(llm_client, fundamental_fetcher)
        self.trader = TraderAgent(llm_client)
        self.risk_manager = RiskManagementAgent(llm_client)

        logger.info("Initialized TradingWorkflow with all agents")

    def analyze(self, symbol: str, period_days: int = 90) -> TradingWorkflowResult:
        """Run complete trading analysis.

        Args:
            symbol: Stock ticker symbol
            period_days: Days of historical data to fetch

        Returns:
            TradingWorkflowResult with all analyses and final decision
        """
        logger.info(f"Starting trading workflow for {symbol}")

        state = self._fetch_data(symbol, period_days)

        state = self._run_technical_analysis(state)

        state = self._run_sentiment_analysis(state)

        state = self._run_news_analysis(state)

        state = self._run_fundamental_analysis(state)

        state = self._make_decision(state)

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
            decision=state["final_decision"],
            risk=state["risk_assessment"],
            order=state.get("order_status"),
        )

        if self.metrics_tracker:
            try:
                self.metrics_tracker.record_decision(result)
            except Exception as e:
                logger.error(f"Failed to record metrics (continuing): {e}")

        return result

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
            final_decision=None,
            risk_assessment=None,
            account_info=None,
            order_status=None,
        )

    def _run_technical_analysis(self, state: TradingState) -> TradingState:
        """Run technical analysis.

        Args:
            state: Current workflow state

        Returns:
            Updated state with technical analysis
        """
        logger.info("Running technical analysis")

        technical = self.technical_analyst.analyze(state["symbol"], state["market_data"])

        state["technical_analysis"] = technical
        return state

    def _run_sentiment_analysis(self, state: TradingState) -> TradingState:
        """Run sentiment analysis.

        Args:
            state: Current workflow state

        Returns:
            Updated state with sentiment analysis
        """
        logger.info("Running sentiment analysis")

        sentiment = self.sentiment_analyst.analyze(state["symbol"], state["news_articles"])

        state["sentiment_analysis"] = sentiment
        return state

    def _run_news_analysis(self, state: TradingState) -> TradingState:
        """Run news analysis.

        Args:
            state: Current workflow state

        Returns:
            Updated state with news analysis
        """
        logger.info("Running news analysis")

        news = self.news_analyst.analyze(state["symbol"], state["news_articles"])

        state["news_analysis"] = news
        return state

    def _run_fundamental_analysis(self, state: TradingState) -> TradingState:
        """Run fundamental analysis.

        Args:
            state: Current workflow state

        Returns:
            Updated state with fundamental analysis
        """
        logger.info("Running fundamental analysis")

        current_price = float(state["market_data"]["Close"].iloc[-1])
        fundamental = self.fundamental_analyst.analyze(state["symbol"], current_price)

        state["fundamental_analysis"] = fundamental
        return state

    def _make_decision(self, state: TradingState) -> TradingState:
        """Make final trading decision.

        Args:
            state: Current workflow state

        Returns:
            Updated state with final decision
        """
        logger.info("Making final trading decision")

        decision = self.trader.decide(
            state["symbol"],
            state["technical_analysis"],
            state["sentiment_analysis"],
            state["news_analysis"],
            state["fundamental_analysis"],
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

        account_info = self._get_account_info()
        state["account_info"] = account_info

        current_price = float(state["market_data"]["Close"].iloc[-1])

        risk_assessment = self.risk_manager.assess(
            symbol=state["symbol"],
            action=state["final_decision"].action,
            current_price=current_price,
            account_info=account_info,
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

    def __repr__(self) -> str:
        """String representation."""
        return "TradingWorkflow(agents=6)"
