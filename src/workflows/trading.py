"""Trading workflow orchestrating all agents."""

import pandas as pd
from loguru import logger
from pydantic import BaseModel
from typing_extensions import TypedDict

from src.agents.news import NewsAnalysis, NewsAnalyst
from src.agents.risk import AccountInfo, RiskAssessment, RiskManagementAgent
from src.agents.sentiment import SentimentAnalysis, SentimentAnalyst
from src.agents.technical import TechnicalAnalysis, TechnicalAnalyst
from src.agents.trader import TraderAgent, TradingDecision
from src.data.market import MarketDataFetcher
from src.data.news import NewsArticle, NewsFetcher
from src.models.llm import LLMClient
from src.models.sentiment import FinBERTSentiment
from src.strategies.momentum import MomentumStrategy


class TradingState(TypedDict):
    """State for trading workflow."""

    symbol: str
    market_data: pd.DataFrame | None
    news_articles: list[NewsArticle] | None
    technical_analysis: TechnicalAnalysis | None
    sentiment_analysis: SentimentAnalysis | None
    news_analysis: NewsAnalysis | None
    final_decision: TradingDecision | None
    risk_assessment: RiskAssessment | None
    account_info: AccountInfo | None


class TradingWorkflowResult(BaseModel):
    """Complete trading analysis result."""

    symbol: str
    technical: TechnicalAnalysis
    sentiment: SentimentAnalysis
    news: NewsAnalysis
    decision: TradingDecision
    risk: RiskAssessment

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


class TradingWorkflow:
    """Orchestrate multi-agent trading analysis."""

    def __init__(
        self,
        llm_client: LLMClient,
        market_fetcher: MarketDataFetcher,
        news_fetcher: NewsFetcher,
        finbert: FinBERTSentiment,
    ) -> None:
        """Initialize trading workflow.

        Args:
            llm_client: LLM client for agents
            market_fetcher: Market data fetcher
            news_fetcher: News data fetcher
            finbert: FinBERT sentiment model
        """
        self.market_fetcher = market_fetcher
        self.news_fetcher = news_fetcher

        strategy = MomentumStrategy()

        self.technical_analyst = TechnicalAnalyst(llm_client, strategy)
        self.sentiment_analyst = SentimentAnalyst(finbert)
        self.news_analyst = NewsAnalyst(llm_client)
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

        state = self._make_decision(state)

        state = self._assess_risk(state)

        logger.info(
            f"Workflow complete: {state['final_decision'].action.value} "
            f"(confidence={state['final_decision'].confidence:.2f}, "
            f"risk_approved={state['risk_assessment'].validation.approved})"
        )

        return TradingWorkflowResult(
            symbol=symbol,
            technical=state["technical_analysis"],
            sentiment=state["sentiment_analysis"],
            news=state["news_analysis"],
            decision=state["final_decision"],
            risk=state["risk_assessment"],
        )

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
            final_decision=None,
            risk_assessment=None,
            account_info=None,
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
            AccountInfo with mocked data
        """
        return AccountInfo(
            balance=100000.0,
            available_cash=100000.0,
            positions={},
            total_exposure=0.0,
        )

    def __repr__(self) -> str:
        """String representation."""
        return "TradingWorkflow(agents=5)"
