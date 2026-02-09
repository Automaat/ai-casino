"""Agent providers for DI container."""

from collections.abc import Callable
from typing import TYPE_CHECKING

from src.agents.meta import StrategyType
from src.daemon.config import DaemonConfig
from src.data.comparative import ComparativeDataFetcher
from src.data.finnhub import FinnhubFetcher
from src.data.fundamental import FundamentalDataFetcher
from src.data.market import MarketDataFetcher
from src.data.reddit import RedditFetcher
from src.models.llm import LLMClient

if TYPE_CHECKING:
    from src.agents.bearish_researcher import BearishResearcher
    from src.agents.bullish_researcher import BullishResearcher
    from src.agents.comparative import ComparativeAnalyst
    from src.agents.event_triage import EventTriageAgent
    from src.agents.fundamental import FundamentalAnalyst
    from src.agents.game_plan import GamePlanAgent
    from src.agents.journal import TradeJournalAgent
    from src.agents.meta import MetaAgent
    from src.agents.news import NewsAnalyst
    from src.agents.risk import RiskManagementAgent
    from src.agents.sentiment import SentimentAnalyst
    from src.agents.social import SocialSentimentAnalyst
    from src.agents.technical import TechnicalAnalyst
    from src.agents.trader import TraderAgent
    from src.agents.trump import TrumpAnalyst
    from src.agents.web_researcher import WebResearchAgent
    from src.metrics.portfolio_var import PortfolioVaRCalculator
    from src.models.sentiment import FinBERTSentiment
    from src.strategies.regime import MarketRegimeDetector
    from src.tools.websearch import WebSearchTool


def create_news_analyst(llm_client: LLMClient) -> "NewsAnalyst":
    """Create NewsAnalyst with LLM client.

    Args:
        llm_client: LLM client for news analysis

    Returns:
        Configured NewsAnalyst
    """
    from src.agents.news import NewsAnalyst

    return NewsAnalyst(llm_client)


def create_sentiment_analyst(finbert_sentiment: "FinBERTSentiment") -> "SentimentAnalyst":
    """Create SentimentAnalyst with FinBERT model.

    Args:
        finbert_sentiment: FinBERT sentiment analyzer

    Returns:
        Configured SentimentAnalyst
    """
    from src.agents.sentiment import SentimentAnalyst

    return SentimentAnalyst(finbert_sentiment)


def create_trump_analyst(llm_client: LLMClient) -> "TrumpAnalyst":
    """Create TrumpAnalyst with LLM client.

    Args:
        llm_client: LLM client for Trump post analysis

    Returns:
        Configured TrumpAnalyst
    """
    from src.agents.trump import TrumpAnalyst

    return TrumpAnalyst(llm_client)


def create_fundamental_analyst(
    llm_client: LLMClient,
    fundamental_fetcher: FundamentalDataFetcher,
) -> "FundamentalAnalyst":
    """Create FundamentalAnalyst with LLM client and data fetcher.

    Args:
        llm_client: LLM client for fundamental analysis
        fundamental_fetcher: Fundamental data fetcher

    Returns:
        Configured FundamentalAnalyst
    """
    from src.agents.fundamental import FundamentalAnalyst

    return FundamentalAnalyst(llm_client, fundamental_fetcher)


def create_social_sentiment_analyst(
    llm_client: LLMClient,
    finnhub_fetcher: FinnhubFetcher,
    reddit_fetcher: RedditFetcher,
    finbert_sentiment: "FinBERTSentiment",
) -> "SocialSentimentAnalyst":
    """Create SocialSentimentAnalyst with all dependencies.

    Args:
        llm_client: LLM client for interpretation
        finnhub_fetcher: Finnhub data fetcher
        reddit_fetcher: Reddit data fetcher
        finbert_sentiment: FinBERT sentiment analyzer

    Returns:
        Configured SocialSentimentAnalyst
    """
    from src.agents.social import SocialSentimentAnalyst

    return SocialSentimentAnalyst(llm_client, finnhub_fetcher, reddit_fetcher, finbert_sentiment)


def create_trader_agent(llm_client: LLMClient) -> "TraderAgent":
    """Create TraderAgent with LLM client.

    Args:
        llm_client: LLM client for trading decisions

    Returns:
        Configured TraderAgent
    """
    from src.agents.trader import TraderAgent

    return TraderAgent(llm_client)


def create_bullish_researcher(llm_client: LLMClient) -> "BullishResearcher":
    """Create BullishResearcher with LLM client.

    Args:
        llm_client: LLM client for bullish research

    Returns:
        Configured BullishResearcher
    """
    from src.agents.bullish_researcher import BullishResearcher

    return BullishResearcher(llm_client)


def create_bearish_researcher(llm_client: LLMClient) -> "BearishResearcher":
    """Create BearishResearcher with LLM client.

    Args:
        llm_client: LLM client for bearish research

    Returns:
        Configured BearishResearcher
    """
    from src.agents.bearish_researcher import BearishResearcher

    return BearishResearcher(llm_client)


def create_event_triage_agent(llm_client: LLMClient) -> "EventTriageAgent":
    """Create EventTriageAgent with LLM client.

    Args:
        llm_client: LLM client for event triage

    Returns:
        Configured EventTriageAgent
    """
    from src.agents.event_triage import EventTriageAgent

    return EventTriageAgent(llm_client)


def create_comparative_analyst(
    llm_client: LLMClient,
    comparative_fetcher: ComparativeDataFetcher,
) -> "ComparativeAnalyst":
    """Create ComparativeAnalyst with LLM client and data fetcher.

    Args:
        llm_client: LLM client for comparative analysis
        comparative_fetcher: Comparative data fetcher

    Returns:
        Configured ComparativeAnalyst
    """
    from src.agents.comparative import ComparativeAnalyst

    return ComparativeAnalyst(llm_client, comparative_fetcher)


def create_game_plan_agent(
    llm_client: LLMClient,
    market_fetcher: MarketDataFetcher,
) -> "GamePlanAgent":
    """Create GamePlanAgent with LLM client and market fetcher.

    Args:
        llm_client: LLM client for game plan generation
        market_fetcher: Market data fetcher

    Returns:
        Configured GamePlanAgent
    """
    from src.agents.game_plan import GamePlanAgent

    return GamePlanAgent(llm_client, market_fetcher)


def create_trade_journal_agent(
    llm_client: LLMClient,
    market_fetcher: MarketDataFetcher,
) -> "TradeJournalAgent":
    """Create TradeJournalAgent with LLM client and market fetcher.

    Args:
        llm_client: LLM client for trade journal analysis
        market_fetcher: Market data fetcher

    Returns:
        Configured TradeJournalAgent
    """
    from src.agents.journal import TradeJournalAgent

    return TradeJournalAgent(llm_client, market_fetcher)


def create_technical_analyst(llm_client: LLMClient) -> Callable[[StrategyType], "TechnicalAnalyst"]:
    """Returns factory accepting strategy parameter.

    TechnicalAnalyst requires a strategy selected by workflow logic (MetaAgent or default).
    Container provides factory that accepts strategy as parameter.

    Args:
        llm_client: LLM client for technical analysis

    Returns:
        Factory function accepting strategy and returning TechnicalAnalyst
    """
    from src.agents.technical import TechnicalAnalyst

    def factory(strategy: StrategyType) -> "TechnicalAnalyst":
        return TechnicalAnalyst(llm_client, strategy)

    return factory


def create_web_research_agent(llm_client: LLMClient, search_tool: "WebSearchTool") -> "WebResearchAgent":
    """Create WebResearchAgent with LLM client and search tool.

    Args:
        llm_client: LLM client for research analysis
        search_tool: Web search tool

    Returns:
        Configured WebResearchAgent
    """
    from src.agents.web_researcher import WebResearchAgent

    return WebResearchAgent(llm_client, search_tool)


def create_meta_agent(llm_client: LLMClient, regime_detector: "MarketRegimeDetector") -> "MetaAgent":
    """Create MetaAgent with LLM client and regime detector.

    Optional dependencies (metrics_tracker, param_store) passed as None.

    Args:
        llm_client: LLM client for meta-agent decisions
        regime_detector: Market regime detector

    Returns:
        Configured MetaAgent
    """
    from src.agents.meta import MetaAgent

    return MetaAgent(llm_client, regime_detector, metrics_tracker=None, param_store=None)


def create_risk_management_agent(
    llm_client: LLMClient,
    daemon_config: DaemonConfig,
    portfolio_var_calculator: "PortfolioVaRCalculator | None" = None,
) -> "RiskManagementAgent":
    """Create RiskManagementAgent with config extraction.

    Extracts position_sizing and portfolio_var configs from daemon_config.

    Args:
        llm_client: LLM client for risk analysis
        daemon_config: Daemon configuration
        portfolio_var_calculator: Optional PortfolioVaRCalculator

    Returns:
        Configured RiskManagementAgent
    """
    from src.agents.risk import PortfolioVaRConfig, RiskManagementAgent

    position_sizing_config = getattr(daemon_config, "position_sizing", None)
    portfolio_var_config = None
    risk_limits = getattr(daemon_config, "risk_limits", None)
    if risk_limits is not None:
        enabled = getattr(risk_limits, "enabled", True)
        if enabled:
            if hasattr(risk_limits, "model_dump"):
                portfolio_var_config = PortfolioVaRConfig(**risk_limits.model_dump())
            else:
                portfolio_var_config = PortfolioVaRConfig(**risk_limits)

    return RiskManagementAgent(
        llm_client,
        portfolio_var_calculator=portfolio_var_calculator,
        portfolio_var_config=portfolio_var_config,
        position_sizing_config=position_sizing_config,
    )
