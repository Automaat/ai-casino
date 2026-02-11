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
    from src.agents.base_researcher import ResearchDirection
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
    from src.agents.thesis_researcher import ThesisResearcher
    from src.agents.trader import TraderAgent
    from src.agents.trump import TrumpAnalyst
    from src.agents.web_researcher import WebResearchAgent
    from src.coordinator.agent import TradingCoordinator
    from src.coordinator.confirmation import TradeConfirmationHandler
    from src.coordinator.pattern_analyzer import PatternAnalyzer
    from src.daemon.notification_channels import TelegramChannel
    from src.daemon.state import DaemonState
    from src.di.container import AppContainer
    from src.metrics.portfolio_var import PortfolioVaRCalculator
    from src.models.sentiment import FinBERTSentiment
    from src.strategies.regime import MarketRegimeDetector
    from src.tools.websearch import WebSearchTool


def create_news_analyst(llm_client: LLMClient) -> NewsAnalyst:
    """Create NewsAnalyst with LLM client.

    Args:
        llm_client: LLM client for news analysis

    Returns:
        Configured NewsAnalyst
    """
    from src.agents.news import NewsAnalyst

    return NewsAnalyst(llm_client)


def create_sentiment_analyst(finbert_sentiment: FinBERTSentiment) -> SentimentAnalyst:
    """Create SentimentAnalyst with FinBERT model.

    Args:
        finbert_sentiment: FinBERT sentiment analyzer

    Returns:
        Configured SentimentAnalyst
    """
    from src.agents.sentiment import SentimentAnalyst

    return SentimentAnalyst(finbert_sentiment)


def create_trump_analyst(llm_client: LLMClient) -> TrumpAnalyst:
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
) -> FundamentalAnalyst:
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
    finbert_sentiment: FinBERTSentiment,
) -> SocialSentimentAnalyst:
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


def create_trader_agent(llm_client: LLMClient) -> TraderAgent:
    """Create TraderAgent with LLM client.

    Args:
        llm_client: LLM client for trading decisions

    Returns:
        Configured TraderAgent
    """
    from src.agents.trader import TraderAgent

    return TraderAgent(llm_client)


def create_thesis_researcher(llm_client: LLMClient, direction: ResearchDirection) -> ThesisResearcher:
    """Create ThesisResearcher with direction.

    Args:
        llm_client: LLM client for thesis research
        direction: Research direction (BULLISH or BEARISH)

    Returns:
        Configured ThesisResearcher
    """
    from src.agents.thesis_researcher import ThesisResearcher

    return ThesisResearcher(llm_client, direction)


def create_event_triage_agent(llm_client: LLMClient) -> EventTriageAgent:
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
) -> ComparativeAnalyst:
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
) -> GamePlanAgent:
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
) -> TradeJournalAgent:
    """Create TradeJournalAgent with LLM client and market fetcher.

    Args:
        llm_client: LLM client for trade journal analysis
        market_fetcher: Market data fetcher

    Returns:
        Configured TradeJournalAgent
    """
    from src.agents.journal import TradeJournalAgent

    return TradeJournalAgent(llm_client, market_fetcher)


def create_technical_analyst(llm_client: LLMClient) -> Callable[[StrategyType], TechnicalAnalyst]:
    """Returns factory accepting strategy parameter.

    TechnicalAnalyst requires a strategy selected by workflow logic (MetaAgent or default).
    Container provides factory that accepts strategy as parameter.

    Args:
        llm_client: LLM client for technical analysis

    Returns:
        Factory function accepting strategy and returning TechnicalAnalyst
    """
    from src.agents.technical import TechnicalAnalyst

    def factory(strategy: StrategyType) -> TechnicalAnalyst:
        return TechnicalAnalyst(llm_client, strategy)

    return factory


def create_web_research_agent(llm_client: LLMClient, search_tool: WebSearchTool) -> WebResearchAgent:
    """Create WebResearchAgent with LLM client and search tool.

    Args:
        llm_client: LLM client for research analysis
        search_tool: Web search tool

    Returns:
        Configured WebResearchAgent
    """
    from src.agents.web_researcher import WebResearchAgent

    return WebResearchAgent(llm_client, search_tool)


def create_meta_agent(llm_client: LLMClient, regime_detector: MarketRegimeDetector) -> MetaAgent:
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
    portfolio_var_calculator: PortfolioVaRCalculator | None = None,
) -> RiskManagementAgent:
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


def create_confirmation_handler(
    daemon_config: DaemonConfig,
    telegram_channel: TelegramChannel | None = None,
) -> TradeConfirmationHandler | None:
    """Create confirmation handler if Telegram enabled.

    Args:
        daemon_config: Daemon configuration
        telegram_channel: Optional Telegram channel

    Returns:
        TradeConfirmationHandler if Telegram configured, None otherwise
    """
    if not telegram_channel or not telegram_channel.is_configured():
        return None

    from src.coordinator.confirmation import TradeConfirmationHandler

    timeout = daemon_config.coordinator.approval_timeout_seconds
    return TradeConfirmationHandler(
        telegram_channel=telegram_channel,
        approval_timeout_seconds=timeout,
    )


def create_pattern_analyzer(
    daemon_config: DaemonConfig,
    container: AppContainer,
) -> PatternAnalyzer | None:
    """Create pattern analyzer if enabled.

    Args:
        daemon_config: Daemon configuration
        container: DI container for repositories

    Returns:
        PatternAnalyzer if pattern detection enabled, None otherwise
    """
    from src.coordinator.pattern_analyzer import PatternAnalyzer

    if not daemon_config.coordinator.pattern_detection.enabled:
        return None

    # Get repositories from container
    analysis_repo = container.analysis_repository()
    trade_repo = container.trade_repository()

    # Memory will be injected at runtime (not available at container build time)
    # Pass None here and set it later when coordinator is created
    min_sample_size = daemon_config.coordinator.pattern_detection.min_sample_size

    return PatternAnalyzer(
        analysis_repo=analysis_repo,
        trade_repo=trade_repo,
        memory=None,  # type: ignore[arg-type]  # Will be set by coordinator
        min_sample_size=min_sample_size,
    )


def create_trading_coordinator(
    llm_client: LLMClient,
    daemon_config: DaemonConfig,
    container: AppContainer,
    daemon_state: DaemonState | None = None,
) -> TradingCoordinator:
    """Create TradingCoordinator with all dependencies.

    If coordinator.model_override is set, creates dedicated LLM client with that model.
    Otherwise uses default llm_client parameter.

    Args:
        llm_client: LLM client for tool calling
        daemon_config: Daemon config for coordinator settings
        container: DI container for tool registry
        daemon_state: Optional daemon state for today's data access

    Returns:
        Configured TradingCoordinator
    """
    import os

    from src.coordinator.agent import TradingCoordinator
    from src.coordinator.memory import CoordinatorMemory
    from src.coordinator.tools import build_coordinator_registry
    from src.di.config import resolve_config_or_env

    # Get dependencies for enhanced memory
    broker = container.alpaca_broker()
    analysis_repo = container.analysis_repository()

    # Create enhanced memory with multi-tier context
    memory = CoordinatorMemory(
        daemon_state=daemon_state,
        analysis_repo=analysis_repo,
        broker=broker,
    )

    # Build tool registry with enhanced memory
    tool_registry = build_coordinator_registry(container, memory)

    # Extract coordinator config
    coordinator_config = daemon_config.coordinator

    # Apply model override if configured
    if coordinator_config.model_override:
        # Resolve API keys same way as create_llm_client
        provider = daemon_config.llm.provider or os.getenv("LLM_PROVIDER", "ollama")
        api_key = None
        if provider == "anthropic":
            api_key = resolve_config_or_env(
                daemon_config.api_keys.anthropic_api_key,
                "ANTHROPIC_API_KEY",
            )
        elif provider == "openai":
            api_key = resolve_config_or_env(
                daemon_config.api_keys.openai_api_key,
                "OPENAI_API_KEY",
            )

        coordinator_llm = LLMClient(
            provider=provider,
            model=coordinator_config.model_override,
            api_key=api_key,
        )
    else:
        coordinator_llm = llm_client

    return TradingCoordinator(
        llm_client=coordinator_llm,
        tool_registry=tool_registry,
        memory=memory,
        config=coordinator_config,
        broker=broker,
    )
