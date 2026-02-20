"""Model providers for DI container."""

from typing import TYPE_CHECKING

from src.daemon.config import DaemonConfig
from src.models.llm import LLMClient

if TYPE_CHECKING:
    from src.backtesting.runner import BacktestRunner
    from src.daemon.state import DaemonState
    from src.data.earnings import EarningsCalendarFetcher
    from src.data.market import MarketDataFetcher
    from src.data.news import NewsFetcher
    from src.data.universe import StockUniverseFetcher
    from src.data.websearch import WebSearchFetcher
    from src.database.engine import DatabaseEngine
    from src.di.container import AppContainer
    from src.metrics.execution import ExecutionMetricsCollector
    from src.metrics.portfolio_var import PortfolioVaRCalculator
    from src.metrics.quantstats_reporter import QuantStatsReporter
    from src.metrics.risk import RiskMetricsCalculator
    from src.metrics.tracker import BaseMetricsTracker
    from src.optimization.optimizer import OptunaOptimizer
    from src.screening.pre_market import PreMarketScreener
    from src.screening.screener import StockScreener
    from src.strategies.regime import MarketRegimeDetector
    from src.tools.registry import ToolRegistry
    from src.tools.websearch import WebSearchTool


def create_llm_client(
    daemon_config: DaemonConfig,
    metrics_collector: ExecutionMetricsCollector | None = None,
) -> LLMClient:
    """Create LLMClient with resolved config.

    Resolves provider/model from daemon_config.llm.* with env fallbacks.
    API key resolution by provider type (anthropic/openai/ollama).

    Args:
        daemon_config: Daemon configuration
        metrics_collector: Optional metrics collector for instrumentation

    Returns:
        Configured LLMClient
    """
    from src.di.config import resolve_config_or_env

    provider = daemon_config.llm.provider or "ollama"

    if provider == "anthropic":
        api_key = resolve_config_or_env(daemon_config.api_keys.anthropic_api_key, "ANTHROPIC_API_KEY")
    elif provider == "openai":
        api_key = resolve_config_or_env(daemon_config.api_keys.openai_api_key, "OPENAI_API_KEY")
    else:
        api_key = None

    cache_ttl = daemon_config.llm.response_cache_ttl_seconds
    llm_client = LLMClient(
        provider=provider,
        model=daemon_config.llm.model or "qwen3:14b",
        base_url=daemon_config.llm.ollama_base_url or "http://localhost:11434",
        api_key=api_key,
        openai_base_url=daemon_config.api_keys.openai_api_base,
        enable_prompt_caching=daemon_config.llm.enable_prompt_caching,
        cache_ttl=cache_ttl,
        cache_max_entries=daemon_config.llm.response_cache_max_entries,
    )

    if metrics_collector is not None:
        llm_client.set_metrics_collector(metrics_collector)

    return llm_client


def create_llm_client_for_agent(
    daemon_config: DaemonConfig,
    agent_name: str,
    metrics_collector: ExecutionMetricsCollector | None = None,
) -> LLMClient:
    """Create LLMClient with per-agent model override support.

    Checks daemon_config.llm.model_overrides for agent_name. If found,
    uses overridden model (and optionally provider). Otherwise falls back
    to default provider/model.

    Args:
        daemon_config: Daemon configuration
        agent_name: Agent identifier for override lookup
        metrics_collector: Optional metrics collector for instrumentation

    Returns:
        Configured LLMClient (possibly with different model/provider)
    """
    from loguru import logger

    from src.di.config import resolve_config_or_env

    override = daemon_config.llm.get_resolved_override(agent_name)
    if override:
        provider = override.provider or daemon_config.llm.provider
        model = override.model
        logger.debug(f"LLM override for {agent_name}: provider={provider}, model={model}")
    else:
        provider = daemon_config.llm.provider or "ollama"
        model = daemon_config.llm.model or "qwen3:14b"

    if provider == "anthropic":
        api_key = resolve_config_or_env(daemon_config.api_keys.anthropic_api_key, "ANTHROPIC_API_KEY")
    elif provider == "openai":
        api_key = resolve_config_or_env(daemon_config.api_keys.openai_api_key, "OPENAI_API_KEY")
    else:
        api_key = None

    cache_ttl = daemon_config.llm.response_cache_ttl_seconds
    llm_client = LLMClient(
        provider=provider,
        model=model,
        base_url=daemon_config.llm.ollama_base_url or "http://localhost:11434",
        api_key=api_key,
        openai_base_url=daemon_config.api_keys.openai_api_base,
        enable_prompt_caching=daemon_config.llm.enable_prompt_caching,
        cache_ttl=cache_ttl,
        cache_max_entries=daemon_config.llm.response_cache_max_entries,
    )

    if metrics_collector is not None:
        llm_client.set_metrics_collector(metrics_collector)

    return llm_client


def create_finbert_sentiment(
    daemon_config: DaemonConfig,
    device: str | None = None,
) -> object:
    """Create FinBERT sentiment analyzer (local or remote based on config).

    Respects daemon_config.finbert.mode to choose between local in-process
    model or remote microservice HTTP client. Maintains backward compatibility.

    Args:
        daemon_config: Daemon configuration
        device: Device for inference (cuda/cpu). Only used in local mode.

    Returns:
        FinBERTProtocol-compatible instance (FinBERTSentiment or FinBERTClient)
    """
    mode = daemon_config.finbert.mode

    if mode == "remote":
        from src.models.sentiment_client import FinBERTClient

        return FinBERTClient(
            base_url=daemon_config.finbert.service_url,
            timeout=daemon_config.finbert.timeout,
        )

    # Local mode: lazy import to avoid loading 440MB model on container creation
    from src.models.sentiment import get_finbert_sentiment

    return get_finbert_sentiment(device=device)


def create_risk_metrics_calculator() -> RiskMetricsCalculator:
    """Create RiskMetricsCalculator with lazy import.

    Returns:
        RiskMetricsCalculator singleton instance
    """
    from src.metrics.risk import RiskMetricsCalculator

    return RiskMetricsCalculator()


def create_portfolio_var_calculator(
    risk_calculator: RiskMetricsCalculator,
    market_fetcher: MarketDataFetcher,
) -> PortfolioVaRCalculator:
    """Create PortfolioVaRCalculator with dependencies.

    Args:
        risk_calculator: Risk metrics calculator
        market_fetcher: Market data fetcher for historical data

    Returns:
        PortfolioVaRCalculator instance
    """
    from src.metrics.portfolio_var import PortfolioVaRCalculator

    return PortfolioVaRCalculator(risk_calculator, market_fetcher)


def create_web_search_tool(websearch_fetcher: WebSearchFetcher) -> WebSearchTool:
    """Create WebSearchTool with websearch fetcher.

    Args:
        websearch_fetcher: WebSearchFetcher for executing searches

    Returns:
        WebSearchTool instance
    """
    from src.tools.websearch import WebSearchTool

    return WebSearchTool(websearch_fetcher)


def create_market_regime_detector() -> MarketRegimeDetector:
    """Create MarketRegimeDetector with lazy import.

    Returns:
        MarketRegimeDetector singleton instance
    """
    from src.strategies.regime import MarketRegimeDetector

    return MarketRegimeDetector()


def create_backtest_runner(cash: float = 100000.0, commission: float = 0.002) -> BacktestRunner:
    """Create BacktestRunner with lazy import.

    Args:
        cash: Initial cash balance
        commission: Commission rate (0.002 = 0.2%)

    Returns:
        BacktestRunner instance
    """
    from src.backtesting.runner import BacktestRunner

    return BacktestRunner(cash=cash, commission=commission)


def create_optuna_optimizer(n_trials: int = 100) -> OptunaOptimizer:
    """Create OptunaOptimizer with lazy import.

    Args:
        n_trials: Number of optimization trials

    Returns:
        OptunaOptimizer instance
    """
    from src.optimization.optimizer import OptunaOptimizer

    return OptunaOptimizer(n_trials=n_trials)


def create_metrics_tracker(
    daemon_config: DaemonConfig,
    database_engine: DatabaseEngine | None,
) -> BaseMetricsTracker:
    """Create appropriate metrics tracker with config.

    Returns DatabaseMetricsTracker if database engine provided and persistence enabled,
    otherwise returns JSONL-based MetricsTracker.

    Args:
        daemon_config: Daemon configuration
        database_engine: Optional database engine for database mode

    Returns:
        Appropriate metrics tracker instance
    """
    from loguru import logger

    from src.metrics.db_tracker import DatabaseMetricsTracker
    from src.metrics.tracker import MetricsTracker

    if daemon_config.database.enable_persistence and database_engine:
        logger.info("Using DatabaseMetricsTracker")
        return DatabaseMetricsTracker(
            database_engine=database_engine,
            risk_free_rate=daemon_config.metrics.risk_free_rate,
        )

    logger.info("Using JSONL MetricsTracker")
    return MetricsTracker(risk_free_rate=daemon_config.metrics.risk_free_rate)


def create_quantstats_reporter(daemon_config: DaemonConfig) -> QuantStatsReporter:
    """Create QuantStatsReporter with config.

    Args:
        daemon_config: Daemon configuration

    Returns:
        QuantStatsReporter instance
    """
    from src.metrics.quantstats_reporter import QuantStatsReporter

    return QuantStatsReporter(risk_free_rate=daemon_config.metrics.risk_free_rate)


def create_stock_screener(
    universe_fetcher: StockUniverseFetcher,
    daemon_config: DaemonConfig,
) -> StockScreener:
    """Create StockScreener with dependencies.

    Args:
        universe_fetcher: Stock universe fetcher
        daemon_config: Daemon configuration for liquidity filters

    Returns:
        StockScreener instance
    """
    from src.screening.screener import StockScreener

    return StockScreener(
        universe_fetcher=universe_fetcher,
        liquidity_filters=daemon_config.liquidity_filters,
    )


def create_pre_market_screener(
    universe_fetcher: StockUniverseFetcher,
    news_fetcher: NewsFetcher,
    earnings_fetcher: EarningsCalendarFetcher,
) -> PreMarketScreener:
    """Create PreMarketScreener with dependencies.

    Args:
        universe_fetcher: Stock universe fetcher
        news_fetcher: News fetcher
        earnings_fetcher: Earnings calendar fetcher

    Returns:
        PreMarketScreener instance
    """
    from src.screening.pre_market import PreMarketScreener

    return PreMarketScreener(
        universe_fetcher=universe_fetcher,
        news_fetcher=news_fetcher,
        earnings_fetcher=earnings_fetcher,
    )


def create_coordinator_tool_registry(
    container: AppContainer,
    daemon_state: DaemonState | None = None,
) -> ToolRegistry:
    """Create coordinator tool registry with DI container.

    Args:
        container: DI container for tool dependency resolution
        daemon_state: Optional daemon state for today's data access

    Returns:
        ToolRegistry with all coordinator tools registered
    """
    from src.v1.coordinator.memory import CoordinatorMemory
    from src.v1.coordinator.tools import build_coordinator_registry

    # Create memory with daemon_state if provided
    memory = None
    if daemon_state is not None:
        broker = container.alpaca_broker()

        # Get database engine for per-request repo creation (avoids session leaks)
        database_engine = None
        daemon_config = container.daemon_config()
        if daemon_config.database.enable_persistence:
            try:
                database_engine = container.database_engine()
            except Exception as e:
                from loguru import logger

                logger.opt(exception=True).warning(f"Failed to get database_engine for memory: {e}")

        memory = CoordinatorMemory(
            daemon_state=daemon_state,
            database_engine=database_engine,
            broker=broker,
        )

    return build_coordinator_registry(container, memory)
