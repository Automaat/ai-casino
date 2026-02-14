"""Model providers for DI container."""

from typing import TYPE_CHECKING

from src.daemon.config import DaemonConfig
from src.models.llm import LLMClient

if TYPE_CHECKING:
    from src.backtesting.runner import BacktestRunner
    from src.daemon.state import DaemonState
    from src.data.market import MarketDataFetcher
    from src.data.universe import StockUniverseFetcher
    from src.database.repositories.trade import TradeRepository
    from src.di.container import AppContainer
    from src.metrics.execution import ExecutionMetricsCollector
    from src.metrics.portfolio_var import PortfolioVaRCalculator
    from src.metrics.quantstats_reporter import QuantStatsReporter
    from src.metrics.risk import RiskMetricsCalculator
    from src.metrics.tracker import BaseMetricsTracker
    from src.optimization.optimizer import OptunaOptimizer
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

    llm_client = LLMClient(
        provider=provider,
        model=daemon_config.llm.model or "qwen3:14b",
        base_url=daemon_config.llm.ollama_base_url or "http://localhost:11434",
        api_key=api_key,
        openai_base_url=daemon_config.api_keys.openai_api_base,
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


def create_web_search_tool(container: AppContainer) -> WebSearchTool:
    """Create WebSearchTool with DI container.

    Args:
        container: DI container for dependency resolution

    Returns:
        WebSearchTool instance
    """
    from src.tools.websearch import WebSearchTool

    return WebSearchTool(container)


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
    trade_repository: TradeRepository | None,
) -> BaseMetricsTracker:
    """Create appropriate metrics tracker with config.

    Returns DatabaseMetricsTracker if repository provided and persistence enabled,
    otherwise returns JSONL-based MetricsTracker.

    Args:
        daemon_config: Daemon configuration
        trade_repository: Optional trade repository for database mode

    Returns:
        Appropriate metrics tracker instance
    """
    from loguru import logger

    from src.metrics.db_tracker import DatabaseMetricsTracker
    from src.metrics.tracker import MetricsTracker

    if daemon_config.database.enable_persistence and trade_repository:
        logger.info("Using DatabaseMetricsTracker")
        return DatabaseMetricsTracker(
            trade_repository=trade_repository,
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
    from src.coordinator.memory import CoordinatorMemory
    from src.coordinator.tools import build_coordinator_registry

    # Create memory with daemon_state if provided
    memory = None
    if daemon_state is not None:
        broker = container.alpaca_broker()
        analysis_repo = container.analysis_repository()

        # Get signal outcome repository if database enabled
        signal_outcome_repo = None
        daemon_config = container.daemon_config()
        if daemon_config.database.enable_persistence:
            try:
                signal_outcome_repo = container.signal_outcome_repository()
            except Exception as e:
                from loguru import logger

                logger.opt(exception=True).warning(
                    f"Failed to create signal_outcome_repository for memory: {e}"
                )

        memory = CoordinatorMemory(
            daemon_state=daemon_state,
            analysis_repo=analysis_repo,
            signal_outcome_repo=signal_outcome_repo,
            broker=broker,
        )

    return build_coordinator_registry(container, memory)
