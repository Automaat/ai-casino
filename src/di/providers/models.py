"""Model providers for DI container."""

import os
from typing import TYPE_CHECKING

from src.daemon.config import DaemonConfig
from src.di.config import resolve_config_or_env
from src.models.llm import LLMClient

if TYPE_CHECKING:
    from src.backtesting.runner import BacktestRunner
    from src.daemon.state import DaemonState
    from src.data.market import MarketDataFetcher
    from src.data.universe import StockUniverseFetcher
    from src.di.container import AppContainer
    from src.metrics.execution import ExecutionMetricsCollector
    from src.metrics.portfolio_var import PortfolioVaRCalculator
    from src.metrics.quantstats_reporter import QuantStatsReporter
    from src.metrics.risk import RiskMetricsCalculator
    from src.metrics.tracker import MetricsTracker
    from src.models.sentiment import FinBERTSentiment
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
    provider = daemon_config.llm.provider or os.getenv("LLM_PROVIDER", "ollama")

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
    else:
        api_key = None

    llm_client = LLMClient(
        provider=daemon_config.llm.provider,
        model=daemon_config.llm.model,
        api_key=api_key,
        openai_base_url=resolve_config_or_env(
            daemon_config.api_keys.openai_api_base,
            "OPENAI_API_BASE",
        ),
    )

    if metrics_collector is not None:
        llm_client.set_metrics_collector(metrics_collector)

    return llm_client


def create_finbert_sentiment(device: str | None = None) -> FinBERTSentiment:
    """Create FinBERT sentiment analyzer with lazy import.

    Thin wrapper over existing get_finbert_sentiment() factory.
    Maintains singleton behavior via existing implementation.
    Uses lazy import to avoid loading 440MB model on container creation.

    Args:
        device: Device for inference (cuda/cpu). Auto-detect if None.

    Returns:
        FinBERTSentiment singleton instance
    """
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


def create_metrics_tracker(risk_free_rate: float | None = None) -> MetricsTracker:
    """Create MetricsTracker with lazy import.

    Args:
        risk_free_rate: Annual risk-free rate for Sharpe ratio (default from env or 0.02)

    Returns:
        MetricsTracker instance
    """
    from src.metrics.tracker import MetricsTracker

    return MetricsTracker(risk_free_rate=risk_free_rate)


def create_quantstats_reporter(risk_free_rate: float | None = None) -> QuantStatsReporter:
    """Create QuantStatsReporter with lazy import.

    Args:
        risk_free_rate: Annual risk-free rate (default from env or 0.02)

    Returns:
        QuantStatsReporter instance
    """
    from src.metrics.quantstats_reporter import QuantStatsReporter

    return QuantStatsReporter(risk_free_rate=risk_free_rate)


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
        daemon_state: Optional daemon state for analysis history tool

    Returns:
        ToolRegistry with all coordinator tools registered
    """
    from src.coordinator.tools import build_coordinator_registry

    return build_coordinator_registry(container, daemon_state)
