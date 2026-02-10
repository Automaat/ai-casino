"""Workflow providers for dependency injection.

This module provides factory functions for creating TradingWorkflow instances
with different configurations (meta, momentum, trump, full).
"""

from typing import TYPE_CHECKING

# ruff: noqa: PLR0913
from src.agents.risk import PortfolioVaRConfig
from src.cache.historical import HistoricalCache
from src.daemon.config import DaemonConfig
from src.daemon.notifications import NotificationService
from src.data.broker import AlpacaBroker
from src.data.fundamental import FundamentalDataFetcher
from src.data.market import MarketDataFetcher
from src.data.news import NewsFetcher
from src.database.repositories.snapshot import PortfolioSnapshotRepository
from src.metrics.portfolio_var import PortfolioVaRCalculator
from src.metrics.tracker import MetricsTracker
from src.models.llm import LLMClient
from src.models.sentiment import FinBERTSentiment
from src.optimization.param_store import OptimizedParamStore
from src.workflows import TradingWorkflow

if TYPE_CHECKING:
    from src.di.container import AppContainer


def _extract_portfolio_var_config(daemon_config: DaemonConfig) -> PortfolioVaRConfig | None:
    """Extract PortfolioVaRConfig from daemon risk_limits config."""
    risk_config = daemon_config.risk_limits
    if not risk_config.enabled:
        return None

    return PortfolioVaRConfig(
        enabled=risk_config.enabled,
        max_var_95=risk_config.max_var_95,
        max_cvar_99=risk_config.max_cvar_99,
        lookback_days=risk_config.lookback_days,
        adaptive_stop_loss=risk_config.adaptive_stop_loss,
        cdar_stop_threshold=risk_config.cdar_stop_threshold,
        atr_multiplier_min=risk_config.atr_multiplier_min,
    )


def create_workflow_meta(
    llm_client: LLMClient,
    market_fetcher: MarketDataFetcher,
    news_fetcher: NewsFetcher,
    finbert_sentiment: FinBERTSentiment,
    fundamental_fetcher: FundamentalDataFetcher,
    historical_cache: HistoricalCache,
    portfolio_var_calculator: PortfolioVaRCalculator,
    daemon_config: DaemonConfig,
    container: "AppContainer",
    broker: AlpacaBroker | None = None,
    metrics_tracker: MetricsTracker | None = None,
    param_store: OptimizedParamStore | None = None,
    snapshot_repository: PortfolioSnapshotRepository | None = None,
    notification_service: NotificationService | None = None,
) -> TradingWorkflow:
    """Create TradingWorkflow with meta-agent enabled."""
    portfolio_var_config = _extract_portfolio_var_config(daemon_config)
    pre_trade_backtest_config = daemon_config.pre_trade_backtesting
    position_sizing_config = daemon_config.position_sizing

    return TradingWorkflow(
        llm_client,
        market_fetcher,
        news_fetcher,
        finbert_sentiment,
        fundamental_fetcher,
        broker=broker,
        metrics_tracker=metrics_tracker,
        use_ensemble=False,
        use_meta_agent=True,
        trump_mode=False,
        snapshot_on_trade=None,
        snapshot_repository=snapshot_repository,
        param_store=param_store,
        historical_cache=historical_cache,
        portfolio_var_calculator=portfolio_var_calculator,
        portfolio_var_config=portfolio_var_config,
        pre_trade_backtest_config=pre_trade_backtest_config,
        notification_service=notification_service,
        position_sizing_config=position_sizing_config,
        finnhub_fetcher=container.finnhub_fetcher() if container else None,
        container=container,
    )


def create_workflow_momentum(
    llm_client: LLMClient,
    market_fetcher: MarketDataFetcher,
    news_fetcher: NewsFetcher,
    finbert_sentiment: FinBERTSentiment,
    fundamental_fetcher: FundamentalDataFetcher,
    historical_cache: HistoricalCache,
    portfolio_var_calculator: PortfolioVaRCalculator,
    daemon_config: DaemonConfig,
    container: "AppContainer",
    broker: AlpacaBroker | None = None,
    metrics_tracker: MetricsTracker | None = None,
    param_store: OptimizedParamStore | None = None,
    snapshot_repository: PortfolioSnapshotRepository | None = None,
    notification_service: NotificationService | None = None,
) -> TradingWorkflow:
    """Create TradingWorkflow with momentum strategy only."""
    portfolio_var_config = _extract_portfolio_var_config(daemon_config)
    pre_trade_backtest_config = daemon_config.pre_trade_backtesting
    position_sizing_config = daemon_config.position_sizing

    return TradingWorkflow(
        llm_client,
        market_fetcher,
        news_fetcher,
        finbert_sentiment,
        fundamental_fetcher,
        broker=broker,
        metrics_tracker=metrics_tracker,
        use_ensemble=False,
        use_meta_agent=False,
        trump_mode=False,
        snapshot_on_trade=None,
        snapshot_repository=snapshot_repository,
        param_store=param_store,
        historical_cache=historical_cache,
        portfolio_var_calculator=portfolio_var_calculator,
        portfolio_var_config=portfolio_var_config,
        pre_trade_backtest_config=pre_trade_backtest_config,
        notification_service=notification_service,
        position_sizing_config=position_sizing_config,
        finnhub_fetcher=container.finnhub_fetcher() if container else None,
        container=container,
    )


def create_workflow_trump(
    llm_client: LLMClient,
    market_fetcher: MarketDataFetcher,
    news_fetcher: NewsFetcher,
    finbert_sentiment: FinBERTSentiment,
    fundamental_fetcher: FundamentalDataFetcher,
    historical_cache: HistoricalCache,
    portfolio_var_calculator: PortfolioVaRCalculator,
    daemon_config: DaemonConfig,
    container: "AppContainer",
    broker: AlpacaBroker | None = None,
    metrics_tracker: MetricsTracker | None = None,
    param_store: OptimizedParamStore | None = None,
    snapshot_repository: PortfolioSnapshotRepository | None = None,
    notification_service: NotificationService | None = None,
) -> TradingWorkflow:
    """Create TradingWorkflow with meta-agent and Trump mode enabled."""
    portfolio_var_config = _extract_portfolio_var_config(daemon_config)
    pre_trade_backtest_config = daemon_config.pre_trade_backtesting
    position_sizing_config = daemon_config.position_sizing

    return TradingWorkflow(
        llm_client,
        market_fetcher,
        news_fetcher,
        finbert_sentiment,
        fundamental_fetcher,
        broker=broker,
        metrics_tracker=metrics_tracker,
        use_ensemble=False,
        use_meta_agent=True,
        trump_mode=True,
        snapshot_on_trade=None,
        snapshot_repository=snapshot_repository,
        param_store=param_store,
        historical_cache=historical_cache,
        portfolio_var_calculator=portfolio_var_calculator,
        portfolio_var_config=portfolio_var_config,
        pre_trade_backtest_config=pre_trade_backtest_config,
        notification_service=notification_service,
        position_sizing_config=position_sizing_config,
        finnhub_fetcher=container.finnhub_fetcher() if container else None,
        container=container,
    )


def create_workflow_full(
    llm_client: LLMClient,
    market_fetcher: MarketDataFetcher,
    news_fetcher: NewsFetcher,
    finbert_sentiment: FinBERTSentiment,
    fundamental_fetcher: FundamentalDataFetcher,
    historical_cache: HistoricalCache,
    portfolio_var_calculator: PortfolioVaRCalculator,
    daemon_config: DaemonConfig,
    container: "AppContainer",
    broker: AlpacaBroker | None = None,
    metrics_tracker: MetricsTracker | None = None,
    param_store: OptimizedParamStore | None = None,
    snapshot_repository: PortfolioSnapshotRepository | None = None,
    notification_service: NotificationService | None = None,
) -> TradingWorkflow:
    """Create TradingWorkflow with all features enabled (alias for trump)."""
    return create_workflow_trump(
        llm_client,
        market_fetcher,
        news_fetcher,
        finbert_sentiment,
        fundamental_fetcher,
        historical_cache,
        portfolio_var_calculator,
        daemon_config,
        container,
        broker=broker,
        metrics_tracker=metrics_tracker,
        param_store=param_store,
        snapshot_repository=snapshot_repository,
        notification_service=notification_service,
    )
