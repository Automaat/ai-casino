"""Workflow providers for dependency injection.

This module provides factory functions for creating TradingWorkflow instances
with different configurations (meta, momentum, trump, full).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.agents.risk import PortfolioVaRConfig
from src.cache.historical import HistoricalCache
from src.daemon.config import DaemonConfig
from src.data.broker import AlpacaBroker
from src.data.fundamental import FundamentalDataFetcher
from src.data.market import MarketDataFetcher
from src.data.news import NewsFetcher
from src.database.repositories.execution_metric import ExecutionMetricRepository
from src.database.repositories.snapshot import PortfolioSnapshotRepository
from src.metrics.portfolio_var import PortfolioVaRCalculator
from src.metrics.tracker import BaseMetricsTracker
from src.models.llm import LLMClient
from src.models.sentiment import FinBERTSentiment
from src.optimization.param_store import OptimizedParamStore
from src.v1.notifications.service import NotificationService
from src.workflows import TradingWorkflow
from src.workflows.config import WorkflowComponents, WorkflowConfig

if TYPE_CHECKING:
    from src.daemon.event_bus import EventBus
    from src.di.container import AppContainer
    from src.v1.trades.service import TradingService
    from src.validators.risk import RiskValidator


@dataclass
class WorkflowFactoryParams:
    """Parameters for workflow factory functions."""

    llm_client: LLMClient
    market_fetcher: MarketDataFetcher
    news_fetcher: NewsFetcher
    finbert_sentiment: FinBERTSentiment
    fundamental_fetcher: FundamentalDataFetcher
    historical_cache: HistoricalCache
    portfolio_var_calculator: PortfolioVaRCalculator
    daemon_config: DaemonConfig
    container: Any  # AppContainer, typed as Any for flexibility
    broker: AlpacaBroker | None = None
    metrics_tracker: BaseMetricsTracker | None = None
    param_store: OptimizedParamStore | None = None
    snapshot_repository: PortfolioSnapshotRepository | None = None
    execution_metric_repository: ExecutionMetricRepository | None = None
    notification_service: NotificationService | None = None
    web_search_fetcher: object | None = None  # WebSearchFetcher (avoid circular import)
    event_bus: EventBus | None = None


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


def _create_risk_validator(daemon_config: DaemonConfig) -> RiskValidator:
    """Create RiskValidator from daemon risk_validation config."""
    from src.validators.risk import RiskValidator

    return RiskValidator(daemon_config.risk_validation)


def _create_trading_service(params: WorkflowFactoryParams) -> TradingService | None:
    """Create TradingService if broker available."""
    if not params.broker:
        return None

    from src.v1.trades.service import TradingService

    database_engine = None
    if params.daemon_config.database.enable_persistence:
        with contextlib.suppress(Exception):
            database_engine = params.container.database_engine()

    return TradingService(
        broker=params.broker,
        daemon_config=params.daemon_config,
        database_engine=database_engine,
        notification_service=params.notification_service,
    )


def create_workflow_meta(params: WorkflowFactoryParams) -> TradingWorkflow:
    """Create TradingWorkflow with meta-agent enabled."""
    portfolio_var_config = _extract_portfolio_var_config(params.daemon_config)
    pre_trade_backtest_config = params.daemon_config.pre_trade_backtesting
    position_sizing_config = params.daemon_config.position_sizing
    risk_validation_config = params.daemon_config.risk_validation
    risk_validator = _create_risk_validator(params.daemon_config)
    analysis_orchestrator_config = params.daemon_config.analysis_orchestration

    config = WorkflowConfig(
        use_ensemble=False,
        use_meta_agent=True,
        trump_mode=False,
        snapshot_on_trade=None,
        pre_trade_backtest_config=pre_trade_backtest_config,
    )

    components = WorkflowComponents(
        llm_client=params.llm_client,
        market_fetcher=params.market_fetcher,
        news_fetcher=params.news_fetcher,
        finbert=params.finbert_sentiment,
        fundamental_fetcher=params.fundamental_fetcher,
        container=params.container,
        broker=params.broker,
        metrics_tracker=params.metrics_tracker,
        snapshot_repository=params.snapshot_repository,
        execution_metric_repository=params.execution_metric_repository,
        param_store=params.param_store,
        historical_cache=params.historical_cache,
        portfolio_var_calculator=params.portfolio_var_calculator,
        portfolio_var_config=portfolio_var_config,
        notification_service=params.notification_service,
        position_sizing_config=position_sizing_config,
        risk_validation_config=risk_validation_config,
        risk_validator=risk_validator,
        analysis_orchestrator_config=analysis_orchestrator_config,
        web_search_fetcher=params.web_search_fetcher,
        event_bus=params.event_bus,
        trading_service=_create_trading_service(params),
    )

    return TradingWorkflow(config, components)


def create_workflow_momentum(params: WorkflowFactoryParams) -> TradingWorkflow:
    """Create TradingWorkflow with momentum strategy only."""
    portfolio_var_config = _extract_portfolio_var_config(params.daemon_config)
    pre_trade_backtest_config = params.daemon_config.pre_trade_backtesting
    position_sizing_config = params.daemon_config.position_sizing
    risk_validation_config = params.daemon_config.risk_validation
    risk_validator = _create_risk_validator(params.daemon_config)
    analysis_orchestrator_config = params.daemon_config.analysis_orchestration

    config = WorkflowConfig(
        use_ensemble=False,
        use_meta_agent=False,
        trump_mode=False,
        snapshot_on_trade=None,
        pre_trade_backtest_config=pre_trade_backtest_config,
    )

    components = WorkflowComponents(
        llm_client=params.llm_client,
        market_fetcher=params.market_fetcher,
        news_fetcher=params.news_fetcher,
        finbert=params.finbert_sentiment,
        fundamental_fetcher=params.fundamental_fetcher,
        container=params.container,
        broker=params.broker,
        metrics_tracker=params.metrics_tracker,
        snapshot_repository=params.snapshot_repository,
        execution_metric_repository=params.execution_metric_repository,
        param_store=params.param_store,
        historical_cache=params.historical_cache,
        portfolio_var_calculator=params.portfolio_var_calculator,
        portfolio_var_config=portfolio_var_config,
        notification_service=params.notification_service,
        position_sizing_config=position_sizing_config,
        risk_validation_config=risk_validation_config,
        risk_validator=risk_validator,
        analysis_orchestrator_config=analysis_orchestrator_config,
        web_search_fetcher=params.web_search_fetcher,
        event_bus=params.event_bus,
        trading_service=_create_trading_service(params),
    )

    return TradingWorkflow(config, components)


def create_workflow_trump(params: WorkflowFactoryParams) -> TradingWorkflow:
    """Create TradingWorkflow with meta-agent and Trump mode enabled."""
    portfolio_var_config = _extract_portfolio_var_config(params.daemon_config)
    pre_trade_backtest_config = params.daemon_config.pre_trade_backtesting
    position_sizing_config = params.daemon_config.position_sizing
    risk_validation_config = params.daemon_config.risk_validation
    risk_validator = _create_risk_validator(params.daemon_config)
    analysis_orchestrator_config = params.daemon_config.analysis_orchestration

    config = WorkflowConfig(
        use_ensemble=False,
        use_meta_agent=True,
        trump_mode=True,
        snapshot_on_trade=None,
        pre_trade_backtest_config=pre_trade_backtest_config,
    )

    components = WorkflowComponents(
        llm_client=params.llm_client,
        market_fetcher=params.market_fetcher,
        news_fetcher=params.news_fetcher,
        finbert=params.finbert_sentiment,
        fundamental_fetcher=params.fundamental_fetcher,
        container=params.container,
        broker=params.broker,
        metrics_tracker=params.metrics_tracker,
        snapshot_repository=params.snapshot_repository,
        execution_metric_repository=params.execution_metric_repository,
        param_store=params.param_store,
        historical_cache=params.historical_cache,
        portfolio_var_calculator=params.portfolio_var_calculator,
        portfolio_var_config=portfolio_var_config,
        notification_service=params.notification_service,
        position_sizing_config=position_sizing_config,
        risk_validation_config=risk_validation_config,
        risk_validator=risk_validator,
        analysis_orchestrator_config=analysis_orchestrator_config,
        web_search_fetcher=params.web_search_fetcher,
        event_bus=params.event_bus,
        trading_service=_create_trading_service(params),
    )

    return TradingWorkflow(config, components)


def create_workflow_full(params: WorkflowFactoryParams) -> TradingWorkflow:
    """Create TradingWorkflow with all features enabled (alias for trump)."""
    return create_workflow_trump(params)


# Wrapper factories for DI container (merge core deps with runtime params)
# These wrappers exist solely for backward compatibility with dependency-injector Factory provider pattern
# The core factories (create_workflow_*) use proper parameter objects


def create_workflow_meta_wrapper(  # noqa: PLR0913 - DI adapter, delegates to clean factory
    llm_client: LLMClient,
    market_fetcher: MarketDataFetcher,
    news_fetcher: NewsFetcher,
    finbert_sentiment: FinBERTSentiment,
    fundamental_fetcher: FundamentalDataFetcher,
    historical_cache: HistoricalCache,
    portfolio_var_calculator: PortfolioVaRCalculator,
    daemon_config: DaemonConfig,
    container: AppContainer | None = None,
    broker: AlpacaBroker | None = None,
    metrics_tracker: BaseMetricsTracker | None = None,
    param_store: OptimizedParamStore | None = None,
    snapshot_repository: PortfolioSnapshotRepository | None = None,
    notification_service: NotificationService | None = None,
    web_search_fetcher: object | None = None,  # WebSearchFetcher (avoid circular import)
    event_bus: EventBus | None = None,
) -> TradingWorkflow:
    """Wrapper for create_workflow_meta that accepts individual parameters."""
    if container is None:
        msg = "container parameter is required"
        raise ValueError(msg)

    params = WorkflowFactoryParams(
        llm_client=llm_client,
        market_fetcher=market_fetcher,
        news_fetcher=news_fetcher,
        finbert_sentiment=finbert_sentiment,
        fundamental_fetcher=fundamental_fetcher,
        historical_cache=historical_cache,
        portfolio_var_calculator=portfolio_var_calculator,
        daemon_config=daemon_config,
        container=container,
        broker=broker,
        metrics_tracker=metrics_tracker,
        param_store=param_store,
        snapshot_repository=snapshot_repository,
        notification_service=notification_service,
        web_search_fetcher=web_search_fetcher,
        event_bus=event_bus,
    )
    return create_workflow_meta(params)


def create_workflow_momentum_wrapper(  # noqa: PLR0913 - DI adapter, delegates to clean factory
    llm_client: LLMClient,
    market_fetcher: MarketDataFetcher,
    news_fetcher: NewsFetcher,
    finbert_sentiment: FinBERTSentiment,
    fundamental_fetcher: FundamentalDataFetcher,
    historical_cache: HistoricalCache,
    portfolio_var_calculator: PortfolioVaRCalculator,
    daemon_config: DaemonConfig,
    container: AppContainer | None = None,
    broker: AlpacaBroker | None = None,
    metrics_tracker: BaseMetricsTracker | None = None,
    param_store: OptimizedParamStore | None = None,
    snapshot_repository: PortfolioSnapshotRepository | None = None,
    notification_service: NotificationService | None = None,
    web_search_fetcher: object | None = None,  # WebSearchFetcher (avoid circular import)
) -> TradingWorkflow:
    """Wrapper for create_workflow_momentum that accepts individual parameters."""
    if container is None:
        msg = "container parameter is required"
        raise ValueError(msg)

    params = WorkflowFactoryParams(
        llm_client=llm_client,
        market_fetcher=market_fetcher,
        news_fetcher=news_fetcher,
        finbert_sentiment=finbert_sentiment,
        fundamental_fetcher=fundamental_fetcher,
        historical_cache=historical_cache,
        portfolio_var_calculator=portfolio_var_calculator,
        daemon_config=daemon_config,
        container=container,
        broker=broker,
        metrics_tracker=metrics_tracker,
        param_store=param_store,
        snapshot_repository=snapshot_repository,
        notification_service=notification_service,
        web_search_fetcher=web_search_fetcher,
    )
    return create_workflow_momentum(params)


def create_workflow_trump_wrapper(  # noqa: PLR0913 - DI adapter, delegates to clean factory
    llm_client: LLMClient,
    market_fetcher: MarketDataFetcher,
    news_fetcher: NewsFetcher,
    finbert_sentiment: FinBERTSentiment,
    fundamental_fetcher: FundamentalDataFetcher,
    historical_cache: HistoricalCache,
    portfolio_var_calculator: PortfolioVaRCalculator,
    daemon_config: DaemonConfig,
    container: AppContainer | None = None,
    broker: AlpacaBroker | None = None,
    metrics_tracker: BaseMetricsTracker | None = None,
    param_store: OptimizedParamStore | None = None,
    snapshot_repository: PortfolioSnapshotRepository | None = None,
    notification_service: NotificationService | None = None,
    web_search_fetcher: object | None = None,  # WebSearchFetcher (avoid circular import)
) -> TradingWorkflow:
    """Wrapper for create_workflow_trump that accepts individual parameters."""
    if container is None:
        msg = "container parameter is required"
        raise ValueError(msg)

    params = WorkflowFactoryParams(
        llm_client=llm_client,
        market_fetcher=market_fetcher,
        news_fetcher=news_fetcher,
        finbert_sentiment=finbert_sentiment,
        fundamental_fetcher=fundamental_fetcher,
        historical_cache=historical_cache,
        portfolio_var_calculator=portfolio_var_calculator,
        daemon_config=daemon_config,
        container=container,
        broker=broker,
        metrics_tracker=metrics_tracker,
        param_store=param_store,
        snapshot_repository=snapshot_repository,
        notification_service=notification_service,
        web_search_fetcher=web_search_fetcher,
    )
    return create_workflow_trump(params)


def create_workflow_full_wrapper(  # noqa: PLR0913 - DI adapter, delegates to clean factory
    llm_client: LLMClient,
    market_fetcher: MarketDataFetcher,
    news_fetcher: NewsFetcher,
    finbert_sentiment: FinBERTSentiment,
    fundamental_fetcher: FundamentalDataFetcher,
    historical_cache: HistoricalCache,
    portfolio_var_calculator: PortfolioVaRCalculator,
    daemon_config: DaemonConfig,
    container: AppContainer | None = None,
    broker: AlpacaBroker | None = None,
    metrics_tracker: BaseMetricsTracker | None = None,
    param_store: OptimizedParamStore | None = None,
    snapshot_repository: PortfolioSnapshotRepository | None = None,
    notification_service: NotificationService | None = None,
    web_search_fetcher: object | None = None,  # WebSearchFetcher (avoid circular import)
) -> TradingWorkflow:
    """Wrapper for create_workflow_full that accepts individual parameters."""
    if container is None:
        msg = "container parameter is required"
        raise ValueError(msg)

    params = WorkflowFactoryParams(
        llm_client=llm_client,
        market_fetcher=market_fetcher,
        news_fetcher=news_fetcher,
        finbert_sentiment=finbert_sentiment,
        fundamental_fetcher=fundamental_fetcher,
        historical_cache=historical_cache,
        portfolio_var_calculator=portfolio_var_calculator,
        daemon_config=daemon_config,
        container=container,
        broker=broker,
        metrics_tracker=metrics_tracker,
        param_store=param_store,
        snapshot_repository=snapshot_repository,
        notification_service=notification_service,
        web_search_fetcher=web_search_fetcher,
    )
    return create_workflow_full(params)
