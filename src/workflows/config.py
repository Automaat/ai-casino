"""Workflow configuration and components."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from src.daemon.config import PreTradeBacktestingConfig

if TYPE_CHECKING:
    from src.agents.risk import PortfolioVaRConfig
    from src.cache.historical import HistoricalCache
    from src.daemon.config import AnalysisOrchestratorConfig, PositionSizingConfig, RiskValidationConfig
    from src.daemon.event_bus import EventBus
    from src.data.broker import AlpacaBroker
    from src.data.fundamental import FundamentalDataFetcher
    from src.data.market import MarketDataFetcher
    from src.data.news import NewsFetcher
    from src.database.repositories.execution_metric import ExecutionMetricRepository
    from src.database.repositories.snapshot import PortfolioSnapshotRepository
    from src.di.container import AppContainer
    from src.metrics.portfolio_var import PortfolioVaRCalculator
    from src.metrics.tracker import BaseMetricsTracker
    from src.models.llm import LLMClient
    from src.models.sentiment import FinBERTSentiment
    from src.optimization.param_store import OptimizedParamStore
    from src.v1.notifications.service import NotificationService
    from src.v1.trades.service import TradingService
    from src.validators.risk import RiskValidator


class WorkflowConfig(BaseModel):
    """Configuration for TradingWorkflow behavior."""

    use_ensemble: bool = Field(default=False, description="Use ensemble strategy")
    use_meta_agent: bool = Field(default=True, description="Use meta-agent for strategy selection")
    trump_mode: bool = Field(default=False, description="Enable Trump social media analysis")
    snapshot_on_trade: bool | None = Field(
        default=None, description="Capture portfolio snapshot after trades"
    )
    execution_metrics_enabled: bool = Field(default=True, description="Enable execution metrics collection")
    pre_trade_backtest_config: PreTradeBacktestingConfig | None = Field(
        default=None, description="Pre-trade backtesting configuration"
    )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"WorkflowConfig(use_ensemble={self.use_ensemble}, "
            f"use_meta_agent={self.use_meta_agent}, trump_mode={self.trump_mode})"
        )


@dataclass
class WorkflowComponents:
    """Dependencies injected into TradingWorkflow."""

    # Required components
    llm_client: LLMClient
    market_fetcher: MarketDataFetcher
    news_fetcher: NewsFetcher
    finbert: FinBERTSentiment
    fundamental_fetcher: FundamentalDataFetcher
    container: AppContainer

    # Optional components
    broker: AlpacaBroker | None = None
    metrics_tracker: BaseMetricsTracker | None = None
    snapshot_repository: PortfolioSnapshotRepository | None = None
    execution_metric_repository: ExecutionMetricRepository | None = None
    param_store: OptimizedParamStore | None = None
    historical_cache: HistoricalCache | None = None
    portfolio_var_calculator: PortfolioVaRCalculator | None = None
    portfolio_var_config: PortfolioVaRConfig | None = None
    notification_service: NotificationService | None = None
    position_sizing_config: PositionSizingConfig | None = None
    risk_validation_config: RiskValidationConfig | None = None
    risk_validator: RiskValidator | None = None
    analysis_orchestrator_config: AnalysisOrchestratorConfig | None = None
    web_search_fetcher: object | None = None  # WebSearchFetcher (avoid circular import)
    event_bus: EventBus | None = None
    trading_service: TradingService | None = None

    def __repr__(self) -> str:
        """String representation."""
        optional_count = sum(
            [
                self.broker is not None,
                self.metrics_tracker is not None,
                self.snapshot_repository is not None,
                self.execution_metric_repository is not None,
                self.param_store is not None,
                self.historical_cache is not None,
                self.portfolio_var_calculator is not None,
                self.portfolio_var_config is not None,
                self.notification_service is not None,
                self.position_sizing_config is not None,
                self.risk_validation_config is not None,
                self.risk_validator is not None,
                self.analysis_orchestrator_config is not None,
                self.event_bus is not None,
                self.trading_service is not None,
            ]
        )
        return f"WorkflowComponents(required=6, optional={optional_count})"
