"""Configuration for the trading daemon."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

# Re-export all config classes (backward compatibility)
from src.circuit_breaker.models import CircuitBreakerConfig
from src.coordinator import CoordinatorConfig
from src.coordinator.models import AdaptiveThresholdConfig, PatternDetectionConfig
from src.daemon.config.analysis import (
    AnalysisOrchestratorConfig,
    AnomalyWatcherConfig,
    FilingsWatcherConfig,
    NewsSourcesConfig,
    NewsWatcherConfig,
    SocialWatcherConfig,
)
from src.daemon.config.base import NotificationTrigger, TradingMode
from src.daemon.config.infrastructure import (
    ApiConfig,
    ApiKeysConfig,
    DatabaseConfig,
    DataSourcesConfig,
    FinBERTConfig,
    LLMConfig,
    PrefetchConfig,
)
from src.daemon.config.logging import LoggingConfig
from src.daemon.config.notifications import NotificationsConfig, TelegramNotificationConfig
from src.daemon.config.portfolio import (
    CorrelationAuditConfig,
    GamePlanConfig,
    PeerAnalysisConfig,
    PortfolioRebalancingConfig,
)
from src.daemon.config.profiling import ProfilingConfig
from src.daemon.config.reporting import HealthConfig, MetricsConfig, ReportingConfig, SignalTrackingConfig
from src.daemon.config.risk import (
    MonteCarloConfig,
    PositionManagementConfig,
    PositionSizingConfig,
    PreTradeBacktestingConfig,
    RiskLimitsConfig,
)
from src.daemon.config.screening import (
    DiscoveryConfig,
    EarningsCalendarConfig,
    LiquidityFilterConfig,
    ScreeningConfig,
    SectorRotationConfig,
)
from src.daemon.config.trading import (
    JournalConfig,
    OptimizationConfig,
    PaperTradingConfig,
    ScheduleConfig,
    StateConfig,
)

__all__ = [
    "AnalysisOrchestratorConfig",
    "AnomalyWatcherConfig",
    "ApiConfig",
    "ApiKeysConfig",
    "CoordinatorConfig",
    "CorrelationAuditConfig",
    "DaemonConfig",
    "DataSourcesConfig",
    "DatabaseConfig",
    "DiscoveryConfig",
    "EarningsCalendarConfig",
    "FilingsWatcherConfig",
    "FinBERTConfig",
    "GamePlanConfig",
    "HealthConfig",
    "JournalConfig",
    "LLMConfig",
    "LiquidityFilterConfig",
    "LoggingConfig",
    "MetricsConfig",
    "MonteCarloConfig",
    "NewsWatcherConfig",
    "NotificationTrigger",
    "NotificationsConfig",
    "OptimizationConfig",
    "PaperTradingConfig",
    "PeerAnalysisConfig",
    "PortfolioRebalancingConfig",
    "PositionManagementConfig",
    "PositionSizingConfig",
    "PreTradeBacktestingConfig",
    "PrefetchConfig",
    "ProfilingConfig",
    "ReportingConfig",
    "RiskLimitsConfig",
    "ScheduleConfig",
    "ScreeningConfig",
    "SectorRotationConfig",
    "SignalTrackingConfig",
    "SocialWatcherConfig",
    "StateConfig",
    "TelegramNotificationConfig",
    "TradingMode",
]


class DaemonConfig(BaseModel):
    """Configuration for the trading daemon."""

    watchlist: list[str] = Field(default_factory=lambda: ["AAPL", "TSLA", "GOOGL", "MSFT"])
    interval_minutes: int = 30
    market_hours_only: bool = True
    auto_trade: bool = False
    max_concurrent_analyses: int = 3
    trading_mode: TradingMode = TradingMode.PAPER
    paper_trading: PaperTradingConfig = Field(default_factory=PaperTradingConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    state: StateConfig = Field(default_factory=StateConfig)
    journal: JournalConfig = Field(default_factory=JournalConfig)
    health: HealthConfig = Field(default_factory=HealthConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    screening: ScreeningConfig = Field(default_factory=ScreeningConfig)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    liquidity_filters: LiquidityFilterConfig = Field(default_factory=LiquidityFilterConfig)
    prefetch: PrefetchConfig = Field(default_factory=PrefetchConfig)
    sector_rotation: SectorRotationConfig = Field(default_factory=SectorRotationConfig)
    earnings_calendar: EarningsCalendarConfig = Field(default_factory=EarningsCalendarConfig)
    peer_analysis: PeerAnalysisConfig = Field(default_factory=PeerAnalysisConfig)
    correlation_audit: CorrelationAuditConfig = Field(default_factory=CorrelationAuditConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    risk_limits: RiskLimitsConfig = Field(default_factory=RiskLimitsConfig)
    rebalancing: PortfolioRebalancingConfig = Field(default_factory=PortfolioRebalancingConfig)
    signal_tracking: SignalTrackingConfig = Field(default_factory=SignalTrackingConfig)
    pre_trade_backtesting: PreTradeBacktestingConfig = Field(default_factory=PreTradeBacktestingConfig)
    game_plan: GamePlanConfig = Field(default_factory=GamePlanConfig)
    position_sizing: PositionSizingConfig = Field(default_factory=PositionSizingConfig)
    position_management: PositionManagementConfig = Field(default_factory=PositionManagementConfig)
    monte_carlo: MonteCarloConfig = Field(default_factory=MonteCarloConfig)
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)
    analysis_orchestration: AnalysisOrchestratorConfig = Field(default_factory=AnalysisOrchestratorConfig)
    news_watcher: NewsWatcherConfig = Field(default_factory=NewsWatcherConfig)
    social_watcher: SocialWatcherConfig = Field(default_factory=SocialWatcherConfig)
    filings_watcher: FilingsWatcherConfig = Field(default_factory=FilingsWatcherConfig)
    anomaly_watcher: AnomalyWatcherConfig = Field(default_factory=AnomalyWatcherConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    finbert: FinBERTConfig = Field(default_factory=FinBERTConfig)
    api_keys: ApiKeysConfig = Field(default_factory=ApiKeysConfig)
    data_sources: DataSourcesConfig = Field(default_factory=DataSourcesConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    coordinator: CoordinatorConfig = Field(default_factory=CoordinatorConfig)
    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> DaemonConfig:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML config file

        Returns:
            DaemonConfig instance
        """
        with path.open("r") as f:
            data = yaml.safe_load(f)

        daemon_data = data.get("daemon", {})

        paper_trading_data = daemon_data.pop("paper_trading", {}) or {}
        schedule_data = daemon_data.pop("schedule", {}) or {}
        state_data = daemon_data.pop("state", {}) or {}
        journal_data = daemon_data.pop("journal", {}) or {}
        health_data = daemon_data.pop("health", {}) or {}
        optimization_data = daemon_data.pop("optimization", {}) or {}
        screening_data = daemon_data.pop("screening", {}) or {}
        discovery_data = daemon_data.pop("discovery", {}) or {}
        liquidity_filters_data = daemon_data.pop("liquidity_filters", {}) or {}
        prefetch_data = daemon_data.pop("prefetch", {}) or {}
        sector_rotation_data = daemon_data.pop("sector_rotation", {}) or {}
        earnings_calendar_data = daemon_data.pop("earnings_calendar", {}) or {}
        peer_analysis_data = daemon_data.pop("peer_analysis", {}) or {}
        correlation_audit_data = daemon_data.pop("correlation_audit", {}) or {}
        reporting_data = daemon_data.pop("reporting", {}) or {}
        risk_limits_data = daemon_data.pop("risk_limits", {}) or {}
        rebalancing_data = daemon_data.pop("rebalancing", {}) or {}
        signal_tracking_data = daemon_data.pop("signal_tracking", {}) or {}
        pre_trade_backtesting_data = daemon_data.pop("pre_trade_backtesting", {}) or {}
        game_plan_data = daemon_data.pop("game_plan", {}) or {}
        position_sizing_data = daemon_data.pop("position_sizing", {}) or {}
        position_management_data = daemon_data.pop("position_management", {}) or {}
        monte_carlo_data = daemon_data.pop("monte_carlo", {}) or {}
        notifications_data = daemon_data.pop("notifications", {}) or {}
        analysis_orchestration_data = daemon_data.pop("analysis_orchestration", {}) or {}
        news_watcher_data = daemon_data.pop("news_watcher", {}) or {}
        social_watcher_data = daemon_data.pop("social_watcher", {}) or {}
        filings_watcher_data = daemon_data.pop("filings_watcher", {}) or {}
        anomaly_watcher_data = daemon_data.pop("anomaly_watcher", {}) or {}
        api_data = daemon_data.pop("api", {}) or {}
        llm_data = daemon_data.pop("llm", {}) or {}
        finbert_data = daemon_data.pop("finbert", {}) or {}
        api_keys_data = daemon_data.pop("api_keys", {}) or {}
        data_sources_data = daemon_data.pop("data_sources", {}) or {}
        database_data = daemon_data.pop("database", {}) or {}
        coordinator_data = daemon_data.pop("coordinator", {}) or {}
        profiling_data = daemon_data.pop("profiling", {}) or {}
        metrics_data = daemon_data.pop("metrics", {}) or {}
        logging_data = daemon_data.pop("logging", {}) or {}

        # Extract nested telegram config from notifications
        telegram_data = notifications_data.pop("telegram", {}) or {}

        # Extract nested sources config from news_watcher
        news_sources_data = news_watcher_data.pop("sources", {}) or {}

        # Extract nested circuit_breaker config from api
        circuit_breaker_data = api_data.pop("circuit_breaker", {}) or {}

        # Extract nested adaptive_thresholds config from coordinator
        adaptive_thresholds_data = coordinator_data.pop("adaptive_thresholds", {}) or {}

        # Extract nested pattern_detection config from coordinator
        pattern_detection_data = coordinator_data.pop("pattern_detection", {}) or {}

        return cls(
            **daemon_data,
            paper_trading=PaperTradingConfig(**paper_trading_data),
            schedule=ScheduleConfig(**schedule_data),
            state=StateConfig(**state_data),
            journal=JournalConfig(**journal_data),
            health=HealthConfig(**health_data),
            optimization=OptimizationConfig(**optimization_data),
            screening=ScreeningConfig(**screening_data),
            discovery=DiscoveryConfig(**discovery_data),
            liquidity_filters=LiquidityFilterConfig(**liquidity_filters_data),
            prefetch=PrefetchConfig(**prefetch_data),
            sector_rotation=SectorRotationConfig(**sector_rotation_data),
            earnings_calendar=EarningsCalendarConfig(**earnings_calendar_data),
            peer_analysis=PeerAnalysisConfig(**peer_analysis_data),
            correlation_audit=CorrelationAuditConfig(**correlation_audit_data),
            reporting=ReportingConfig(**reporting_data),
            risk_limits=RiskLimitsConfig(**risk_limits_data),
            rebalancing=PortfolioRebalancingConfig(**rebalancing_data),
            signal_tracking=SignalTrackingConfig(**signal_tracking_data),
            pre_trade_backtesting=PreTradeBacktestingConfig(**pre_trade_backtesting_data),
            game_plan=GamePlanConfig(**game_plan_data),
            position_sizing=PositionSizingConfig(**position_sizing_data),
            position_management=PositionManagementConfig(**position_management_data),
            monte_carlo=MonteCarloConfig(**monte_carlo_data),
            notifications=NotificationsConfig(
                **notifications_data, telegram=TelegramNotificationConfig(**telegram_data)
            ),
            analysis_orchestration=AnalysisOrchestratorConfig(**analysis_orchestration_data),
            news_watcher=NewsWatcherConfig(
                **news_watcher_data, sources=NewsSourcesConfig(**news_sources_data)
            ),
            social_watcher=SocialWatcherConfig(**social_watcher_data),
            filings_watcher=FilingsWatcherConfig(**filings_watcher_data),
            anomaly_watcher=AnomalyWatcherConfig(**anomaly_watcher_data),
            api=ApiConfig(**api_data, circuit_breaker=CircuitBreakerConfig(**circuit_breaker_data)),
            llm=LLMConfig(**llm_data),
            finbert=FinBERTConfig(**finbert_data),
            api_keys=ApiKeysConfig(**api_keys_data),
            data_sources=DataSourcesConfig(**data_sources_data),
            database=DatabaseConfig(**database_data),
            coordinator=CoordinatorConfig(
                **coordinator_data,
                adaptive_thresholds=AdaptiveThresholdConfig(**adaptive_thresholds_data),
                pattern_detection=PatternDetectionConfig(**pattern_detection_data),
            ),
            profiling=ProfilingConfig(**profiling_data),
            metrics=MetricsConfig(**metrics_data),
            logging=LoggingConfig(**logging_data),
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"DaemonConfig(watchlist={self.watchlist}, "
            f"interval={self.interval_minutes}m, auto_trade={self.auto_trade})"
        )
