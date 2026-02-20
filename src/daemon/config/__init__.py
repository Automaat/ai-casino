"""Configuration for the trading daemon."""

from pathlib import Path

import yaml
from loguru import logger
from pydantic import BaseModel, Field, PrivateAttr

# Re-export all config classes (backward compatibility)
from src.circuit_breaker.models import CircuitBreakerConfig
from src.daemon.config.analysis import (
    AnalysisOrchestratorConfig,
    AnomalyWatcherConfig,
    EconomicCalendarWatcherConfig,
    FilingsWatcherConfig,
    NewsSourcesConfig,
    NewsTrendingWatcherConfig,
    NewsWatcherConfig,
    OptionsFlowWatcherConfig,
    SocialSentimentWatcherConfig,
    SocialWatcherConfig,
    TrumpWatcherConfig,
)
from src.daemon.config.base import TradingMode
from src.daemon.config.events import EventWatcherIntegrationConfig
from src.daemon.config.infrastructure import (
    ApiConfig,
    ApiKeysConfig,
    DatabaseConfig,
    DataSourcesConfig,
    FinBERTConfig,
    FinnhubSourcesConfig,
    LLMConfig,
    PrefetchConfig,
)
from src.daemon.config.logging import LoggingConfig
from src.daemon.config.notifications import NotificationsConfig, TelegramNotificationConfig
from src.daemon.config.portfolio import (
    CorrelationAuditConfig,
    GamePlanConfig,
    PeerAnalysisConfig,
    PortfolioHealthConfig,
    PortfolioRebalancingConfig,
)
from src.daemon.config.pre_market import PreMarketScreeningConfig
from src.daemon.config.profiling import ProfilingConfig
from src.daemon.config.reddit import RedditScraperConfig
from src.daemon.config.reporting import HealthConfig, MetricsConfig, ReportingConfig, SignalTrackingConfig
from src.daemon.config.risk import (
    MonteCarloConfig,
    PositionCircuitBreakerConfig,
    PositionManagementConfig,
    PositionSizingConfig,
    PreTradeBacktestingConfig,
    RiskLimitsConfig,
)
from src.daemon.config.risk_validation import RiskValidationConfig
from src.daemon.config.screening import (
    DiscoveryConfig,
    EarningsCalendarConfig,
    LiquidityFilterConfig,
    SectorRotationConfig,
)
from src.daemon.config.trading import (
    JournalConfig,
    OptimizationConfig,
    PaperTradingConfig,
    ScheduleConfig,
    StateConfig,
)
from src.daemon.config.workflow import WorkflowConfigDaemon
from src.v1.coordinator import CoordinatorConfig
from src.v1.coordinator.models import AdaptiveThresholdConfig, PatternDetectionConfig, SweepPassConfig

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
    "EconomicCalendarWatcherConfig",
    "EventWatcherIntegrationConfig",
    "FilingsWatcherConfig",
    "FinBERTConfig",
    "FinnhubSourcesConfig",
    "GamePlanConfig",
    "HealthConfig",
    "JournalConfig",
    "LLMConfig",
    "LiquidityFilterConfig",
    "LoggingConfig",
    "MetricsConfig",
    "MonteCarloConfig",
    "NewsTrendingWatcherConfig",
    "NewsWatcherConfig",
    "NotificationsConfig",
    "OptimizationConfig",
    "OptionsFlowWatcherConfig",
    "PaperTradingConfig",
    "PeerAnalysisConfig",
    "PortfolioHealthConfig",
    "PortfolioRebalancingConfig",
    "PositionCircuitBreakerConfig",
    "PositionManagementConfig",
    "PositionSizingConfig",
    "PreMarketScreeningConfig",
    "PreTradeBacktestingConfig",
    "PrefetchConfig",
    "ProfilingConfig",
    "ReportingConfig",
    "RiskLimitsConfig",
    "RiskValidationConfig",
    "ScheduleConfig",
    "SectorRotationConfig",
    "SignalTrackingConfig",
    "SocialSentimentWatcherConfig",
    "SocialWatcherConfig",
    "StateConfig",
    "SweepPassConfig",
    "TelegramNotificationConfig",
    "TradingMode",
    "TrumpWatcherConfig",
    "WorkflowConfigDaemon",
]


class DaemonConfig(BaseModel):
    """Configuration for the trading daemon."""

    _config_path: Path | None = PrivateAttr(default=None)

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
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    liquidity_filters: LiquidityFilterConfig = Field(default_factory=LiquidityFilterConfig)
    prefetch: PrefetchConfig = Field(default_factory=PrefetchConfig)
    sector_rotation: SectorRotationConfig = Field(default_factory=SectorRotationConfig)
    earnings_calendar: EarningsCalendarConfig = Field(default_factory=EarningsCalendarConfig)
    peer_analysis: PeerAnalysisConfig = Field(default_factory=PeerAnalysisConfig)
    correlation_audit: CorrelationAuditConfig = Field(default_factory=CorrelationAuditConfig)
    portfolio_health: PortfolioHealthConfig = Field(default_factory=PortfolioHealthConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    risk_limits: RiskLimitsConfig = Field(default_factory=RiskLimitsConfig)
    rebalancing: PortfolioRebalancingConfig = Field(default_factory=PortfolioRebalancingConfig)
    signal_tracking: SignalTrackingConfig = Field(default_factory=SignalTrackingConfig)
    pre_trade_backtesting: PreTradeBacktestingConfig = Field(default_factory=PreTradeBacktestingConfig)
    game_plan: GamePlanConfig = Field(default_factory=GamePlanConfig)
    position_sizing: PositionSizingConfig = Field(default_factory=PositionSizingConfig)
    position_management: PositionManagementConfig = Field(default_factory=PositionManagementConfig)
    monte_carlo: MonteCarloConfig = Field(default_factory=MonteCarloConfig)
    pre_market: PreMarketScreeningConfig = Field(default_factory=PreMarketScreeningConfig)
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)
    analysis_orchestration: AnalysisOrchestratorConfig = Field(default_factory=AnalysisOrchestratorConfig)
    news_watcher: NewsWatcherConfig = Field(default_factory=NewsWatcherConfig)
    social_watcher: SocialWatcherConfig = Field(default_factory=SocialWatcherConfig)
    reddit_scraper: RedditScraperConfig = Field(default_factory=RedditScraperConfig)
    trump_watcher: TrumpWatcherConfig = Field(default_factory=TrumpWatcherConfig)
    filings_watcher: FilingsWatcherConfig = Field(default_factory=FilingsWatcherConfig)
    anomaly_watcher: AnomalyWatcherConfig = Field(default_factory=AnomalyWatcherConfig)
    news_trending_watcher: NewsTrendingWatcherConfig = Field(default_factory=NewsTrendingWatcherConfig)
    economic_calendar_watcher: EconomicCalendarWatcherConfig = Field(
        default_factory=EconomicCalendarWatcherConfig
    )
    options_flow_watcher: OptionsFlowWatcherConfig = Field(default_factory=OptionsFlowWatcherConfig)
    social_sentiment_watcher: SocialSentimentWatcherConfig = Field(
        default_factory=SocialSentimentWatcherConfig
    )
    event_integration: EventWatcherIntegrationConfig = Field(default_factory=EventWatcherIntegrationConfig)
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
    risk_validation: RiskValidationConfig = Field(default_factory=RiskValidationConfig)
    workflow: WorkflowConfigDaemon = Field(default_factory=WorkflowConfigDaemon)

    @staticmethod
    def _extract_config_sections(daemon_data: dict) -> dict[str, dict]:
        """Extract all config sections from daemon data.

        Args:
            daemon_data: Raw daemon configuration dict

        Returns:
            Dict mapping section names to their config data
        """
        # Extract top-level sections
        sections = {
            "paper_trading": daemon_data.pop("paper_trading", {}) or {},
            "schedule": daemon_data.pop("schedule", {}) or {},
            "state": daemon_data.pop("state", {}) or {},
            "journal": daemon_data.pop("journal", {}) or {},
            "health": daemon_data.pop("health", {}) or {},
            "optimization": daemon_data.pop("optimization", {}) or {},
            "discovery": daemon_data.pop("discovery", {}) or {},
            "liquidity_filters": daemon_data.pop("liquidity_filters", {}) or {},
            "prefetch": daemon_data.pop("prefetch", {}) or {},
            "sector_rotation": daemon_data.pop("sector_rotation", {}) or {},
            "earnings_calendar": daemon_data.pop("earnings_calendar", {}) or {},
            "peer_analysis": daemon_data.pop("peer_analysis", {}) or {},
            "correlation_audit": daemon_data.pop("correlation_audit", {}) or {},
            "portfolio_health": daemon_data.pop("portfolio_health", {}) or {},
            "reporting": daemon_data.pop("reporting", {}) or {},
            "risk_limits": daemon_data.pop("risk_limits", {}) or {},
            "rebalancing": daemon_data.pop("rebalancing", {}) or {},
            "signal_tracking": daemon_data.pop("signal_tracking", {}) or {},
            "pre_trade_backtesting": daemon_data.pop("pre_trade_backtesting", {}) or {},
            "game_plan": daemon_data.pop("game_plan", {}) or {},
            "position_sizing": daemon_data.pop("position_sizing", {}) or {},
            "position_management": daemon_data.pop("position_management", {}) or {},
            "monte_carlo": daemon_data.pop("monte_carlo", {}) or {},
            "pre_market": daemon_data.pop("pre_market", {}) or {},
            "notifications": daemon_data.pop("notifications", {}) or {},
            "analysis_orchestration": daemon_data.pop("analysis_orchestration", {}) or {},
            "news_watcher": daemon_data.pop("news_watcher", {}) or {},
            "social_watcher": daemon_data.pop("social_watcher", {}) or {},
            "reddit_scraper": daemon_data.pop("reddit_scraper", {}) or {},
            "trump_watcher": daemon_data.pop("trump_watcher", {}) or {},
            "filings_watcher": daemon_data.pop("filings_watcher", {}) or {},
            "anomaly_watcher": daemon_data.pop("anomaly_watcher", {}) or {},
            "news_trending_watcher": daemon_data.pop("news_trending_watcher", {}) or {},
            "economic_calendar_watcher": daemon_data.pop("economic_calendar_watcher", {}) or {},
            "options_flow_watcher": daemon_data.pop("options_flow_watcher", {}) or {},
            "social_sentiment_watcher": daemon_data.pop("social_sentiment_watcher", {}) or {},
            "event_integration": daemon_data.pop("event_integration", {}) or {},
            "api": daemon_data.pop("api", {}) or {},
            "llm": daemon_data.pop("llm", {}) or {},
            "finbert": daemon_data.pop("finbert", {}) or {},
            "api_keys": daemon_data.pop("api_keys", {}) or {},
            "data_sources": daemon_data.pop("data_sources", {}) or {},
            "database": daemon_data.pop("database", {}) or {},
            "coordinator": daemon_data.pop("coordinator", {}) or {},
            "profiling": daemon_data.pop("profiling", {}) or {},
            "metrics": daemon_data.pop("metrics", {}) or {},
            "logging": daemon_data.pop("logging", {}) or {},
            "risk_validation": daemon_data.pop("risk_validation", {}) or {},
            "workflow": daemon_data.pop("workflow", {}) or {},
        }

        # Extract nested sections
        sections["telegram"] = sections["notifications"].pop("telegram", {}) or {}
        sections["news_sources"] = sections["news_watcher"].pop("sources", {}) or {}
        sections["circuit_breaker"] = sections["api"].pop("circuit_breaker", {}) or {}
        sections["adaptive_thresholds"] = sections["coordinator"].pop("adaptive_thresholds", {}) or {}
        sections["pattern_detection"] = sections["coordinator"].pop("pattern_detection", {}) or {}
        sections["sweep_pass"] = sections["coordinator"].pop("sweep_pass", {}) or {}
        sections["finnhub_premium"] = sections["data_sources"].pop("finnhub_premium", {}) or {}
        sections["position_circuit_breaker"] = (
            sections["position_management"].pop("circuit_breaker", {}) or {}
        )

        return sections

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
        sections = cls._extract_config_sections(daemon_data)

        config = cls(
            **daemon_data,
            paper_trading=PaperTradingConfig(**sections["paper_trading"]),
            schedule=ScheduleConfig(**sections["schedule"]),
            state=StateConfig(**sections["state"]),
            journal=JournalConfig(**sections["journal"]),
            health=HealthConfig(**sections["health"]),
            optimization=OptimizationConfig(**sections["optimization"]),
            discovery=DiscoveryConfig(**sections["discovery"]),
            liquidity_filters=LiquidityFilterConfig(**sections["liquidity_filters"]),
            prefetch=PrefetchConfig(**sections["prefetch"]),
            sector_rotation=SectorRotationConfig(**sections["sector_rotation"]),
            earnings_calendar=EarningsCalendarConfig(**sections["earnings_calendar"]),
            peer_analysis=PeerAnalysisConfig(**sections["peer_analysis"]),
            correlation_audit=CorrelationAuditConfig(**sections["correlation_audit"]),
            portfolio_health=PortfolioHealthConfig(**sections["portfolio_health"]),
            reporting=ReportingConfig(**sections["reporting"]),
            risk_limits=RiskLimitsConfig(**sections["risk_limits"]),
            rebalancing=PortfolioRebalancingConfig(**sections["rebalancing"]),
            signal_tracking=SignalTrackingConfig(**sections["signal_tracking"]),
            pre_trade_backtesting=PreTradeBacktestingConfig(**sections["pre_trade_backtesting"]),
            game_plan=GamePlanConfig(**sections["game_plan"]),
            position_sizing=PositionSizingConfig(**sections["position_sizing"]),
            position_management=PositionManagementConfig(
                **sections["position_management"],
                circuit_breaker=PositionCircuitBreakerConfig(**sections["position_circuit_breaker"]),
            ),
            monte_carlo=MonteCarloConfig(**sections["monte_carlo"]),
            pre_market=PreMarketScreeningConfig(**sections["pre_market"]),
            notifications=NotificationsConfig(
                **sections["notifications"], telegram=TelegramNotificationConfig(**sections["telegram"])
            ),
            analysis_orchestration=AnalysisOrchestratorConfig(**sections["analysis_orchestration"]),
            news_watcher=NewsWatcherConfig(
                **sections["news_watcher"], sources=NewsSourcesConfig(**sections["news_sources"])
            ),
            social_watcher=SocialWatcherConfig(**sections["social_watcher"]),
            reddit_scraper=RedditScraperConfig(**sections["reddit_scraper"]),
            trump_watcher=TrumpWatcherConfig(**sections["trump_watcher"]),
            filings_watcher=FilingsWatcherConfig(**sections["filings_watcher"]),
            anomaly_watcher=AnomalyWatcherConfig(**sections["anomaly_watcher"]),
            news_trending_watcher=NewsTrendingWatcherConfig(**sections["news_trending_watcher"]),
            economic_calendar_watcher=EconomicCalendarWatcherConfig(**sections["economic_calendar_watcher"]),
            options_flow_watcher=OptionsFlowWatcherConfig(**sections["options_flow_watcher"]),
            social_sentiment_watcher=SocialSentimentWatcherConfig(**sections["social_sentiment_watcher"]),
            event_integration=EventWatcherIntegrationConfig(**sections["event_integration"]),
            api=ApiConfig(
                **sections["api"], circuit_breaker=CircuitBreakerConfig(**sections["circuit_breaker"])
            ),
            llm=LLMConfig(**sections["llm"]),
            finbert=FinBERTConfig(**sections["finbert"]),
            api_keys=ApiKeysConfig(**sections["api_keys"]),
            data_sources=DataSourcesConfig(
                **sections["data_sources"],
                finnhub_premium=FinnhubSourcesConfig(**sections["finnhub_premium"]),
            ),
            database=DatabaseConfig(**sections["database"]),
            coordinator=CoordinatorConfig(
                **sections["coordinator"],
                adaptive_thresholds=AdaptiveThresholdConfig(**sections["adaptive_thresholds"]),
                pattern_detection=PatternDetectionConfig(**sections["pattern_detection"]),
                sweep_pass=SweepPassConfig(**sections["sweep_pass"]),
            ),
            profiling=ProfilingConfig(**sections["profiling"]),
            metrics=MetricsConfig(**sections["metrics"]),
            logging=LoggingConfig(**sections["logging"]),
            risk_validation=RiskValidationConfig(**sections["risk_validation"]),
            workflow=WorkflowConfigDaemon(**sections["workflow"]),
        )
        config._config_path = path
        return config

    def remove_watchlist_symbol(self, symbol: str) -> None:
        """Remove symbol from watchlist in-memory and persist to config file.

        Args:
            symbol: Ticker to remove
        """
        if symbol not in self.watchlist:
            return
        self.watchlist.remove(symbol)
        logger.warning(f"Removed {symbol} from watchlist (no market data available)")
        if self._config_path is None:
            return
        try:
            with self._config_path.open("r") as f:
                data = yaml.safe_load(f) or {}
            daemon_config = data.get("daemon") or {}
            watchlist: list[str] = daemon_config.get("watchlist", []) or []
            if symbol in watchlist:
                watchlist.remove(symbol)
                daemon_config["watchlist"] = watchlist
                data["daemon"] = daemon_config
            with self._config_path.open("w") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            logger.info(f"Persisted watchlist removal of {symbol} to {self._config_path}")
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to persist watchlist removal of {symbol}: {e}")

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"DaemonConfig(watchlist={self.watchlist}, "
            f"interval={self.interval_minutes}m, auto_trade={self.auto_trade})"
        )
