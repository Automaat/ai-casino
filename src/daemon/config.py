"""Configuration for the trading daemon."""

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from loguru import logger
from pydantic import BaseModel, Field, field_validator, model_validator


class ScheduleConfig(BaseModel):
    """Schedule configuration for trading hours."""

    start_time: str = "09:30"
    end_time: str = "16:00"
    timezone: str = "America/New_York"
    enable_pre_market: bool = False
    enable_after_hours: bool = False


class ScreeningConfig(BaseModel):
    """Configuration for after-hours watchlist screening."""

    enabled: bool = False
    screen_time: str = "16:30"
    screen_days: list[str] = Field(default_factory=lambda: ["mon", "tue", "wed", "thu", "fri"])
    criteria: Literal["momentum", "value", "breakout"] = "momentum"
    universe: Literal["SP500", "NASDAQ100", "COMBINED"] = "COMBINED"
    top_n: int = 10
    watchlist_name: str = "daemon-screening"

    @model_validator(mode="after")
    def validate_screen_time(self) -> "ScreeningConfig":
        """Validate screen_time is within 16:00-20:00."""
        if not self.enabled:
            return self

        # Strict HH:MM format validation
        import re

        pattern = r"^([0-1][0-9]|2[0-3]):([0-5][0-9])$"
        match = re.match(pattern, self.screen_time)
        if not match:
            msg = f"screen_time must be in HH:MM format (00:00-23:59), got {self.screen_time}"
            raise ValueError(msg)

        hour, minute = int(match.group(1)), int(match.group(2))

        # Validate 16:00-20:00 range
        if not (16 <= hour < 20 or (hour == 20 and minute == 0)):
            msg = f"screen_time must be between 16:00-20:00, got {self.screen_time}"
            raise ValueError(msg)

        return self


class DiscoveryConfig(BaseModel):
    """Configuration for automated stock discovery."""

    enabled: bool = False
    discovery_time: str = "16:30"
    discovery_days: list[str] = Field(default_factory=lambda: ["mon", "wed", "fri"])

    # Source enablement
    enable_technical_screening: bool = True
    enable_reddit_trending: bool = False
    enable_earnings_calendar: bool = True
    enable_sector_rotation: bool = True
    enable_volume_spikes: bool = False
    enable_price_gaps: bool = False
    enable_news_trending: bool = False

    # Technical screening
    screening_criteria: list[str] = Field(default_factory=lambda: ["momentum"])
    screening_universe: Literal["SP500", "NASDAQ100", "COMBINED"] = "COMBINED"
    screening_top_n: int = 20

    # Social/Reddit
    reddit_min_mentions: int = 5
    reddit_min_upvote_ratio: float = 0.75

    # Earnings
    earnings_lookahead_days: int = 7

    # Trigger thresholds for intraday detection
    volume_spike_threshold: float = 2.0
    price_gap_threshold: float = 5.0

    # Scoring weights
    scoring_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "technical_weight": 0.35,
            "liquidity_weight": 0.25,
            "timing_weight": 0.20,
            "social_weight": 0.15,
            "volatility_weight": 0.05,
        }
    )

    # Limits
    max_discovered_per_cycle: int = 5
    min_composite_score: float = 0.60
    max_watchlist_size: int = 50

    # Portfolio filters
    portfolio_filters: Any = Field(
        default_factory=lambda: {
            "max_sector_concentration": 0.30,
            "min_market_cap": 1e9,
            "min_avg_volume": 1_000_000,
            "price_range": [10.0, 500.0],
            "exclude_sectors": [],
        }
    )

    # Lifecycle management
    candidate_ttl_days: int = 7
    auto_remove_on_signal: bool = False

    # State tracking
    track_outcomes: bool = True
    outcome_lookback_days: int = 90

    @model_validator(mode="after")
    def validate_discovery_time(self) -> "DiscoveryConfig":
        """Validate discovery_time is within 16:00-20:00."""
        if not self.enabled:
            return self

        import re

        pattern = r"^([0-1][0-9]|2[0-3]):([0-5][0-9])$"
        match = re.match(pattern, self.discovery_time)
        if not match:
            msg = f"discovery_time must be in HH:MM format (00:00-23:59), got {self.discovery_time}"
            raise ValueError(msg)

        hour, minute = int(match.group(1)), int(match.group(2))

        if not (16 <= hour < 20 or (hour == 20 and minute == 0)):
            msg = f"discovery_time must be between 16:00-20:00, got {self.discovery_time}"
            raise ValueError(msg)

        return self


class StateConfig(BaseModel):
    """State persistence configuration."""

    state_file: str = "~/.ai-casino/daemon-state.json"


class JournalConfig(BaseModel):
    """Configuration for after-hours trade journal."""

    enabled: bool = True
    journal_dir: str = "~/.ai-casino/journal"
    run_offset_minutes: int = Field(
        default=15,
        ge=0,
        lt=24 * 60,
        description="Minutes after market close to start journal window (0-1439)",
    )


class HealthConfig(BaseModel):
    """Configuration for API health checks and state cleanup."""

    enabled: bool = True
    run_time: str = "17:00"
    archive_days: int = 30
    log_max_size_mb: int = 5
    health_dir: str = "~/.ai-casino/health"
    archive_dir: str = "~/.ai-casino/archive"


class OptimizationConfig(BaseModel):
    """Configuration for after-hours strategy parameter optimization."""

    enabled: bool = False
    optimization_time: str = "17:00"
    optimization_days: list[str] = Field(default_factory=lambda: ["sat"])
    n_trials: int = 100
    min_trades: int = 100
    params_file: str = "~/.ai-casino/optimized-params.json"
    refresh_days: int = 30
    strategies: list[str] = Field(default_factory=lambda: ["momentum", "mean_reversion", "trend_following"])


class PortfolioRebalancingConfig(BaseModel):
    """Configuration for portfolio rebalancing."""

    enabled: bool = False
    method: Literal["max_sharpe", "min_volatility", "hrp"] = "max_sharpe"
    run_time: str = "16:45"
    run_days: list[str] = Field(default_factory=lambda: ["mon"])
    rebalance_threshold: float = Field(default=0.01, ge=0.001, le=0.20)
    lookback_days: int = Field(default=90, ge=30, le=365)

    @model_validator(mode="after")
    def validate_run_time(self) -> "PortfolioRebalancingConfig":
        """Validate run_time is in HH:MM format within 16:00-20:00."""
        if not self.enabled:
            return self

        import re

        pattern = r"^([0-1][0-9]|2[0-3]):([0-5][0-9])$"
        match = re.match(pattern, self.run_time)
        if not match:
            msg = f"run_time must be in HH:MM format (00:00-23:59), got {self.run_time}"
            raise ValueError(msg)

        hour, minute = int(match.group(1)), int(match.group(2))

        if not (16 <= hour < 20 or (hour == 20 and minute == 0)):
            msg = f"run_time must be between 16:00-20:00, got {self.run_time}"
            raise ValueError(msg)

        return self


class PrefetchConfig(BaseModel):
    """Configuration for after-hours data prefetching."""

    enabled: bool = False
    prefetch_time: str = "16:30"
    enable_pre_market_refresh: bool = False
    pre_market_refresh_time: str = "04:00"
    cache_dir: str = "data/cache/prefetch"
    warm_finbert: bool = True
    check_connectivity: bool = True


class ApiConfig(BaseModel):
    """Configuration for embedded API server."""

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = Field(
        default=8484,
        ge=1,
        le=65535,
        description="TCP port for embedded API server (1-65535)",
    )

    @field_validator("host")
    @classmethod
    def warn_non_localhost(cls, v: str) -> str:
        """Warn if API binds to non-localhost (security risk)."""
        if v not in ("127.0.0.1", "localhost", "::1"):
            logger.warning(
                f"API host '{v}' is not localhost - daemon exposed to network without auth. "
                "Only use for development in trusted environments."
            )
        return v


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: str | None = None
    model: str | None = None


class DataSourcesConfig(BaseModel):
    """Data sources configuration."""

    market_data: Literal["yfinance", "alpha_vantage"] = "yfinance"


class ApiKeysConfig(BaseModel):
    """API keys configuration.

    All fields are optional and fall back to environment variables.
    Config values take priority when both config and env vars are set.
    """

    alpha_vantage_api_key: str | None = None
    marketaux_api_key: str | None = None
    finnhub_api_key: str | None = None
    alpaca_api_key: str | None = None
    alpaca_secret_key: str | None = None
    alpaca_paper_api_key: str | None = None
    alpaca_paper_secret_key: str | None = None
    reddit_client_id: str | None = None
    reddit_client_secret: str | None = None
    reddit_user_agent: str | None = None
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    openai_api_base: str | None = None


class SectorRotationConfig(BaseModel):
    """Configuration for sector rotation analysis."""

    enabled: bool = False
    run_time: str = "16:15"
    run_days: list[str] = Field(default_factory=lambda: ["mon", "tue", "wed", "thu", "fri"])
    boost_factor: float = Field(
        default=0.15, ge=0.0, le=1.0, description="Sector weight boost factor (0.0-1.0)"
    )

    @model_validator(mode="after")
    def validate_run_time(self) -> "SectorRotationConfig":
        """Validate run_time is in HH:MM format within 16:00-20:00."""
        if not self.enabled:
            return self

        import re

        pattern = r"^([0-1][0-9]|2[0-3]):([0-5][0-9])$"
        match = re.match(pattern, self.run_time)
        if not match:
            msg = f"run_time must be in HH:MM format (00:00-23:59), got {self.run_time}"
            raise ValueError(msg)

        hour, minute = int(match.group(1)), int(match.group(2))

        if not (16 <= hour < 20 or (hour == 20 and minute == 0)):
            msg = f"run_time must be between 16:00-20:00, got {self.run_time}"
            raise ValueError(msg)

        return self


class PeerAnalysisConfig(BaseModel):
    """Configuration for weekly deep peer benchmarking analysis."""

    enabled: bool = False
    run_time: str = "17:30"
    run_days: list[str] = Field(default_factory=lambda: ["sun"])
    max_peers: int = 10
    output_dir: str = "~/.ai-casino/peer-analysis"
    rate_limit_sleep: float = 13.0

    @model_validator(mode="after")
    def validate_run_time(self) -> "PeerAnalysisConfig":
        """Validate run_time is in HH:MM format."""
        if not self.enabled:
            return self

        import re

        pattern = r"^([0-1][0-9]|2[0-3]):([0-5][0-9])$"
        match = re.match(pattern, self.run_time)
        if not match:
            msg = f"run_time must be in HH:MM format (00:00-23:59), got {self.run_time}"
            raise ValueError(msg)

        return self


class CorrelationAuditConfig(BaseModel):
    """Configuration for weekly portfolio correlation audit."""

    enabled: bool = False
    run_time: str = "17:45"
    run_days: list[str] = Field(default_factory=lambda: ["sun"])
    correlation_threshold: float = Field(default=0.8, ge=0.5, le=0.95)
    lookback_days: int = Field(default=90, ge=30, le=180)
    output_dir: str = "~/.ai-casino/correlation-audits"

    @model_validator(mode="after")
    def validate_run_time(self) -> "CorrelationAuditConfig":
        """Validate run_time is in HH:MM format."""
        if not self.enabled:
            return self

        import re

        pattern = r"^([0-1][0-9]|2[0-3]):([0-5][0-9])$"
        match = re.match(pattern, self.run_time)
        if not match:
            msg = f"run_time must be in HH:MM format (00:00-23:59), got {self.run_time}"
            raise ValueError(msg)

        return self


class ReportingConfig(BaseModel):
    """Configuration for automated performance reporting."""

    enabled: bool = False
    tearsheet_time: str = "16:30"
    benchmark: str = "SPY"
    retention_days: int = 30

    @model_validator(mode="after")
    def validate_tearsheet_time(self) -> "ReportingConfig":
        """Validate tearsheet_time is in HH:MM format within 16:00-20:00 and retention_days >= 1."""
        if not self.enabled:
            return self

        import re

        pattern = r"^([0-1][0-9]|2[0-3]):([0-5][0-9])$"
        match = re.match(pattern, self.tearsheet_time)
        if not match:
            msg = f"tearsheet_time must be in HH:MM format (00:00-23:59), got {self.tearsheet_time}"
            raise ValueError(msg)

        hour, minute = int(match.group(1)), int(match.group(2))

        if not (16 <= hour < 20 or (hour == 20 and minute == 0)):
            msg = f"tearsheet_time must be between 16:00-20:00, got {self.tearsheet_time}"
            raise ValueError(msg)

        if self.retention_days < 1:
            msg = "retention_days must be >= 1 when reporting enabled"
            raise ValueError(msg)

        return self


class RiskLimitsConfig(BaseModel):
    """Configuration for portfolio-level VaR risk limits."""

    enabled: bool = False
    max_var_95: float = Field(default=0.03, ge=0.001, le=0.20)
    max_cvar_99: float = Field(default=0.05, ge=0.001, le=0.30)
    lookback_days: int = Field(default=90, ge=20, le=365)
    adaptive_stop_loss: bool = True
    cdar_stop_threshold: float = Field(default=0.10, ge=0.01, le=0.50)
    atr_multiplier_min: float = Field(default=1.0, ge=0.5, le=2.0)
    report_dir: str = "~/.ai-casino/risk-reports"


class SignalTrackingConfig(BaseModel):
    """Configuration for signal accuracy tracking."""

    enabled: bool = True
    tracking_time: str = "17:00"

    @model_validator(mode="after")
    def validate_tracking_time(self) -> "SignalTrackingConfig":
        """Validate tracking_time is in HH:MM format within 16:00-20:00."""
        if not self.enabled:
            return self

        import re

        pattern = r"^([0-1][0-9]|2[0-3]):([0-5][0-9])$"
        match = re.match(pattern, self.tracking_time)
        if not match:
            msg = f"tracking_time must be in HH:MM format (00:00-23:59), got {self.tracking_time}"
            raise ValueError(msg)

        hour, minute = int(match.group(1)), int(match.group(2))

        if not (16 <= hour < 20 or (hour == 20 and minute == 0)):
            msg = f"tracking_time must be between 16:00-20:00, got {self.tracking_time}"
            raise ValueError(msg)

        return self


class PreTradeBacktestingConfig(BaseModel):
    """Configuration for pre-trade backtesting validation."""

    enabled: bool = False
    lookback_days: int = Field(default=180, ge=30, le=365)
    min_sharpe_threshold: float = Field(default=0.5, ge=-1.0, le=3.0)
    max_drawdown_threshold: float = Field(default=0.25, ge=0.05, le=0.50)
    confidence_penalty_multiplier: float = Field(default=0.7, ge=0.1, le=1.0)


class GamePlanConfig(BaseModel):
    """Configuration for game plan generation."""

    enabled: bool = False
    generation_time: str = "04:00"
    plan_dir: str = "~/.ai-casino/game-plans"
    futures_symbols: list[str] = Field(default_factory=lambda: ["ES=F", "NQ=F"])
    lookback_hours: int = 16

    @model_validator(mode="after")
    def validate_generation_time(self) -> "GamePlanConfig":
        """Validate generation_time is in pre-market window (04:00-09:30)."""
        if not self.enabled:
            return self

        import re

        pattern = r"^([0-1][0-9]|2[0-3]):([0-5][0-9])$"
        match = re.match(pattern, self.generation_time)
        if not match:
            msg = f"generation_time must be HH:MM format, got {self.generation_time}"
            raise ValueError(msg)

        hour, minute = int(match.group(1)), int(match.group(2))

        if not (4 <= hour < 9 or (hour == 9 and minute < 30)):
            msg = f"generation_time must be 04:00-09:30, got {self.generation_time}"
            raise ValueError(msg)

        return self


class PositionSizingConfig(BaseModel):
    """Configuration for position sizing strategy."""

    primary_goal: Literal["maximize_returns", "minimize_risk", "balanced"] = "balanced"
    risk_tolerance: Literal["conservative", "moderate", "aggressive"] = "moderate"
    complexity: Literal["simple", "advanced"] = "simple"

    max_risk_per_trade_pct: float = Field(default=2.0, ge=0.1, le=10.0)
    max_single_position_pct: float = Field(default=20.0, ge=1.0, le=50.0)
    max_total_exposure_pct: float = Field(default=80.0, ge=10.0, le=100.0)

    blend_weight_optimization: float = Field(default=0.5, ge=0.0, le=1.0)
    blend_weight_risk_based: float = Field(default=0.5, ge=0.0, le=1.0)

    confidence_scaling_enabled: bool = False
    confidence_high_threshold: float = Field(default=0.8, ge=0.5, le=1.0)
    confidence_low_threshold: float = Field(default=0.6, ge=0.3, le=0.9)
    confidence_low_reduction_factor: float = Field(default=0.5, ge=0.1, le=0.9)

    use_monte_carlo_adjustment: bool = False
    monte_carlo_risk_multiplier: float = Field(default=0.7, ge=0.1, le=1.0)

    @model_validator(mode="after")
    def validate_blend_weights(self) -> "PositionSizingConfig":
        """Validate blend weights sum to 1.0."""
        total = self.blend_weight_optimization + self.blend_weight_risk_based
        if not (0.99 <= total <= 1.01):
            msg = f"Blend weights must sum to 1.0, got {total:.2f}"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_confidence_thresholds(self) -> "PositionSizingConfig":
        """Validate confidence thresholds are properly ordered."""
        if self.confidence_low_threshold >= self.confidence_high_threshold:
            msg = (
                f"confidence_low_threshold ({self.confidence_low_threshold}) must be < "
                f"confidence_high_threshold ({self.confidence_high_threshold})"
            )
            raise ValueError(msg)
        return self


class PositionManagementConfig(BaseModel):
    """Configuration for position management."""

    enabled: bool = False
    trailing_stop_enabled: bool = True
    trailing_stop_percent: float = Field(default=3.0, ge=0.5, le=10.0)
    partial_profit_enabled: bool = True
    profit_target_1_percent: float = Field(default=5.0, ge=1.0, le=20.0)
    profit_target_1_sell_pct: float = Field(default=0.5, ge=0.1, le=1.0)
    profit_target_2_percent: float = Field(default=10.0, ge=5.0, le=50.0)
    profit_target_2_sell_pct: float = Field(default=1.0, ge=0.1, le=1.0)
    time_exit_enabled: bool = True
    max_holding_days: int = Field(default=30, ge=1, le=180)
    breakeven_enabled: bool = True
    breakeven_activation_percent: float = Field(default=5.0, ge=1.0, le=20.0)
    conviction_scaling_enabled: bool = True
    conviction_decrease_threshold: float = Field(default=0.15, ge=0.05, le=0.5)
    conviction_scale_out_percent: float = Field(default=0.5, ge=0.1, le=1.0)


class MonteCarloConfig(BaseModel):
    """Monte Carlo stress testing configuration."""

    enabled: bool = False
    schedule_time: str = "17:00"
    schedule_days: list[str] = Field(default_factory=lambda: ["sun"])

    # Simulation parameters
    num_simulations: int = Field(default=5000, ge=100, le=10000)
    horizon_days: int = Field(default=252, ge=1, le=504)
    simulation_method: str = "PARAMETRIC"
    min_historical_days: int = Field(default=90, ge=30)
    random_seed: int | None = None

    # Risk thresholds
    loss_threshold: float = Field(default=0.10, ge=0.0, le=1.0)
    max_acceptable_prob: float = Field(default=0.15, ge=0.0, le=1.0)

    # Position sizing integration (opt-in)
    adjust_position_sizing: bool = False

    # History retention
    max_history_records: int = Field(default=52, ge=1, le=520)

    @model_validator(mode="after")
    def validate_schedule_time(self) -> "MonteCarloConfig":
        """Validate schedule_time is within 16:00-20:00 for after-hours or any time for weekends."""
        if not self.enabled:
            return self

        import re

        pattern = r"^([0-1][0-9]|2[0-3]):([0-5][0-9])$"
        match = re.match(pattern, self.schedule_time)
        if not match:
            msg = f"schedule_time must be HH:MM format, got {self.schedule_time}"
            raise ValueError(msg)

        hour = int(match.group(1))
        minute = int(match.group(2))

        # Allow any time for weekend runs, restrict to 16:00-20:00 for weekdays
        weekday_days = ["mon", "tue", "wed", "thu", "fri"]
        is_weekday_only = any(d.lower() in weekday_days for d in self.schedule_days)

        if is_weekday_only and not (16 <= hour < 20 or (hour == 20 and minute == 0)):
            msg = f"Weekday schedule_time must be 16:00-20:00, got {self.schedule_time}"
            raise ValueError(msg)

        return self


class TradingMode(StrEnum):
    """Trading mode for broker execution."""

    PAPER = "paper"
    LIVE = "live"


class PaperTradingConfig(BaseModel):
    """Paper trading validation configuration."""

    min_duration_days: int = Field(default=30, ge=1, le=365)
    min_trades: int = Field(default=20, ge=5, le=1000)
    min_sharpe: float = Field(default=0.5, ge=-1.0, le=5.0)
    max_drawdown_percent: float = Field(default=15.0, ge=1.0, le=50.0)
    min_win_rate: float = Field(default=0.45, ge=0.0, le=1.0)


class NotificationTrigger(StrEnum):
    """Trigger types for notifications."""

    SIGNAL = "signal"
    RISK_REJECTION = "risk_rejection"
    PORTFOLIO_VAR_BREACH = "portfolio_var_breach"
    HEALTH_FAILURE = "health_failure"
    PAPER_TRADING_READY = "paper_trading_ready"


class TelegramNotificationConfig(BaseModel):
    """Telegram notification channel configuration."""

    bot_token: str | None = None
    chat_id: str | None = None


class NotificationsConfig(BaseModel):
    """Notification system configuration."""

    enabled: bool = False
    channels: list[str] = Field(default_factory=lambda: ["telegram"])
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    notify_on: list[NotificationTrigger] = Field(
        default_factory=lambda: [
            NotificationTrigger.SIGNAL,
            NotificationTrigger.RISK_REJECTION,
            NotificationTrigger.PORTFOLIO_VAR_BREACH,
            NotificationTrigger.HEALTH_FAILURE,
        ]
    )
    rate_limit_enabled: bool = True
    rate_limit_per_symbol_minutes: int = 60
    telegram: TelegramNotificationConfig = Field(default_factory=TelegramNotificationConfig)


class EarningsCalendarConfig(BaseModel):
    """Configuration for earnings calendar preparation."""

    enabled: bool = False
    fetch_time: str = "16:45"
    fetch_days: list[str] = Field(default_factory=lambda: ["mon"])
    lookahead_days: int = Field(default=3, ge=1, le=14)
    reduce_position_t1: bool = False
    position_reduction_factor: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_fetch_time(self) -> "EarningsCalendarConfig":
        """Validate fetch_time is in HH:MM format within 16:00-20:00."""
        if not self.enabled:
            return self

        import re

        pattern = r"^([0-1][0-9]|2[0-3]):([0-5][0-9])$"
        match = re.match(pattern, self.fetch_time)
        if not match:
            msg = f"fetch_time must be in HH:MM format (00:00-23:59), got {self.fetch_time}"
            raise ValueError(msg)

        hour, minute = int(match.group(1)), int(match.group(2))

        if not (16 <= hour < 20 or (hour == 20 and minute == 0)):
            msg = f"fetch_time must be between 16:00-20:00, got {self.fetch_time}"
            raise ValueError(msg)

        return self


class AnalysisOrchestratorConfig(BaseModel):
    """Configuration for analysis orchestration."""

    max_concurrent_analyses: int = Field(default=3, ge=1, le=10)
    target_allocation_ttl_days: int = Field(default=7, ge=1, le=30)
    enable_position_sync: bool = True


class NewsWatcherConfig(BaseModel):
    """Configuration for news watcher."""

    enabled: bool = False
    poll_interval_minutes: int = Field(default=5, ge=1, le=60)
    breaking_threshold_minutes: int = Field(default=15, ge=5, le=120)
    relevance_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    cooldown_minutes: int = Field(default=15, ge=1, le=120)
    max_concurrent_analyses: int = Field(default=2, ge=1, le=10)


class SocialWatcherConfig(BaseModel):
    """Configuration for social media watcher."""

    enabled: bool = False
    poll_interval_minutes: int = Field(default=15, ge=5, le=60)
    volume_spike_threshold: float = Field(default=0.5, ge=0.1, le=2.0)
    viral_score_threshold: int = Field(default=1000, ge=100, le=10000)
    viral_upvote_ratio: float = Field(default=0.8, ge=0.5, le=1.0)
    subreddits: list[str] = Field(default_factory=lambda: ["wallstreetbets", "stocks"])
    relevance_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    cooldown_minutes: int = Field(default=15, ge=1, le=120)
    max_concurrent_analyses: int = Field(default=2, ge=1, le=10)


class FilingsWatcherConfig(BaseModel):
    """Configuration for SEC filings watcher."""

    enabled: bool = False
    poll_interval_minutes: int = Field(default=10, ge=5, le=60)
    filing_types: list[str] = Field(default_factory=lambda: ["8-K", "4", "13D"])
    cik_ticker_mapping_file: str = "~/.ai-casino/cik_ticker_map.json"
    relevance_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    cooldown_minutes: int = Field(default=15, ge=1, le=120)
    max_concurrent_analyses: int = Field(default=2, ge=1, le=10)


class AnomalyWatcherConfig(BaseModel):
    """Configuration for market anomaly watcher."""

    enabled: bool = False
    poll_interval_minutes: int = Field(default=15, ge=5, le=60)
    volume_spike_multiplier: float = Field(default=2.0, ge=1.5, le=5.0)
    price_move_threshold_pct: float = Field(default=5.0, ge=2.0, le=20.0)
    gap_threshold_pct: float = Field(default=3.0, ge=1.0, le=10.0)
    max_symbols_per_cycle: int = Field(default=5, ge=1, le=50)
    relevance_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    cooldown_minutes: int = Field(default=15, ge=1, le=120)
    max_concurrent_analyses: int = Field(default=2, ge=1, le=10)


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
    api_keys: ApiKeysConfig = Field(default_factory=ApiKeysConfig)
    data_sources: DataSourcesConfig = Field(default_factory=DataSourcesConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "DaemonConfig":
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
        api_keys_data = daemon_data.pop("api_keys", {}) or {}
        data_sources_data = daemon_data.pop("data_sources", {}) or {}

        # Extract nested telegram config from notifications
        telegram_data = notifications_data.pop("telegram", {}) or {}

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
            news_watcher=NewsWatcherConfig(**news_watcher_data),
            social_watcher=SocialWatcherConfig(**social_watcher_data),
            filings_watcher=FilingsWatcherConfig(**filings_watcher_data),
            anomaly_watcher=AnomalyWatcherConfig(**anomaly_watcher_data),
            api=ApiConfig(**api_data),
            llm=LLMConfig(**llm_data),
            api_keys=ApiKeysConfig(**api_keys_data),
            data_sources=DataSourcesConfig(**data_sources_data),
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"DaemonConfig(watchlist={self.watchlist}, "
            f"interval={self.interval_minutes}m, auto_trade={self.auto_trade})"
        )
