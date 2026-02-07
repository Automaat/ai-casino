"""Configuration for the trading daemon."""

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


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


class DaemonConfig(BaseModel):
    """Configuration for the trading daemon."""

    watchlist: list[str] = Field(default_factory=lambda: ["AAPL", "TSLA", "GOOGL", "MSFT"])
    interval_minutes: int = 30
    market_hours_only: bool = True
    auto_trade: bool = False
    max_concurrent_analyses: int = 3
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    state: StateConfig = Field(default_factory=StateConfig)
    journal: JournalConfig = Field(default_factory=JournalConfig)
    health: HealthConfig = Field(default_factory=HealthConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    screening: ScreeningConfig = Field(default_factory=ScreeningConfig)
    prefetch: PrefetchConfig = Field(default_factory=PrefetchConfig)
    sector_rotation: SectorRotationConfig = Field(default_factory=SectorRotationConfig)
    earnings_calendar: EarningsCalendarConfig = Field(default_factory=EarningsCalendarConfig)
    peer_analysis: PeerAnalysisConfig = Field(default_factory=PeerAnalysisConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    risk_limits: RiskLimitsConfig = Field(default_factory=RiskLimitsConfig)
    rebalancing: PortfolioRebalancingConfig = Field(default_factory=PortfolioRebalancingConfig)

    @classmethod
    def from_toml(cls, path: Path) -> "DaemonConfig":
        """Load configuration from TOML file.

        Args:
            path: Path to TOML config file

        Returns:
            DaemonConfig instance
        """
        with path.open("rb") as f:
            data = tomllib.load(f)

        daemon_data = data.get("daemon", {})

        schedule_data = daemon_data.pop("schedule", {})
        state_data = daemon_data.pop("state", {})
        journal_data = daemon_data.pop("journal", {})
        health_data = daemon_data.pop("health", {})
        optimization_data = daemon_data.pop("optimization", {})
        screening_data = daemon_data.pop("screening", {})
        prefetch_data = daemon_data.pop("prefetch", {})
        sector_rotation_data = daemon_data.pop("sector_rotation", {})
        earnings_calendar_data = daemon_data.pop("earnings_calendar", {})
        peer_analysis_data = daemon_data.pop("peer_analysis", {})
        reporting_data = daemon_data.pop("reporting", {})
        risk_limits_data = daemon_data.pop("risk_limits", {})
        rebalancing_data = daemon_data.pop("rebalancing", {})

        return cls(
            **daemon_data,
            schedule=ScheduleConfig(**schedule_data),
            state=StateConfig(**state_data),
            journal=JournalConfig(**journal_data),
            health=HealthConfig(**health_data),
            optimization=OptimizationConfig(**optimization_data),
            screening=ScreeningConfig(**screening_data),
            prefetch=PrefetchConfig(**prefetch_data),
            sector_rotation=SectorRotationConfig(**sector_rotation_data),
            earnings_calendar=EarningsCalendarConfig(**earnings_calendar_data),
            peer_analysis=PeerAnalysisConfig(**peer_analysis_data),
            reporting=ReportingConfig(**reporting_data),
            risk_limits=RiskLimitsConfig(**risk_limits_data),
            rebalancing=PortfolioRebalancingConfig(**rebalancing_data),
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"DaemonConfig(watchlist={self.watchlist}, "
            f"interval={self.interval_minutes}m, auto_trade={self.auto_trade})"
        )
