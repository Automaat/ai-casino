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
    after_hours_screen_time: str = "16:30"
    after_hours_screen_days: list[str] = Field(default_factory=lambda: ["mon", "tue", "wed", "thu", "fri"])
    after_hours_criteria: Literal["momentum", "value", "breakout"] = "momentum"
    after_hours_universe: Literal["SP500", "NASDAQ100", "COMBINED"] = "COMBINED"
    after_hours_top_n: int = 10

    @model_validator(mode="after")
    def validate_after_hours_screen_time(self) -> "ScheduleConfig":
        """Validate after_hours_screen_time is within 16:00-20:00."""
        if not self.enable_after_hours:
            return self

        try:
            hour, minute = map(int, self.after_hours_screen_time.split(":"))
        except ValueError as e:
            msg = f"after_hours_screen_time must be in HH:MM format, got {self.after_hours_screen_time}"
            raise ValueError(msg) from e

        if not (16 <= hour < 20 or (hour == 20 and minute == 0)):
            msg = f"after_hours_screen_time must be between 16:00-20:00, got {self.after_hours_screen_time}"
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

        return cls(
            **daemon_data,
            schedule=ScheduleConfig(**schedule_data),
            state=StateConfig(**state_data),
            journal=JournalConfig(**journal_data),
            health=HealthConfig(**health_data),
            optimization=OptimizationConfig(**optimization_data),
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"DaemonConfig(watchlist={self.watchlist}, "
            f"interval={self.interval_minutes}m, auto_trade={self.auto_trade})"
        )
