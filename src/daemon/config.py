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
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    screening: ScreeningConfig = Field(default_factory=ScreeningConfig)

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
        optimization_data = daemon_data.pop("optimization", {})
        screening_data = daemon_data.pop("screening", {})

        return cls(
            **daemon_data,
            schedule=ScheduleConfig(**schedule_data),
            state=StateConfig(**state_data),
            journal=JournalConfig(**journal_data),
            optimization=OptimizationConfig(**optimization_data),
            screening=ScreeningConfig(**screening_data),
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"DaemonConfig(watchlist={self.watchlist}, "
            f"interval={self.interval_minutes}m, auto_trade={self.auto_trade})"
        )
