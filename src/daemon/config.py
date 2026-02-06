"""Configuration for the trading daemon."""

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field


class ScheduleConfig(BaseModel):
    """Schedule configuration for trading hours."""

    start_time: str = "09:30"
    end_time: str = "16:00"
    timezone: str = "America/New_York"
    enable_pre_market: bool = False
    enable_after_hours: bool = False
    after_hours_screen_time: str = "16:30"
    after_hours_screen_days: list[str] = Field(default_factory=lambda: ["mon", "tue", "wed", "thu", "fri"])
    after_hours_criteria: str = "momentum"
    after_hours_universe: str = "COMBINED"
    after_hours_top_n: int = 10


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

        return cls(
            **daemon_data,
            schedule=ScheduleConfig(**schedule_data),
            state=StateConfig(**state_data),
            journal=JournalConfig(**journal_data),
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"DaemonConfig(watchlist={self.watchlist}, "
            f"interval={self.interval_minutes}m, auto_trade={self.auto_trade})"
        )
