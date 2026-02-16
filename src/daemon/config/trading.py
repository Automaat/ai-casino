"""Trading schedule, state, journal, and paper trading configuration."""

from pydantic import BaseModel, Field


class ScheduleConfig(BaseModel):
    """Schedule configuration for trading hours."""

    start_time: str = "09:30"
    end_time: str = "16:00"
    timezone: str = "America/New_York"
    enable_pre_market: bool = False
    enable_after_hours: bool = False


class StateConfig(BaseModel):
    """State persistence configuration."""

    state_file: str = "~/.ai-casino/daemon-state.json"
    cleanup_enabled: bool = True
    cleanup_retention_days: int = Field(
        default=90, ge=1, le=365, description="Data retention in days (1-365)"
    )
    cleanup_hour: int = Field(default=3, ge=0, le=23, description="Hour to run cleanup (0-23, default 3 AM)")


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


class PaperTradingConfig(BaseModel):
    """Paper trading validation configuration."""

    min_duration_days: int = Field(default=30, ge=1, le=365)
    min_trades: int = Field(default=20, ge=5, le=1000)
    min_sharpe: float = Field(default=0.5, ge=-1.0, le=5.0)
    max_drawdown_percent: float = Field(default=15.0, ge=1.0, le=50.0)
    min_win_rate: float = Field(default=0.45, ge=0.0, le=1.0)


class PaperTradingValidationConfig(BaseModel):
    """Paper trading validation report configuration."""

    enabled: bool = False
    enable_file_export: bool = Field(
        default=False,
        description="Export validation reports to JSON files (deprecated, use database)",
    )
    output_dir: str = Field(
        default="~/.ai-casino/paper-trading-validation",
        description="Directory for JSON exports if enable_file_export=true",
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
