"""Portfolio management configuration."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from src.daemon.config._validators import validate_time_format, validate_time_range


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
        if self.enabled:
            validate_time_range(self.run_time, "run_time", "after_hours")
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
        if self.enabled:
            validate_time_format(self.run_time, "run_time")
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
        if self.enabled:
            validate_time_format(self.run_time, "run_time")
        return self


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
        if self.enabled:
            validate_time_range(self.generation_time, "generation_time", "pre_market")
        return self
