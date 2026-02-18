"""Coordinator configuration and result models."""

from typing import Literal

from pydantic import BaseModel, Field


class PatternDetectionConfig(BaseModel):
    """Pattern detection configuration."""

    enabled: bool = True
    detection_frequency: int = Field(default=5, ge=1, description="Run every Nth cycle")
    lookback_days: int = Field(default=30, ge=7, le=90)
    min_sample_size: int = Field(default=10, ge=5, le=50)

    def __repr__(self) -> str:
        """String representation."""
        return f"PatternDetectionConfig(enabled={self.enabled}, frequency={self.detection_frequency})"


class AdaptiveThresholdConfig(BaseModel):
    """Adaptive confidence threshold configuration."""

    enabled: bool = False
    adaptation_interval_cycles: int = Field(default=5, ge=1, le=50)
    min_sample_size: int = Field(default=20, ge=10, le=100)
    buy_increase_step: float = Field(default=0.1, ge=0.0, le=0.2)
    sell_decrease_step: float = Field(default=0.05, ge=0.0, le=0.2)
    buy_accuracy_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    sell_accuracy_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    weekly_reset_enabled: bool = True
    max_threshold: float = Field(default=0.95, ge=0.5, le=1.0)
    min_threshold: float = Field(default=0.5, ge=0.0, le=0.9)

    def __repr__(self) -> str:
        """String representation."""
        return f"AdaptiveThresholdConfig(enabled={self.enabled}, interval={self.adaptation_interval_cycles})"


class SweepPassConfig(BaseModel):
    """Configuration for the watchlist sweep pass after coordinator cycle."""

    enabled: bool = True
    stale_hours: int = Field(default=4, ge=1, le=24)
    max_symbols: int = Field(default=20, ge=1, le=50)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SweepPassConfig(enabled={self.enabled}, stale_hours={self.stale_hours}, max={self.max_symbols})"
        )


class CoordinatorConfig(BaseModel):
    """Configuration for the coordinator agent."""

    enabled: bool = False
    max_tool_calls: int = Field(default=25, ge=5, le=50, description="Maximum tool calls per cycle")
    temperature: float = Field(default=0.5, ge=0.0, le=1.0, description="LLM temperature for coordinator")
    model_override: str | None = Field(
        default=None, description="Optional model override for coordinator (uses daemon LLM config if None)"
    )
    confirmation_mode: Literal["auto", "manual"] = Field(
        default="auto", description="Trade confirmation mode: auto or manual"
    )
    approval_timeout_seconds: int = Field(
        default=60,
        ge=30,
        le=300,
        description="Timeout for manual trade approvals (seconds)",
    )
    cycle_timeout_seconds: int = Field(
        default=600, ge=60, le=3600, description="Maximum cycle duration in seconds before timeout"
    )
    max_daily_trades: int = Field(default=10, ge=1, le=100, description="Maximum trades executed per day")
    max_position_pct: float = Field(
        default=10.0, ge=1.0, le=100.0, description="Maximum position size as % of portfolio"
    )
    min_confidence_to_trade: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Minimum signal confidence required to execute trade"
    )
    pattern_detection: PatternDetectionConfig = Field(default_factory=PatternDetectionConfig)
    adaptive_thresholds: AdaptiveThresholdConfig = Field(default_factory=AdaptiveThresholdConfig)
    sweep_pass: SweepPassConfig = Field(default_factory=SweepPassConfig)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"CoordinatorConfig(enabled={self.enabled}, max_tool_calls={self.max_tool_calls}, "
            f"confirmation_mode={self.confirmation_mode})"
        )


class CoordinatorCycleResult(BaseModel):
    """Result from coordinator-driven cycle."""

    summary: str = Field(description="Human-readable cycle summary")
    symbols_analyzed: list[str] = Field(default_factory=list, description="Symbols analyzed in this cycle")
    trades_proposed: int = Field(default=0, ge=0, description="Number of trades proposed")
    trades_executed: int = Field(default=0, ge=0, description="Number of trades successfully executed")
    tool_calls_made: int = Field(default=0, ge=0, description="Total tool calls made by coordinator")
    game_plan_generated: bool = Field(default=False, description="Whether a game plan was generated")
    cycle_duration_seconds: float = Field(default=0.0, ge=0.0, description="Cycle duration in seconds")

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"CoordinatorCycleResult(symbols={len(self.symbols_analyzed)}, "
            f"trades={self.trades_executed}/{self.trades_proposed}, "
            f"tool_calls={self.tool_calls_made})"
        )
