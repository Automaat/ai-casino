"""Coordinator configuration and result models."""

from typing import Literal

from pydantic import BaseModel, Field


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
    cycle_timeout_seconds: int = Field(
        default=600, ge=60, description="Maximum cycle duration in seconds before timeout"
    )
    max_daily_trades: int = Field(default=10, ge=1, description="Maximum trades executed per day")
    max_position_pct: float = Field(
        default=10.0, ge=1.0, le=100.0, description="Maximum position size as % of portfolio"
    )
    min_confidence_to_trade: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum signal confidence required to execute trade"
    )

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

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"CoordinatorCycleResult(symbols={len(self.symbols_analyzed)}, "
            f"trades={self.trades_executed}/{self.trades_proposed}, "
            f"tool_calls={self.tool_calls_made})"
        )
