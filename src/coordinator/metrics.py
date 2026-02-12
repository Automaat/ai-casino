"""Coordinator cycle metrics persistence."""

from datetime import datetime
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field


class CoordinatorCycleMetrics(BaseModel):
    """Coordinator cycle metrics."""

    cycle_num: int = Field(ge=0, description="Cycle number")
    timestamp: datetime
    symbols_analyzed: list[str] = Field(default_factory=list)
    tool_calls_made: int = Field(ge=0)
    trades_proposed: int = Field(ge=0)
    trades_executed: int = Field(ge=0)
    trades_pending: int = Field(ge=0, default=0)
    game_plan_generated: bool
    cycle_duration_seconds: float = Field(ge=0.0)
    patterns_detected: int = Field(ge=0, default=0)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CoordinatorCycleMetrics(cycle={self.cycle_num}, "
            f"symbols={len(self.symbols_analyzed)}, "
            f"trades={self.trades_executed}/{self.trades_proposed})"
        )


def save_metrics_jsonl(metrics: CoordinatorCycleMetrics, path: Path) -> None:
    """Append metrics to JSONL file.

    Args:
        metrics: Metrics to save
        path: Path to JSONL file
    """
    try:
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Append to file
        with path.open("a") as f:
            f.write(metrics.model_dump_json() + "\n")

        logger.debug(f"Saved metrics to {path}")

    except Exception as e:
        logger.opt(exception=True).error(f"Failed to save metrics: {e}")


def load_recent_metrics(path: Path, limit: int = 50) -> list[CoordinatorCycleMetrics]:
    """Load recent metrics from JSONL.

    Args:
        path: Path to JSONL file
        limit: Maximum number of records to load

    Returns:
        List of recent metrics (most recent first)
    """
    if not path.exists():
        return []

    try:
        lines = path.read_text().splitlines()
        recent = lines[-limit:] if len(lines) > limit else lines

        metrics_list = []
        for line in recent:
            if not line.strip():
                continue
            try:
                metrics_list.append(CoordinatorCycleMetrics.model_validate_json(line))
            except Exception as e:
                logger.opt(exception=True).warning(f"Failed to parse metrics line: {e}")
                continue

        # Reverse to get most recent first
        return list(reversed(metrics_list))

    except Exception as e:
        logger.opt(exception=True).error(f"Failed to load metrics: {e}")
        return []
