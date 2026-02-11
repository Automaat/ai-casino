"""Profiling metrics models and aggregation."""

from datetime import datetime

from pydantic import BaseModel, Field


class FunctionStats(BaseModel):
    """Statistics for a single function."""

    function: str
    cumtime: float
    ncalls: int
    percall: float


class PhaseMetrics(BaseModel):
    """Metrics for a single phase."""

    name: str
    duration_seconds: float


class ProfilingMetrics(BaseModel):
    """Per-cycle profiling metrics."""

    cycle_number: int
    timestamp: datetime
    duration_seconds: float
    profiling_overhead_percent: float
    phases: list[PhaseMetrics] = Field(default_factory=list)
    top_functions: list[FunctionStats] = Field(default_factory=list)
    p50_function_time: float | None = None
    p95_function_time: float | None = None
    p99_function_time: float | None = None

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"ProfilingMetrics(cycle={self.cycle_number}, "
            f"duration={self.duration_seconds:.2f}s, overhead={self.profiling_overhead_percent:.1f}%)"
        )
