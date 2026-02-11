"""Profiling configuration for daemon cycles."""

from pydantic import BaseModel, Field


class ProfilingConfig(BaseModel):
    """Daemon profiling configuration."""

    enabled: bool = False
    output_dir: str = "~/.ai-casino/profiles"
    profiler_type: str = "yappi"
    clock_type: str = "wall"
    formats: list[str] = Field(default_factory=lambda: ["pstats", "json"])

    # Retention
    retention_days: int = Field(default=7, ge=1, le=90)
    max_files: int = Field(default=1000, ge=10, le=10000)
    max_disk_mb: int = Field(default=500, ge=10, le=5000)

    # Sampling
    sample_rate: int = Field(default=1, ge=1, le=100)

    # Output
    top_n_functions: int = Field(default=50, ge=10, le=500)
    enable_metrics_aggregation: bool = True
    metrics_window_size: int = Field(default=100, ge=10, le=1000)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ProfilingConfig(enabled={self.enabled}, profiler={self.profiler_type})"
