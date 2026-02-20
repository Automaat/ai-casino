"""Task scheduling and result models."""

from enum import StrEnum

from pydantic import BaseModel, Field


class DayOfWeek(StrEnum):
    """Days of the week."""

    MON = "mon"
    TUE = "tue"
    WED = "wed"
    THU = "thu"
    FRI = "fri"
    SAT = "sat"
    SUN = "sun"


WEEKDAYS = [DayOfWeek.MON, DayOfWeek.TUE, DayOfWeek.WED, DayOfWeek.THU, DayOfWeek.FRI]


class DedupStrategy(StrEnum):
    """Deduplication strategy for task execution."""

    DAILY = "daily"
    INTERVAL = "interval"
    NONE = "none"


class TaskSchedule(BaseModel):
    """Schedule definition for a task."""

    time: str = Field(description="Scheduled time in HH:MM format")
    days: list[DayOfWeek] = Field(description="Days to run on")
    enabled: bool = True
    dedup: DedupStrategy = DedupStrategy.DAILY
    dedup_interval_minutes: int | None = Field(
        default=None, description="Minutes between runs (required for INTERVAL strategy)"
    )

    def __repr__(self) -> str:
        """String representation."""
        return f"TaskSchedule(time={self.time}, days={[d.value for d in self.days]}, dedup={self.dedup})"


class TaskResult(BaseModel):
    """Result from a task execution."""

    task_name: str
    success: bool
    duration_seconds: float
    message: str | None = None

    def __repr__(self) -> str:
        """String representation."""
        status = "OK" if self.success else "FAIL"
        return f"TaskResult({self.task_name}: {status}, {self.duration_seconds:.1f}s)"
