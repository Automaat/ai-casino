"""Task scheduling and result models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


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

    time: str | None = Field(
        default=None, description="HH:MM time (required for DAILY/NONE, omit for INTERVAL)"
    )
    days: list[DayOfWeek] = Field(description="Days to run on")
    enabled: bool = True
    dedup: DedupStrategy = DedupStrategy.DAILY
    dedup_interval_minutes: int | None = Field(
        default=None, description="Minutes between runs (required for INTERVAL strategy)"
    )

    @model_validator(mode="after")
    def _validate(self) -> TaskSchedule:
        """Validate time/interval consistency."""
        if self.dedup == DedupStrategy.INTERVAL:
            if self.dedup_interval_minutes is None:
                msg = "dedup_interval_minutes required for INTERVAL strategy"
                raise ValueError(msg)
            if self.time is not None:
                msg = "time must be omitted (set to None) for INTERVAL strategy"
                raise ValueError(msg)
            return self

        # Non-INTERVAL strategies (DAILY, NONE): require time.
        if self.time is None:
            msg = "time required for non-INTERVAL strategies"
            raise ValueError(msg)
        return self

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
