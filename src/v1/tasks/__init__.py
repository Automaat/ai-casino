"""V1 task framework — self-contained scheduled tasks."""

from src.v1.tasks.interface import Task
from src.v1.tasks.models import WEEKDAYS, DayOfWeek, DedupStrategy, TaskResult, TaskSchedule
from src.v1.tasks.runner import TaskRunner

__all__ = [
    "WEEKDAYS",
    "DayOfWeek",
    "DedupStrategy",
    "Task",
    "TaskResult",
    "TaskRunner",
    "TaskSchedule",
]
