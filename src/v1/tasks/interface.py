"""Task abstract base class."""

from abc import ABC, abstractmethod
from datetime import datetime

from src.v1.tasks.models import TaskResult, TaskSchedule


class Task(ABC):
    """Abstract base for scheduled daemon tasks."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique task identifier."""

    @property
    @abstractmethod
    def schedule(self) -> TaskSchedule:
        """Task schedule definition."""

    @abstractmethod
    async def execute(self) -> TaskResult:
        """Run the task and return result."""

    @abstractmethod
    async def last_run_at(self) -> datetime | None:
        """Timestamp of the most recent execution (for dedup)."""

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.name})"
