"""Base repository abstract class."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from sqlalchemy.ext.asyncio import AsyncSession

T = TypeVar("T")


class BaseRepository(ABC, Generic[T]):
    """Abstract base class for repositories."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        self._session = session

    @abstractmethod
    async def create(self, entity: T) -> T:
        """Create new entity."""

    @abstractmethod
    async def get_by_id(self, entity_id: str) -> T | None:
        """Get entity by ID."""

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}(session={self._session})"
