"""Base repository abstract class."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

T = TypeVar("T")


class BaseRepository(ABC, Generic[T]):
    """Abstract base class for repositories.

    Note: Sessions are currently held by repositories and not explicitly closed.
    This is a known architectural limitation. Connections are eventually reclaimed
    via pool_recycle (default 3600s) and garbage collection. For proper cleanup,
    consider using repositories as async context managers in future refactoring.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        self._session = session

    def _recreate_session_if_needed(self, error: RuntimeError) -> bool:
        """Recreate session if event loop error detected.

        Args:
            error: RuntimeError from database operation

        Returns:
            True if session was recreated, False otherwise
        """
        error_msg = str(error)
        if "bound to a different event loop" in error_msg or "attached to a different loop" in error_msg:
            logger.warning(f"{self.__class__.__name__}: Event loop error, recreating session")
            from src.database.connection import get_db_engine

            engine = get_db_engine()
            self._session = engine.session()
            return True
        return False

    @abstractmethod
    async def create(self, entity: T) -> T:
        """Create new entity."""

    @abstractmethod
    async def get_by_id(self, entity_id: str) -> T | None:
        """Get entity by ID."""

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}(session={self._session})"
