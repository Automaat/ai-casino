"""Base repository abstract class."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

T = TypeVar("T")


class BaseRepository(ABC, Generic[T]):
    """Abstract base class for repositories with async context manager support.

    Repositories can be used in two ways:
    1. Direct instantiation (session held until GC) - deprecated
    2. Async context manager (session properly closed) - recommended

    Example:
        # Using the repository as an async context manager (recommended):
        async with MyRepository(session) as repo:
            await repo.create(entity)
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        self._session = session
        self.owns_session = False

    async def __aenter__(self) -> BaseRepository[T]:
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object) -> None:
        """Exit async context manager and close session if owned."""
        if self.owns_session:
            await self._session.close()

    async def close(self) -> None:
        """Explicitly close the session if this repository owns it.

        If the session is not owned by this repository (owns_session is False),
        this method will be a no-op to avoid closing a shared or externally
        managed session.
        """
        if self.owns_session:
            await self._session.close()
        else:
            logger.warning(
                f"{self.__class__.__name__}.close() called on non-owned session; "
                "no action taken to avoid closing an external session."
            )

    @abstractmethod
    async def create(self, entity: T) -> T:
        """Create new entity."""

    @abstractmethod
    async def get_by_id(self, entity_id: str) -> T | None:
        """Get entity by ID."""

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}(session={self._session})"
