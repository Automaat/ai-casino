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
        async with get_session() as session:
            repo = MyRepository(session)
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
        """Explicitly close the session."""
        await self._session.close()

    def __del__(self) -> None:
        """Cleanup on garbage collection."""
        if hasattr(self, "_session") and hasattr(self, "owns_session") and self.owns_session:
            try:
                import asyncio

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    task = loop.create_task(self._session.close())
                    task.add_done_callback(lambda _: None)
                else:
                    loop.run_until_complete(self._session.close())
            except Exception as e:
                logger.opt(exception=False).debug(f"Failed to close session during GC: {e}")

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
            self.owns_session = True
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
