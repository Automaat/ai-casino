"""FastAPI dependencies for database access."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession

from src.database.connection import get_session
from src.database.repositories.metadata import MetadataRepository
from src.database.repositories.supervisor_metrics import SupervisorMetricsRepository


@asynccontextmanager
async def get_db_session() -> AsyncIterator[AsyncSession]:
    """Get database session for API requests.

    This ensures the session is created in the FastAPI event loop,
    avoiding "bound to different event loop" errors.

    Yields:
        AsyncSession for database operations
    """
    async with get_session() as session:
        yield session


@asynccontextmanager
async def get_metadata_repo() -> AsyncIterator[MetadataRepository]:
    """Get metadata repository with fresh session for API requests.

    This ensures the session is created in the FastAPI event loop,
    avoiding "bound to different event loop" errors.

    Yields:
        MetadataRepository with fresh session
    """
    async with get_session() as session:
        yield MetadataRepository(session)


@asynccontextmanager
async def get_supervisor_metrics_repo() -> AsyncIterator[SupervisorMetricsRepository]:
    """Get supervisor metrics repository with fresh session for API requests.

    This ensures the session is created in the FastAPI event loop,
    avoiding "bound to different event loop" errors.

    Yields:
        SupervisorMetricsRepository with fresh session
    """
    async with get_session() as session:
        yield SupervisorMetricsRepository(session)
