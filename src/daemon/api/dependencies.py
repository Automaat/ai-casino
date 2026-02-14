"""FastAPI dependencies for database access."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from src.database.connection import get_session
from src.database.repositories.metadata import MetadataRepository


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
