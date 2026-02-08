"""Database connection management."""

import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.engine import DatabaseEngine


class _DatabaseEngineHolder:
    """Singleton holder for DatabaseEngine instance."""

    instance: DatabaseEngine | None = None
    lock = threading.Lock()


def get_db_engine() -> DatabaseEngine:
    """Get or create singleton DatabaseEngine."""
    if _DatabaseEngineHolder.instance is not None:
        return _DatabaseEngineHolder.instance

    with _DatabaseEngineHolder.lock:
        if _DatabaseEngineHolder.instance is not None:
            return _DatabaseEngineHolder.instance
        _DatabaseEngineHolder.instance = DatabaseEngine()
        return _DatabaseEngineHolder.instance


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """Get database session as async context manager."""
    engine = get_db_engine()
    async with engine.session() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Session error: {e}")
            raise
