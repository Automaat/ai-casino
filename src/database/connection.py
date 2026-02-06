"""Database connection management."""

import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.engine import DatabaseEngine

_db_engine_instance: DatabaseEngine | None = None
_db_engine_lock = threading.Lock()


def get_db_engine() -> DatabaseEngine:
    """Get or create singleton DatabaseEngine."""
    global _db_engine_instance  # noqa: PLW0603

    if _db_engine_instance is not None:
        return _db_engine_instance

    with _db_engine_lock:
        if _db_engine_instance is not None:
            return _db_engine_instance
        _db_engine_instance = DatabaseEngine()
        return _db_engine_instance


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
