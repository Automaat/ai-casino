"""Fixtures for metrics tests using SQLite in-memory."""

import uuid

import pytest
from sqlalchemy import JSON, Float, String, event
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.sql import sqltypes

from src.database.models import Base


def _should_remove_default(default_value, patterns):
    """Check if server default should be removed based on patterns."""
    if default_value is None:
        return False
    default_text = str(default_value.arg)
    if isinstance(patterns, list):
        return any(pattern in default_text for pattern in patterns)
    return default_text in patterns


def _adapt_types_for_sqlite(target, connection, **kw):
    """Replace PostgreSQL-specific types with SQLite-compatible equivalents."""
    for table in target.tables.values():
        for column in table.columns:
            # JSONB → JSON
            if isinstance(column.type, postgresql.JSONB):
                column.type = JSON()
                if _should_remove_default(column.server_default, ["::jsonb", "::json"]):
                    column.server_default = None

            # ARRAY → JSON
            elif isinstance(column.type, postgresql.ARRAY):
                column.type = JSON()
                if _should_remove_default(column.server_default, ["'{}'", "ARRAY[]", "::text[]"]):
                    column.server_default = None

            # UUID → String(36)
            elif isinstance(column.type, postgresql.UUID):
                column.type = String(36)
                if _should_remove_default(column.server_default, ["uuid_generate_v4"]):
                    column.server_default = None

            # DECIMAL → Float
            elif isinstance(column.type, sqltypes.DECIMAL):
                column.type = Float()


@pytest.fixture
async def async_engine():
    """Create async SQLite in-memory engine for testing."""
    from src.database.connection import _DatabaseEngineHolder
    from src.database.engine import DatabaseEngine

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )

    event.listens_for(Base.metadata, "before_create", once=True)(_adapt_types_for_sqlite)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Initialize global database engine with test engine for SignalAnalyticsService
    db_engine = DatabaseEngine()
    db_engine._engine = engine  # pyrefly: ignore[reportAttributeAccessIssue]
    db_engine._session_factory = async_sessionmaker(  # pyrefly: ignore[reportAttributeAccessIssue]
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    _DatabaseEngineHolder.initialize(db_engine)

    yield engine

    # Clean up
    _DatabaseEngineHolder.instance = None
    await engine.dispose()


@pytest.fixture
async def db_session(async_engine) -> AsyncSession:
    """Create async database session for testing."""
    session_maker = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Convert UUID objects to strings before flush for SQLite compatibility
    def _convert_uuids_to_strings(session, flush_context, instances):
        """Convert UUID values to strings for SQLite."""
        for obj in session.new | session.dirty:
            mapper = obj.__mapper__
            for column in mapper.columns:
                if isinstance(column.type, String) and hasattr(obj, column.key):
                    value = getattr(obj, column.key)
                    if isinstance(value, uuid.UUID):
                        setattr(obj, column.key, str(value))

    async with session_maker() as session:
        # Attach listener to sync session for this specific async session instance
        sync_session = session.sync_session
        event.listen(sync_session, "before_flush", _convert_uuids_to_strings)

        yield session

        # Remove listener to prevent cross-test side effects
        event.remove(sync_session, "before_flush", _convert_uuids_to_strings)
        await session.rollback()
