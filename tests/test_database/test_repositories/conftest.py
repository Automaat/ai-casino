"""Fixtures for database repository tests using SQLite in-memory."""

import uuid

import pytest
from sqlalchemy import JSON, Float, String, event
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session
from sqlalchemy.sql import sqltypes

from src.database.models import Base


def _adapt_types_for_sqlite(target, connection, **kw):
    """Replace PostgreSQL-specific types with SQLite-compatible equivalents."""
    for table in target.tables.values():
        for column in table.columns:
            # JSONB → JSON
            if isinstance(column.type, postgresql.JSONB):
                column.type = JSON()
                # Remove PostgreSQL-specific default
                if column.server_default is not None:
                    default_text = str(column.server_default.arg)
                    if "::jsonb" in default_text or "::json" in default_text:
                        column.server_default = None

            # UUID → String(36)
            if isinstance(column.type, postgresql.UUID):
                column.type = String(36)
                # Remove uuid_generate_v4() default
                if column.server_default is not None:
                    default_text = str(column.server_default.arg)
                    if "uuid_generate_v4" in default_text:
                        column.server_default = None

            # DECIMAL → Float
            if isinstance(column.type, sqltypes.DECIMAL):
                column.type = Float()


@pytest.fixture
async def async_engine():
    """Create async SQLite in-memory engine for testing."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )

    event.listens_for(Base.metadata, "before_create")(_adapt_types_for_sqlite)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()


@pytest.fixture
async def async_session(async_engine) -> AsyncSession:
    """Create async database session for testing."""
    session_maker = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Convert UUID objects to strings before flush for SQLite compatibility
    @event.listens_for(Session, "before_flush")
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
        yield session
        await session.rollback()
