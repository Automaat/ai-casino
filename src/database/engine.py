"""Database engine with auto-migration support."""

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool


@dataclass
class PoolConfig:
    """Database connection pool configuration."""

    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: float = 30.0
    pool_recycle: int = 3600
    pool_pre_ping: bool = True


class MissingDatabaseURLError(ValueError):
    """Raised when DATABASE_URL is not configured."""

    def __init__(self) -> None:
        """Initialize with standard error message."""
        super().__init__("DATABASE_URL must be provided or set in environment")


class DatabaseEngine:
    """Database engine with connection pooling and auto-migration."""

    def __init__(
        self,
        database_url: str | None = None,
        pool_config: PoolConfig | None = None,
        poolclass: type | None = None,
    ) -> None:
        """Initialize database engine with AsyncAdaptedQueuePool.

        Reuses connections across requests/analyses to reduce overhead.
        Thread-safe: each event loop (main + API thread) maintains separate queue.

        Args:
            database_url: PostgreSQL connection URL (or from DATABASE_URL env)
            pool_config: Pool configuration (size, timeout, recycle, etc.)
            poolclass: Pool class override (for testing only, e.g., NullPool)

        Raises:
            MissingDatabaseURLError: If no database URL provided
        """
        self._database_url = database_url or os.getenv("DATABASE_URL")
        if not self._database_url:
            raise MissingDatabaseURLError

        pool_config = pool_config or PoolConfig()
        engine_kwargs: dict[str, Any] = {"pool_pre_ping": pool_config.pool_pre_ping}

        # Only pass pool params for databases that support pooling (PostgreSQL/MySQL)
        # SQLite uses StaticPool which doesn't accept pool parameters
        supports_pooling = self._database_url.startswith(("postgresql", "mysql"))
        if supports_pooling and poolclass is not NullPool:
            engine_kwargs.update(
                {
                    "pool_size": pool_config.pool_size,
                    "max_overflow": pool_config.max_overflow,
                    "pool_timeout": pool_config.pool_timeout,
                    "pool_recycle": pool_config.pool_recycle,
                }
            )

        if poolclass is not None:
            engine_kwargs["poolclass"] = poolclass

        self._engine: AsyncEngine = create_async_engine(self._database_url, **engine_kwargs)
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        self._migrations_applied = False
        self._migration_lock = asyncio.Lock()
        logger.info(f"DatabaseEngine initialized with {self._database_url.split('@')[-1]}")

    @property
    def engine(self) -> AsyncEngine:
        """Return underlying SQLAlchemy async engine."""
        return self._engine

    def session(self) -> AsyncSession:
        """Create new database session."""
        return self._session_factory()

    async def run_migrations(self) -> None:
        """Apply pending SQL migrations from migrations directory."""
        async with self._migration_lock:
            if self._migrations_applied:
                return
            await self._apply_migrations()
            self._migrations_applied = True

    async def _apply_migrations(self) -> None:
        """Apply pending SQL migrations (must be called with _migration_lock held)."""
        migrations_dir = Path(__file__).parent / "migrations"
        if not migrations_dir.exists():
            logger.warning(f"Migrations directory not found: {migrations_dir}")
            return

        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version VARCHAR(50) PRIMARY KEY,
                        applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)
            )

            result = await conn.execute(text("SELECT version FROM schema_migrations"))
            applied = {row[0] for row in result.fetchall()}

            migration_files = sorted(migrations_dir.glob("*.sql"))
            for migration_file in migration_files:
                version = migration_file.stem
                if version in applied:
                    continue

                logger.info(f"Applying migration: {version}")
                sql = migration_file.read_text()
                for stmt in sql.split(";"):
                    stmt_clean = stmt.strip()
                    if stmt_clean:
                        await conn.execute(text(stmt_clean))

                await conn.execute(
                    text("INSERT INTO schema_migrations (version) VALUES (:version)"),
                    {"version": version},
                )
                logger.info(f"Migration {version} applied successfully")

    async def ensure_migrated(self) -> None:
        """Ensure migrations are applied (call on first use)."""
        if not self._migrations_applied:
            await self.run_migrations()

    def get_pool_stats(self) -> dict[str, int]:
        """Return pool statistics for monitoring.

        Note: Pool methods (size, checkedout, overflow) are available on QueuePool
        but not part of the base Pool interface. This works at runtime but type
        checker can't verify it due to SQLAlchemy's complex pool type hierarchy.

        Returns:
            Pool statistics, or error message if pool doesn't support stats.
        """
        pool: Any = self._engine.pool
        # NullPool and StaticPool don't support these methods
        if isinstance(pool, NullPool):
            return {
                "pool_size": 0,
                "connections_in_use": 0,
                "connections_available": 0,
                "overflow_count": 0,
            }
        try:
            return {
                "pool_size": pool.size(),
                "connections_in_use": pool.checkedout(),
                "connections_available": pool.size() - pool.checkedout(),
                "overflow_count": pool.overflow(),
            }
        except AttributeError:
            # Fallback for pool types that don't support these methods
            return {
                "pool_size": 0,
                "connections_in_use": 0,
                "connections_available": 0,
                "overflow_count": 0,
            }

    async def close(self) -> None:
        """Close database connections."""
        await self._engine.dispose()
        logger.info("Database connections closed")

    def __repr__(self) -> str:
        """Return string representation (hides credentials)."""
        if self._database_url is None:
            return "DatabaseEngine(url=None)"
        return f"DatabaseEngine(url={self._database_url.split('@')[-1]})"
