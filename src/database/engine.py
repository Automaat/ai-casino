"""Database engine with auto-migration support."""

import os
from pathlib import Path

from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool


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
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_pre_ping: bool = True,
    ) -> None:
        """Initialize database engine.

        Uses NullPool to avoid event loop issues when API server runs in separate thread.
        Each request creates a fresh connection - pool_size/max_overflow ignored.

        Args:
            database_url: PostgreSQL connection URL (or from DATABASE_URL env)
            pool_size: Ignored (using NullPool)
            max_overflow: Ignored (using NullPool)
            pool_pre_ping: Verify connections before use

        Raises:
            MissingDatabaseURLError: If no database URL provided
        """
        self._database_url = database_url or os.getenv("DATABASE_URL")
        if not self._database_url:
            raise MissingDatabaseURLError

        self._engine: AsyncEngine = create_async_engine(
            self._database_url,
            poolclass=NullPool,  # No pooling - each request gets fresh connection
            pool_pre_ping=pool_pre_ping,
        )
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        self._migrations_applied = False
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
        if self._migrations_applied:
            return

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

        self._migrations_applied = True

    async def ensure_migrated(self) -> None:
        """Ensure migrations are applied (call on first use)."""
        if not self._migrations_applied:
            await self.run_migrations()

    async def close(self) -> None:
        """Close database connections."""
        await self._engine.dispose()
        logger.info("Database connections closed")

    def __repr__(self) -> str:
        """Return string representation (hides credentials)."""
        if self._database_url is None:
            return "DatabaseEngine(url=None)"
        return f"DatabaseEngine(url={self._database_url.split('@')[-1]})"
