"""Tests for database engine."""

import pytest

from src.database.engine import DatabaseEngine, MissingDatabaseURLError, PoolConfig


class TestDatabaseEngine:
    """Tests for DatabaseEngine."""

    def test_init_without_url_raises(self, monkeypatch):
        """Test that init without DATABASE_URL raises error."""
        monkeypatch.delenv("DATABASE_URL", raising=False)
        with pytest.raises(MissingDatabaseURLError, match="DATABASE_URL must be provided"):
            DatabaseEngine()

    def test_init_with_url(self, monkeypatch):
        """Test initialization with valid URL."""
        url = "postgresql+asyncpg://localhost:5432/test"
        monkeypatch.setenv("DATABASE_URL", url)
        engine = DatabaseEngine()
        assert engine._database_url == url

    def test_init_with_explicit_url(self, monkeypatch):
        """Test initialization with explicit URL parameter."""
        monkeypatch.delenv("DATABASE_URL", raising=False)
        url = "postgresql+asyncpg://localhost:5432/test"
        engine = DatabaseEngine(database_url=url)
        assert engine._database_url == url

    def test_repr(self, monkeypatch):
        """Test string representation hides credentials."""
        url = "postgresql+asyncpg://user:pass@localhost:5432/test"
        monkeypatch.setenv("DATABASE_URL", url)
        engine = DatabaseEngine()
        assert "pass" not in repr(engine)
        assert "localhost:5432/test" in repr(engine)

    def test_uses_asyncadapted_queuepool_by_default(self, monkeypatch):
        """Test that engine uses AsyncAdaptedQueuePool by default."""
        from sqlalchemy.pool import AsyncAdaptedQueuePool

        url = "postgresql+asyncpg://localhost:5432/test"
        monkeypatch.setenv("DATABASE_URL", url)
        engine = DatabaseEngine()

        assert isinstance(engine.engine.pool, AsyncAdaptedQueuePool)
        assert engine.engine.pool.size() == 5  # default pool_size

    def test_pool_params_applied(self, monkeypatch):
        """Test that pool parameters are correctly applied."""
        url = "postgresql+asyncpg://localhost:5432/test"
        monkeypatch.delenv("DATABASE_URL", raising=False)

        pool_config = PoolConfig(
            pool_size=10,
            max_overflow=20,
            pool_timeout=60.0,
            pool_recycle=1800,
        )
        engine = DatabaseEngine(database_url=url, pool_config=pool_config)

        pool = engine.engine.pool
        assert pool.size() == 10
        assert pool._max_overflow == 20
        assert pool._timeout == 60.0
        assert pool._recycle == 1800

    def test_poolclass_override_for_testing(self, monkeypatch):
        """Test that poolclass can be overridden for testing (e.g., NullPool)."""
        from sqlalchemy.pool import NullPool

        url = "postgresql+asyncpg://localhost:5432/test"
        monkeypatch.delenv("DATABASE_URL", raising=False)

        engine = DatabaseEngine(database_url=url, poolclass=NullPool)

        assert isinstance(engine.engine.pool, NullPool)

    @pytest.mark.integration
    async def test_concurrent_sessions_reuse_connections(self):
        """Test that concurrent sessions reuse pooled connections."""
        import asyncio
        import os

        from sqlalchemy import text

        url = os.getenv("DATABASE_URL")
        if not url:
            pytest.skip("DATABASE_URL not set")

        # Only test with PostgreSQL (pooling not relevant for SQLite)
        if not url.startswith("postgresql"):
            pytest.skip("Pooling test requires PostgreSQL")

        pool_config = PoolConfig(pool_size=3, max_overflow=0)
        engine = DatabaseEngine(database_url=url, pool_config=pool_config)
        await engine.ensure_migrated()

        async def query_db():
            async with engine.session() as session:
                result = await session.execute(text("SELECT 1"))
                return result.scalar()

        # 5 concurrent queries with pool_size=3, max_overflow=0
        # Should queue and reuse, not fail
        results = await asyncio.gather(*[query_db() for _ in range(5)])
        assert len(results) == 5
        assert all(r == 1 for r in results)

        await engine.close()

    def test_get_pool_stats(self, monkeypatch):
        """Test pool statistics retrieval."""
        url = "postgresql+asyncpg://localhost:5432/test"
        monkeypatch.setenv("DATABASE_URL", url)

        pool_config = PoolConfig(pool_size=5, max_overflow=10)
        engine = DatabaseEngine(pool_config=pool_config)
        stats = engine.get_pool_stats()

        assert "pool_size" in stats
        assert "connections_in_use" in stats
        assert "connections_available" in stats
        assert "overflow_count" in stats
        assert stats["pool_size"] >= 0
