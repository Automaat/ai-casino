"""Tests for database engine."""

import pytest

from src.database.engine import DatabaseEngine, MissingDatabaseURLError


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
