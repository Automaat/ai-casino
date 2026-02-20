"""Tests for MetadataRepository datetime parsing."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.database.repositories.metadata import MetadataRepository


@pytest.fixture
def mock_session():
    """Mock SQLAlchemy async session."""
    session = MagicMock()
    session.execute = AsyncMock()
    return session


def _make_orm_result(session: MagicMock, data_value: object) -> None:
    """Wire mock session to return an ORM with given data value."""
    orm = MagicMock()
    orm.value = {"data": data_value}
    result = MagicMock()
    result.scalar_one_or_none.return_value = orm
    session.execute.return_value = result


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_parses_z_suffix_datetime(mock_session: MagicMock) -> None:
    """String ending with Z is parsed as UTC datetime."""
    _make_orm_result(mock_session, "2024-01-15T10:00:00Z")

    repo = MetadataRepository(mock_session)
    result = await repo.get("key")

    assert isinstance(result, datetime)
    assert result == datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
    assert result.tzinfo is UTC


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_parses_plus00_suffix_datetime(mock_session: MagicMock) -> None:
    """String with +00:00 suffix is parsed as UTC datetime."""
    _make_orm_result(mock_session, "2024-01-15T10:00:00+00:00")

    repo = MetadataRepository(mock_session)
    result = await repo.get("key")

    assert isinstance(result, datetime)
    assert result == datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
    assert result.tzinfo is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_returns_plain_string_as_is(mock_session: MagicMock) -> None:
    """Non-datetime string is returned unchanged."""
    _make_orm_result(mock_session, "some-plain-string")

    repo = MetadataRepository(mock_session)
    result = await repo.get("key")

    assert result == "some-plain-string"
    assert isinstance(result, str)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_parses_naive_datetime_adds_utc(mock_session: MagicMock) -> None:
    """Naive datetime string (no tz info) gets UTC assigned."""
    _make_orm_result(mock_session, "2024-01-15T10:00:00")

    repo = MetadataRepository(mock_session)
    result = await repo.get("key")

    assert isinstance(result, datetime)
    assert result.tzinfo is UTC
    assert result == datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_returns_none_when_key_missing(mock_session: MagicMock) -> None:
    """Returns None when key not found."""
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = result_mock

    repo = MetadataRepository(mock_session)
    result = await repo.get("missing_key")

    assert result is None
