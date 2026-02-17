"""Tests for DiscoveryHistoryRepository."""

from datetime import UTC, datetime, timedelta

import pytest

from src.daemon.state import DiscoveryHistoryRecord
from src.discovery.models import DiscoverySource


@pytest.fixture
def discovery_record() -> DiscoveryHistoryRecord:
    """Create sample discovery history record."""
    return DiscoveryHistoryRecord(
        symbol="NVDA",
        discovered_at=datetime.now(UTC),
        composite_score=0.78,
        sources=[DiscoverySource.TECHNICAL_SCREENING, DiscoverySource.EARNINGS_UPCOMING],
        added_to_watchlist=True,
        ttl_expires_at=datetime.now(UTC) + timedelta(days=7),
        first_signal="BUY",
        first_signal_date=datetime.now(UTC),
        outcome_7d=5.2,
        outcome_30d=12.8,
    )


@pytest.mark.asyncio
async def test_create_discovery_record(async_session, discovery_record: DiscoveryHistoryRecord) -> None:
    """Test creating discovery history record."""
    from src.database.repositories.discovery import DiscoveryHistoryRepository

    repo = DiscoveryHistoryRepository(async_session)
    result = await repo.create(discovery_record)

    assert result.symbol == discovery_record.symbol
    assert result.composite_score == discovery_record.composite_score
    assert result.added_to_watchlist is True


@pytest.mark.asyncio
async def test_get_by_symbol(async_session, discovery_record: DiscoveryHistoryRecord) -> None:
    """Test retrieving discovery records by symbol."""
    from src.database.repositories.discovery import DiscoveryHistoryRepository

    repo = DiscoveryHistoryRepository(async_session)
    await repo.create(discovery_record)

    results = await repo.get_by_symbol("NVDA")

    assert len(results) == 1
    assert results[0].symbol == "NVDA"
    assert results[0].composite_score == 0.78


@pytest.mark.asyncio
async def test_update_outcome(async_session, discovery_record: DiscoveryHistoryRecord) -> None:
    """Test updating outcome metrics."""
    from src.database.repositories.discovery import DiscoveryHistoryRepository

    repo = DiscoveryHistoryRepository(async_session)
    await repo.create(discovery_record)

    result = await repo.update_outcome("NVDA", outcome_7d=8.5, outcome_30d=15.2)

    assert result is not None
    assert result.outcome_7d == 8.5
    assert result.outcome_30d == 15.2


@pytest.mark.asyncio
async def test_delete_before(async_session, discovery_record: DiscoveryHistoryRecord) -> None:
    """Test cleanup of old discovery records."""
    from src.database.repositories.discovery import DiscoveryHistoryRepository

    repo = DiscoveryHistoryRepository(async_session)
    await repo.create(discovery_record)

    cutoff = datetime.now(UTC) + timedelta(days=1)
    deleted_count = await repo.delete_before(cutoff)

    assert deleted_count == 1


@pytest.mark.asyncio
async def test_mark_added_to_watchlist(async_session) -> None:
    """Test marking discovery record as added to watchlist."""
    from src.database.repositories.discovery import DiscoveryHistoryRepository

    repo = DiscoveryHistoryRepository(async_session)
    discovered_at = datetime.now(UTC)
    record = DiscoveryHistoryRecord(
        symbol="AAPL",
        discovered_at=discovered_at,
        composite_score=0.65,
        sources=[DiscoverySource.EVENT_WATCHLIST],
        added_to_watchlist=False,
        ttl_expires_at=datetime.now(UTC) + timedelta(days=3),
    )
    await repo.create(record)

    result = await repo.mark_added_to_watchlist("AAPL", discovered_at)
    assert result is True

    records = await repo.get_by_symbol("AAPL")
    assert len(records) == 1
    assert records[0].added_to_watchlist is True


@pytest.mark.asyncio
async def test_mark_added_to_watchlist_missing_record(async_session) -> None:
    """Test marking non-existent discovery record returns False."""
    from src.database.repositories.discovery import DiscoveryHistoryRepository

    repo = DiscoveryHistoryRepository(async_session)
    result = await repo.mark_added_to_watchlist("NONEXISTENT", datetime.now(UTC))
    assert result is False
