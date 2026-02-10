"""Tests for DiscoveryHistoryRepository."""

from datetime import UTC, datetime, timedelta

import pytest

from src.daemon.state import DiscoveryHistoryRecord
from src.database.repositories.discovery import DiscoveryHistoryRepository
from src.discovery.models import DiscoverySource


@pytest.fixture
def discovery_record() -> DiscoveryHistoryRecord:
    """Create sample discovery history record."""
    return DiscoveryHistoryRecord(
        symbol="NVDA",
        discovered_at=datetime.now(UTC),
        composite_score=0.78,
        sources=[DiscoverySource.TECHNICAL_SCREENING, DiscoverySource.EARNINGS_CALENDAR],
        added_to_watchlist=True,
        ttl_expires_at=datetime.now(UTC) + timedelta(days=7),
        first_signal="BUY",
        first_signal_date=datetime.now(UTC),
        outcome_7d=5.2,
        outcome_30d=12.8,
    )


@pytest.mark.asyncio
async def test_create_discovery_record(discovery_record: DiscoveryHistoryRecord) -> None:
    """Test creating discovery history record."""
    pass


@pytest.mark.asyncio
async def test_get_by_symbol() -> None:
    """Test retrieving discovery records by symbol."""
    pass


@pytest.mark.asyncio
async def test_update_outcome() -> None:
    """Test updating outcome metrics."""
    pass


@pytest.mark.asyncio
async def test_delete_before() -> None:
    """Test cleanup of old discovery records."""
    pass
