"""Tests for PositionRecordRepository."""

from datetime import UTC, datetime

import pytest

from src.daemon.positions import PositionRecord


@pytest.fixture
def position_record() -> PositionRecord:
    """Create sample position record."""
    return PositionRecord(
        symbol="TSLA",
        entry_timestamp=datetime.now(UTC),
        entry_price=250.50,
        entry_signal="BUY",
        entry_confidence=0.90,
        current_qty=10.0,
        current_stop_loss=240.00,
        initial_stop_loss=240.00,
        profit_targets=[260.0, 270.0, 280.0],
        days_held=5,
        last_updated=datetime.now(UTC),
        trailing_stop_activated=False,
        breakeven_activated=False,
    )


@pytest.mark.asyncio
async def test_create_position_record(async_session, position_record: PositionRecord) -> None:
    """Test creating position record."""
    from src.database.repositories.position import PositionRecordRepository

    repo = PositionRecordRepository(async_session)
    result = await repo.create(position_record)

    assert result.symbol == position_record.symbol
    assert result.entry_price == position_record.entry_price
    assert result.current_qty == position_record.current_qty


@pytest.mark.asyncio
async def test_update_position_record(async_session, position_record: PositionRecord) -> None:
    """Test updating position record."""
    from src.database.repositories.position import PositionRecordRepository

    repo = PositionRecordRepository(async_session)
    await repo.create(position_record)

    position_record.current_qty = 5.0
    position_record.current_stop_loss = 245.00
    result = await repo.update(position_record)

    assert result.current_qty == 5.0
    assert result.current_stop_loss == 245.00


@pytest.mark.asyncio
async def test_get_by_symbol(async_session, position_record: PositionRecord) -> None:
    """Test retrieving position by symbol."""
    from src.database.repositories.position import PositionRecordRepository

    repo = PositionRecordRepository(async_session)
    await repo.create(position_record)

    result = await repo.get_by_symbol("TSLA")

    assert result is not None
    assert result.symbol == "TSLA"
    assert result.entry_price == 250.50


@pytest.mark.asyncio
async def test_delete_by_symbol(async_session, position_record: PositionRecord) -> None:
    """Test deleting position by symbol."""
    from src.database.repositories.position import PositionRecordRepository

    repo = PositionRecordRepository(async_session)
    await repo.create(position_record)

    deleted = await repo.delete_by_symbol("TSLA")
    assert deleted is True

    result = await repo.get_by_symbol("TSLA")
    assert result is None
