"""Tests for PortfolioHealthRecordRepository."""

from datetime import UTC, datetime

import pytest

from src.daemon.state.models import PortfolioHealthRecord


@pytest.fixture
def health_record() -> PortfolioHealthRecord:
    """Create sample portfolio health record."""
    return PortfolioHealthRecord(
        timestamp=datetime.now(UTC),
        total_positions=3,
        portfolio_value=50000.0,
        cash_percent=15.0,
        max_concentration_percent=40.0,
        max_concentration_symbol="AAPL",
        total_pnl_percent=5.2,
        biggest_drawdown_symbol="TSLA",
        biggest_drawdown_percent=-8.5,
        health_status="WARNING",
        recommendations=["Reduce AAPL concentration"],
        constraints=["reduce:AAPL"],
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_health_record(async_session, health_record: PortfolioHealthRecord) -> None:
    """Test creating portfolio health record."""
    from src.database.repositories.portfolio_health import PortfolioHealthRecordRepository

    repo = PortfolioHealthRecordRepository(async_session)
    result = await repo.create(health_record)

    assert result.health_status == health_record.health_status
    assert result.total_positions == health_record.total_positions
    assert result.portfolio_value == health_record.portfolio_value


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_latest_returns_none_when_empty(async_session) -> None:
    """Test get_latest returns None when no records exist."""
    from src.database.repositories.portfolio_health import PortfolioHealthRecordRepository

    repo = PortfolioHealthRecordRepository(async_session)
    result = await repo.get_latest()

    assert result is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_latest_returns_most_recent(async_session) -> None:
    """Test get_latest returns most recent record."""
    from src.database.repositories.portfolio_health import PortfolioHealthRecordRepository

    repo = PortfolioHealthRecordRepository(async_session)

    older = PortfolioHealthRecord(
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        total_positions=1,
        portfolio_value=10000.0,
        cash_percent=20.0,
        max_concentration_percent=80.0,
        max_concentration_symbol="AAPL",
        total_pnl_percent=2.0,
        biggest_drawdown_symbol=None,
        biggest_drawdown_percent=0.0,
        health_status="HEALTHY",
        recommendations=[],
        constraints=[],
    )
    newer = PortfolioHealthRecord(
        timestamp=datetime(2025, 6, 1, tzinfo=UTC),
        total_positions=2,
        portfolio_value=25000.0,
        cash_percent=10.0,
        max_concentration_percent=60.0,
        max_concentration_symbol="TSLA",
        total_pnl_percent=-3.0,
        biggest_drawdown_symbol="TSLA",
        biggest_drawdown_percent=-5.0,
        health_status="CRITICAL",
        recommendations=["Reduce TSLA exposure"],
        constraints=["reduce:TSLA"],
    )
    await repo.create(older)
    await repo.create(newer)

    result = await repo.get_latest()

    assert result is not None
    assert result.health_status == "CRITICAL"
    assert result.total_positions == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_recent_returns_ordered_records(async_session) -> None:
    """Test get_recent returns records ordered by timestamp descending."""
    from src.database.repositories.portfolio_health import PortfolioHealthRecordRepository

    repo = PortfolioHealthRecordRepository(async_session)

    for i, status in enumerate(["HEALTHY", "WARNING", "CRITICAL"]):
        record = PortfolioHealthRecord(
            timestamp=datetime(2025, i + 1, 1, tzinfo=UTC),
            total_positions=i + 1,
            portfolio_value=10000.0 * (i + 1),
            cash_percent=20.0,
            max_concentration_percent=30.0,
            max_concentration_symbol="AAPL",
            total_pnl_percent=0.0,
            biggest_drawdown_symbol=None,
            biggest_drawdown_percent=0.0,
            health_status=status,
            recommendations=[],
            constraints=[],
        )
        await repo.create(record)

    results = await repo.get_recent(limit=2)

    assert len(results) == 2
    assert results[0].health_status == "CRITICAL"
    assert results[1].health_status == "WARNING"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_recent_empty(async_session) -> None:
    """Test get_recent returns empty list when no records."""
    from src.database.repositories.portfolio_health import PortfolioHealthRecordRepository

    repo = PortfolioHealthRecordRepository(async_session)
    results = await repo.get_recent()

    assert results == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_record_with_null_drawdown_symbol(async_session) -> None:
    """Test creating record with no biggest drawdown symbol."""
    from src.database.repositories.portfolio_health import PortfolioHealthRecordRepository

    repo = PortfolioHealthRecordRepository(async_session)
    record = PortfolioHealthRecord(
        timestamp=datetime.now(UTC),
        total_positions=0,
        portfolio_value=50000.0,
        cash_percent=100.0,
        max_concentration_percent=0.0,
        max_concentration_symbol="N/A",
        total_pnl_percent=0.0,
        biggest_drawdown_symbol=None,
        biggest_drawdown_percent=0.0,
        health_status="HEALTHY",
        recommendations=["Portfolio health is within all thresholds"],
        constraints=[],
    )
    result = await repo.create(record)

    assert result.biggest_drawdown_symbol is None
    assert result.total_positions == 0
