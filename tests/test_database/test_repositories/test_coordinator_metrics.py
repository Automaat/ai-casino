"""Tests for coordinator metrics repository."""

from datetime import UTC, datetime

import pytest

from src.database.repositories.coordinator_metrics import CoordinatorMetricsRepository
from src.v1.coordinator.metrics import CoordinatorCycleMetrics


@pytest.mark.asyncio
async def test_coordinator_metrics_repository_create(async_session):
    """Test creating coordinator metrics record."""
    repo = CoordinatorMetricsRepository(async_session)

    metrics = CoordinatorCycleMetrics(
        cycle_num=1,
        timestamp=datetime.now(UTC),
        symbols_analyzed=["AAPL", "MSFT"],
        tool_calls_made=5,
        trades_proposed=2,
        trades_executed=1,
        trades_pending=1,
        game_plan_generated=True,
        cycle_duration_seconds=45.5,
        patterns_detected=3,
    )

    result = await repo.create(metrics)

    assert result.id is not None
    assert result.created_at is not None
    assert result.cycle_num == 1
    assert result.symbols_analyzed == ["AAPL", "MSFT"]
    assert result.tool_calls_made == 5
    assert result.trades_executed == 1


@pytest.mark.asyncio
async def test_coordinator_metrics_repository_get_recent(async_session):
    """Test fetching recent coordinator metrics."""
    repo = CoordinatorMetricsRepository(async_session)

    # Create multiple records
    for i in range(3):
        metrics = CoordinatorCycleMetrics(
            cycle_num=i + 1,
            timestamp=datetime.now(UTC),
            symbols_analyzed=[f"SYM{i}"],
            tool_calls_made=i + 1,
            trades_proposed=i,
            trades_executed=i,
            trades_pending=0,
            game_plan_generated=True,
            cycle_duration_seconds=30.0 + i,
            patterns_detected=i,
        )
        await repo.create(metrics)

    # Fetch recent
    recent = await repo.get_recent(limit=2)

    assert len(recent) == 2
    # Should be newest first
    assert recent[0].cycle_num == 3
    assert recent[1].cycle_num == 2


@pytest.mark.asyncio
async def test_coordinator_metrics_repository_get_by_cycle_num(async_session):
    """Test filtering by cycle number."""
    repo = CoordinatorMetricsRepository(async_session)

    # Create records with different cycle numbers
    for i in range(3):
        metrics = CoordinatorCycleMetrics(
            cycle_num=1,  # All same cycle
            timestamp=datetime.now(UTC),
            symbols_analyzed=["TEST"],
            tool_calls_made=i + 1,
            trades_proposed=i,
            trades_executed=i,
            trades_pending=0,
            game_plan_generated=False,
            cycle_duration_seconds=20.0,
            patterns_detected=0,
        )
        await repo.create(metrics)

    # Different cycle
    other = CoordinatorCycleMetrics(
        cycle_num=2,
        timestamp=datetime.now(UTC),
        symbols_analyzed=["OTHER"],
        tool_calls_made=99,
        trades_proposed=0,
        trades_executed=0,
        trades_pending=0,
        game_plan_generated=False,
        cycle_duration_seconds=10.0,
        patterns_detected=0,
    )
    await repo.create(other)

    # Fetch by cycle
    cycle_1 = await repo.get_by_cycle_num(1)

    assert len(cycle_1) == 3
    assert all(m.cycle_num == 1 for m in cycle_1)
