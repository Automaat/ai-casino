"""Unit tests for GamePlanTask retry logic."""

from datetime import UTC, date, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.game_plan import GamePlan
from src.daemon.config.portfolio import GamePlanConfig
from src.v1.tasks.implementations.game_plan import GamePlanTask


def _make_game_plan(confidence: float) -> GamePlan:
    return GamePlan(
        date=date.today(),
        priority_symbols=["AAPL", "TSLA"],
        risk_stance="NEUTRAL",
        sector_focus=["Technology"],
        key_levels={"AAPL": 175.0},
        overnight_summary="Futures flat",
        reasoning="Tech momentum intact",
        confidence=confidence,
        generated_at=datetime.now(UTC),
    )


def _make_task(min_confidence: float = 0.7, max_retries: int = 3) -> GamePlanTask:
    config = GamePlanConfig(min_confidence=min_confidence, max_retries=max_retries)
    agent = MagicMock()
    state = MagicMock()
    state.record_game_plan = AsyncMock()
    state.get_last_game_plan = AsyncMock(return_value=None)
    broker_manager = MagicMock()
    broker_manager.get_merged_watchlist = AsyncMock(return_value=["AAPL", "TSLA"])
    scheduler = MagicMock()
    scheduler.timezone = "America/New_York"
    return GamePlanTask(
        agent=agent,
        state=state,
        broker_manager=broker_manager,
        config=config,
        scheduler=scheduler,
    )


@pytest.mark.unit
async def test_stops_early_when_confidence_sufficient() -> None:
    """Exits retry loop immediately when confidence >= min_confidence."""
    task = _make_task(min_confidence=0.6, max_retries=3)
    high_confidence_plan = _make_game_plan(confidence=0.8)
    task._agent.generate = AsyncMock(return_value=high_confidence_plan)

    result = await task.execute()

    assert result.success is True
    task._agent.generate.assert_awaited_once()
    task._state.record_game_plan.assert_awaited_once()


@pytest.mark.unit
async def test_retries_until_max_attempts_on_low_confidence() -> None:
    """Retries max_retries times when confidence stays below min_confidence."""
    max_retries = 3
    task = _make_task(min_confidence=0.7, max_retries=max_retries)
    low_confidence_plan = _make_game_plan(confidence=0.4)
    task._agent.generate = AsyncMock(return_value=low_confidence_plan)

    result = await task.execute()

    assert result.success is True
    assert task._agent.generate.await_count == max_retries
    task._state.record_game_plan.assert_awaited_once()


@pytest.mark.unit
async def test_persists_last_low_confidence_plan() -> None:
    """Persists the last candidate even when all attempts have low confidence."""
    task = _make_task(min_confidence=0.8, max_retries=2)
    first_plan = _make_game_plan(confidence=0.3)
    second_plan = _make_game_plan(confidence=0.5)
    task._agent.generate = AsyncMock(side_effect=[first_plan, second_plan])

    result = await task.execute()

    assert result.success is True
    persisted = task._state.record_game_plan.call_args[0][0]
    assert persisted.confidence == second_plan.confidence


@pytest.mark.unit
async def test_succeeds_on_second_attempt() -> None:
    """Stops retrying once a high-confidence plan is returned."""
    task = _make_task(min_confidence=0.7, max_retries=3)
    low = _make_game_plan(confidence=0.4)
    high = _make_game_plan(confidence=0.9)
    task._agent.generate = AsyncMock(side_effect=[low, high])

    result = await task.execute()

    assert result.success is True
    assert task._agent.generate.await_count == 2
