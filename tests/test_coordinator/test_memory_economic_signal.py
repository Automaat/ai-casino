"""Tests for CoordinatorMemory.get_current_economic_signal()."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.daemon.events import (
    EconomicEvent,
    EconomicEventSignal,
    EconomicImpact,
    EconomicRecommendation,
    EconomicRiskLevel,
)
from src.v1.coordinator.memory import CoordinatorMemory


def _make_signal(
    risk_level: EconomicRiskLevel = EconomicRiskLevel.HIGH,
    recommendation: EconomicRecommendation = EconomicRecommendation.AVOID_NEW_POSITIONS,
    reason: str = "High-impact event imminent",
    computed_at: datetime | None = None,
    avoid_until: datetime | None = None,
    events: list[EconomicEvent] | None = None,
) -> EconomicEventSignal:
    """Build EconomicEventSignal for testing."""
    if computed_at is None:
        computed_at = datetime.now(UTC)
    if events is None:
        events = [
            EconomicEvent(
                event_id="US_CPI_2026",
                country="US",
                event="CPI",
                impact=EconomicImpact.HIGH,
                scheduled_at=datetime.now(UTC) + timedelta(hours=1),
            )
        ]
    return EconomicEventSignal(
        upcoming_events=events,
        risk_level=risk_level,
        recommendation=recommendation,
        reason=reason,
        computed_at=computed_at,
        avoid_until=avoid_until,
    )


def _make_memory(
    tmp_path,
    signal: EconomicEventSignal | None = None,
    raise_exc: Exception | None = None,
) -> tuple[CoordinatorMemory, MagicMock]:
    """Create CoordinatorMemory with mocked engine + repo class."""
    mock_engine = MagicMock()
    mock_session = MagicMock()
    mock_engine.session.return_value = mock_session

    mock_repo = AsyncMock()
    mock_repo.owns_session = False
    mock_repo.__aenter__ = AsyncMock(return_value=mock_repo)
    mock_repo.__aexit__ = AsyncMock(return_value=False)
    if raise_exc:
        mock_repo.get_latest = AsyncMock(side_effect=raise_exc)
    else:
        mock_repo.get_latest = AsyncMock(return_value=signal)

    mock_repo_class = MagicMock(return_value=mock_repo)

    memory = CoordinatorMemory(
        memory_file=tmp_path / "memory.jsonl",
        database_engine=mock_engine,
    )
    return memory, mock_repo_class


@pytest.mark.unit
@pytest.mark.asyncio
async def test_returns_none_without_engine(tmp_path) -> None:
    """Returns None when no database engine configured."""
    memory = CoordinatorMemory(memory_file=tmp_path / "memory.jsonl")
    result = await memory.get_current_economic_signal()
    assert result is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_returns_none_when_no_signal(tmp_path) -> None:
    """Returns None when database has no signal."""
    memory, mock_repo_class = _make_memory(tmp_path, signal=None)
    with patch(
        "src.database.repositories.economic_calendar.EconomicCalendarSignalRepository", mock_repo_class
    ):
        result = await memory.get_current_economic_signal()
    assert result is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_returns_none_for_stale_signal(tmp_path) -> None:
    """Returns None when signal is older than 3 hours."""
    stale_signal = _make_signal(computed_at=datetime.now(UTC) - timedelta(hours=4))
    memory, mock_repo_class = _make_memory(tmp_path, signal=stale_signal)
    with patch(
        "src.database.repositories.economic_calendar.EconomicCalendarSignalRepository", mock_repo_class
    ):
        result = await memory.get_current_economic_signal()
    assert result is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_returns_none_for_low_risk(tmp_path) -> None:
    """Returns None when risk level is LOW."""
    low_signal = _make_signal(
        risk_level=EconomicRiskLevel.LOW,
        recommendation=EconomicRecommendation.TRADE_NORMALLY,
    )
    memory, mock_repo_class = _make_memory(tmp_path, signal=low_signal)
    with patch(
        "src.database.repositories.economic_calendar.EconomicCalendarSignalRepository", mock_repo_class
    ):
        result = await memory.get_current_economic_signal()
    assert result is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_returns_formatted_string_for_high_risk(tmp_path) -> None:
    """Returns formatted string with uppercase recommendation name for HIGH risk."""
    signal = _make_signal(
        risk_level=EconomicRiskLevel.HIGH,
        recommendation=EconomicRecommendation.AVOID_NEW_POSITIONS,
        reason="Fed meeting imminent",
    )
    memory, mock_repo_class = _make_memory(tmp_path, signal=signal)
    with patch(
        "src.database.repositories.economic_calendar.EconomicCalendarSignalRepository", mock_repo_class
    ):
        result = await memory.get_current_economic_signal()

    assert result is not None
    assert "ECONOMIC RISK: HIGH" in result
    assert "AVOID_NEW_POSITIONS" in result  # .name is uppercase
    assert "avoid_new_positions" not in result  # .value is NOT used
    assert "Fed meeting imminent" in result
    assert "CPI" in result


@pytest.mark.unit
@pytest.mark.asyncio
async def test_returns_formatted_string_for_medium_risk(tmp_path) -> None:
    """Returns formatted string for MEDIUM risk."""
    signal = _make_signal(
        risk_level=EconomicRiskLevel.MEDIUM,
        recommendation=EconomicRecommendation.REDUCE_SIZE,
        reason="GDP release ahead",
    )
    memory, mock_repo_class = _make_memory(tmp_path, signal=signal)
    with patch(
        "src.database.repositories.economic_calendar.EconomicCalendarSignalRepository", mock_repo_class
    ):
        result = await memory.get_current_economic_signal()

    assert result is not None
    assert "ECONOMIC RISK: MEDIUM" in result
    assert "REDUCE_SIZE" in result


@pytest.mark.unit
@pytest.mark.asyncio
async def test_includes_avoid_until_in_output(tmp_path) -> None:
    """Formatted string includes avoid_until when set."""
    avoid_time = datetime(2026, 2, 22, 15, 30, tzinfo=UTC)
    signal = _make_signal(
        risk_level=EconomicRiskLevel.HIGH,
        recommendation=EconomicRecommendation.AVOID_NEW_POSITIONS,
        reason="Fed decision",
        avoid_until=avoid_time,
    )
    memory, mock_repo_class = _make_memory(tmp_path, signal=signal)
    with patch(
        "src.database.repositories.economic_calendar.EconomicCalendarSignalRepository", mock_repo_class
    ):
        result = await memory.get_current_economic_signal()

    assert result is not None
    assert "Avoid until: 15:30 UTC" in result


@pytest.mark.unit
@pytest.mark.asyncio
async def test_swallows_db_exception_and_returns_none(tmp_path) -> None:
    """Returns None and does not propagate database exceptions."""
    memory, mock_repo_class = _make_memory(tmp_path, raise_exc=RuntimeError("DB connection lost"))
    with patch(
        "src.database.repositories.economic_calendar.EconomicCalendarSignalRepository", mock_repo_class
    ):
        result = await memory.get_current_economic_signal()
    assert result is None
