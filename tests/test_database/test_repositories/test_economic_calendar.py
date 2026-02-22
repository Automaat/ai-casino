"""Tests for EconomicCalendarSignalRepository."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.daemon.events import (
    EconomicEvent,
    EconomicEventSignal,
    EconomicImpact,
    EconomicRecommendation,
    EconomicRiskLevel,
)
from src.database.repositories.economic_calendar import EconomicCalendarSignalRepository


@pytest.fixture
def mock_session() -> AsyncMock:
    """Create mock async session."""
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.execute = AsyncMock()
    return session


@pytest.fixture
def sample_event() -> EconomicEvent:
    """Create sample EconomicEvent."""
    return EconomicEvent(
        event_id="US_CPI_2026",
        country="US",
        event="CPI",
        impact=EconomicImpact.HIGH,
        scheduled_at=datetime(2026, 2, 22, 13, 30, tzinfo=UTC),
        actual="3.1%",
        estimate="3.0%",
        prev="2.9%",
    )


@pytest.fixture
def sample_signal(sample_event: EconomicEvent) -> EconomicEventSignal:
    """Create sample EconomicEventSignal."""
    return EconomicEventSignal(
        upcoming_events=[sample_event],
        risk_level=EconomicRiskLevel.HIGH,
        recommendation=EconomicRecommendation.AVOID_NEW_POSITIONS,
        reason="High-impact CPI release imminent",
        computed_at=datetime(2026, 2, 22, 12, 0, tzinfo=UTC),
        avoid_until=datetime(2026, 2, 22, 15, 30, tzinfo=UTC),
    )


@pytest.mark.asyncio
class TestEconomicCalendarSignalRepositoryCreate:
    """Tests for EconomicCalendarSignalRepository.create()."""

    async def test_create_adds_orm_and_commits(
        self, mock_session: AsyncMock, sample_signal: EconomicEventSignal
    ) -> None:
        """create() adds ORM record and commits session."""
        repo = EconomicCalendarSignalRepository(mock_session)
        result = await repo.create(sample_signal)

        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        assert result is sample_signal

    async def test_create_serializes_events_to_json(
        self, mock_session: AsyncMock, sample_signal: EconomicEventSignal
    ) -> None:
        """create() serializes upcoming_events as list of dicts with correct JSONB shape."""
        repo = EconomicCalendarSignalRepository(mock_session)
        await repo.create(sample_signal)

        orm = mock_session.add.call_args[0][0]
        assert isinstance(orm.upcoming_events, list)
        assert len(orm.upcoming_events) == 1

        event_dict = orm.upcoming_events[0]
        assert event_dict["event_id"] == "US_CPI_2026"
        assert event_dict["country"] == "US"
        assert event_dict["event"] == "CPI"
        assert event_dict["impact"] == "high"  # StrEnum .value
        assert event_dict["scheduled_at"] == "2026-02-22T13:30:00+00:00"
        assert event_dict["actual"] == "3.1%"
        assert event_dict["estimate"] == "3.0%"
        assert event_dict["prev"] == "2.9%"

    async def test_create_sets_risk_and_recommendation_values(
        self, mock_session: AsyncMock, sample_signal: EconomicEventSignal
    ) -> None:
        """create() stores StrEnum .value for risk_level and recommendation."""
        repo = EconomicCalendarSignalRepository(mock_session)
        await repo.create(sample_signal)

        orm = mock_session.add.call_args[0][0]
        assert orm.risk_level == "HIGH"
        assert orm.recommendation == "avoid_new_positions"
        assert orm.reason == "High-impact CPI release imminent"
        assert orm.avoid_until == datetime(2026, 2, 22, 15, 30, tzinfo=UTC)

    async def test_create_empty_events_list(self, mock_session: AsyncMock) -> None:
        """create() handles signal with no upcoming events."""
        signal = EconomicEventSignal(
            upcoming_events=[],
            risk_level=EconomicRiskLevel.LOW,
            recommendation=EconomicRecommendation.TRADE_NORMALLY,
            reason="No events",
            computed_at=datetime.now(UTC),
        )
        repo = EconomicCalendarSignalRepository(mock_session)
        result = await repo.create(signal)

        orm = mock_session.add.call_args[0][0]
        assert orm.upcoming_events == []
        assert result is signal


@pytest.mark.asyncio
class TestEconomicCalendarSignalRepositoryGetLatest:
    """Tests for EconomicCalendarSignalRepository.get_latest()."""

    async def test_get_latest_returns_none_when_empty(self, mock_session: AsyncMock) -> None:
        """get_latest() returns None when no records exist."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        repo = EconomicCalendarSignalRepository(mock_session)
        result = await repo.get_latest()

        assert result is None

    async def test_get_latest_returns_domain_model(self, mock_session: AsyncMock) -> None:
        """get_latest() deserializes ORM to domain EconomicEventSignal."""
        orm = MagicMock()
        orm.risk_level = "HIGH"
        orm.recommendation = "avoid_new_positions"
        orm.reason = "High-impact CPI release imminent"
        orm.computed_at = datetime(2026, 2, 22, 12, 0, tzinfo=UTC)
        orm.avoid_until = datetime(2026, 2, 22, 15, 30, tzinfo=UTC)
        orm.upcoming_events = [
            {
                "event_id": "US_CPI_2026",
                "country": "US",
                "event": "CPI",
                "impact": "high",
                "scheduled_at": "2026-02-22T13:30:00+00:00",
                "actual": "3.1%",
                "estimate": "3.0%",
                "prev": "2.9%",
            }
        ]

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = orm
        mock_session.execute.return_value = mock_result

        repo = EconomicCalendarSignalRepository(mock_session)
        result = await repo.get_latest()

        assert result is not None
        assert result.risk_level == EconomicRiskLevel.HIGH
        assert result.recommendation == EconomicRecommendation.AVOID_NEW_POSITIONS
        assert result.reason == "High-impact CPI release imminent"
        assert result.avoid_until == datetime(2026, 2, 22, 15, 30, tzinfo=UTC)
        assert len(result.upcoming_events) == 1
        assert result.upcoming_events[0].event == "CPI"
        assert result.upcoming_events[0].impact == EconomicImpact.HIGH


class TestEconomicCalendarSignalRepositoryToDomain:
    """Tests for EconomicCalendarSignalRepository._to_domain()."""

    def test_to_domain_empty_events(self, mock_session: AsyncMock) -> None:
        """_to_domain() handles empty upcoming_events list."""
        orm = MagicMock()
        orm.risk_level = "LOW"
        orm.recommendation = "trade_normally"
        orm.reason = "No events"
        orm.computed_at = datetime.now(UTC)
        orm.avoid_until = None
        orm.upcoming_events = []

        repo = EconomicCalendarSignalRepository(mock_session)
        result = repo._to_domain(orm)

        assert result.risk_level == EconomicRiskLevel.LOW
        assert result.recommendation == EconomicRecommendation.TRADE_NORMALLY
        assert result.upcoming_events == []
        assert result.avoid_until is None

    def test_to_domain_missing_scheduled_at_falls_back_to_now(self, mock_session: AsyncMock) -> None:
        """_to_domain() uses datetime.now() when scheduled_at is missing."""
        before = datetime.now(UTC) - timedelta(seconds=1)
        orm = MagicMock()
        orm.risk_level = "MEDIUM"
        orm.recommendation = "reduce_size"
        orm.reason = "Test"
        orm.computed_at = datetime.now(UTC)
        orm.avoid_until = None
        orm.upcoming_events = [
            {
                "event_id": "US_NFP_2026",
                "country": "US",
                "event": "NFP",
                "impact": "high",
                "scheduled_at": "",  # empty triggers fallback
            }
        ]

        repo = EconomicCalendarSignalRepository(mock_session)
        result = repo._to_domain(orm)

        assert len(result.upcoming_events) == 1
        assert result.upcoming_events[0].scheduled_at >= before

    def test_repr(self, mock_session: AsyncMock) -> None:
        """__repr__ returns expected string."""
        repo = EconomicCalendarSignalRepository(mock_session)
        assert repr(repo) == "EconomicCalendarSignalRepository()"
