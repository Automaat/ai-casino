"""Unit tests for EconomicCalendarWatcher."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.daemon.events import (
    EconomicImpact,
    EconomicRecommendation,
    EconomicRiskLevel,
)
from src.data.economic_calendar import EconomicCalendarEntry, EconomicCalendarFetcher
from src.v1.watchers.economic_calendar_watcher import (
    EconomicCalendarWatcher,
    EconomicCalendarWatcherConfig,
)


@pytest.fixture
def mock_fetcher() -> MagicMock:
    """Mock EconomicCalendarFetcher."""
    return MagicMock(spec=EconomicCalendarFetcher)


@pytest.fixture
def config() -> EconomicCalendarWatcherConfig:
    """Default watcher config."""
    return EconomicCalendarWatcherConfig(
        poll_interval_minutes=60,
        lookahead_hours=24,
        high_impact_avoid_hours=2.0,
    )


@pytest.fixture
def watcher(mock_fetcher: MagicMock, config: EconomicCalendarWatcherConfig) -> EconomicCalendarWatcher:
    """Create EconomicCalendarWatcher with mocked fetcher."""
    return EconomicCalendarWatcher(fetcher=mock_fetcher, config=config)


def _entry(event: str, impact: str, hours_from_now: float, country: str = "US") -> EconomicCalendarEntry:
    """Helper to create EconomicCalendarEntry."""
    return EconomicCalendarEntry(
        country=country,
        event=event,
        impact=impact,
        scheduled_at=datetime.now(UTC) + timedelta(hours=hours_from_now),
    )


@pytest.mark.unit
def test_init(watcher: EconomicCalendarWatcher) -> None:
    """current_signal is None on init."""
    assert watcher.current_signal is None
    assert watcher.running is False


@pytest.mark.unit
def test_classify_impact(watcher: EconomicCalendarWatcher) -> None:
    """Impact classification handles numeric and string values."""
    assert watcher._classify_impact("1") == EconomicImpact.HIGH
    assert watcher._classify_impact("high") == EconomicImpact.HIGH
    assert watcher._classify_impact("HIGH") == EconomicImpact.HIGH
    assert watcher._classify_impact("2") == EconomicImpact.MEDIUM
    assert watcher._classify_impact("medium") == EconomicImpact.MEDIUM
    assert watcher._classify_impact("3") == EconomicImpact.LOW
    assert watcher._classify_impact("low") == EconomicImpact.LOW
    assert watcher._classify_impact("unknown") == EconomicImpact.LOW
    assert watcher._classify_impact("") == EconomicImpact.LOW


@pytest.mark.unit
def test_filter_upcoming_excludes_past(watcher: EconomicCalendarWatcher) -> None:
    """Past events are excluded from upcoming."""
    entries = [
        _entry("CPI", "high", -1),  # 1 hour ago
        _entry("NFP", "high", 12),  # 12 hours ahead
    ]
    result = watcher._filter_upcoming(entries)
    assert len(result) == 1
    assert result[0].event == "NFP"


@pytest.mark.unit
def test_filter_upcoming_excludes_non_us(watcher: EconomicCalendarWatcher) -> None:
    """Non-US events are excluded."""
    entries = [
        _entry("ECB Rate Decision", "high", 5, country="EU"),
        _entry("CPI", "high", 5, country="US"),
    ]
    result = watcher._filter_upcoming(entries)
    assert len(result) == 1
    assert result[0].country == "US"


@pytest.mark.unit
def test_filter_upcoming_excludes_low_impact(watcher: EconomicCalendarWatcher) -> None:
    """LOW impact events are excluded."""
    entries = [
        _entry("Minor Report", "low", 5),
        _entry("CPI", "high", 5),
    ]
    result = watcher._filter_upcoming(entries)
    assert len(result) == 1
    assert result[0].event == "CPI"


@pytest.mark.unit
def test_compute_signal_avoid_imminent(watcher: EconomicCalendarWatcher) -> None:
    """HIGH impact event within avoid_hours triggers AVOID_NEW_POSITIONS with HIGH risk."""
    entries = [_entry("CPI", "high", 1.5)]  # 1.5h away, within 2h threshold
    events = watcher._filter_upcoming(entries)
    signal = watcher._compute_signal(events)

    assert signal.risk_level == EconomicRiskLevel.HIGH
    assert signal.recommendation == EconomicRecommendation.AVOID_NEW_POSITIONS
    assert signal.avoid_until is not None


@pytest.mark.unit
def test_compute_signal_reduce_upcoming(watcher: EconomicCalendarWatcher) -> None:
    """HIGH impact event within lookahead but beyond avoid_hours triggers REDUCE_SIZE."""
    entries = [_entry("Nonfarm Payroll", "high", 12)]  # 12h away, beyond 2h threshold
    events = watcher._filter_upcoming(entries)
    signal = watcher._compute_signal(events)

    assert signal.risk_level == EconomicRiskLevel.MEDIUM
    assert signal.recommendation == EconomicRecommendation.REDUCE_SIZE


@pytest.mark.unit
def test_compute_signal_medium_imminent(watcher: EconomicCalendarWatcher) -> None:
    """MEDIUM impact event within 4h triggers REDUCE_SIZE."""
    entries = [_entry("GDP", "medium", 3)]  # 3h away, within 4h medium threshold
    events = watcher._filter_upcoming(entries)
    signal = watcher._compute_signal(events)

    assert signal.risk_level == EconomicRiskLevel.MEDIUM
    assert signal.recommendation == EconomicRecommendation.REDUCE_SIZE


@pytest.mark.unit
def test_compute_signal_low(watcher: EconomicCalendarWatcher) -> None:
    """No events → TRADE_NORMALLY with LOW risk."""
    signal = watcher._compute_signal([])

    assert signal.risk_level == EconomicRiskLevel.LOW
    assert signal.recommendation == EconomicRecommendation.TRADE_NORMALLY
    assert signal.upcoming_events == []


@pytest.mark.unit
def test_compute_signal_medium_not_imminent(watcher: EconomicCalendarWatcher) -> None:
    """MEDIUM impact event beyond 4h → TRADE_NORMALLY."""
    entries = [_entry("GDP", "medium", 10)]  # 10h away, beyond 4h threshold
    events = watcher._filter_upcoming(entries)
    signal = watcher._compute_signal(events)

    assert signal.risk_level == EconomicRiskLevel.LOW
    assert signal.recommendation == EconomicRecommendation.TRADE_NORMALLY


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fetch_and_assess(watcher: EconomicCalendarWatcher, mock_fetcher: MagicMock) -> None:
    """_tick calls fetcher via asyncio.to_thread and updates current_signal."""
    mock_fetcher.fetch_economic_calendar.return_value = []

    await watcher._tick()

    mock_fetcher.fetch_economic_calendar.assert_called_once()
    assert watcher.current_signal is not None
    assert watcher.current_signal.risk_level == EconomicRiskLevel.LOW


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_updates_signal(watcher: EconomicCalendarWatcher, mock_fetcher: MagicMock) -> None:
    """run() executes one cycle and sets current_signal."""
    import asyncio

    mock_fetcher.fetch_economic_calendar.return_value = []

    async def stop_after_first_cycle() -> None:
        await asyncio.sleep(0.05)
        watcher.running = False

    await asyncio.gather(
        watcher.run(),
        stop_after_first_cycle(),
    )

    assert watcher.current_signal is not None
    mock_fetcher.fetch_economic_calendar.assert_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tick_persists_signal_with_database_engine(
    mock_fetcher: MagicMock, config: EconomicCalendarWatcherConfig
) -> None:
    """_tick() calls repository.create() when database_engine is provided."""
    mock_engine = MagicMock()
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_engine.session.return_value = mock_session

    mock_repo = AsyncMock()
    mock_repo.create = AsyncMock()

    mock_fetcher.fetch_economic_calendar.return_value = []
    watcher = EconomicCalendarWatcher(fetcher=mock_fetcher, config=config, database_engine=mock_engine)

    with patch(
        "src.database.repositories.economic_calendar.EconomicCalendarSignalRepository",
        return_value=mock_repo,
    ):
        await watcher._tick()

    mock_repo.create.assert_called_once()
    created_signal = mock_repo.create.call_args[0][0]
    assert created_signal.risk_level == EconomicRiskLevel.LOW


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tick_swallows_persistence_exception(
    mock_fetcher: MagicMock, config: EconomicCalendarWatcherConfig
) -> None:
    """_tick() swallows DB exception and still updates current_signal."""
    mock_engine = MagicMock()
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_engine.session.return_value = mock_session

    mock_repo = AsyncMock()
    mock_repo.create = AsyncMock(side_effect=RuntimeError("DB write failed"))

    mock_fetcher.fetch_economic_calendar.return_value = []
    watcher = EconomicCalendarWatcher(fetcher=mock_fetcher, config=config, database_engine=mock_engine)

    with patch(
        "src.database.repositories.economic_calendar.EconomicCalendarSignalRepository",
        return_value=mock_repo,
    ):
        await watcher._tick()  # must not raise

    assert watcher.current_signal is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tick_skips_persistence_without_database_engine(
    mock_fetcher: MagicMock, config: EconomicCalendarWatcherConfig
) -> None:
    """_tick() does not attempt DB persistence when database_engine is None."""
    mock_fetcher.fetch_economic_calendar.return_value = []
    watcher = EconomicCalendarWatcher(fetcher=mock_fetcher, config=config)

    with patch(
        "src.database.repositories.economic_calendar.EconomicCalendarSignalRepository"
    ) as mock_repo_cls:
        await watcher._tick()

    mock_repo_cls.assert_not_called()
    assert watcher.current_signal is not None
