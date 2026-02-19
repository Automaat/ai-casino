"""Economic calendar watcher - monitors macro events to manage position risk."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from loguru import logger

from src.daemon.events import (
    EconomicEvent,
    EconomicEventSignal,
    EconomicImpact,
    EconomicRecommendation,
    EconomicRiskLevel,
)
from src.data.economic_calendar import EconomicCalendarEntry, EconomicCalendarFetcher
from src.watchers.base import PeriodicWatcher


@dataclass
class EconomicCalendarWatcherConfig:
    """Configuration for economic calendar watcher."""

    poll_interval_minutes: int = 60
    lookahead_hours: int = 24
    high_impact_avoid_hours: float = 2.0


class EconomicCalendarWatcher(PeriodicWatcher):
    """Background service that polls FRED economic calendar and computes risk signals."""

    def __init__(
        self,
        fetcher: EconomicCalendarFetcher,
        config: EconomicCalendarWatcherConfig,
    ) -> None:
        """Initialize economic calendar watcher.

        Args:
            fetcher: FRED economic calendar fetcher
            config: Watcher configuration
        """
        super().__init__(poll_interval=config.poll_interval_minutes * 60)
        self._fetcher = fetcher
        self._config = config
        self._current_signal: EconomicEventSignal | None = None

    @property
    def name(self) -> str:
        """Watcher display name."""
        return "EconomicCalendarWatcher"

    @property
    def current_signal(self) -> EconomicEventSignal | None:
        """Return current economic event signal (sync, no await)."""
        return self._current_signal

    def _classify_impact(self, raw: str) -> EconomicImpact:
        """Convert raw impact string to EconomicImpact enum.

        Args:
            raw: Raw impact string ("high", "medium", "low", "1", "2", "3")

        Returns:
            EconomicImpact enum value
        """
        mapping: dict[str, EconomicImpact] = {
            "high": EconomicImpact.HIGH,
            "1": EconomicImpact.HIGH,
            "medium": EconomicImpact.MEDIUM,
            "2": EconomicImpact.MEDIUM,
            "low": EconomicImpact.LOW,
            "3": EconomicImpact.LOW,
        }
        return mapping.get(raw.lower(), EconomicImpact.LOW)

    def _filter_upcoming(self, entries: list[EconomicCalendarEntry]) -> list[EconomicEvent]:
        """Filter entries to US events within lookahead window with HIGH or MEDIUM impact.

        Args:
            entries: Raw calendar entries

        Returns:
            Filtered and converted EconomicEvent list
        """
        now = datetime.now(UTC)
        window_end = now + timedelta(hours=self._config.lookahead_hours)

        events: list[EconomicEvent] = []
        for entry in entries:
            if entry.country != "US":
                continue
            impact = self._classify_impact(entry.impact)
            if impact == EconomicImpact.LOW:
                continue
            if entry.scheduled_at < now or entry.scheduled_at > window_end:
                continue

            event_id = f"US_{entry.event}_{entry.scheduled_at.isoformat()}"
            events.append(
                EconomicEvent(
                    event_id=event_id,
                    country=entry.country,
                    event=entry.event,
                    impact=impact,
                    scheduled_at=entry.scheduled_at,
                    actual=entry.actual,
                )
            )

        events.sort(key=lambda e: e.scheduled_at)
        return events

    def _compute_signal(self, events: list[EconomicEvent]) -> EconomicEventSignal:
        """Compute risk signal from upcoming events.

        Args:
            events: Upcoming economic events (filtered)

        Returns:
            EconomicEventSignal with risk assessment
        """
        now = datetime.now(UTC)

        if not events:
            return EconomicEventSignal(
                upcoming_events=[],
                risk_level=EconomicRiskLevel.LOW,
                recommendation=EconomicRecommendation.TRADE_NORMALLY,
                reason="No high/medium impact events in lookahead window",
            )

        high_impact = [e for e in events if e.impact == EconomicImpact.HIGH]
        medium_impact = [e for e in events if e.impact == EconomicImpact.MEDIUM]

        # HIGH impact event within avoid_hours → AVOID
        for event in high_impact:
            hours_away = (event.scheduled_at - now).total_seconds() / 3600
            if hours_away <= self._config.high_impact_avoid_hours:
                return EconomicEventSignal(
                    upcoming_events=events,
                    risk_level=EconomicRiskLevel.HIGH,
                    recommendation=EconomicRecommendation.AVOID_NEW_POSITIONS,
                    reason=(f"{event.event} in {hours_away:.1f}h - avoid new positions until after release"),
                    avoid_until=event.scheduled_at + timedelta(hours=1),
                )

        # HIGH impact event within lookahead → REDUCE_SIZE
        if high_impact:
            nearest = high_impact[0]
            hours_away = (nearest.scheduled_at - now).total_seconds() / 3600
            return EconomicEventSignal(
                upcoming_events=events,
                risk_level=EconomicRiskLevel.MEDIUM,
                recommendation=EconomicRecommendation.REDUCE_SIZE,
                reason=(
                    f"{nearest.event} in {hours_away:.1f}h - reduce position sizes ahead of high-impact event"
                ),
            )

        # MEDIUM impact event within 4h → REDUCE_SIZE
        for event in medium_impact:
            hours_away = (event.scheduled_at - now).total_seconds() / 3600
            if hours_away <= 4.0:
                return EconomicEventSignal(
                    upcoming_events=events,
                    risk_level=EconomicRiskLevel.MEDIUM,
                    recommendation=EconomicRecommendation.REDUCE_SIZE,
                    reason=(f"{event.event} in {hours_away:.1f}h - medium impact event approaching"),
                )

        return EconomicEventSignal(
            upcoming_events=events,
            risk_level=EconomicRiskLevel.LOW,
            recommendation=EconomicRecommendation.TRADE_NORMALLY,
            reason="Events present but not imminent enough to restrict trading",
        )

    async def _tick(self) -> None:
        """Fetch calendar and compute signal."""
        now = datetime.now(UTC)
        from_date = now.strftime("%Y-%m-%d")
        to_date = (now + timedelta(hours=self._config.lookahead_hours + 24)).strftime("%Y-%m-%d")

        entries = await asyncio.to_thread(self._fetcher.fetch_economic_calendar, from_date, to_date)
        events = self._filter_upcoming(entries)
        signal = self._compute_signal(events)
        self._current_signal = signal

        logger.info(
            f"Economic calendar assessed: risk={signal.risk_level}, "
            f"events={len(events)}, recommendation={signal.recommendation}"
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"EconomicCalendarWatcher(running={self.running}, signal={self._current_signal})"
