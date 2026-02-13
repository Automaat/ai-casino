"""Daemon integration for earnings calendar."""

from datetime import UTC, datetime, timedelta

from loguru import logger

from src.agents.models import EarningsFlags
from src.data.earnings import EarningsCalendar, EarningsCalendarFetcher, EarningsEvent


class DaemonEarningsCalendar:
    """Daemon wrapper for earnings calendar with context formatting."""

    def __init__(self) -> None:
        """Initialize daemon earnings calendar."""
        self._fetcher = EarningsCalendarFetcher()
        logger.info("Initialized DaemonEarningsCalendar")

    def fetch(self, symbols: list[str]) -> EarningsCalendar:
        """Fetch earnings dates for symbols.

        Args:
            symbols: Stock ticker symbols

        Returns:
            EarningsCalendar with upcoming events
        """
        return self._fetcher.fetch_earnings_dates(symbols)

    def get_upcoming(self, events: list[EarningsEvent], days_ahead: int = 3) -> list[EarningsEvent]:
        """Filter events within days_ahead of today.

        Args:
            events: All earnings events
            days_ahead: Number of calendar days to look ahead

        Returns:
            Events within the lookahead window
        """
        today = datetime.now(UTC).date()
        cutoff = today + timedelta(days=days_ahead)
        return [e for e in events if today <= e.earnings_date <= cutoff]

    def format_context(self, upcoming: list[EarningsEvent]) -> str:
        """Format upcoming earnings as text for trader prompt.

        Args:
            upcoming: Upcoming earnings events

        Returns:
            Formatted context string (empty if no upcoming)
        """
        if not upcoming:
            return ""

        today = datetime.now(UTC).date()
        lines: list[str] = []

        for event in upcoming:
            days_until = (event.earnings_date - today).days
            eps_line = f" (EPS estimate: ${event.estimate_eps:.2f})" if event.estimate_eps is not None else ""
            lines.append(f"{event.symbol} reports on {event.earnings_date} ({days_until}d away){eps_line}")

        return "\n".join(lines)

    def get_earnings_flags(self, events: list[EarningsEvent], symbol: str) -> EarningsFlags:
        """Get earnings flags for a specific symbol.

        Args:
            events: All earnings events
            symbol: Stock ticker to check

        Returns:
            EarningsFlags with upcoming_earnings, days_until_earnings, pre_earnings_zone
        """
        today = datetime.now(UTC).date()

        for event in events:
            if event.symbol != symbol:
                continue

            days_until = (event.earnings_date - today).days
            if days_until < 0:
                continue

            zone: str | None = None
            if days_until <= 1:
                zone = "T-1"
            elif days_until <= 3:
                zone = "T-3"

            return EarningsFlags(
                upcoming_earnings=True,
                days_until_earnings=days_until,
                pre_earnings_zone=zone,
            )

        return EarningsFlags(
            upcoming_earnings=False,
            days_until_earnings=None,
            pre_earnings_zone=None,
        )

    def __repr__(self) -> str:
        """String representation."""
        return "DaemonEarningsCalendar()"
