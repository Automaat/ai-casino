"""Data pipeline state manager."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import Field

from src.daemon.state.managers.base import StateManager
from src.daemon.state.models import (
    EarningsCalendarRecord,
    EarningsEventRecord,
    PrefetchRecord,
    ScreeningRecord,
)
from src.screening.screener import ScreeningResult


class DataPipelineStateManager(StateManager):
    """Background data operations (prefetch, screening, earnings)."""

    # Prefetch
    last_prefetch: datetime | None = None
    prefetch_history: list[PrefetchRecord] = Field(default_factory=list)
    last_pre_market_refresh: datetime | None = None

    # Screening
    last_after_hours_screening: datetime | None = None
    screening_history: list[ScreeningRecord] = Field(default_factory=list)

    # Earnings
    last_earnings_fetch: datetime | None = None
    earnings_calendar_history: list[EarningsCalendarRecord] = Field(default_factory=list)

    def record_prefetch(
        self,
        symbols_prefetched: int,
        symbols_failed: int,
        finbert_ready: bool,
        total_duration_seconds: float,
    ) -> None:
        """Record a data prefetch run.

        Args:
            symbols_prefetched: Number of symbols successfully prefetched
            symbols_failed: Number of symbols that failed
            finbert_ready: Whether FinBERT was warmed up
            total_duration_seconds: Total prefetch duration
        """
        now = datetime.now(UTC)

        self.prefetch_history.append(
            PrefetchRecord(
                timestamp=now,
                symbols_prefetched=symbols_prefetched,
                symbols_failed=symbols_failed,
                finbert_ready=finbert_ready,
                total_duration_seconds=total_duration_seconds,
            )
        )
        self.last_prefetch = now
        self.prefetch_history = self._cap_history(self.prefetch_history, 30, 30)

    def record_after_hours_screening(
        self,
        criteria: str,
        universe: str,
        candidates: list[ScreeningResult],
        top_n: int = 10,
        screened_at: datetime | None = None,
    ) -> None:
        """Record after-hours screening results.

        Args:
            criteria: Screening criteria
            universe: Universe screened
            candidates: Candidate list (typically top-N from screening)
            top_n: Number of top symbols to track
            screened_at: Timestamp when screening was performed (defaults to now)
        """
        now = datetime.now(UTC)
        top_symbols = [c.symbol for c in candidates[:top_n]]

        self.screening_history.append(
            ScreeningRecord(
                timestamp=now,
                criteria=criteria,
                universe=universe,
                top_symbols=top_symbols,
                candidates=candidates[:top_n],
                screened_at=screened_at or now,
            )
        )
        self.last_after_hours_screening = now
        self.screening_history = self._cap_history(self.screening_history, 30, 30)

    def record_earnings_fetch(
        self,
        events: list[EarningsEventRecord],
        symbols_fetched: int,
        symbols_failed: int,
    ) -> None:
        """Record an earnings calendar fetch run.

        Args:
            events: Earnings event records
            symbols_fetched: Number of symbols with earnings data
            symbols_failed: Number of symbols that failed to fetch
        """
        now = datetime.now(UTC)

        self.earnings_calendar_history.append(
            EarningsCalendarRecord(
                timestamp=now,
                events=events,
                symbols_fetched=symbols_fetched,
                symbols_failed=symbols_failed,
            )
        )
        self.last_earnings_fetch = now
        self.earnings_calendar_history = self._cap_history(self.earnings_calendar_history, 10, 10)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"DataPipelineStateManager(prefetches={len(self.prefetch_history)})"
