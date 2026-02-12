"""Discovery state manager."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import Field, PrivateAttr

from src.daemon.state.managers.base import StateManager, _make_task_cleanup_callback
from src.daemon.state.models import DiscoveryHistoryRecord
from src.discovery.models import DiscoveryCandidate

if TYPE_CHECKING:
    from src.database.repositories.discovery import DiscoveryHistoryRepository


class DiscoveryStateManager(StateManager):
    """Stock discovery with TTL management."""

    last_discovery: datetime | None = None
    discovery_history: list[DiscoveryHistoryRecord] = Field(default_factory=list)
    active_discovery_candidates: list[DiscoveryCandidate] = Field(default_factory=list)

    _discovery_repository: DiscoveryHistoryRepository | None = PrivateAttr(default=None)

    def set_repository(self, repository: DiscoveryHistoryRepository) -> None:
        """Inject discovery repository.

        Args:
            repository: Discovery history repository
        """
        self._discovery_repository = repository
        logger.debug("Discovery repository injected")

    def record_discovery(self, candidates: list[DiscoveryCandidate], added_symbols: list[str]) -> None:
        """Record discovery run and update active candidates.

        Args:
            candidates: Discovery candidates to record
            added_symbols: Symbols actually added to watchlist
        """
        # Add new history records
        for candidate in candidates:
            history_record = DiscoveryHistoryRecord(
                symbol=candidate.symbol,
                discovered_at=candidate.discovery_timestamp,
                composite_score=candidate.composite_score,
                sources=candidate.sources,
                added_to_watchlist=candidate.symbol in added_symbols,
                ttl_expires_at=candidate.ttl_expires_at,
            )

            # Persist to database if repository available
            if self._discovery_repository:
                try:
                    task = asyncio.create_task(self._discovery_repository.create(history_record))  # type: ignore[bad-argument-type]
                    self._pending_tasks.add(task)
                    task.add_done_callback(_make_task_cleanup_callback(self._pending_tasks))
                    logger.debug(f"Scheduled discovery history persistence to database: {candidate.symbol}")
                except Exception as e:
                    logger.opt(exception=True).error(f"Failed to schedule discovery history persistence: {e}")
                    raise

            # Keep in-memory list (capped for transition period)
            self.discovery_history.append(history_record)

        # Update active candidates (replace old with new)
        self.active_discovery_candidates = candidates

        # Limit history to last 100 records
        self.discovery_history = self._cap_history(self.discovery_history, 100, 100)

        logger.info(f"Recorded discovery: {len(candidates)} candidates, {len(added_symbols)} added")

    def expire_stale_candidates(self) -> list[str]:
        """Remove candidates past TTL, return expired symbols.

        Returns:
            List of expired symbols
        """
        now = datetime.now(UTC)
        expired_symbols: list[str] = []

        # Filter out expired candidates
        active_candidates: list[DiscoveryCandidate] = []
        for candidate in self.active_discovery_candidates:
            ttl_expires_at = candidate.ttl_expires_at
            if ttl_expires_at.tzinfo is None:
                ttl_expires_at = ttl_expires_at.replace(tzinfo=UTC)
            if ttl_expires_at > now:
                active_candidates.append(candidate)
            else:
                expired_symbols.append(candidate.symbol)

        self.active_discovery_candidates = active_candidates

        if expired_symbols:
            logger.info(f"Expired {len(expired_symbols)} discovery candidates")

        return expired_symbols

    def get_active_discovery_symbols(self) -> list[str]:
        """Get symbols from active discovery candidates.

        Returns:
            List of active discovery symbols
        """
        return [c.symbol for c in self.active_discovery_candidates]

    def __repr__(self) -> str:
        """Return string representation."""
        return f"DiscoveryStateManager(active={len(self.active_discovery_candidates)})"
