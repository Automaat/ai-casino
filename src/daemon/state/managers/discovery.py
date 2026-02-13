"""Discovery state manager."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import PrivateAttr

from src.daemon.state.managers.base import StateManager
from src.daemon.state.models import DiscoveryHistoryRecord
from src.discovery.models import ActiveDiscoveryCandidate, DiscoveryCandidate, DiscoverySourceDetail

if TYPE_CHECKING:
    from src.database.repositories.active_discovery import ActiveDiscoveryCandidateRepository
    from src.database.repositories.discovery import DiscoveryHistoryRepository
    from src.database.repositories.metadata import MetadataRepository


class DiscoveryStateManager(StateManager):
    """Stock discovery with TTL management."""

    _metadata_repository: MetadataRepository | None = PrivateAttr(default=None)
    _discovery_repository: DiscoveryHistoryRepository | None = PrivateAttr(default=None)
    _active_discovery_repository: ActiveDiscoveryCandidateRepository | None = PrivateAttr(default=None)

    _discovery_cache: list[DiscoveryHistoryRecord] | None = PrivateAttr(default=None)

    def set_repositories(
        self,
        metadata_repository: MetadataRepository,
        discovery_repository: DiscoveryHistoryRepository,
        active_discovery_repository: ActiveDiscoveryCandidateRepository,
    ) -> None:
        """Inject repositories."""
        self._metadata_repository = metadata_repository
        self._discovery_repository = discovery_repository
        self._active_discovery_repository = active_discovery_repository
        logger.debug("DiscoveryStateManager repositories injected")

    async def get_last_discovery(self) -> datetime | None:
        """Get last discovery timestamp from DB."""
        if not self._metadata_repository:
            return None
        return await self._metadata_repository.get_datetime("discovery.last_discovery")

    async def set_last_discovery(self, value: datetime | None) -> None:
        """Set last discovery timestamp in DB."""
        if self._metadata_repository and value is not None:
            await self._metadata_repository.set("discovery.last_discovery", value)

    async def get_discovery_history(self, limit: int = 100) -> list[DiscoveryHistoryRecord]:
        """Get discovery history with lazy loading."""
        if not self._discovery_repository:
            return []
        if self._discovery_cache is None:
            self._discovery_cache = await self._discovery_repository.get_recent_discoveries(days=30)
        return self._discovery_cache

    async def get_active_discovery_candidates(self) -> list[DiscoveryCandidate]:
        """Get all active discovery candidates from DB, converted to DiscoveryCandidate."""
        if not self._active_discovery_repository:
            return []
        active_candidates = await self._active_discovery_repository.get_all_active()
        # Convert ActiveDiscoveryCandidate to DiscoveryCandidate for backward compatibility
        from src.discovery.models import DiscoverySource

        return [
            DiscoveryCandidate(
                symbol=c.symbol,
                name="Unknown",
                sector="Unknown",
                sources=[DiscoverySource(s.source_type) for s in c.sources],
                composite_score=c.composite_score,
                source_scores={s.source_type: s.weight for s in c.sources},
                discovery_timestamp=c.discovered_at,
                ttl_expires_at=c.ttl_expires_at,
            )
            for c in active_candidates
        ]

    async def set_active_discovery_candidates(self, value: list[DiscoveryCandidate]) -> None:
        """Set active discovery candidates in DB (replaces all)."""
        if not self._active_discovery_repository:
            return

        # Delete all existing
        await self._active_discovery_repository.delete_expired(datetime.max.replace(tzinfo=UTC))

        # Insert new ones
        for candidate in value:
            sources = [
                DiscoverySourceDetail(
                    source_type=str(source.value),
                    weight=candidate.source_scores.get(str(source.value), 0.0),
                    metadata={},
                )
                for source in candidate.sources
            ]

            active_candidate = ActiveDiscoveryCandidate(
                symbol=candidate.symbol,
                discovered_at=candidate.discovery_timestamp,
                composite_score=candidate.composite_score,
                sources=sources,
                ttl_expires_at=candidate.ttl_expires_at,
            )
            await self._active_discovery_repository.create(active_candidate)

    async def record_discovery(self, candidates: list[DiscoveryCandidate], added_symbols: list[str]) -> None:
        """Record discovery run and update active candidates."""
        now = datetime.now(UTC)

        # Add history records to DB
        for candidate in candidates:
            history_record = DiscoveryHistoryRecord(
                symbol=candidate.symbol,
                discovered_at=candidate.discovery_timestamp,
                composite_score=candidate.composite_score,
                sources=candidate.sources,
                added_to_watchlist=candidate.symbol in added_symbols,
                ttl_expires_at=candidate.ttl_expires_at,
            )

            if self._discovery_repository:
                await self._discovery_repository.create(history_record)

        # Update active candidates in DB
        if self._active_discovery_repository:
            for candidate in candidates:
                # Convert DiscoverySource enums to DiscoverySourceDetail
                sources = [
                    DiscoverySourceDetail(
                        source_type=str(source.value),
                        weight=candidate.source_scores.get(str(source.value), 0.0),
                        metadata={},
                    )
                    for source in candidate.sources
                ]

                active_candidate = ActiveDiscoveryCandidate(
                    symbol=candidate.symbol,
                    discovered_at=candidate.discovery_timestamp,
                    composite_score=candidate.composite_score,
                    sources=sources,
                    ttl_expires_at=candidate.ttl_expires_at,
                )

                # Check if exists, update or create
                existing = await self._active_discovery_repository.get_by_symbol(candidate.symbol)
                if existing:
                    await self._active_discovery_repository.delete_by_symbol(candidate.symbol)
                await self._active_discovery_repository.create(active_candidate)

        # Update metadata
        if self._metadata_repository:
            await self._metadata_repository.set("discovery.last_discovery", now)

        # Invalidate cache
        self._discovery_cache = None

        logger.info(f"Recorded discovery: {len(candidates)} candidates, {len(added_symbols)} added")

    async def expire_stale_candidates(self) -> list[str]:
        """Remove candidates past TTL, return expired symbols."""
        if not self._active_discovery_repository:
            return []

        now = datetime.now(UTC)
        deleted_count = await self._active_discovery_repository.delete_expired(now)

        logger.info(f"Expired {deleted_count} discovery candidates")
        return []  # Return empty list since we don't track individual expired symbols

    async def get_active_discovery_symbols(self) -> list[str]:
        """Get symbols from active discovery candidates."""
        candidates = await self.get_active_discovery_candidates()
        return [c.symbol for c in candidates]

    def __repr__(self) -> str:
        """Return string representation."""
        return "DiscoveryStateManager()"
