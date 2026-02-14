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
    from sqlalchemy.ext.asyncio import AsyncSession


class DiscoveryStateManager(StateManager):
    """Stock discovery with TTL management."""

    _discovery_cache: list[DiscoveryHistoryRecord] | None = PrivateAttr(default=None)

    async def get_last_discovery(self, session: AsyncSession | None = None) -> datetime | None:
        """Get last discovery timestamp from DB."""
        from src.database.repositories.metadata import MetadataRepository

        if session:
            repo = MetadataRepository(session)
            return await repo.get_datetime("discovery.last_discovery")

        try:
            from src.database.connection import get_session

            async with get_session() as fresh_session:
                repo = MetadataRepository(fresh_session)
                return await repo.get_datetime("discovery.last_discovery")
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to get last discovery: {e}")
            return None

    async def set_last_discovery(self, value: datetime | None) -> None:
        """Set last discovery timestamp in DB."""
        if value is None:
            return
        try:
            from src.database.connection import get_session
            from src.database.repositories.metadata import MetadataRepository

            async with get_session() as session:
                await MetadataRepository(session).set("discovery.last_discovery", value)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to set last discovery: {e}")

    async def get_discovery_history(
        self, limit: int = 100, session: AsyncSession | None = None
    ) -> list[DiscoveryHistoryRecord]:
        """Get discovery history with lazy loading."""
        from src.database.repositories.discovery import DiscoveryHistoryRepository

        if session:
            repo = DiscoveryHistoryRepository(session)
            return await repo.get_recent_discoveries(days=30)

        if self._discovery_cache is None:
            try:
                from src.database.connection import get_session

                async with get_session() as fresh_session:
                    repo = DiscoveryHistoryRepository(fresh_session)
                    self._discovery_cache = await repo.get_recent_discoveries(days=30)
            except Exception as e:
                logger.opt(exception=True).warning(f"Failed to get discovery history: {e}")
                return []
        return self._discovery_cache

    def _to_discovery_candidate(self, active_candidate: ActiveDiscoveryCandidate) -> DiscoveryCandidate:
        """Convert ActiveDiscoveryCandidate to DiscoveryCandidate.

        Args:
            active_candidate: ActiveDiscoveryCandidate from repository

        Returns:
            DiscoveryCandidate with converted fields
        """
        from src.discovery.models import DiscoverySource

        return DiscoveryCandidate(
            symbol=active_candidate.symbol,
            name="Unknown",
            sector="Unknown",
            sources=[DiscoverySource(s.source_type) for s in active_candidate.sources],
            composite_score=active_candidate.composite_score,
            source_scores={s.source_type: s.weight for s in active_candidate.sources},
            discovery_timestamp=active_candidate.discovered_at,
            ttl_expires_at=active_candidate.ttl_expires_at,
        )

    async def get_active_discovery_candidates(
        self, session: AsyncSession | None = None
    ) -> list[DiscoveryCandidate]:
        """Get all active discovery candidates from DB, converted to DiscoveryCandidate."""
        from src.database.repositories.active_discovery import ActiveDiscoveryCandidateRepository

        if session:
            repo = ActiveDiscoveryCandidateRepository(session)
            active_candidates = await repo.get_all_active()
            return [self._to_discovery_candidate(c) for c in active_candidates]

        try:
            from src.database.connection import get_session

            async with get_session() as fresh_session:
                repo = ActiveDiscoveryCandidateRepository(fresh_session)
                active_candidates = await repo.get_all_active()
                return [self._to_discovery_candidate(c) for c in active_candidates]
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to get active discovery candidates: {e}")
            return []

    async def set_active_discovery_candidates(self, value: list[DiscoveryCandidate]) -> None:
        """Set active discovery candidates in DB (replaces all)."""
        try:
            from src.database.connection import get_session
            from src.database.repositories.active_discovery import ActiveDiscoveryCandidateRepository

            async with get_session() as session:
                repo = ActiveDiscoveryCandidateRepository(session)

                # Delete all existing
                await repo.delete_expired(datetime.max.replace(tzinfo=UTC))

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
                    await repo.create(active_candidate)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to set active discovery candidates: {e}")

    async def record_discovery(self, candidates: list[DiscoveryCandidate], added_symbols: list[str]) -> None:
        """Record discovery run and update active candidates."""
        now = datetime.now(UTC)

        try:
            from src.database.connection import get_session
            from src.database.repositories.active_discovery import ActiveDiscoveryCandidateRepository
            from src.database.repositories.discovery import DiscoveryHistoryRepository
            from src.database.repositories.metadata import MetadataRepository

            async with get_session() as session:
                discovery_repo = DiscoveryHistoryRepository(session)
                active_repo = ActiveDiscoveryCandidateRepository(session)
                metadata_repo = MetadataRepository(session)

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
                    await discovery_repo.create(history_record)

                # Update active candidates in DB
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
                    existing = await active_repo.get_by_symbol(candidate.symbol)
                    if existing:
                        await active_repo.delete_by_symbol(candidate.symbol)
                    await active_repo.create(active_candidate)

                # Update metadata
                await metadata_repo.set("discovery.last_discovery", now)

        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to record discovery: {e}")

        # Invalidate cache
        self._discovery_cache = None

        logger.info(f"Recorded discovery: {len(candidates)} candidates, {len(added_symbols)} added")

    async def expire_stale_candidates(self) -> list[str]:
        """Remove candidates past TTL, return expired symbols."""
        try:
            from src.database.connection import get_session
            from src.database.repositories.active_discovery import ActiveDiscoveryCandidateRepository

            now = datetime.now(UTC)

            async with get_session() as session:
                repo = ActiveDiscoveryCandidateRepository(session)
                deleted_count = await repo.delete_expired(now)

            logger.info(f"Expired {deleted_count} discovery candidates")
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to expire stale candidates: {e}")

        return []  # Return empty list since we don't track individual expired symbols

    async def get_active_discovery_symbols(self, session: AsyncSession | None = None) -> list[str]:
        """Get symbols from active discovery candidates."""
        candidates = await self.get_active_discovery_candidates(session=session)
        return [c.symbol for c in candidates]

    def __repr__(self) -> str:
        """Return string representation."""
        return "DiscoveryStateManager()"
