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

    from src.agents.supervisor.models import CandidateRanking


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

    async def record_discovery(
        self,
        candidates: list[DiscoveryCandidate],
        added_symbols: list[str],
        supervisor_ranking: CandidateRanking | None = None,
    ) -> None:
        """Record discovery run and update active candidates.

        Args:
            candidates: Discovery candidates
            added_symbols: Symbols approved for watchlist
            supervisor_ranking: Optional supervisor CandidateRanking with evaluations
        """
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

                # Build supervisor eval map if available
                supervisor_evals = {}
                if supervisor_ranking:
                    for eval_item in supervisor_ranking.evaluations:
                        supervisor_evals[eval_item.symbol] = eval_item

                # Add history records to DB
                for candidate in candidates:
                    supervisor_eval = supervisor_evals.get(candidate.symbol)

                    history_record = DiscoveryHistoryRecord(
                        symbol=candidate.symbol,
                        discovered_at=candidate.discovery_timestamp,
                        composite_score=candidate.composite_score,
                        sources=candidate.sources,
                        added_to_watchlist=candidate.symbol in added_symbols,
                        ttl_expires_at=candidate.ttl_expires_at,
                        supervisor_evaluation_score=(
                            (
                                supervisor_eval.quality_score
                                + supervisor_eval.momentum_score
                                + supervisor_eval.risk_score
                                + supervisor_eval.portfolio_fit_score
                            )
                            / 4.0
                            if supervisor_eval
                            else None
                        ),
                        supervisor_recommendation=(
                            supervisor_eval.recommendation.value if supervisor_eval else None
                        ),
                        evaluation_reasoning=supervisor_eval.reasoning if supervisor_eval else None,
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

    async def record_pre_market_candidates(
        self, candidates: list, expires_at: datetime, session: AsyncSession | None = None
    ) -> None:
        """Store pre-market candidates with TTL (expires 9:30 AM ET).

        Args:
            candidates: List of PreMarketCandidate objects
            expires_at: Expiration timestamp (9:30 AM ET)
            session: Optional session for transaction context
        """
        from src.discovery.models import DiscoveryCandidate, DiscoverySource

        discovery_candidates = []
        for candidate in candidates:
            dc = DiscoveryCandidate(
                symbol=candidate.symbol,
                name=candidate.name,
                sector=candidate.sector,
                sources=[DiscoverySource.PRICE_GAP],
                composite_score=candidate.composite_score,
                source_scores={"PRE_MARKET": candidate.composite_score},
                discovery_timestamp=datetime.now(UTC),
                ttl_expires_at=expires_at,
                metadata={
                    "gap_percent": candidate.gap_percent,
                    "volume_ratio": candidate.volume_ratio,
                    "priority": candidate.priority,
                },
            )
            discovery_candidates.append(dc)

        try:
            from src.database.connection import get_session
            from src.database.repositories.active_discovery import ActiveDiscoveryCandidateRepository

            if session:
                repo = ActiveDiscoveryCandidateRepository(session)
                for dc in discovery_candidates:
                    sources = [
                        DiscoverySourceDetail(
                            source_type="pre_market",
                            weight=dc.composite_score,
                            metadata=dc.metadata,
                        )
                    ]
                    active_candidate = ActiveDiscoveryCandidate(
                        symbol=dc.symbol,
                        discovered_at=dc.discovery_timestamp,
                        composite_score=dc.composite_score,
                        sources=sources,
                        ttl_expires_at=dc.ttl_expires_at,
                    )
                    existing = await repo.get_by_symbol(dc.symbol)
                    if existing:
                        await repo.delete_by_symbol(dc.symbol)
                    await repo.create(active_candidate)
            else:
                async with get_session() as fresh_session:
                    repo = ActiveDiscoveryCandidateRepository(fresh_session)
                    for dc in discovery_candidates:
                        sources = [
                            DiscoverySourceDetail(
                                source_type="pre_market",
                                weight=dc.composite_score,
                                metadata=dc.metadata,
                            )
                        ]
                        active_candidate = ActiveDiscoveryCandidate(
                            symbol=dc.symbol,
                            discovered_at=dc.discovery_timestamp,
                            composite_score=dc.composite_score,
                            sources=sources,
                            ttl_expires_at=dc.ttl_expires_at,
                        )
                        existing = await repo.get_by_symbol(dc.symbol)
                        if existing:
                            await repo.delete_by_symbol(dc.symbol)
                        await repo.create(active_candidate)

            logger.info(f"Recorded {len(discovery_candidates)} pre-market candidates (expires {expires_at})")
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to record pre-market candidates: {e}")

    async def get_active_pre_market_candidates(
        self, session: AsyncSession | None = None
    ) -> list[DiscoveryCandidate]:
        """Get non-expired pre-market candidates.

        Args:
            session: Optional session for transaction context

        Returns:
            List of active pre-market discovery candidates
        """
        from src.database.repositories.active_discovery import ActiveDiscoveryCandidateRepository

        if session:
            repo = ActiveDiscoveryCandidateRepository(session)
            active_candidates = await repo.get_all_active()
            return [
                self._to_discovery_candidate(c)
                for c in active_candidates
                if any(s.source_type == "pre_market" for s in c.sources)
            ]

        try:
            from src.database.connection import get_session

            async with get_session() as fresh_session:
                repo = ActiveDiscoveryCandidateRepository(fresh_session)
                active_candidates = await repo.get_all_active()
                return [
                    self._to_discovery_candidate(c)
                    for c in active_candidates
                    if any(s.source_type == "pre_market" for s in c.sources)
                ]
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to get active pre-market candidates: {e}")
            return []

    async def add_event_candidates(
        self, candidates: list[DiscoveryCandidate], session: AsyncSession | None = None
    ) -> None:
        """Add event-generated candidates to active pool.

        Merges with existing candidates:
        - Same symbol from multiple sources → boost score by 10%
        - Extend TTL if new event has higher urgency
        - Track sources to prevent double-counting

        Args:
            candidates: Event-generated discovery candidates
            session: Optional session for transaction context
        """
        from src.database.repositories.active_discovery import ActiveDiscoveryCandidateRepository

        score_boost_factor = 1.1

        if session:
            repo = ActiveDiscoveryCandidateRepository(session)
            for candidate in candidates:
                existing = await repo.get_by_symbol(candidate.symbol)

                if existing:
                    new_sources = {str(s.value) for s in candidate.sources}
                    existing_sources = {s.source_type for s in existing.sources}
                    all_sources = new_sources.union(existing_sources)

                    boosted_score = min(1.0, existing.composite_score * score_boost_factor)
                    new_ttl = max(candidate.ttl_expires_at, existing.ttl_expires_at)

                    merged_sources = [
                        DiscoverySourceDetail(
                            source_type=source_type,
                            weight=candidate.source_scores.get(source_type, 0.0),
                            metadata={},
                        )
                        for source_type in all_sources
                    ]

                    await repo.delete_by_symbol(candidate.symbol)
                    updated_candidate = ActiveDiscoveryCandidate(
                        symbol=candidate.symbol,
                        discovered_at=existing.discovered_at,
                        composite_score=boosted_score,
                        sources=merged_sources,
                        ttl_expires_at=new_ttl,
                    )
                    await repo.create(updated_candidate)
                    logger.info(
                        f"Merged event candidate {candidate.symbol}: "
                        f"score {existing.composite_score:.2f}→{boosted_score:.2f}, "
                        f"sources {len(existing_sources)}→{len(all_sources)}"
                    )
                else:
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
                    logger.info(
                        f"Added event candidate {candidate.symbol} score={candidate.composite_score:.2f}"
                    )
        else:
            from src.database.connection import get_session

            async with get_session() as fresh_session:
                repo = ActiveDiscoveryCandidateRepository(fresh_session)
                for candidate in candidates:
                    existing = await repo.get_by_symbol(candidate.symbol)

                    if existing:
                        new_sources = {str(s.value) for s in candidate.sources}
                        existing_sources = {s.source_type for s in existing.sources}
                        all_sources = new_sources.union(existing_sources)

                        boosted_score = min(1.0, existing.composite_score * score_boost_factor)
                        new_ttl = max(candidate.ttl_expires_at, existing.ttl_expires_at)

                        merged_sources = [
                            DiscoverySourceDetail(
                                source_type=source_type,
                                weight=candidate.source_scores.get(source_type, 0.0),
                                metadata={},
                            )
                            for source_type in all_sources
                        ]

                        await repo.delete_by_symbol(candidate.symbol)
                        updated_candidate = ActiveDiscoveryCandidate(
                            symbol=candidate.symbol,
                            discovered_at=existing.discovered_at,
                            composite_score=boosted_score,
                            sources=merged_sources,
                            ttl_expires_at=new_ttl,
                        )
                        await repo.create(updated_candidate)
                        logger.info(
                            f"Merged event candidate {candidate.symbol}: "
                            f"score {existing.composite_score:.2f}→{boosted_score:.2f}, "
                            f"sources {len(existing_sources)}→{len(all_sources)}"
                        )
                    else:
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
                        logger.info(
                            f"Added event candidate {candidate.symbol} score={candidate.composite_score:.2f}"
                        )

    async def get_last_discovery_outcome_tracking(
        self, session: AsyncSession | None = None
    ) -> datetime | None:
        """Get last discovery outcome tracking timestamp from DB."""
        from src.database.repositories.metadata import MetadataRepository

        if session:
            repo = MetadataRepository(session)
            return await repo.get_datetime("discovery.last_outcome_tracking")

        try:
            from src.database.connection import get_session

            async with get_session() as fresh_session:
                repo = MetadataRepository(fresh_session)
                return await repo.get_datetime("discovery.last_outcome_tracking")
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to get last outcome tracking: {e}")
            return None

    async def set_last_discovery_outcome_tracking(self, value: datetime | None) -> None:
        """Set last discovery outcome tracking timestamp in DB."""
        if value is None:
            return
        try:
            from src.database.connection import get_session
            from src.database.repositories.metadata import MetadataRepository

            async with get_session() as session:
                await MetadataRepository(session).set("discovery.last_outcome_tracking", value)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to set last outcome tracking: {e}")

    def __repr__(self) -> str:
        """Return string representation."""
        return "DiscoveryStateManager()"
