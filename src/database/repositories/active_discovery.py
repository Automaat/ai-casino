"""Active discovery candidate repository for database operations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import delete, select

from src.database.models import ActiveDiscoveryCandidateORM
from src.database.repositories.base import BaseRepository
from src.discovery.models import ActiveDiscoveryCandidate, DiscoverySourceDetail

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class ActiveDiscoveryCandidateRepository(BaseRepository[ActiveDiscoveryCandidate]):
    """Repository for active discovery candidate persistence."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(session)

    async def create(self, entity: ActiveDiscoveryCandidate) -> ActiveDiscoveryCandidate:
        """Create new active discovery candidate.

        Args:
            entity: ActiveDiscoveryCandidate to persist

        Returns:
            Created ActiveDiscoveryCandidate
        """
        # Serialize sources to JSONB
        sources_json = [
            {
                "source_type": s.source_type,
                "weight": s.weight,
                "metadata": s.metadata,
            }
            for s in entity.sources
        ]

        orm = ActiveDiscoveryCandidateORM(
            id=uuid.uuid4(),
            symbol=entity.symbol,
            discovered_at=entity.discovered_at,
            composite_score=Decimal(str(entity.composite_score)),
            sources=sources_json,
            ttl_expires_at=entity.ttl_expires_at,
            created_at=datetime.now(UTC),
        )
        self._session.add(orm)
        await self._session.commit()
        logger.info(f"Created active discovery candidate: {entity.symbol}")
        return entity

    async def get_by_id(self, entity_id: str) -> ActiveDiscoveryCandidate | None:
        """Get active discovery candidate by ID.

        Args:
            entity_id: Active discovery candidate UUID string

        Returns:
            ActiveDiscoveryCandidate if found, None otherwise
        """
        result = await self._session.execute(
            select(ActiveDiscoveryCandidateORM).where(ActiveDiscoveryCandidateORM.id == uuid.UUID(entity_id))
        )
        orm = result.scalar_one_or_none()
        return self._to_record(orm) if orm else None

    async def get_by_symbol(self, symbol: str) -> ActiveDiscoveryCandidate | None:
        """Get active discovery candidate by symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            ActiveDiscoveryCandidate if found, None otherwise
        """
        result = await self._session.execute(
            select(ActiveDiscoveryCandidateORM).where(ActiveDiscoveryCandidateORM.symbol == symbol)
        )
        orm = result.scalar_one_or_none()
        return self._to_record(orm) if orm else None

    async def get_all_active(self) -> list[ActiveDiscoveryCandidate]:
        """Get all active discovery candidates.

        Returns:
            List of all ActiveDiscoveryCandidates
        """
        result = await self._session.execute(
            select(ActiveDiscoveryCandidateORM).order_by(ActiveDiscoveryCandidateORM.discovered_at.desc())
        )
        return [self._to_record(orm) for orm in result.scalars().all()]

    async def delete_expired(self, cutoff: datetime) -> int:
        """Delete expired active discovery candidates.

        Args:
            cutoff: Delete candidates with ttl_expires_at < cutoff

        Returns:
            Number of candidates deleted
        """
        result = await self._session.execute(
            delete(ActiveDiscoveryCandidateORM).where(ActiveDiscoveryCandidateORM.ttl_expires_at < cutoff)
        )
        await self._session.commit()
        deleted_count: int = getattr(result, "rowcount", 0) or 0
        if deleted_count > 0:
            logger.info(f"Deleted {deleted_count} expired discovery candidates")
        return deleted_count

    async def delete_by_symbol(self, symbol: str) -> bool:
        """Delete active discovery candidate by symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            True if deleted, False if not found
        """
        result = await self._session.execute(
            delete(ActiveDiscoveryCandidateORM).where(ActiveDiscoveryCandidateORM.symbol == symbol)
        )
        await self._session.commit()
        deleted_count: int = getattr(result, "rowcount", 0) or 0
        deleted = deleted_count > 0
        if deleted:
            logger.info(f"Deleted active discovery candidate: {symbol}")
        return deleted

    def _to_record(self, orm: ActiveDiscoveryCandidateORM) -> ActiveDiscoveryCandidate:
        """Convert ORM model to ActiveDiscoveryCandidate.

        Args:
            orm: ActiveDiscoveryCandidateORM instance

        Returns:
            ActiveDiscoveryCandidate
        """
        sources = []
        if isinstance(orm.sources, list):
            for s_dict in orm.sources:
                sources.append(
                    DiscoverySourceDetail(
                        source_type=s_dict["source_type"],
                        weight=s_dict["weight"],
                        metadata=s_dict.get("metadata", {}),
                    )
                )

        return ActiveDiscoveryCandidate(
            symbol=orm.symbol,
            discovered_at=orm.discovered_at,
            composite_score=float(orm.composite_score),
            sources=sources,
            ttl_expires_at=orm.ttl_expires_at,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "ActiveDiscoveryCandidateRepository()"
