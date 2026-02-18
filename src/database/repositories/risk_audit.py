"""Risk audit repository for database operations."""

import uuid
from datetime import datetime
from decimal import Decimal

from loguru import logger
from sqlalchemy import select

from src.agents.risk.models import RiskAuditRecord
from src.database.models import RiskAuditORM
from src.database.repositories.base import BaseRepository
from src.strategies.signal import Signal


class RiskAuditRepository(BaseRepository[RiskAuditRecord]):
    """Repository for risk audit records."""

    async def create(self, entity: RiskAuditRecord) -> RiskAuditRecord:
        """Insert risk audit record.

        Args:
            entity: RiskAuditRecord to persist

        Returns:
            Created RiskAuditRecord with ID
        """
        orm = RiskAuditORM(
            id=uuid.uuid4(),
            timestamp=entity.timestamp,
            symbol=entity.symbol,
            action=entity.action.value,
            current_price=Decimal(str(entity.current_price)),
            approved=entity.approved,
            risk_level=entity.risk_level,
            risk_score=Decimal(str(entity.risk_score)),
            confidence=Decimal(str(entity.confidence)),
            recommended_shares=entity.recommended_shares,
            position_value=Decimal(str(entity.position_value)),
            risk_amount=Decimal(str(entity.risk_amount)),
            risk_percent=Decimal(str(entity.risk_percent)),
            stop_loss_price=Decimal(str(entity.stop_loss_price)),
            take_profit_price=(
                Decimal(str(entity.take_profit_price)) if entity.take_profit_price is not None else None
            ),
            reward_risk_ratio=(
                Decimal(str(entity.reward_risk_ratio)) if entity.reward_risk_ratio is not None else None
            ),
            warnings=entity.warnings,
            portfolio_var_95=Decimal(str(entity.portfolio_var_95)) if entity.portfolio_var_95 else None,
            portfolio_cvar_99=Decimal(str(entity.portfolio_cvar_99)) if entity.portfolio_cvar_99 else None,
            portfolio_cdar_95=Decimal(str(entity.portfolio_cdar_95)) if entity.portfolio_cdar_95 else None,
        )
        self._session.add(orm)
        await self._session.commit()
        await self._session.refresh(orm)
        entity.id = str(orm.id)
        entity.created_at = orm.created_at
        logger.debug(f"Created risk audit: {entity.symbol} {entity.action.value}")
        return entity

    async def get_by_id(self, entity_id: str) -> RiskAuditRecord | None:
        """Get risk audit by ID.

        Args:
            entity_id: Risk audit UUID string

        Returns:
            RiskAuditRecord if found, None otherwise
        """
        result = await self._session.execute(
            select(RiskAuditORM).where(RiskAuditORM.id == uuid.UUID(entity_id))
        )
        orm = result.scalar_one_or_none()
        return self._to_record(orm) if orm else None

    async def get_by_symbol(self, symbol: str, limit: int = 100) -> list[RiskAuditRecord]:
        """Get audit logs for symbol.

        Args:
            symbol: Stock ticker symbol
            limit: Maximum records to return

        Returns:
            List of RiskAuditRecords for symbol
        """
        result = await self._session.execute(
            select(RiskAuditORM)
            .where(RiskAuditORM.symbol == symbol)
            .order_by(RiskAuditORM.timestamp.desc())
            .limit(limit)
        )
        return [self._to_record(orm) for orm in result.scalars().all()]

    async def get_violations(
        self, symbol: str | None = None, start_date: datetime | None = None, limit: int = 100
    ) -> list[RiskAuditRecord]:
        """Get violation records (approved=false).

        Args:
            symbol: Optional symbol filter
            start_date: Optional start date filter
            limit: Maximum records to return

        Returns:
            List of violation RiskAuditRecords
        """
        query = select(RiskAuditORM).where(RiskAuditORM.approved.is_(False))

        if symbol:
            query = query.where(RiskAuditORM.symbol == symbol)
        if start_date:
            query = query.where(RiskAuditORM.timestamp >= start_date)

        query = query.order_by(RiskAuditORM.timestamp.desc()).limit(limit)

        result = await self._session.execute(query)
        return [self._to_record(orm) for orm in result.scalars().all()]

    def _to_record(self, orm: RiskAuditORM) -> RiskAuditRecord:
        """Convert ORM to domain model.

        Args:
            orm: RiskAuditORM instance

        Returns:
            RiskAuditRecord
        """
        return RiskAuditRecord(
            id=str(orm.id),
            timestamp=orm.timestamp,
            symbol=orm.symbol,
            action=Signal(orm.action),
            current_price=float(orm.current_price),
            approved=orm.approved,
            risk_level=orm.risk_level,
            risk_score=float(orm.risk_score),
            confidence=float(orm.confidence),
            recommended_shares=orm.recommended_shares,
            position_value=float(orm.position_value),
            risk_amount=float(orm.risk_amount),
            risk_percent=float(orm.risk_percent),
            stop_loss_price=float(orm.stop_loss_price),
            take_profit_price=float(orm.take_profit_price) if orm.take_profit_price is not None else None,
            reward_risk_ratio=float(orm.reward_risk_ratio) if orm.reward_risk_ratio is not None else None,
            warnings=orm.warnings,
            portfolio_var_95=float(orm.portfolio_var_95) if orm.portfolio_var_95 else None,
            portfolio_cvar_99=float(orm.portfolio_cvar_99) if orm.portfolio_cvar_99 else None,
            portfolio_cdar_95=float(orm.portfolio_cdar_95) if orm.portfolio_cdar_95 else None,
            created_at=orm.created_at,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "RiskAuditRepository()"
