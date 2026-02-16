"""Signal outcome repository for persistent learning database operations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import and_, select

from src.daemon.state.models import SignalOutcome, SignalUpdateRecord
from src.database.models import SignalOutcomeORM
from src.database.repositories.base import BaseRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class SignalRecordInput(BaseModel):
    """Input parameters for recording a trading signal."""

    symbol: str = Field(description="Stock ticker symbol")
    timestamp: datetime = Field(description="Signal timestamp")
    signal: str = Field(description="BUY/SELL/HOLD")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    price_at_signal: float = Field(gt=0.0, description="Price when signal was generated")
    strategy_used: str | None = Field(default=None, description="Strategy name")
    regime: str | None = Field(default=None, description="Market regime")
    trading_session: str = Field(default="REGULAR", description="REGULAR or PRE_MARKET")
    technical_signal: str | None = Field(default=None, description="Technical analysis signal")
    sentiment_signal: str | None = Field(default=None, description="Sentiment analysis signal")
    news_signal: str | None = Field(default=None, description="News analysis signal")
    technical_reasoning: str | None = Field(default=None, description="Technical analysis interpretation")
    sentiment_reasoning: str | None = Field(default=None, description="Sentiment analysis summary")
    news_reasoning: str | None = Field(default=None, description="News impact assessment")


class SignalOutcomeRepository(BaseRepository[SignalOutcome]):
    """Repository for signal outcome persistence and learning queries."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(session)

    async def record_signal(self, input_data: SignalRecordInput) -> SignalOutcome:
        """Record a new trading signal for outcome tracking.

        Args:
            input_data: Signal record input parameters

        Returns:
            Created SignalOutcome
        """
        orm = SignalOutcomeORM(
            id=uuid.uuid4(),
            symbol=input_data.symbol,
            timestamp=input_data.timestamp,
            signal=input_data.signal,
            confidence=Decimal(str(input_data.confidence)),
            price_at_signal=Decimal(str(input_data.price_at_signal)),
            strategy_used=input_data.strategy_used,
            regime=input_data.regime,
            trading_session=input_data.trading_session,
            technical_signal=input_data.technical_signal,
            sentiment_signal=input_data.sentiment_signal,
            news_signal=input_data.news_signal,
            technical_reasoning=input_data.technical_reasoning,
            sentiment_reasoning=input_data.sentiment_reasoning,
            news_reasoning=input_data.news_reasoning,
            created_at=datetime.now(UTC),
        )
        self._session.add(orm)
        await self._session.commit()

        logger.info(
            f"Recorded signal outcome: {input_data.symbol} {input_data.signal} @ "
            f"{input_data.timestamp} (conf={input_data.confidence:.2f})"
        )
        return self._to_domain(orm)

    async def get_by_id(self, entity_id: str) -> SignalOutcome | None:
        """Get signal outcome by ID.

        Args:
            entity_id: Signal outcome UUID string

        Returns:
            SignalOutcome if found, None otherwise
        """
        result = await self._session.execute(
            select(SignalOutcomeORM).where(SignalOutcomeORM.id == uuid.UUID(entity_id))
        )
        orm = result.scalar_one_or_none()
        return self._to_domain(orm) if orm else None

    async def get_by_symbol(
        self,
        symbol: str,
        limit: int = 50,
        start_date: datetime | None = None,
    ) -> list[SignalOutcome]:
        """Get signal outcomes for specific symbol.

        Args:
            symbol: Stock ticker symbol
            limit: Maximum number of records to return
            start_date: Optional start date filter (inclusive)

        Returns:
            List of SignalOutcomes for symbol
        """
        try:
            stmt = select(SignalOutcomeORM).where(SignalOutcomeORM.symbol == symbol)

            if start_date:
                stmt = stmt.where(SignalOutcomeORM.timestamp >= start_date)

            stmt = stmt.order_by(SignalOutcomeORM.timestamp.desc()).limit(limit)

            result = await self._session.execute(stmt)
            return [self._to_domain(orm) for orm in result.scalars().all()]
        except RuntimeError as e:
            if self._recreate_session_if_needed(e):
                return await self.get_by_symbol(symbol, limit, start_date)
            raise

    async def get_signals_needing_update(self, horizon: str) -> list[SignalUpdateRecord]:
        """Get signals that need outcome price updates for given horizon.

        Args:
            horizon: Time horizon - "1d", "5d", or "20d"

        Returns:
            List of SignalUpdateRecords needing updates
        """
        if horizon not in ["1d", "5d", "20d"]:
            msg = f"Invalid horizon: {horizon}. Must be one of: 1d, 5d, 20d"
            raise ValueError(msg)
        horizon_days = {"1d": 1, "5d": 5, "20d": 20}
        days = horizon_days[horizon]
        price_field = f"price_at_{horizon}"

        try:
            now = datetime.now(UTC)
            cutoff = now - timedelta(days=days + 2)  # +2 buffer for weekends

            stmt = (
                select(SignalOutcomeORM.id, SignalOutcomeORM.symbol, SignalOutcomeORM.timestamp)
                .where(
                    and_(
                        SignalOutcomeORM.timestamp <= cutoff,
                        getattr(SignalOutcomeORM, price_field).is_(None),
                    )
                )
                .order_by(SignalOutcomeORM.timestamp.desc())
                .limit(100)
            )

            result = await self._session.execute(stmt)
            records = []
            for row in result.all():
                target_date = row.timestamp + timedelta(days=days)
                records.append(
                    SignalUpdateRecord(
                        id=str(row.id),
                        symbol=row.symbol,
                        timestamp=row.timestamp,
                        target_date=target_date,
                    )
                )
            return records
        except RuntimeError as e:
            if self._recreate_session_if_needed(e):
                return await self.get_signals_needing_update(horizon)
            raise

    async def update_outcome_prices(
        self,
        signal_id: str,
        price_at_1d: float | None = None,
        price_at_5d: float | None = None,
        price_at_20d: float | None = None,
    ) -> None:
        """Update outcome prices for a signal.

        Args:
            signal_id: Signal outcome UUID string
            price_at_1d: Price at 1 day (optional)
            price_at_5d: Price at 5 days (optional)
            price_at_20d: Price at 20 days (optional)
        """
        result = await self._session.execute(
            select(SignalOutcomeORM).where(SignalOutcomeORM.id == uuid.UUID(signal_id))
        )
        orm = result.scalar_one_or_none()

        if not orm:
            logger.warning(f"Signal outcome {signal_id} not found for update")
            return

        if price_at_1d is not None:
            orm.price_at_1d = Decimal(str(price_at_1d))
        if price_at_5d is not None:
            orm.price_at_5d = Decimal(str(price_at_5d))
        if price_at_20d is not None:
            orm.price_at_20d = Decimal(str(price_at_20d))

        orm.outcome_updated_at = datetime.now(UTC)
        await self._session.commit()
        logger.debug(f"Updated outcome prices for signal {signal_id}")

    async def get_success_rate_by_regime(
        self,
        regime: str,
        horizon: str = "5d",
        min_confidence: float | None = None,
        days_back: int = 90,
        signal_type: str | None = None,
    ) -> dict[str, float]:
        """Calculate success rate by regime.

        Args:
            regime: Market regime filter
            horizon: Time horizon - "1d", "5d", or "20d"
            min_confidence: Minimum confidence filter (optional)
            days_back: Number of days to look back
            signal_type: BUY/SELL filter (optional)

        Returns:
            Dict with success_rate, total_decisions, hit_count, miss_count
        """
        if horizon not in ["1d", "5d", "20d"]:
            msg = f"Invalid horizon: {horizon}. Must be one of: 1d, 5d, 20d"
            raise ValueError(msg)
        price_field = f"price_at_{horizon}"

        try:
            cutoff = datetime.now(UTC) - timedelta(days=days_back)

            stmt = select(
                SignalOutcomeORM.signal,
                SignalOutcomeORM.price_at_signal,
                getattr(SignalOutcomeORM, price_field),
            ).where(
                and_(
                    SignalOutcomeORM.regime == regime,
                    SignalOutcomeORM.timestamp >= cutoff,
                    getattr(SignalOutcomeORM, price_field).is_not(None),
                )
            )

            if min_confidence is not None:
                stmt = stmt.where(SignalOutcomeORM.confidence >= Decimal(str(min_confidence)))

            if signal_type:
                stmt = stmt.where(SignalOutcomeORM.signal == signal_type)

            result = await self._session.execute(stmt)
            rows = result.all()

            if not rows:
                return {"success_rate": 0.0, "total_decisions": 0, "hit_count": 0, "miss_count": 0}

            hit_count = 0
            miss_count = 0

            for row in rows:
                signal = row[0]
                entry_price = float(row[1])
                exit_price = float(row[2])

                if signal == "BUY":
                    is_hit = exit_price > entry_price
                elif signal == "SELL":
                    is_hit = exit_price < entry_price
                else:  # HOLD
                    continue

                if is_hit:
                    hit_count += 1
                else:
                    miss_count += 1

            total = hit_count + miss_count
            success_rate = hit_count / total if total > 0 else 0.0

            return {
                "success_rate": success_rate,
                "total_decisions": total,
                "hit_count": hit_count,
                "miss_count": miss_count,
            }
        except RuntimeError as e:
            if self._recreate_session_if_needed(e):
                return await self.get_success_rate_by_regime(
                    regime, horizon, min_confidence, days_back, signal_type
                )
            raise

    async def get_success_rate_by_strategy(
        self,
        strategy: str,
        horizon: str = "5d",
        min_confidence: float | None = None,
        days_back: int = 90,
        signal_type: str | None = None,
    ) -> dict[str, float]:
        """Calculate success rate by strategy.

        Args:
            strategy: Strategy name filter
            horizon: Time horizon - "1d", "5d", or "20d"
            min_confidence: Minimum confidence filter (optional)
            days_back: Number of days to look back
            signal_type: BUY/SELL filter (optional)

        Returns:
            Dict with success_rate, total_decisions, hit_count, miss_count
        """
        if horizon not in ["1d", "5d", "20d"]:
            msg = f"Invalid horizon: {horizon}. Must be one of: 1d, 5d, 20d"
            raise ValueError(msg)
        price_field = f"price_at_{horizon}"

        try:
            cutoff = datetime.now(UTC) - timedelta(days=days_back)

            stmt = select(
                SignalOutcomeORM.signal,
                SignalOutcomeORM.price_at_signal,
                getattr(SignalOutcomeORM, price_field),
            ).where(
                and_(
                    SignalOutcomeORM.strategy_used == strategy,
                    SignalOutcomeORM.timestamp >= cutoff,
                    getattr(SignalOutcomeORM, price_field).is_not(None),
                )
            )

            if min_confidence is not None:
                stmt = stmt.where(SignalOutcomeORM.confidence >= Decimal(str(min_confidence)))

            if signal_type:
                stmt = stmt.where(SignalOutcomeORM.signal == signal_type)

            result = await self._session.execute(stmt)
            rows = result.all()

            if not rows:
                return {"success_rate": 0.0, "total_decisions": 0, "hit_count": 0, "miss_count": 0}

            hit_count = 0
            miss_count = 0

            for row in rows:
                signal = row[0]
                entry_price = float(row[1])
                exit_price = float(row[2])

                if signal == "BUY":
                    is_hit = exit_price > entry_price
                elif signal == "SELL":
                    is_hit = exit_price < entry_price
                else:  # HOLD
                    continue

                if is_hit:
                    hit_count += 1
                else:
                    miss_count += 1

            total = hit_count + miss_count
            success_rate = hit_count / total if total > 0 else 0.0

            return {
                "success_rate": success_rate,
                "total_decisions": total,
                "hit_count": hit_count,
                "miss_count": miss_count,
            }
        except RuntimeError as e:
            if self._recreate_session_if_needed(e):
                return await self.get_success_rate_by_strategy(
                    strategy, horizon, min_confidence, days_back, signal_type
                )
            raise

    async def get_recent_outcomes(
        self,
        window: int = 90,
        signal_type: str | None = None,
        min_confidence: float | None = None,
    ) -> list[SignalOutcome]:
        """Get recent signal outcomes within time window.

        Args:
            window: Number of days to look back
            signal_type: BUY/SELL/HOLD filter (optional)
            min_confidence: Minimum confidence filter (optional)

        Returns:
            List of SignalOutcomes
        """
        try:
            cutoff = datetime.now(UTC) - timedelta(days=window)
            stmt = select(SignalOutcomeORM).where(SignalOutcomeORM.timestamp >= cutoff)

            if signal_type:
                stmt = stmt.where(SignalOutcomeORM.signal == signal_type)

            if min_confidence is not None:
                stmt = stmt.where(SignalOutcomeORM.confidence >= Decimal(str(min_confidence)))

            stmt = stmt.order_by(SignalOutcomeORM.timestamp.desc())

            result = await self._session.execute(stmt)
            return [self._to_domain(orm) for orm in result.scalars().all()]
        except RuntimeError as e:
            if self._recreate_session_if_needed(e):
                return await self.get_recent_outcomes(window, signal_type, min_confidence)
            raise

    async def create(self, entity: SignalOutcome) -> SignalOutcome:
        """Create new signal outcome (alternative to record_signal).

        Args:
            entity: SignalOutcome to persist

        Returns:
            Created SignalOutcome
        """
        input_data = SignalRecordInput(
            symbol=entity.symbol,
            timestamp=entity.timestamp,
            signal=entity.signal,
            confidence=entity.confidence,
            price_at_signal=entity.price_at_signal,
            strategy_used=entity.strategy_used,
            regime=entity.regime,
            trading_session=entity.trading_session,
            technical_signal=entity.technical_signal,
            sentiment_signal=entity.sentiment_signal,
            news_signal=entity.news_signal,
        )
        return await self.record_signal(input_data)

    def _to_domain(self, orm: SignalOutcomeORM) -> SignalOutcome:
        """Convert ORM model to SignalOutcome domain model.

        Args:
            orm: SignalOutcomeORM instance

        Returns:
            SignalOutcome
        """
        return SignalOutcome(
            symbol=orm.symbol,
            timestamp=orm.timestamp,
            signal=orm.signal,
            confidence=float(orm.confidence),
            price_at_signal=float(orm.price_at_signal),
            strategy_used=orm.strategy_used,
            regime=orm.regime,
            trading_session=orm.trading_session,
            technical_signal=orm.technical_signal,
            sentiment_signal=orm.sentiment_signal,
            news_signal=orm.news_signal,
            technical_reasoning=orm.technical_reasoning,
            sentiment_reasoning=orm.sentiment_reasoning,
            news_reasoning=orm.news_reasoning,
            price_at_1d=float(orm.price_at_1d) if orm.price_at_1d is not None else None,
            price_at_5d=float(orm.price_at_5d) if orm.price_at_5d is not None else None,
            price_at_20d=float(orm.price_at_20d) if orm.price_at_20d is not None else None,
            actual_exit_price=float(orm.actual_exit_price) if orm.actual_exit_price is not None else None,
            actual_exit_date=orm.actual_exit_date,
            outcome_updated_at=orm.outcome_updated_at,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "SignalOutcomeRepository()"
