"""ORM model for economic calendar signals."""

import uuid
from datetime import UTC, datetime

from sqlalchemy import TIMESTAMP, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.database.models.base import Base
from src.database.types import JSONB, UUID


class EconomicCalendarSignalORM(Base):
    """Economic calendar signal ORM model."""

    __tablename__ = "economic_calendar_signals"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID,
        primary_key=True,
        default=uuid.uuid4,
    )
    risk_level: Mapped[str] = mapped_column(String(10), nullable=False)
    recommendation: Mapped[str] = mapped_column(String(30), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    upcoming_events: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    avoid_until: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    computed_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    __table_args__ = (
        Index(
            "idx_economic_calendar_signals_computed_at", "computed_at", postgresql_ops={"computed_at": "DESC"}
        ),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"EconomicCalendarSignalORM(id={self.id}, risk={self.risk_level}, computed_at={self.computed_at})"
        )
