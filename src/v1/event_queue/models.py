"""Domain model and ORM for market event queue."""

import uuid
from datetime import datetime

from pydantic import BaseModel
from sqlalchemy import TIMESTAMP, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from src.database.models.base import Base
from src.database.types import JSONB, UUID


class QueueEventObservability(BaseModel):
    """Enriched queue event for observability API responses."""

    event_id: str
    event_type: str
    enqueued_at: datetime
    expires_at: datetime
    consumed_at: datetime | None
    status: str  # "pending" | "consumed" | "expired"
    symbols: list[str]
    urgency: str
    sentiment: str
    confidence: float
    reasoning: str
    ttl_remaining_seconds: float | None

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"QueueEventObservability(event_id={self.event_id!r}, "
            f"event_type={self.event_type!r}, status={self.status!r})"
        )


class QueuedMarketEvent(BaseModel):
    """Domain model for a pending market event in the queue."""

    event_id: str
    event_type: str
    payload: dict
    enqueued_at: datetime
    process_after: datetime | None = None

    def __repr__(self) -> str:
        """Return string representation."""
        return f"QueuedMarketEvent(event_id={self.event_id!r}, event_type={self.event_type!r})"


class MarketEventQueueORM(Base):
    """ORM model for the market_event_queue table."""

    __tablename__ = "market_event_queue"

    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, default=uuid.uuid4)
    event_id: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    enqueued_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    consumed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    process_after: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)

    __table_args__ = (
        Index(
            "idx_market_event_queue_pending",
            "enqueued_at",
            postgresql_where="consumed_at IS NULL",
        ),
        Index("idx_market_event_queue_expires", "expires_at"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"MarketEventQueueORM(event_id={self.event_id!r}, "
            f"event_type={self.event_type!r}, consumed_at={self.consumed_at!r})"
        )
