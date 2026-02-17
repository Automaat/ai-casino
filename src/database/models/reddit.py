"""ORM models for Reddit data storage."""

import uuid
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from sqlalchemy import DECIMAL, TIMESTAMP, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import text

from src.database.models.base import Base


class ExtractionMethod(StrEnum):
    """Ticker extraction method."""

    LLM = "LLM"
    REGEX = "REGEX"


class RedditPostORM(Base):
    """Reddit post ORM model."""

    __tablename__ = "reddit_posts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )
    reddit_id: Mapped[str] = mapped_column(String(20), nullable=False, unique=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    body: Mapped[str | None] = mapped_column(Text, nullable=True)
    subreddit: Mapped[str] = mapped_column(String(50), nullable=False)
    score: Mapped[int] = mapped_column(Integer, nullable=False)
    upvote_ratio: Mapped[Decimal] = mapped_column(DECIMAL(5, 4), nullable=False)
    num_comments: Mapped[int] = mapped_column(Integer, nullable=False)
    url: Mapped[str] = mapped_column(String(500), nullable=False)
    created_utc: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    __table_args__ = (
        Index("idx_reddit_posts_reddit_id", "reddit_id", unique=True),
        Index("idx_reddit_posts_subreddit", "subreddit"),
        Index("idx_reddit_posts_created_utc", "created_utc", postgresql_using="btree"),
        Index("idx_reddit_posts_score", "score", postgresql_using="btree"),
        Index("idx_reddit_posts_subreddit_created", "subreddit", "created_utc"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"RedditPostORM(id={self.id}, reddit_id={self.reddit_id}, score={self.score})"


class RedditCommentORM(Base):
    """Reddit comment ORM model."""

    __tablename__ = "reddit_comments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )
    reddit_id: Mapped[str] = mapped_column(String(20), nullable=False, unique=True)
    parent_post_reddit_id: Mapped[str] = mapped_column(String(20), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    score: Mapped[int] = mapped_column(Integer, nullable=False)
    created_utc: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    __table_args__ = (
        Index("idx_reddit_comments_reddit_id", "reddit_id", unique=True),
        Index("idx_reddit_comments_parent_post", "parent_post_reddit_id"),
        Index("idx_reddit_comments_created_utc", "created_utc", postgresql_using="btree"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"RedditCommentORM(id={self.id}, reddit_id={self.reddit_id})"


class RedditTickerMentionORM(Base):
    """Reddit ticker mention ORM model."""

    __tablename__ = "reddit_ticker_mentions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )
    symbol: Mapped[str] = mapped_column(String(10), nullable=False)
    source_type: Mapped[str] = mapped_column(String(20), nullable=False)  # 'post' or 'comment'
    source_reddit_id: Mapped[str] = mapped_column(String(20), nullable=False)
    subreddit: Mapped[str] = mapped_column(String(50), nullable=False)
    sentiment: Mapped[str] = mapped_column(String(20), nullable=False)  # BULLISH/BEARISH/NEUTRAL
    mention_context: Mapped[str | None] = mapped_column(String(200), nullable=True)
    confidence: Mapped[Decimal] = mapped_column(DECIMAL(5, 4), nullable=False)
    extraction_method: Mapped[str] = mapped_column(String(10), nullable=False)  # LLM or REGEX
    created_utc: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    extracted_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    __table_args__ = (
        Index("idx_reddit_ticker_mentions_symbol", "symbol"),
        Index("idx_reddit_ticker_mentions_symbol_created", "symbol", "created_utc"),
        Index("idx_reddit_ticker_mentions_subreddit", "subreddit"),
        Index("idx_reddit_ticker_mentions_created_utc", "created_utc", postgresql_using="btree"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"RedditTickerMentionORM(id={self.id}, symbol={self.symbol}, sentiment={self.sentiment})"


class RedditTickerSentimentORM(Base):
    """Reddit ticker sentiment aggregation ORM model."""

    __tablename__ = "reddit_ticker_sentiment"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )
    symbol: Mapped[str] = mapped_column(String(10), nullable=False)
    subreddit: Mapped[str] = mapped_column(String(50), nullable=False)
    window_start: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    mention_count: Mapped[int] = mapped_column(Integer, nullable=False)
    avg_sentiment: Mapped[Decimal] = mapped_column(DECIMAL(5, 4), nullable=False)
    bullish_count: Mapped[int] = mapped_column(Integer, nullable=False)
    bearish_count: Mapped[int] = mapped_column(Integer, nullable=False)
    neutral_count: Mapped[int] = mapped_column(Integer, nullable=False)
    avg_confidence: Mapped[Decimal] = mapped_column(DECIMAL(5, 4), nullable=False)
    mention_velocity: Mapped[Decimal | None] = mapped_column(DECIMAL(8, 4), nullable=True)
    computed_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    __table_args__ = (
        Index("idx_reddit_ticker_sentiment_symbol", "symbol"),
        Index("idx_reddit_ticker_sentiment_symbol_window", "symbol", "window_start"),
        Index("idx_reddit_ticker_sentiment_mention_count", "mention_count", postgresql_using="btree"),
        Index("idx_reddit_ticker_sentiment_window_start", "window_start", postgresql_using="btree"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"RedditTickerSentimentORM(id={self.id}, symbol={self.symbol}, "
            f"mentions={self.mention_count}, velocity={self.mention_velocity})"
        )
