"""Reddit data repositories for database operations."""

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert

from src.daemon.events import Sentiment
from src.data.reddit import RedditComment, RedditPost, TickerMention
from src.database.models.reddit import (
    ExtractionMethod,
    RedditCommentORM,
    RedditPostORM,
    RedditTickerMentionORM,
    RedditTickerSentimentORM,
)
from src.database.repositories.base import BaseRepository

if TYPE_CHECKING:
    from sqlalchemy.engine import Result
    from sqlalchemy.ext.asyncio import AsyncSession


class RedditPostRepository(BaseRepository[RedditPost]):
    """Repository for Reddit post persistence."""

    def __init__(self, session: "AsyncSession") -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(session)

    async def create(self, entity: RedditPost) -> RedditPost:
        """Create new Reddit post.

        Args:
            entity: RedditPost to persist

        Returns:
            Created RedditPost
        """
        orm = RedditPostORM(
            id=uuid.uuid4(),
            reddit_id=entity.id,
            title=entity.title,
            body=entity.body,
            subreddit=entity.subreddit,
            score=entity.score,
            upvote_ratio=Decimal(str(entity.upvote_ratio)),
            num_comments=entity.num_comments,
            url=entity.url,
            created_utc=entity.created_utc,
            fetched_at=datetime.now(UTC),
        )
        self._session.add(orm)
        await self._session.commit()
        logger.debug(f"Created Reddit post: {entity.id} from r/{entity.subreddit}")
        return entity

    async def bulk_insert(self, posts: list[RedditPost]) -> int:
        """Bulk insert Reddit posts with conflict ignore.

        Args:
            posts: List of RedditPosts to insert

        Returns:
            Number of posts inserted
        """
        if not posts:
            return 0

        values = [
            {
                "id": uuid.uuid4(),
                "reddit_id": post.id,
                "title": post.title,
                "body": post.body,
                "subreddit": post.subreddit,
                "score": post.score,
                "upvote_ratio": Decimal(str(post.upvote_ratio)),
                "num_comments": post.num_comments,
                "url": post.url,
                "created_utc": post.created_utc,
                "fetched_at": datetime.now(UTC),
            }
            for post in posts
        ]

        stmt = insert(RedditPostORM).values(values).on_conflict_do_nothing(index_elements=["reddit_id"])

        result: Result = await self._session.execute(stmt)
        await self._session.commit()

        inserted_count = getattr(result, "rowcount", 0) or 0
        logger.info(f"Bulk inserted {inserted_count}/{len(posts)} Reddit posts (deduped by reddit_id)")
        return inserted_count

    async def get_by_id(self, entity_id: str) -> RedditPost | None:
        """Get Reddit post by database ID.

        Args:
            entity_id: Database UUID string

        Returns:
            RedditPost if found, None otherwise
        """
        result = await self._session.execute(
            select(RedditPostORM).where(RedditPostORM.id == uuid.UUID(entity_id))
        )
        orm = result.scalar_one_or_none()
        return self._to_post(orm) if orm else None

    async def get_by_reddit_id(self, reddit_id: str) -> RedditPost | None:
        """Get Reddit post by Reddit ID.

        Args:
            reddit_id: Reddit post ID (e.g., t3_abc123)

        Returns:
            RedditPost if found, None otherwise
        """
        result = await self._session.execute(
            select(RedditPostORM).where(RedditPostORM.reddit_id == reddit_id)
        )
        orm = result.scalar_one_or_none()
        return self._to_post(orm) if orm else None

    async def get_posts_in_window(
        self,
        window_minutes: int,
        subreddits: list[str] | None = None,
    ) -> list[RedditPost]:
        """Get posts within time window.

        Args:
            window_minutes: Time window in minutes
            subreddits: Optional list of subreddits to filter

        Returns:
            List of RedditPosts within time window
        """
        cutoff = datetime.now(UTC) - timedelta(minutes=window_minutes)

        query = select(RedditPostORM).where(RedditPostORM.created_utc >= cutoff)

        if subreddits:
            query = query.where(RedditPostORM.subreddit.in_(subreddits))

        query = query.order_by(RedditPostORM.created_utc.desc())

        result = await self._session.execute(query)
        return [self._to_post(orm) for orm in result.scalars().all()]

    async def delete_before(self, cutoff: datetime) -> int:
        """Delete posts older than cutoff date.

        Args:
            cutoff: Delete posts with created_utc < cutoff

        Returns:
            Number of posts deleted
        """
        result: Result = await self._session.execute(
            delete(RedditPostORM).where(RedditPostORM.created_utc < cutoff)
        )
        await self._session.commit()
        deleted_count = getattr(result, "rowcount", 0) or 0
        logger.info(f"Deleted {deleted_count} Reddit posts before {cutoff}")
        return deleted_count

    def _to_post(self, orm: RedditPostORM) -> RedditPost:
        """Convert ORM model to RedditPost.

        Args:
            orm: RedditPostORM instance

        Returns:
            RedditPost
        """
        return RedditPost(
            id=orm.reddit_id,
            title=orm.title,
            body=orm.body or "",
            subreddit=orm.subreddit,
            score=orm.score,
            upvote_ratio=float(orm.upvote_ratio),
            url=orm.url,
            created_utc=orm.created_utc,
            num_comments=orm.num_comments,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "RedditPostRepository()"


class RedditCommentRepository(BaseRepository[RedditComment]):
    """Repository for Reddit comment persistence."""

    def __init__(self, session: "AsyncSession") -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(session)

    async def create(self, entity: RedditComment) -> RedditComment:
        """Create new Reddit comment.

        Args:
            entity: RedditComment to persist

        Returns:
            Created RedditComment
        """
        orm = RedditCommentORM(
            id=uuid.uuid4(),
            reddit_id=entity.id,
            parent_post_reddit_id=entity.parent_post_id,
            body=entity.body,
            score=entity.score,
            created_utc=entity.created_utc,
            fetched_at=datetime.now(UTC),
        )
        self._session.add(orm)
        await self._session.commit()
        logger.debug(f"Created Reddit comment: {entity.id}")
        return entity

    async def bulk_insert(self, comments: list[RedditComment]) -> int:
        """Bulk insert Reddit comments with conflict ignore.

        Args:
            comments: List of RedditComments to insert

        Returns:
            Number of comments inserted
        """
        if not comments:
            return 0

        values = [
            {
                "id": uuid.uuid4(),
                "reddit_id": comment.id,
                "parent_post_reddit_id": comment.parent_post_id,
                "body": comment.body,
                "score": comment.score,
                "created_utc": comment.created_utc,
                "fetched_at": datetime.now(UTC),
            }
            for comment in comments
        ]

        stmt = insert(RedditCommentORM).values(values).on_conflict_do_nothing(index_elements=["reddit_id"])

        result: Result = await self._session.execute(stmt)
        await self._session.commit()

        inserted_count = getattr(result, "rowcount", 0) or 0
        logger.info(f"Bulk inserted {inserted_count}/{len(comments)} Reddit comments (deduped by reddit_id)")
        return inserted_count

    async def get_by_id(self, entity_id: str) -> RedditComment | None:
        """Get Reddit comment by database ID.

        Args:
            entity_id: Database UUID string

        Returns:
            RedditComment if found, None otherwise
        """
        result = await self._session.execute(
            select(RedditCommentORM).where(RedditCommentORM.id == uuid.UUID(entity_id))
        )
        orm = result.scalar_one_or_none()
        return self._to_comment(orm) if orm else None

    async def get_by_post_id(self, post_reddit_id: str) -> list[RedditComment]:
        """Get comments for a specific post.

        Args:
            post_reddit_id: Reddit post ID

        Returns:
            List of RedditComments for the post
        """
        result = await self._session.execute(
            select(RedditCommentORM)
            .where(RedditCommentORM.parent_post_reddit_id == post_reddit_id)
            .order_by(RedditCommentORM.score.desc())
        )
        return [self._to_comment(orm) for orm in result.scalars().all()]

    def _to_comment(self, orm: RedditCommentORM) -> RedditComment:
        """Convert ORM model to RedditComment.

        Args:
            orm: RedditCommentORM instance

        Returns:
            RedditComment
        """
        return RedditComment(
            id=orm.reddit_id,
            parent_post_id=orm.parent_post_reddit_id,
            body=orm.body,
            score=orm.score,
            created_utc=orm.created_utc,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "RedditCommentRepository()"


class RedditTickerMentionRepository(BaseRepository[TickerMention]):
    """Repository for Reddit ticker mention persistence."""

    def __init__(self, session: "AsyncSession") -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(session)

    async def create(self, entity: TickerMention) -> TickerMention:
        """Create new ticker mention (not typically used, prefer bulk_insert).

        Args:
            entity: TickerMention to persist

        Returns:
            Created TickerMention
        """
        msg = "Use bulk_insert_from_post() or bulk_insert_from_comment() instead"
        raise NotImplementedError(msg)

    async def bulk_insert_from_post(
        self,
        post: RedditPost,
        mentions: list[TickerMention],
        extraction_method: ExtractionMethod = ExtractionMethod.LLM,
    ) -> int:
        """Bulk insert ticker mentions from a post.

        Args:
            post: RedditPost source
            mentions: List of TickerMentions
            extraction_method: LLM or REGEX

        Returns:
            Number of mentions inserted
        """
        if not mentions:
            return 0

        values = [
            {
                "id": uuid.uuid4(),
                "symbol": mention.symbol.upper(),
                "source_type": "post",
                "source_reddit_id": post.id,
                "subreddit": post.subreddit,
                "sentiment": mention.sentiment,
                "mention_context": mention.context[:200] if mention.context else None,
                "confidence": Decimal(str(mention.confidence)),
                "extraction_method": extraction_method.value,
                "created_utc": post.created_utc,
                "extracted_at": datetime.now(UTC),
            }
            for mention in mentions
        ]

        stmt = insert(RedditTickerMentionORM).values(values)
        result: Result = await self._session.execute(stmt)
        await self._session.commit()

        inserted_count = getattr(result, "rowcount", 0) or 0
        logger.debug(f"Inserted {inserted_count} ticker mentions from post {post.id}")
        return inserted_count

    async def get_by_id(self, entity_id: str) -> TickerMention | None:
        """Get ticker mention by ID.

        Args:
            entity_id: Database UUID string

        Returns:
            TickerMention if found, None otherwise
        """
        result = await self._session.execute(
            select(RedditTickerMentionORM).where(RedditTickerMentionORM.id == uuid.UUID(entity_id))
        )
        orm = result.scalar_one_or_none()
        return self._to_mention(orm) if orm else None

    async def get_mentions_in_window(
        self,
        window_minutes: int,
        symbol: str | None = None,
    ) -> list[tuple[str, int]]:
        """Get mention counts within time window, grouped by symbol.

        Args:
            window_minutes: Time window in minutes
            symbol: Optional symbol filter

        Returns:
            List of (symbol, count) tuples sorted by count desc
        """
        from sqlalchemy import func

        cutoff = datetime.now(UTC) - timedelta(minutes=window_minutes)

        query = (
            select(RedditTickerMentionORM.symbol, func.count(RedditTickerMentionORM.id).label("count"))
            .where(RedditTickerMentionORM.created_utc >= cutoff)
            .group_by(RedditTickerMentionORM.symbol)
            .order_by(func.count(RedditTickerMentionORM.id).desc())
        )

        if symbol:
            query = query.where(RedditTickerMentionORM.symbol == symbol.upper())

        result = await self._session.execute(query)
        return [(row[0], row[1]) for row in result.all()]

    def _to_mention(self, orm: RedditTickerMentionORM) -> TickerMention:
        """Convert ORM model to TickerMention.

        Args:
            orm: RedditTickerMentionORM instance

        Returns:
            TickerMention
        """
        return TickerMention(
            symbol=orm.symbol,
            sentiment=orm.sentiment,
            context=orm.mention_context or "",
            confidence=float(orm.confidence),
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "RedditTickerMentionRepository()"


class RedditTickerSentimentRepository(BaseRepository[dict]):
    """Repository for Reddit ticker sentiment aggregates."""

    def __init__(self, session: "AsyncSession") -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(session)

    async def create(self, entity: dict) -> dict:
        """Create sentiment aggregate (not typically used).

        Args:
            entity: Sentiment aggregate dict

        Returns:
            Created entity
        """
        msg = "Use compute_hourly_aggregates() instead"
        raise NotImplementedError(msg)

    async def get_by_id(self, entity_id: str) -> dict | None:
        """Get sentiment aggregate by ID.

        Args:
            entity_id: Database UUID string

        Returns:
            Sentiment aggregate dict if found, None otherwise
        """
        result = await self._session.execute(
            select(RedditTickerSentimentORM).where(RedditTickerSentimentORM.id == uuid.UUID(entity_id))
        )
        orm = result.scalar_one_or_none()
        return self._to_dict(orm) if orm else None

    async def compute_hourly_aggregates(self, lookback_hours: int = 1) -> int:
        """Compute sentiment aggregates from mentions for recent hours.

        Args:
            lookback_hours: Hours to look back

        Returns:
            Number of aggregates created
        """
        from sqlalchemy import case, func

        cutoff = datetime.now(UTC) - timedelta(hours=lookback_hours)

        # Aggregate mentions by symbol and subreddit for 1-hour windows
        query = (
            select(
                RedditTickerMentionORM.symbol,
                RedditTickerMentionORM.subreddit,
                func.count(RedditTickerMentionORM.id).label("mention_count"),
                func.sum(case((RedditTickerMentionORM.sentiment == Sentiment.BULLISH, 1), else_=0)).label(
                    "bullish_count"
                ),
                func.sum(case((RedditTickerMentionORM.sentiment == Sentiment.BEARISH, 1), else_=0)).label(
                    "bearish_count"
                ),
                func.sum(case((RedditTickerMentionORM.sentiment == Sentiment.NEUTRAL, 1), else_=0)).label(
                    "neutral_count"
                ),
                func.avg(RedditTickerMentionORM.confidence).label("avg_confidence"),
                func.min(RedditTickerMentionORM.created_utc).label("window_start"),
                func.max(RedditTickerMentionORM.created_utc).label("window_end"),
            )
            .where(RedditTickerMentionORM.extracted_at >= cutoff)
            .group_by(RedditTickerMentionORM.symbol, RedditTickerMentionORM.subreddit)
        )

        result = await self._session.execute(query)
        rows = result.all()

        if not rows:
            logger.debug("No ticker mentions to aggregate")
            return 0

        # Compute sentiment score (bullish=1, neutral=0, bearish=-1, normalized to 0-1)
        aggregates = []
        for row in rows:
            bullish = row.bullish_count or 0
            bearish = row.bearish_count or 0
            neutral = row.neutral_count or 0
            total = bullish + bearish + neutral

            if total == 0:
                continue

            # Sentiment score: (bullish - bearish) / total normalized to 0-1 range
            # -1 (all bearish) -> 0.0, 0 (neutral) -> 0.5, +1 (all bullish) -> 1.0
            raw_score = (bullish - bearish) / total
            avg_sentiment = (raw_score + 1) / 2  # Map [-1, 1] to [0, 1]

            aggregates.append(
                {
                    "id": uuid.uuid4(),
                    "symbol": row.symbol,
                    "subreddit": row.subreddit,
                    "window_start": row.window_start,
                    "window_end": row.window_end,
                    "mention_count": row.mention_count,
                    "avg_sentiment": Decimal(str(avg_sentiment)),
                    "bullish_count": bullish,
                    "bearish_count": bearish,
                    "neutral_count": neutral,
                    "avg_confidence": Decimal(str(row.avg_confidence)),
                    "mention_velocity": None,  # Computed later by comparing windows
                    "computed_at": datetime.now(UTC),
                }
            )

        if aggregates:
            stmt = insert(RedditTickerSentimentORM).values(aggregates)
            result: Result = await self._session.execute(stmt)
            await self._session.commit()
            inserted_count = getattr(result, "rowcount", 0) or 0
            logger.info(f"Computed {inserted_count} sentiment aggregates")
            return inserted_count

        return 0

    def _to_dict(self, orm: RedditTickerSentimentORM) -> dict:
        """Convert ORM model to dict.

        Args:
            orm: RedditTickerSentimentORM instance

        Returns:
            Sentiment aggregate dict
        """
        return {
            "id": str(orm.id),
            "symbol": orm.symbol,
            "subreddit": orm.subreddit,
            "window_start": orm.window_start,
            "window_end": orm.window_end,
            "mention_count": orm.mention_count,
            "avg_sentiment": float(orm.avg_sentiment),
            "bullish_count": orm.bullish_count,
            "bearish_count": orm.bearish_count,
            "neutral_count": orm.neutral_count,
            "avg_confidence": float(orm.avg_confidence),
            "mention_velocity": float(orm.mention_velocity) if orm.mention_velocity else None,
            "computed_at": orm.computed_at,
        }

    def __repr__(self) -> str:
        """Return string representation."""
        return "RedditTickerSentimentRepository()"
