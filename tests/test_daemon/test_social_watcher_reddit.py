"""Integration tests for SocialWatcher with Reddit DB backend."""

from datetime import UTC, datetime, timedelta

import pytest

from src.cache.historical import HistoricalCache
from src.data.reddit import RedditPost, TickerMention
from src.database.connection import get_db_engine
from src.database.models.reddit import (
    ExtractionMethod,
    RedditCommentORM,
    RedditPostORM,
    RedditTickerMentionORM,
)
from src.database.repositories.reddit import (
    RedditPostRepository,
    RedditTickerMentionRepository,
)
from src.watchers.social_watcher import SocialWatcher, SocialWatcherConfig


@pytest.fixture
async def setup_reddit_tables():
    """Setup Reddit tables in test database."""
    from sqlalchemy import MetaData

    engine = get_db_engine()

    # Create metadata with only Reddit tables
    reddit_metadata = MetaData()
    reddit_metadata._add_table(
        RedditPostORM.__table__.name, RedditPostORM.__table__.schema, RedditPostORM.__table__
    )
    reddit_metadata._add_table(
        RedditCommentORM.__table__.name, RedditCommentORM.__table__.schema, RedditCommentORM.__table__
    )
    reddit_metadata._add_table(
        RedditTickerMentionORM.__table__.name,
        RedditTickerMentionORM.__table__.schema,
        RedditTickerMentionORM.__table__,
    )

    # Create tables
    async with engine.engine.begin() as conn:
        await conn.run_sync(reddit_metadata.create_all)

    yield

    # Cleanup
    async with engine.engine.begin() as conn:
        await conn.run_sync(reddit_metadata.drop_all)


@pytest.fixture
def historical_cache(tmp_path):
    """Create test historical cache."""
    cache_path = tmp_path / "test_cache.db"
    return HistoricalCache(str(cache_path))


@pytest.fixture
def watcher_config():
    """Create test watcher config."""
    return SocialWatcherConfig(
        poll_interval=900,  # 15 minutes
        volume_spike_threshold=0.5,
        viral_score_threshold=1000,
        viral_upvote_ratio=0.8,
        subreddits=["wallstreetbets", "stocks"],
    )


@pytest.fixture
async def social_watcher(historical_cache, watcher_config):
    """Create SocialWatcher instance."""
    from unittest.mock import Mock

    from src.watchers.pipeline import EventTriagePipeline

    return SocialWatcher(
        pipeline=Mock(spec=EventTriagePipeline),
        historical_cache=historical_cache,
        config=watcher_config,
    )


async def insert_test_post(
    post_id: str,
    title: str,
    score: int,
    upvote_ratio: float,
    age_minutes: int,
    subreddit: str = "wallstreetbets",
) -> RedditPost:
    """Helper to insert test post into DB."""
    # Create timestamp with UTC timezone
    timestamp = datetime.now(UTC) - timedelta(minutes=age_minutes)

    post = RedditPost(
        id=post_id,
        title=title,
        body=f"Test post about {post_id}",
        subreddit=subreddit,
        score=score,
        upvote_ratio=upvote_ratio,
        url=f"https://reddit.com/r/{subreddit}/comments/{post_id}",
        created_utc=timestamp,
        num_comments=50,
    )

    async with get_db_engine().session() as session:
        repo = RedditPostRepository(session)
        await repo.create(post)

    # Return post with timezone-aware datetime (in case DB strips it)
    post.created_utc = timestamp.replace(tzinfo=UTC) if post.created_utc.tzinfo is None else post.created_utc
    return post


async def insert_test_mention(
    symbol: str,
    post_id: str,
    sentiment: str,
    confidence: float,
    age_minutes: int,
    subreddit: str = "wallstreetbets",
) -> None:
    """Helper to insert test ticker mention into DB."""
    mention = TickerMention(
        symbol=symbol,
        sentiment=sentiment,
        context=f"Test mention of {symbol}",
        confidence=confidence,
    )

    # Check if post exists first, create if not
    async with get_db_engine().session() as session:
        post_repo = RedditPostRepository(session)
        post = await post_repo.get_by_reddit_id(post_id)

    if not post:
        await insert_test_post(
            post_id=post_id,
            title=f"{symbol} discussion",
            score=500,
            upvote_ratio=0.85,
            age_minutes=age_minutes,
            subreddit=subreddit,
        )

    async with get_db_engine().session() as session:
        mention_repo = RedditTickerMentionRepository(session)
        post_repo = RedditPostRepository(session)

        post = await post_repo.get_by_reddit_id(post_id)
        if post:
            await mention_repo.bulk_insert_from_post(
                post=post,
                mentions=[mention],
                extraction_method=ExtractionMethod.LLM,
            )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_volume_spike_detection_from_db(setup_reddit_tables, social_watcher):
    """Test volume spike detection using DB queries."""
    # Insert baseline mentions (first poll)
    await insert_test_mention("TSLA", "post1", "BULLISH", 0.9, age_minutes=10)
    await insert_test_mention("TSLA", "post2", "NEUTRAL", 0.85, age_minutes=8)

    events = await social_watcher._fetch_events()
    assert len(events) == 0  # No spike on first poll (no baseline)
    assert social_watcher._previous_mention_counts["TSLA"] == 2

    # Insert more mentions (100% increase)
    await insert_test_mention("TSLA", "post3", "BULLISH", 0.95, age_minutes=5)
    await insert_test_mention("TSLA", "post4", "BULLISH", 0.88, age_minutes=3)

    events = await social_watcher._fetch_events()

    # Should detect volume spike (2 -> 4 = 100% increase > 50% threshold)
    spike_events = [e for e in events if e.mention_delta_pct is not None]
    assert len(spike_events) >= 1
    assert spike_events[0].symbol == "TSLA"
    assert spike_events[0].mention_count == 4
    assert spike_events[0].mention_delta_pct == 100.0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_viral_post_detection_from_db(setup_reddit_tables, social_watcher):
    """Test viral post detection using DB queries."""
    # Insert viral post (recent, high score, high ratio) - within 15min window
    viral_post = await insert_test_post(
        post_id="viral123",
        title="TSLA to the moon! 🚀",
        score=2500,
        upvote_ratio=0.95,
        age_minutes=10,
    )

    # Insert ticker mention for this post
    await insert_test_mention("TSLA", "viral123", "BULLISH", 0.95, age_minutes=10)

    events = await social_watcher._fetch_events()

    # Should detect viral post
    viral_events = [e for e in events if e.viral_post is not None]
    assert len(viral_events) >= 1
    assert viral_events[0].symbol == "TSLA"
    assert viral_events[0].viral_post.id == "viral123"
    assert viral_events[0].viral_post.score == 2500


@pytest.mark.integration
@pytest.mark.asyncio
async def test_viral_post_filters_old_posts(setup_reddit_tables, social_watcher):
    """Test viral posts older than 1hr are filtered."""
    # Insert old viral post (>1hr)
    await insert_test_post(
        post_id="old123",
        title="Old viral post",
        score=3000,
        upvote_ratio=0.95,
        age_minutes=90,  # >1hr
    )
    await insert_test_mention("NVDA", "old123", "BULLISH", 0.9, age_minutes=90)

    events = await social_watcher._fetch_events()

    # Should NOT detect old post
    viral_events = [e for e in events if e.viral_post is not None]
    assert len(viral_events) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_viral_post_filters_low_score(setup_reddit_tables, social_watcher):
    """Test posts below score threshold are filtered."""
    # Insert low-score post
    await insert_test_post(
        post_id="low123",
        title="Low score post",
        score=500,  # Below 1000 threshold
        upvote_ratio=0.95,
        age_minutes=30,
    )
    await insert_test_mention("AMD", "low123", "BULLISH", 0.9, age_minutes=30)

    events = await social_watcher._fetch_events()

    # Should NOT detect low score post
    viral_events = [e for e in events if e.viral_post is not None]
    assert len(viral_events) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_viral_post_deduplication(setup_reddit_tables, social_watcher):
    """Test duplicate viral posts are filtered."""
    # Insert viral post - within 15min window
    await insert_test_post(
        post_id="dup123",
        title="Duplicate check",
        score=2000,
        upvote_ratio=0.95,
        age_minutes=10,
    )
    await insert_test_mention("GOOGL", "dup123", "BULLISH", 0.9, age_minutes=10)

    # First fetch
    events1 = await social_watcher._fetch_events()
    viral_events1 = [e for e in events1 if e.viral_post is not None]
    assert len(viral_events1) >= 1
    assert "dup123" in social_watcher._seen_post_ids

    # Second fetch (same post)
    events2 = await social_watcher._fetch_events()
    viral_events2 = [e for e in events2 if e.viral_post is not None and e.viral_post.id == "dup123"]

    # Should be filtered (already seen)
    assert len(viral_events2) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multiple_symbols_same_window(setup_reddit_tables, social_watcher):
    """Test handling of multiple symbols in same window."""
    # Insert mentions for multiple symbols
    await insert_test_mention("TSLA", "p1", "BULLISH", 0.9, age_minutes=10)
    await insert_test_mention("NVDA", "p2", "BULLISH", 0.85, age_minutes=8)
    await insert_test_mention("AAPL", "p3", "NEUTRAL", 0.80, age_minutes=5)

    events = await social_watcher._fetch_events()

    # Should track all symbols
    assert "TSLA" in social_watcher._previous_mention_counts
    assert "NVDA" in social_watcher._previous_mention_counts
    assert "AAPL" in social_watcher._previous_mention_counts


@pytest.mark.integration
@pytest.mark.asyncio
async def test_combined_volume_and_viral_events(setup_reddit_tables, social_watcher):
    """Test detecting both volume spike and viral post for same symbol."""
    # Establish baseline - within 15min window
    await insert_test_mention("AAPL", "base1", "NEUTRAL", 0.8, age_minutes=12)
    await insert_test_mention("AAPL", "base2", "NEUTRAL", 0.8, age_minutes=11)

    events1 = await social_watcher._fetch_events()
    assert len(events1) == 0  # No baseline yet
    assert social_watcher._previous_mention_counts["AAPL"] == 2

    # Insert spike + viral post
    await insert_test_mention("AAPL", "spike1", "BULLISH", 0.9, age_minutes=5)
    await insert_test_mention("AAPL", "spike2", "BULLISH", 0.9, age_minutes=4)

    viral_post = await insert_test_post(
        post_id="viral_aapl",
        title="AAPL earnings beat expectations!",
        score=2500,
        upvote_ratio=0.95,
        age_minutes=10,
    )
    await insert_test_mention("AAPL", "viral_aapl", "BULLISH", 0.95, age_minutes=10)

    events2 = await social_watcher._fetch_events()

    # Should detect both volume spike and viral post
    volume_events = [e for e in events2 if e.mention_delta_pct is not None]
    viral_events = [e for e in events2 if e.viral_post is not None]

    assert len(volume_events) >= 1  # Volume spike (2 -> 4 = 100%)
    assert len(viral_events) >= 1  # Viral post
