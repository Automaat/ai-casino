"""Add Reddit data tables for web scraping integration.

This migration creates tables for storing Reddit posts, comments, ticker mentions,
and sentiment aggregates to support the Playwright-based Reddit scraping feature.

Tables:
- reddit_posts: Raw Reddit post data
- reddit_comments: Reddit comment data
- reddit_ticker_mentions: Extracted ticker mentions with sentiment
- reddit_ticker_sentiment: Aggregated sentiment metrics by symbol/subreddit

Usage:
    python -m src.database.migrations.add_reddit_tables
"""

import asyncio

from loguru import logger
from sqlalchemy import text

from src.database.engine import DatabaseEngine


async def migrate() -> None:
    """Apply migration to create Reddit tables."""
    engine = DatabaseEngine()

    async with engine.engine.begin() as conn:
        # Check if tables already exist
        result = await conn.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'reddit_posts'
                )
                """
            )
        )
        exists = result.scalar()

        if exists:
            logger.info("Reddit tables already exist, skipping migration")
            return

        logger.info("Creating Reddit tables...")

        # Create reddit_posts table
        await conn.execute(
            text(
                """
                CREATE TABLE reddit_posts (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    reddit_id VARCHAR(20) NOT NULL UNIQUE,
                    title VARCHAR(500) NOT NULL,
                    body TEXT,
                    subreddit VARCHAR(50) NOT NULL,
                    score INTEGER NOT NULL,
                    upvote_ratio DECIMAL(5,4) NOT NULL,
                    num_comments INTEGER NOT NULL,
                    url TEXT NOT NULL,
                    created_utc TIMESTAMP WITH TIME ZONE NOT NULL,
                    fetched_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
                )
                """
            )
        )
        logger.info("Created reddit_posts table")

        # Create indexes for reddit_posts
        await conn.execute(text("CREATE UNIQUE INDEX idx_reddit_posts_reddit_id ON reddit_posts(reddit_id)"))
        await conn.execute(text("CREATE INDEX idx_reddit_posts_subreddit ON reddit_posts(subreddit)"))
        await conn.execute(
            text("CREATE INDEX idx_reddit_posts_created_utc ON reddit_posts USING btree(created_utc)")
        )
        await conn.execute(text("CREATE INDEX idx_reddit_posts_score ON reddit_posts USING btree(score)"))
        await conn.execute(
            text("CREATE INDEX idx_reddit_posts_subreddit_created ON reddit_posts(subreddit, created_utc)")
        )
        logger.info("Created indexes for reddit_posts")

        # Create reddit_comments table
        await conn.execute(
            text(
                """
                CREATE TABLE reddit_comments (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    reddit_id VARCHAR(20) NOT NULL UNIQUE,
                    parent_post_reddit_id VARCHAR(20) NOT NULL,
                    body TEXT NOT NULL,
                    score INTEGER NOT NULL,
                    created_utc TIMESTAMP WITH TIME ZONE NOT NULL,
                    fetched_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
                )
                """
            )
        )
        logger.info("Created reddit_comments table")

        # Create indexes for reddit_comments
        await conn.execute(
            text("CREATE UNIQUE INDEX idx_reddit_comments_reddit_id ON reddit_comments(reddit_id)")
        )
        await conn.execute(
            text("CREATE INDEX idx_reddit_comments_parent_post ON reddit_comments(parent_post_reddit_id)")
        )
        await conn.execute(
            text("CREATE INDEX idx_reddit_comments_created_utc ON reddit_comments USING btree(created_utc)")
        )
        logger.info("Created indexes for reddit_comments")

        # Create reddit_ticker_mentions table
        await conn.execute(
            text(
                """
                CREATE TABLE reddit_ticker_mentions (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    symbol VARCHAR(10) NOT NULL,
                    source_type VARCHAR(20) NOT NULL,
                    source_reddit_id VARCHAR(20) NOT NULL,
                    subreddit VARCHAR(50) NOT NULL,
                    sentiment VARCHAR(20) NOT NULL,
                    mention_context VARCHAR(200),
                    confidence DECIMAL(5,4) NOT NULL,
                    extraction_method VARCHAR(10) NOT NULL,
                    created_utc TIMESTAMP WITH TIME ZONE NOT NULL,
                    extracted_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
                )
                """
            )
        )
        logger.info("Created reddit_ticker_mentions table")

        # Create indexes for reddit_ticker_mentions
        await conn.execute(
            text("CREATE INDEX idx_reddit_ticker_mentions_symbol ON reddit_ticker_mentions(symbol)")
        )
        await conn.execute(
            text(
                "CREATE INDEX idx_reddit_ticker_mentions_symbol_created "
                "ON reddit_ticker_mentions(symbol, created_utc)"
            )
        )
        await conn.execute(
            text("CREATE INDEX idx_reddit_ticker_mentions_subreddit ON reddit_ticker_mentions(subreddit)")
        )
        await conn.execute(
            text(
                "CREATE INDEX idx_reddit_ticker_mentions_created_utc "
                "ON reddit_ticker_mentions USING btree(created_utc)"
            )
        )
        logger.info("Created indexes for reddit_ticker_mentions")

        # Create reddit_ticker_sentiment table
        await conn.execute(
            text(
                """
                CREATE TABLE reddit_ticker_sentiment (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    symbol VARCHAR(10) NOT NULL,
                    subreddit VARCHAR(50) NOT NULL,
                    window_start TIMESTAMP WITH TIME ZONE NOT NULL,
                    window_end TIMESTAMP WITH TIME ZONE NOT NULL,
                    mention_count INTEGER NOT NULL,
                    avg_sentiment DECIMAL(5,4) NOT NULL,
                    bullish_count INTEGER NOT NULL,
                    bearish_count INTEGER NOT NULL,
                    neutral_count INTEGER NOT NULL,
                    avg_confidence DECIMAL(5,4) NOT NULL,
                    mention_velocity DECIMAL(8,4),
                    computed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
                )
                """
            )
        )
        logger.info("Created reddit_ticker_sentiment table")

        # Create indexes for reddit_ticker_sentiment
        await conn.execute(
            text("CREATE INDEX idx_reddit_ticker_sentiment_symbol ON reddit_ticker_sentiment(symbol)")
        )
        await conn.execute(
            text(
                "CREATE INDEX idx_reddit_ticker_sentiment_symbol_window "
                "ON reddit_ticker_sentiment(symbol, window_start)"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX idx_reddit_ticker_sentiment_mention_count "
                "ON reddit_ticker_sentiment USING btree(mention_count)"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX idx_reddit_ticker_sentiment_window_start "
                "ON reddit_ticker_sentiment USING btree(window_start)"
            )
        )
        logger.info("Created indexes for reddit_ticker_sentiment")

        logger.info("Migration completed successfully - all Reddit tables created")


async def main() -> None:
    """Run migration."""
    try:
        await migrate()
    except Exception:
        logger.opt(exception=True).error("Migration failed")
        raise


if __name__ == "__main__":
    asyncio.run(main())
