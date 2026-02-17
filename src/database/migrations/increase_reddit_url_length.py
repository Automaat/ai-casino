"""Increase reddit_posts.url column length to handle long tracking URLs."""

import asyncio
from datetime import UTC, datetime

from loguru import logger
from sqlalchemy import text

from src.database.connection import DatabaseEngine


async def apply_migration() -> None:
    """Apply migration to increase url column length."""
    engine = DatabaseEngine()

    async with engine.engine.begin() as conn:
        logger.info("Increasing reddit_posts.url column length...")

        # Alter column to support longer URLs (tracking params can exceed 500 chars)
        await conn.execute(
            text(
                """
                ALTER TABLE reddit_posts
                ALTER COLUMN url TYPE VARCHAR(2000)
                """
            )
        )

        logger.info("Successfully increased reddit_posts.url to VARCHAR(2000)")


def main() -> None:
    """Run migration."""
    start = datetime.now(UTC)
    logger.info("Starting migration: increase_reddit_url_length")

    try:
        asyncio.run(apply_migration())
        duration = (datetime.now(UTC) - start).total_seconds()
        logger.success(f"Migration completed in {duration:.2f}s")
    except Exception:
        logger.opt(exception=True).error("Migration failed")
        raise


if __name__ == "__main__":
    main()
