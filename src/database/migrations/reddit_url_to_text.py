"""Change reddit_posts.url column from VARCHAR(2000) to TEXT to support unlimited-length URLs."""

import asyncio
from datetime import UTC, datetime

from loguru import logger
from sqlalchemy import text

from src.database.connection import DatabaseEngine


async def apply_migration() -> None:
    """Apply migration to change url column to TEXT."""
    engine = DatabaseEngine()

    async with engine.engine.begin() as conn:
        logger.info("Changing reddit_posts.url column to TEXT...")

        await conn.execute(
            text(
                """
                ALTER TABLE reddit_posts
                ALTER COLUMN url TYPE TEXT
                """
            )
        )

        logger.info("Successfully changed reddit_posts.url to TEXT")


def main() -> None:
    """Run migration."""
    start = datetime.now(UTC)
    logger.info("Starting migration: reddit_url_to_text")

    try:
        asyncio.run(apply_migration())
        duration = (datetime.now(UTC) - start).total_seconds()
        logger.success(f"Migration completed in {duration:.2f}s")
    except Exception:
        logger.opt(exception=True).error("Migration failed")
        raise


if __name__ == "__main__":
    main()
