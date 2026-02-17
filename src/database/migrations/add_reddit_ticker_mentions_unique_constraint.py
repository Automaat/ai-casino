"""Add unique constraint to reddit_ticker_mentions for deduplication."""

import asyncio
from datetime import UTC, datetime

from loguru import logger
from sqlalchemy import text

from src.database.connection import DatabaseEngine


async def apply_migration() -> None:
    """Apply migration to add unique constraint."""
    engine = DatabaseEngine()

    async with engine.engine.begin() as conn:
        logger.info("Adding unique constraint to reddit_ticker_mentions...")

        # Add unique constraint for ON CONFLICT deduplication
        await conn.execute(
            text(
                """
                CREATE UNIQUE INDEX idx_reddit_ticker_mentions_dedup
                ON reddit_ticker_mentions(source_type, source_reddit_id, symbol, extraction_method)
                """
            )
        )

        logger.info("Successfully added unique constraint to reddit_ticker_mentions")


def main() -> None:
    """Run migration."""
    start = datetime.now(UTC)
    logger.info("Starting migration: add_reddit_ticker_mentions_unique_constraint")

    try:
        asyncio.run(apply_migration())
        duration = (datetime.now(UTC) - start).total_seconds()
        logger.success(f"Migration completed in {duration:.2f}s")
    except Exception:
        logger.opt(exception=True).error("Migration failed")
        raise


if __name__ == "__main__":
    main()
