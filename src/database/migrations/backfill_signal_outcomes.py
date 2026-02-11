"""Backfill signal outcomes from SQLite historical cache to PostgreSQL.

Usage:
    python -m src.database.migrations.backfill_signal_outcomes --sqlite-path ~/.ai-casino/cache/historical.db
"""

import argparse
import asyncio
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from sqlalchemy.exc import IntegrityError

from src.cache.historical import HistoricalCache
from src.daemon.state.models import SignalOutcome
from src.database.engine import DatabaseEngine
from src.database.repositories.signal_outcome import SignalOutcomeRepository


async def backfill_signal_outcomes(
    sqlite_path: Path,
    database_url: str,
) -> dict[str, int]:
    """Backfill signal outcomes from SQLite to PostgreSQL.

    Args:
        sqlite_path: Path to SQLite historical.db file
        database_url: PostgreSQL connection URL

    Returns:
        Dict with migration statistics
    """
    logger.info(f"Starting signal outcome migration from {sqlite_path}")

    # Initialize SQLite cache
    cache = HistoricalCache(db_path=str(sqlite_path))

    # Get all signal outcomes from SQLite
    sqlite_records = cache.get_signal_outcomes(window="all")
    logger.info(f"Found {len(sqlite_records)} signal outcomes in SQLite")

    if not sqlite_records:
        logger.warning("No signal outcomes found in SQLite, nothing to migrate")
        return {"total": 0, "migrated": 0, "skipped": 0, "failed": 0}

    # Initialize PostgreSQL engine and repository
    engine = DatabaseEngine(database_url=database_url)
    await engine.ensure_migrated()

    session = engine.session()
    repository = SignalOutcomeRepository(session)

    stats = {"total": len(sqlite_records), "migrated": 0, "skipped": 0, "failed": 0}

    # Migrate each record
    for record in sqlite_records:
        try:
            # Convert SQLite record to SignalOutcome domain model
            timestamp = datetime.fromisoformat(record.timestamp)

            signal_outcome = SignalOutcome(
                symbol=record.symbol,
                timestamp=timestamp,
                signal=record.signal,
                confidence=record.confidence,
                price_at_signal=record.price_at_signal,
                strategy_used=record.strategy_used,
                regime=record.regime,
                trading_session="REGULAR",  # SQLite doesn't have this field
                technical_signal=None,  # SQLite doesn't have this field
                sentiment_signal=None,  # SQLite doesn't have this field
                news_signal=None,  # SQLite doesn't have this field
                price_at_1d=record.price_at_1d,
                price_at_5d=record.price_at_5d,
                price_at_20d=record.price_at_20d,
                actual_exit_price=record.actual_exit_price,
                actual_exit_date=None,  # SQLite doesn't have this field
                outcome_updated_at=datetime.now(UTC) if record.price_at_1d else None,
            )

            # Insert into PostgreSQL - handle duplicates via IntegrityError
            try:
                await repository.create(signal_outcome)
                stats["migrated"] += 1
            except IntegrityError:
                logger.debug(f"Skipping duplicate: {record.symbol} @ {timestamp}")
                stats["skipped"] += 1
                continue

            if stats["migrated"] % 50 == 0:
                logger.info(f"Migration progress: {stats['migrated']}/{stats['total']}")

        except Exception as e:
            logger.opt(exception=True).error(f"Failed to migrate record {record.id}: {e}")
            stats["failed"] += 1

    # Close database connection
    await engine.engine.dispose()

    logger.info(
        f"Migration complete: {stats['migrated']} migrated, "
        f"{stats['skipped']} skipped, {stats['failed']} failed"
    )
    return stats


async def verify_migration(sqlite_path: Path, database_url: str) -> bool:
    """Verify migration by comparing counts.

    Args:
        sqlite_path: Path to SQLite historical.db file
        database_url: PostgreSQL connection URL

    Returns:
        True if counts match, False otherwise
    """
    logger.info("Verifying migration...")

    # SQLite count
    cache = HistoricalCache(db_path=str(sqlite_path))
    sqlite_count = len(cache.get_signal_outcomes(window="all"))

    # PostgreSQL count
    engine = DatabaseEngine(database_url=database_url)
    session = engine.session()
    repository = SignalOutcomeRepository(session)

    postgres_records = await repository.get_recent_outcomes(window=365 * 10)  # 10 years
    postgres_count = len(postgres_records)

    await engine.engine.dispose()

    logger.info(f"SQLite count: {sqlite_count}, PostgreSQL count: {postgres_count}")

    if sqlite_count == postgres_count:
        logger.info("✅ Verification passed: counts match")
        return True

    logger.warning(f"⚠️ Verification failed: counts don't match (diff: {sqlite_count - postgres_count})")
    return False


def main() -> None:
    """CLI entry point for migration script."""
    parser = argparse.ArgumentParser(description="Backfill signal outcomes from SQLite to PostgreSQL")
    parser.add_argument(
        "--sqlite-path",
        type=Path,
        default=Path.home() / ".ai-casino" / "cache" / "historical.db",
        help="Path to SQLite historical.db file",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        help="PostgreSQL connection URL (or use DATABASE_URL env var)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verification after migration",
    )

    args = parser.parse_args()

    # Resolve database URL
    import os

    database_url = args.database_url or os.getenv("DATABASE_URL")
    if not database_url:
        logger.error("Database URL not provided. Use --database-url or set DATABASE_URL env var")
        return

    if not args.sqlite_path.exists():
        logger.error(f"SQLite file not found: {args.sqlite_path}")
        return

    # Run migration
    stats = asyncio.run(backfill_signal_outcomes(args.sqlite_path, database_url))

    logger.info(f"Final stats: {stats}")

    # Run verification if requested
    if args.verify and stats["migrated"] > 0:
        asyncio.run(verify_migration(args.sqlite_path, database_url))


if __name__ == "__main__":
    main()
