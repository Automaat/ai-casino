#!/usr/bin/env python3
"""One-time migration script from JSONL to PostgreSQL database."""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.database.engine import DatabaseEngine
from src.database.repositories.trade import TradeRepository
from src.metrics.tracker import TradeRecord


async def migrate_trades(jsonl_path: Path, db_engine: DatabaseEngine) -> tuple[int, int, int]:
    """Migrate trades from JSONL file to database.

    Args:
        jsonl_path: Path to trades.jsonl file
        db_engine: Database engine

    Returns:
        Tuple of (migrated count, skipped count, failed count)
    """
    if not jsonl_path.exists():
        logger.warning(f"JSONL file not found: {jsonl_path}")
        return 0, 0, 0

    await db_engine.ensure_migrated()

    async with db_engine.session() as session:
        repo = TradeRepository(session)
        migrated = 0
        skipped = 0
        failed = 0

        existing_trades = await repo.get_all()
        existing_keys = {
            (t.timestamp.isoformat(), t.symbol, t.action.value, float(t.entry_price))
            for t in existing_trades
        }

        with jsonl_path.open() as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                    trade = TradeRecord(**data)

                    trade_key = (
                        trade.timestamp.isoformat(),
                        trade.symbol,
                        trade.action.value,
                        float(trade.entry_price),
                    )

                    if trade_key in existing_keys:
                        skipped += 1
                        logger.debug(
                            f"Skipped duplicate trade: {trade.symbol} {trade.action.value} "
                            f"at {trade.timestamp}"
                        )
                        continue

                    await repo.create(trade)
                    migrated += 1
                    logger.debug(f"Migrated trade: {trade.symbol} {trade.action.value}")
                except Exception as e:
                    logger.opt(exception=True).error(
                        f"Failed to migrate trade at line {line_num}: {e}"
                    )
                    failed += 1

    return migrated, skipped, failed


async def main() -> None:
    """Run migration."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable not set")
        sys.exit(1)

    jsonl_path = Path.home() / ".ai-casino" / "logs" / "trades.jsonl"
    logger.info(f"Migrating trades from {jsonl_path} to database")

    db_engine = DatabaseEngine(database_url)

    try:
        migrated, skipped, failed = await migrate_trades(jsonl_path, db_engine)
        logger.info(
            f"Migration complete: {migrated} trades migrated, {skipped} duplicates skipped, {failed} failed"
        )

        if migrated > 0:
            backup_path = jsonl_path.with_suffix(".jsonl.bak")
            jsonl_path.rename(backup_path)
            logger.info(f"Original file backed up to {backup_path}")

    finally:
        await db_engine.close()


if __name__ == "__main__":
    asyncio.run(main())
