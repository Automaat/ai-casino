#!/usr/bin/env python
"""Import existing Alpaca positions into daemon database.

This script fetches current positions from Alpaca and creates position records
in the database with default metadata so they appear in the frontend.
"""

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from src.daemon.config import DaemonConfig
from src.daemon.positions import PositionRecord
from src.data.broker import AlpacaBroker
from src.database.engine import DatabaseEngine
from src.database.repositories.position import PositionRecordRepository
from src.strategies.signal import Signal


async def main() -> None:
    """Import Alpaca positions into database."""
    # Load config
    config_path = Path.home() / ".ai-casino" / "daemon-production.yaml"
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return

    daemon_config = DaemonConfig.from_yaml(config_path)

    # Initialize broker
    broker = AlpacaBroker(
        api_key=daemon_config.api_keys.alpaca_api_key or "",
        secret_key=daemon_config.api_keys.alpaca_secret_key or "",
        paper=daemon_config.trading_mode.value == "paper",
    )

    # Initialize database
    database_url = daemon_config.database.database_url
    if not database_url:
        logger.error("Database URL not configured")
        return

    db_engine = DatabaseEngine(database_url=database_url)
    await db_engine.ensure_migrated()

    # Fetch Alpaca positions
    logger.info("Fetching positions from Alpaca...")
    try:
        account_info = broker.get_account_info()
        alpaca_positions = list(account_info.positions.values())
        logger.info(f"Found {len(alpaca_positions)} positions in Alpaca")
    except Exception as e:
        logger.error(f"Failed to fetch Alpaca positions: {e}")
        return

    if not alpaca_positions:
        logger.info("No positions to import")
        return

    # Create position records
    async with db_engine.session() as session:
        repo = PositionRecordRepository(session)

        for pos in alpaca_positions:
            symbol = pos.symbol
            logger.info(f"Importing {symbol}: qty={pos.qty}, avg_entry_price=${pos.avg_entry_price}")

            # Check if position already exists
            existing = await repo.get_by_symbol(symbol)
            if existing:
                logger.warning(f"Position {symbol} already exists in database, skipping")
                continue

            # Create position record with default metadata
            now = datetime.now(UTC)
            entry_price = float(pos.avg_entry_price)
            # Default stop loss: 5% below entry (will be managed by position manager)
            default_stop_loss = entry_price * 0.95

            position_record = PositionRecord(
                symbol=symbol,
                entry_price=entry_price,
                entry_timestamp=now,  # Default to now (unknown actual entry)
                current_qty=float(pos.qty),
                entry_signal=Signal.HOLD.value,  # Unknown original signal
                entry_confidence=0.5,  # Unknown confidence
                current_stop_loss=default_stop_loss,
                initial_stop_loss=default_stop_loss,
                last_updated=now,
                trailing_stop_activated=False,
                breakeven_activated=False,
                profit_targets=[],
            )

            # Save to database
            try:
                await repo.create(position_record)
                logger.info(f"✓ Imported {symbol} into database")
            except Exception as e:
                logger.error(f"Failed to save {symbol}: {e}")

    await db_engine.close()
    logger.info("Import complete - positions should now appear in frontend")


if __name__ == "__main__":
    asyncio.run(main())
