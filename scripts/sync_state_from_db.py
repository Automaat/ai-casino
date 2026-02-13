#!/usr/bin/env python
"""Sync daemon state from database positions."""

import asyncio
import json
from pathlib import Path

from loguru import logger

from src.daemon.config import DaemonConfig
from src.database.engine import DatabaseEngine
from src.database.repositories.position import PositionRecordRepository


async def main() -> None:
    """Load positions from database and update daemon state file."""
    # Load config
    config_path = Path.home() / ".ai-casino" / "daemon-production.yaml"
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return

    daemon_config = DaemonConfig.from_yaml(config_path)

    # Initialize database
    database_url = daemon_config.database.database_url
    if not database_url:
        logger.error("Database URL not configured")
        return

    db_engine = DatabaseEngine(database_url=database_url)
    await db_engine.ensure_migrated()

    # Fetch positions from database
    logger.info("Fetching positions from database...")
    async with db_engine.session() as session:
        repo = PositionRecordRepository(session)
        positions = await repo.get_all_active()
        logger.info(f"Found {len(positions)} active positions in database")

    if not positions:
        logger.info("No positions to sync")
        await db_engine.close()
        return

    # Load current state file
    state_file = Path.home() / ".ai-casino" / "daemon-state.json"
    if state_file.exists():
        state_data = json.loads(state_file.read_text())
        logger.info(f"Loaded existing state from {state_file}")
    else:
        state_data = {}
        logger.info("Creating new state file")

    # Convert positions to state format
    active_positions = {}
    for pos in positions:
        active_positions[pos.symbol] = pos.model_dump(mode="json")
        logger.info(f"Adding {pos.symbol} to state")

    # Update state
    state_data["active_positions"] = active_positions

    # Write state file
    state_file.write_text(json.dumps(state_data, indent=2))
    logger.info(f"Updated state file: {state_file}")
    logger.info("State updated - restart daemon to load new state")

    await db_engine.close()


if __name__ == "__main__":
    asyncio.run(main())
