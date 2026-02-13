"""Backfill historical data from JSON state file to database.

Usage:
    python -m src.database.migrations.backfill_from_json --state-file ~/.ai-casino/daemon-state.json
"""

import argparse
import asyncio
import json
from pathlib import Path

from loguru import logger

from src.daemon.positions import PositionManagementAction
from src.daemon.state.models import AnalysisRecord, DiscoveryHistoryRecord
from src.database.engine import DatabaseEngine
from src.database.repositories.analysis import AnalysisRecordRepository
from src.database.repositories.discovery import DiscoveryHistoryRepository
from src.database.repositories.position_action import PositionManagementActionRepository


async def backfill_analyses(
    analyses_data: list[dict],
    repository: AnalysisRecordRepository,
) -> tuple[int, int]:
    """Backfill analysis records from JSON data.

    Args:
        analyses_data: List of analysis dicts from JSON
        repository: Analysis record repository

    Returns:
        Tuple of (success_count, error_count)
    """
    logger.info(f"Backfilling {len(analyses_data)} analysis records")
    success = 0
    errors = 0

    for analysis_dict in analyses_data:
        try:
            analysis = AnalysisRecord.model_validate(analysis_dict)
            await repository.create(analysis)
            success += 1
        except Exception as e:
            symbol = analysis_dict.get("symbol", "unknown")
            logger.opt(exception=True).error(f"Failed to backfill analysis {symbol}: {e}")
            errors += 1

    logger.info(f"Analysis backfill complete: {success} success, {errors} errors")
    return success, errors


async def backfill_discovery_history(
    discovery_data: list[dict],
    repository: DiscoveryHistoryRepository,
) -> tuple[int, int]:
    """Backfill discovery history records from JSON data.

    Args:
        discovery_data: List of discovery dicts from JSON
        repository: Discovery history repository

    Returns:
        Tuple of (success_count, error_count)
    """
    logger.info(f"Backfilling {len(discovery_data)} discovery history records")
    success = 0
    errors = 0

    for record_dict in discovery_data:
        try:
            record = DiscoveryHistoryRecord.model_validate(record_dict)
            await repository.create(record)
            success += 1
        except Exception as e:
            symbol = record_dict.get("symbol", "unknown")
            logger.opt(exception=True).error(f"Failed to backfill discovery {symbol}: {e}")
            errors += 1

    logger.info(f"Discovery history backfill complete: {success} success, {errors} errors")
    return success, errors


async def backfill_position_actions(
    actions_data: list[dict],
    repository: PositionManagementActionRepository,
) -> tuple[int, int]:
    """Backfill position management actions from JSON data.

    Args:
        actions_data: List of action dicts from JSON
        repository: Position action repository

    Returns:
        Tuple of (success_count, error_count)
    """
    logger.info(f"Backfilling {len(actions_data)} position actions")
    success = 0
    errors = 0

    for action_dict in actions_data:
        try:
            action = PositionManagementAction.model_validate(action_dict)
            await repository.create(action)
            success += 1
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to backfill position action: {e}")
            errors += 1

    logger.info(f"Position actions backfill complete: {success} success, {errors} errors")
    return success, errors


async def run_backfill(state_file: str, database_url: str) -> dict[str, tuple[int, int]]:
    """Run all backfill tasks.

    Args:
        state_file: Path to JSON state file
        database_url: Database connection URL

    Returns:
        Dict mapping backfill type to (success, error) counts
    """
    logger.info(f"Starting backfill from {state_file}")

    # Load JSON directly
    with Path(state_file).open() as f:
        state_data = json.load(f)

    analyses_data = state_data.get("analyses", [])
    discovery_data = state_data.get("discovery_history", [])
    actions_data = state_data.get("position_management_history", [])

    logger.info(
        f"Loaded state: {len(analyses_data)} analyses, "
        f"{len(discovery_data)} discoveries, "
        f"{len(actions_data)} position actions"
    )

    # Initialize database engine
    engine = DatabaseEngine(database_url=database_url)
    await engine.ensure_migrated()

    results = {}

    # Backfill analyses
    if analyses_data:
        session = engine.session()
        analysis_repo = AnalysisRecordRepository(session)
        results["analyses"] = await backfill_analyses(analyses_data, analysis_repo)
        await session.close()

    # Backfill discovery history
    if discovery_data:
        session = engine.session()
        discovery_repo = DiscoveryHistoryRepository(session)
        results["discovery_history"] = await backfill_discovery_history(discovery_data, discovery_repo)
        await session.close()

    # Backfill position actions
    if actions_data:
        session = engine.session()
        position_action_repo = PositionManagementActionRepository(session)
        results["position_actions"] = await backfill_position_actions(actions_data, position_action_repo)
        await session.close()

    await engine.close()

    # Summary
    total_success = sum(r[0] for r in results.values())
    total_errors = sum(r[1] for r in results.values())
    logger.info(f"Backfill complete: {total_success} total success, {total_errors} total errors")
    logger.info(f"Results: {results}")

    return results


def main() -> None:
    """CLI entry point for backfill script."""
    parser = argparse.ArgumentParser(description="Backfill daemon state from JSON to PostgreSQL")
    parser.add_argument(
        "--state-file",
        default="~/.ai-casino/daemon-state.json",
        help="Path to JSON state file (default: ~/.ai-casino/daemon-state.json)",
    )
    parser.add_argument(
        "--database-url",
        required=True,
        help="PostgreSQL connection URL (postgresql+asyncpg://...)",
    )

    args = parser.parse_args()

    # Expand paths
    state_file = str(Path(args.state_file).expanduser())

    # Run backfill
    asyncio.run(run_backfill(state_file, args.database_url))


if __name__ == "__main__":
    main()
