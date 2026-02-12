"""Backfill historical data from JSON state file to database.

Usage:
    python -m src.database.migrations.backfill_from_json --state-file ~/.ai-casino/daemon-state.json
"""

import argparse
import asyncio
from pathlib import Path

from loguru import logger

from src.daemon.state import DaemonState
from src.database.engine import DatabaseEngine
from src.database.repositories.analysis import AnalysisRecordRepository
from src.database.repositories.discovery import DiscoveryHistoryRepository
from src.database.repositories.position_action import PositionManagementActionRepository


async def backfill_analyses(
    state: DaemonState,
    repository: AnalysisRecordRepository,
) -> tuple[int, int]:
    """Backfill analysis records from daemon state JSON.

    Args:
        state: Loaded daemon state
        repository: Analysis record repository

    Returns:
        Tuple of (success_count, error_count)
    """
    logger.info(f"Backfilling {len(state.analyses)} analysis records")
    success = 0
    errors = 0

    for analysis in state.analyses:
        try:
            await repository.create(analysis)
            success += 1
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to backfill analysis {analysis.symbol}: {e}")
            errors += 1

    logger.info(f"Analysis backfill complete: {success} success, {errors} errors")
    return success, errors


async def backfill_discovery_history(
    state: DaemonState,
    repository: DiscoveryHistoryRepository,
) -> tuple[int, int]:
    """Backfill discovery history records from daemon state JSON.

    Args:
        state: Loaded daemon state
        repository: Discovery history repository

    Returns:
        Tuple of (success_count, error_count)
    """
    logger.info(f"Backfilling {len(state.discovery_history)} discovery history records")
    success = 0
    errors = 0

    for record in state.discovery_history:
        try:
            await repository.create(record)
            success += 1
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to backfill discovery {record.symbol}: {e}")
            errors += 1

    logger.info(f"Discovery history backfill complete: {success} success, {errors} errors")
    return success, errors


async def backfill_position_actions(
    state: DaemonState,
    repository: PositionManagementActionRepository,
) -> tuple[int, int]:
    """Backfill position management actions from daemon state JSON.

    Args:
        state: Loaded daemon state
        repository: Position action repository

    Returns:
        Tuple of (success_count, error_count)
    """
    from src.daemon.positions import PositionManagementAction

    logger.info(f"Backfilling {len(state.position_management_history)} position actions")
    success = 0
    errors = 0

    for action_dict in state.position_management_history:
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

    # Load state from JSON
    state = DaemonState.load(state_file)
    logger.info(
        f"Loaded state: {len(state.analyses)} analyses, "
        f"{len(state.discovery_history)} discoveries, "
        f"{len(state.position_management_history)} position actions"
    )

    # Initialize database engine
    engine = DatabaseEngine(database_url=database_url)
    await engine.ensure_migrated()

    results = {}

    # Backfill analyses
    if state.analyses:
        session = engine.session()
        analysis_repo = AnalysisRecordRepository(session)
        results["analyses"] = await backfill_analyses(state, analysis_repo)
        await session.close()

    # Backfill discovery history
    if state.discovery_history:
        session = engine.session()
        discovery_repo = DiscoveryHistoryRepository(session)
        results["discovery_history"] = await backfill_discovery_history(state, discovery_repo)
        await session.close()

    # Backfill position actions
    if state.position_management_history:
        session = engine.session()
        position_action_repo = PositionManagementActionRepository(session)
        results["position_actions"] = await backfill_position_actions(state, position_action_repo)
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
