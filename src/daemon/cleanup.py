"""Database cleanup tasks for retention policy."""

from datetime import UTC, datetime, timedelta

from loguru import logger

from src.database.repositories.analysis import AnalysisRecordRepository
from src.database.repositories.discovery import DiscoveryHistoryRepository
from src.database.repositories.position_action import PositionManagementActionRepository


async def cleanup_old_analysis_records(
    repository: AnalysisRecordRepository,
    retention_days: int = 90,
) -> int:
    """Delete analysis records older than retention_days.

    Args:
        repository: Analysis record repository
        retention_days: Number of days to retain (default 90)

    Returns:
        Number of records deleted
    """
    cutoff = datetime.now(UTC) - timedelta(days=retention_days)
    deleted = await repository.delete_before(cutoff)
    logger.info(f"Deleted {deleted} analysis records older than {retention_days} days")
    return deleted


async def cleanup_old_position_actions(
    repository: PositionManagementActionRepository,
    retention_days: int = 90,
) -> int:
    """Delete position actions older than retention_days.

    Args:
        repository: Position action repository
        retention_days: Number of days to retain (default 90)

    Returns:
        Number of actions deleted
    """
    cutoff = datetime.now(UTC) - timedelta(days=retention_days)
    deleted = await repository.delete_before(cutoff)
    logger.info(f"Deleted {deleted} position actions older than {retention_days} days")
    return deleted


async def cleanup_old_discovery_history(
    repository: DiscoveryHistoryRepository,
    retention_days: int = 90,
) -> int:
    """Delete discovery history records older than retention_days.

    Args:
        repository: Discovery history repository
        retention_days: Number of days to retain (default 90)

    Returns:
        Number of records deleted
    """
    cutoff = datetime.now(UTC) - timedelta(days=retention_days)
    deleted = await repository.delete_before(cutoff)
    logger.info(f"Deleted {deleted} discovery history records older than {retention_days} days")
    return deleted


async def run_cleanup_all(
    analysis_repository: AnalysisRecordRepository | None = None,
    position_action_repository: PositionManagementActionRepository | None = None,
    discovery_repository: DiscoveryHistoryRepository | None = None,
    retention_days: int = 90,
) -> dict[str, int]:
    """Run all cleanup tasks.

    Args:
        analysis_repository: Analysis record repository
        position_action_repository: Position action repository
        discovery_repository: Discovery history repository
        retention_days: Number of days to retain (default 90)

    Returns:
        Dict mapping cleanup type to deleted count
    """
    results = {}

    if analysis_repository:
        results["analysis_records"] = await cleanup_old_analysis_records(analysis_repository, retention_days)

    if position_action_repository:
        results["position_actions"] = await cleanup_old_position_actions(
            position_action_repository, retention_days
        )

    if discovery_repository:
        results["discovery_history"] = await cleanup_old_discovery_history(
            discovery_repository, retention_days
        )

    total_deleted = sum(results.values())
    logger.info(f"Cleanup complete: {total_deleted} total records deleted - {results}")
    return results
