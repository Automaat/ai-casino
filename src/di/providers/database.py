"""Database provider functions for DI container."""

from typing import TYPE_CHECKING

from loguru import logger

from src.daemon.config import DaemonConfig
from src.database.engine import DatabaseEngine, MissingDatabaseURLError

if TYPE_CHECKING:
    from src.database.repositories.analysis import AnalysisRecordRepository
    from src.database.repositories.discovery import DiscoveryHistoryRepository
    from src.database.repositories.execution_graph import ExecutionGraphRepository
    from src.database.repositories.position import PositionRecordRepository
    from src.database.repositories.position_action import PositionManagementActionRepository
    from src.database.repositories.signal_outcome import SignalOutcomeRepository
    from src.database.repositories.snapshot import PortfolioSnapshotRepository
    from src.database.repositories.trade import TradeRepository


def create_database_engine(daemon_config: DaemonConfig) -> DatabaseEngine:
    """Create DatabaseEngine with config.

    Args:
        daemon_config: Daemon configuration

    Returns:
        Configured DatabaseEngine instance

    Raises:
        MissingDatabaseURLError: If enable_persistence=True but no database URL
    """
    database_url = daemon_config.database.database_url

    if daemon_config.database.enable_persistence and not database_url:
        logger.error(
            "Database persistence enabled but DATABASE_URL not configured. "
            "Set database.database_url in daemon.yaml."
        )
        raise MissingDatabaseURLError

    if not database_url:
        logger.warning("Database URL not configured - database features disabled")
        raise MissingDatabaseURLError

    engine = DatabaseEngine(
        database_url=database_url,
        pool_size=daemon_config.database.pool_size,
        max_overflow=daemon_config.database.max_overflow,
        pool_pre_ping=daemon_config.database.pool_pre_ping,
    )

    # Run migrations on startup
    import asyncio

    try:
        asyncio.get_running_loop()
        # If already in event loop, schedule migration

        def _log_migration_result(t: asyncio.Task) -> None:
            # Avoid calling t.exception() on cancelled tasks (raises CancelledError)
            if t.cancelled():
                logger.warning("DB migration task was cancelled before completion")
                return
            exc = t.exception()
            if exc:
                # Log full traceback for easier debugging
                logger.opt(exception=exc).error("DB migration failed")

        task = asyncio.create_task(engine.ensure_migrated())
        task.add_done_callback(_log_migration_result)
    except RuntimeError:
        # No event loop yet, run in new loop
        asyncio.run(engine.ensure_migrated())

    logger.info("DatabaseEngine initialized and migrations applied")
    return engine


def create_analysis_repository(database_engine: DatabaseEngine) -> AnalysisRecordRepository:
    """Create AnalysisRecordRepository with database session.

    Args:
        database_engine: Database engine instance

    Returns:
        AnalysisRecordRepository instance
    """
    from src.database.repositories.analysis import AnalysisRecordRepository

    session = database_engine.session()
    return AnalysisRecordRepository(session)


def create_position_repository(database_engine: DatabaseEngine) -> PositionRecordRepository:
    """Create PositionRecordRepository with database session.

    Args:
        database_engine: Database engine instance

    Returns:
        PositionRecordRepository instance
    """
    from src.database.repositories.position import PositionRecordRepository

    session = database_engine.session()
    return PositionRecordRepository(session)


def create_position_action_repository(
    database_engine: DatabaseEngine,
) -> PositionManagementActionRepository:
    """Create PositionManagementActionRepository with database session.

    Args:
        database_engine: Database engine instance

    Returns:
        PositionManagementActionRepository instance
    """
    from src.database.repositories.position_action import PositionManagementActionRepository

    session = database_engine.session()
    return PositionManagementActionRepository(session)


def create_discovery_repository(database_engine: DatabaseEngine) -> DiscoveryHistoryRepository:
    """Create DiscoveryHistoryRepository with database session.

    Args:
        database_engine: Database engine instance

    Returns:
        DiscoveryHistoryRepository instance
    """
    from src.database.repositories.discovery import DiscoveryHistoryRepository

    session = database_engine.session()
    return DiscoveryHistoryRepository(session)


def create_snapshot_repository(database_engine: DatabaseEngine) -> PortfolioSnapshotRepository:
    """Create PortfolioSnapshotRepository with database session.

    Args:
        database_engine: Database engine instance

    Returns:
        PortfolioSnapshotRepository instance
    """
    from src.database.repositories.snapshot import PortfolioSnapshotRepository

    session = database_engine.session()
    return PortfolioSnapshotRepository(session)


def create_trade_repository(database_engine: DatabaseEngine) -> TradeRepository:
    """Create TradeRepository with database session.

    Args:
        database_engine: Database engine instance

    Returns:
        TradeRepository instance
    """
    from src.database.repositories.trade import TradeRepository

    session = database_engine.session()
    return TradeRepository(session)


def create_signal_outcome_repository(database_engine: DatabaseEngine) -> SignalOutcomeRepository:
    """Create SignalOutcomeRepository with database session.

    Args:
        database_engine: Database engine instance

    Returns:
        SignalOutcomeRepository instance
    """
    from src.database.repositories.signal_outcome import SignalOutcomeRepository

    session = database_engine.session()
    return SignalOutcomeRepository(session)


def create_execution_graph_repository(database_engine: DatabaseEngine) -> ExecutionGraphRepository:
    """Create ExecutionGraphRepository with database session.

    Args:
        database_engine: Database engine instance

    Returns:
        ExecutionGraphRepository instance
    """
    from src.database.repositories.execution_graph import ExecutionGraphRepository

    session = database_engine.session()
    return ExecutionGraphRepository(session)
