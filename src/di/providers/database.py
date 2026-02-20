"""Database provider functions for DI container."""

from typing import TYPE_CHECKING

from loguru import logger

from src.daemon.config import DaemonConfig
from src.database.engine import DatabaseEngine, MissingDatabaseURLError, PoolConfig

if TYPE_CHECKING:
    from src.database.repositories.analysis import AnalysisRecordRepository
    from src.database.repositories.coordinator_metrics import CoordinatorMetricsRepository
    from src.database.repositories.signal_outcome import SignalOutcomeRepository
    from src.database.repositories.trade import TradeRepository
    from src.v1.event_queue.service import MarketEventQueue


def create_database_engine(daemon_config: DaemonConfig) -> DatabaseEngine:
    """Create DatabaseEngine with config.

    Args:
        daemon_config: Daemon configuration

    Returns:
        Configured DatabaseEngine instance

    Raises:
        MissingDatabaseURLError: If enable_persistence=True but no database URL
    """
    from src.di.config import resolve_config_or_env

    database_url = resolve_config_or_env(daemon_config.database.database_url, "DATABASE_URL")

    if daemon_config.database.enable_persistence and not database_url:
        logger.error(
            "Database persistence enabled but DATABASE_URL not configured. "
            "Set database.database_url in daemon.yaml or DATABASE_URL env var."
        )
        raise MissingDatabaseURLError

    if not database_url:
        logger.warning("Database URL not configured - database features disabled")
        raise MissingDatabaseURLError

    pool_config = PoolConfig(
        pool_size=daemon_config.database.pool_size,
        max_overflow=daemon_config.database.max_overflow,
        pool_pre_ping=daemon_config.database.pool_pre_ping,
        pool_timeout=daemon_config.database.pool_timeout,
        pool_recycle=daemon_config.database.pool_recycle,
    )

    engine = DatabaseEngine(database_url=database_url, pool_config=pool_config)

    logger.info(
        f"Database pool: size={pool_config.pool_size}, "
        f"overflow={pool_config.max_overflow}, "
        f"timeout={pool_config.pool_timeout}s, "
        f"recycle={pool_config.pool_recycle}s"
    )

    # Migrations are deferred to lifecycle.startup() via ensure_migrated()
    # to avoid running asyncio.run() here which would bind asyncpg pool
    # connections to a temporary event loop, causing "Future attached to
    # a different loop" errors in the main daemon event loop.

    return engine


def create_market_event_queue(database_engine: DatabaseEngine) -> MarketEventQueue:
    """Create MarketEventQueue singleton.

    Args:
        database_engine: Database engine singleton

    Returns:
        MarketEventQueue instance
    """
    from src.v1.event_queue.service import MarketEventQueue

    return MarketEventQueue(database_engine)


def create_analysis_repository(database_engine: DatabaseEngine) -> AnalysisRecordRepository:
    """Create AnalysisRecordRepository with fresh session.

    Args:
        database_engine: Database engine singleton

    Returns:
        Repository with new session (caller must close)

    Note:
        Session must be closed by caller via await repo.close() or use as context manager
    """
    from src.database.repositories.analysis import AnalysisRecordRepository

    repo = AnalysisRecordRepository(database_engine.session())
    repo.owns_session = True
    return repo


def create_trade_repository(database_engine: DatabaseEngine) -> TradeRepository:
    """Create TradeRepository with fresh session.

    Args:
        database_engine: Database engine singleton

    Returns:
        Repository with new session (caller must close)

    Note:
        Session must be closed by caller via await repo.close() or use as context manager
    """
    from src.database.repositories.trade import TradeRepository

    repo = TradeRepository(database_engine.session())
    repo.owns_session = True
    return repo


def create_signal_outcome_repository(database_engine: DatabaseEngine) -> SignalOutcomeRepository:
    """Create SignalOutcomeRepository with fresh session.

    Args:
        database_engine: Database engine singleton

    Returns:
        Repository with new session (caller must close)

    Note:
        Session must be closed by caller via await repo.close() or use as context manager
    """
    from src.database.repositories.signal_outcome import SignalOutcomeRepository

    repo = SignalOutcomeRepository(database_engine.session())
    repo.owns_session = True
    return repo


def create_coordinator_metrics_repository(database_engine: DatabaseEngine) -> CoordinatorMetricsRepository:
    """Create CoordinatorMetricsRepository with fresh session.

    Args:
        database_engine: Database engine singleton

    Returns:
        Repository with new session (caller must close)

    Note:
        Session must be closed by caller via await repo.close() or use as context manager
    """
    from src.database.repositories.coordinator_metrics import CoordinatorMetricsRepository

    repo = CoordinatorMetricsRepository(database_engine.session())
    repo.owns_session = True
    return repo
