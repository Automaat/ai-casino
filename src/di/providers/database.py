"""Database provider functions for DI container."""

from loguru import logger

from src.daemon.config import DaemonConfig
from src.database.engine import DatabaseEngine, MissingDatabaseURLError


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

    engine = DatabaseEngine(
        database_url=database_url,
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
