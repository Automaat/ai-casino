"""Database provider functions for DI container."""

from typing import TYPE_CHECKING

from loguru import logger

from src.daemon.config import DaemonConfig
from src.database.engine import DatabaseEngine, MissingDatabaseURLError

if TYPE_CHECKING:
    from src.database.repositories.active_discovery import ActiveDiscoveryCandidateRepository
    from src.database.repositories.analysis import AnalysisRecordRepository
    from src.database.repositories.correlation_audit import CorrelationAuditRecordRepository
    from src.database.repositories.degradation import DegradationRecordRepository
    from src.database.repositories.discovery import DiscoveryHistoryRepository
    from src.database.repositories.earnings_calendar import EarningsCalendarRecordRepository
    from src.database.repositories.execution_graph import ExecutionGraphRepository
    from src.database.repositories.execution_metric import ExecutionMetricRepository
    from src.database.repositories.game_plan import GamePlanRecordRepository
    from src.database.repositories.metadata import MetadataRepository
    from src.database.repositories.monte_carlo import MonteCarloRecordRepository
    from src.database.repositories.optimization import OptimizationRecordRepository
    from src.database.repositories.peer_analysis import PeerAnalysisRecordRepository
    from src.database.repositories.position import PositionRecordRepository
    from src.database.repositories.position_action import PositionManagementActionRepository
    from src.database.repositories.prefetch import PrefetchRecordRepository
    from src.database.repositories.profiling import ProfilingRecordRepository
    from src.database.repositories.rebalancing import RebalancingRecordRepository
    from src.database.repositories.risk_report import RiskReportRecordRepository
    from src.database.repositories.screening import ScreeningRecordRepository
    from src.database.repositories.sector_rotation import SectorRotationRecordRepository
    from src.database.repositories.signal_outcome import SignalOutcomeRepository
    from src.database.repositories.snapshot import PortfolioSnapshotRepository
    from src.database.repositories.tearsheet import TearSheetRepository
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


def create_tearsheet_repository(database_engine: DatabaseEngine) -> TearSheetRepository:
    """Create TearSheetRepository with database session.

    Args:
        database_engine: Database engine instance

    Returns:
        TearSheetRepository instance
    """
    from src.database.repositories.tearsheet import TearSheetRepository

    session = database_engine.session()
    return TearSheetRepository(session)


def create_metadata_repository(database_engine: DatabaseEngine) -> MetadataRepository:
    """Create MetadataRepository with database session.

    Args:
        database_engine: Database engine instance

    Returns:
        MetadataRepository instance
    """
    from src.database.repositories.metadata import MetadataRepository

    session = database_engine.session()
    return MetadataRepository(session)


def create_optimization_repository(database_engine: DatabaseEngine) -> OptimizationRecordRepository:
    """Create OptimizationRecordRepository with database session.

    Args:
        database_engine: Database engine instance

    Returns:
        OptimizationRecordRepository instance
    """
    from src.database.repositories.optimization import OptimizationRecordRepository

    session = database_engine.session()
    return OptimizationRecordRepository(session)


def create_rebalancing_repository(database_engine: DatabaseEngine) -> RebalancingRecordRepository:
    """Create RebalancingRecordRepository with database session.

    Args:
        database_engine: Database engine instance

    Returns:
        RebalancingRecordRepository instance
    """
    from src.database.repositories.rebalancing import RebalancingRecordRepository

    session = database_engine.session()
    return RebalancingRecordRepository(session)


def create_sector_rotation_repository(database_engine: DatabaseEngine) -> SectorRotationRecordRepository:
    """Create SectorRotationRecordRepository with database session.

    Args:
        database_engine: Database engine instance

    Returns:
        SectorRotationRecordRepository instance
    """
    from src.database.repositories.sector_rotation import SectorRotationRecordRepository

    session = database_engine.session()
    return SectorRotationRecordRepository(session)


def create_peer_analysis_repository(database_engine: DatabaseEngine) -> PeerAnalysisRecordRepository:
    """Create PeerAnalysisRecordRepository with database session.

    Args:
        database_engine: Database engine instance

    Returns:
        PeerAnalysisRecordRepository instance
    """
    from src.database.repositories.peer_analysis import PeerAnalysisRecordRepository

    session = database_engine.session()
    return PeerAnalysisRecordRepository(session)


def create_correlation_audit_repository(database_engine: DatabaseEngine) -> CorrelationAuditRecordRepository:
    """Create CorrelationAuditRecordRepository with database session.

    Args:
        database_engine: Database engine instance

    Returns:
        CorrelationAuditRecordRepository instance
    """
    from src.database.repositories.correlation_audit import CorrelationAuditRecordRepository

    session = database_engine.session()
    return CorrelationAuditRecordRepository(session)


def create_risk_report_repository(database_engine: DatabaseEngine) -> RiskReportRecordRepository:
    """Create RiskReportRecordRepository with database session.

    Args:
        database_engine: Database engine instance

    Returns:
        RiskReportRecordRepository instance
    """
    from src.database.repositories.risk_report import RiskReportRecordRepository

    session = database_engine.session()
    return RiskReportRecordRepository(session)


def create_monte_carlo_repository(database_engine: DatabaseEngine) -> MonteCarloRecordRepository:
    """Create MonteCarloRecordRepository with database session.

    Args:
        database_engine: Database engine instance

    Returns:
        MonteCarloRecordRepository instance
    """
    from src.database.repositories.monte_carlo import MonteCarloRecordRepository

    session = database_engine.session()
    return MonteCarloRecordRepository(session)


def create_prefetch_repository(database_engine: DatabaseEngine) -> PrefetchRecordRepository:
    """Create PrefetchRecordRepository with database session.

    Args:
        database_engine: Database engine instance

    Returns:
        PrefetchRecordRepository instance
    """
    from src.database.repositories.prefetch import PrefetchRecordRepository

    session = database_engine.session()
    return PrefetchRecordRepository(session)


def create_screening_repository(database_engine: DatabaseEngine) -> ScreeningRecordRepository:
    """Create ScreeningRecordRepository with database session.

    Args:
        database_engine: Database engine instance

    Returns:
        ScreeningRecordRepository instance
    """
    from src.database.repositories.screening import ScreeningRecordRepository

    session = database_engine.session()
    return ScreeningRecordRepository(session)


def create_earnings_calendar_repository(database_engine: DatabaseEngine) -> EarningsCalendarRecordRepository:
    """Create EarningsCalendarRecordRepository with database session.

    Args:
        database_engine: Database engine instance

    Returns:
        EarningsCalendarRecordRepository instance
    """
    from src.database.repositories.earnings_calendar import EarningsCalendarRecordRepository

    session = database_engine.session()
    return EarningsCalendarRecordRepository(session)


def create_profiling_repository(database_engine: DatabaseEngine) -> ProfilingRecordRepository:
    """Create ProfilingRecordRepository with database session.

    Args:
        database_engine: Database engine instance

    Returns:
        ProfilingRecordRepository instance
    """
    from src.database.repositories.profiling import ProfilingRecordRepository

    session = database_engine.session()
    return ProfilingRecordRepository(session)


def create_game_plan_repository(database_engine: DatabaseEngine) -> GamePlanRecordRepository:
    """Create GamePlanRecordRepository with database session.

    Args:
        database_engine: Database engine instance

    Returns:
        GamePlanRecordRepository instance
    """
    from src.database.repositories.game_plan import GamePlanRecordRepository

    session = database_engine.session()
    return GamePlanRecordRepository(session)


def create_degradation_repository(database_engine: DatabaseEngine) -> DegradationRecordRepository:
    """Create DegradationRecordRepository with database session.

    Args:
        database_engine: Database engine instance

    Returns:
        DegradationRecordRepository instance
    """
    from src.database.repositories.degradation import DegradationRecordRepository

    session = database_engine.session()
    return DegradationRecordRepository(session)


def create_active_discovery_repository(database_engine: DatabaseEngine) -> ActiveDiscoveryCandidateRepository:
    """Create ActiveDiscoveryCandidateRepository with database session.

    Args:
        database_engine: Database engine instance

    Returns:
        ActiveDiscoveryCandidateRepository instance
    """
    from src.database.repositories.active_discovery import ActiveDiscoveryCandidateRepository

    session = database_engine.session()
    return ActiveDiscoveryCandidateRepository(session)


def create_execution_metric_repository(database_engine: DatabaseEngine) -> ExecutionMetricRepository:
    """Create ExecutionMetricRepository with database session.

    Args:
        database_engine: Database engine instance

    Returns:
        ExecutionMetricRepository instance
    """
    from src.database.repositories.execution_metric import ExecutionMetricRepository

    session = database_engine.session()
    return ExecutionMetricRepository(session)
