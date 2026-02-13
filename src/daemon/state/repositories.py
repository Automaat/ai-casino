"""Repository bundle for state manager dependency injection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.database.repositories.active_discovery import ActiveDiscoveryCandidateRepository
    from src.database.repositories.analysis import AnalysisRecordRepository
    from src.database.repositories.correlation_audit import CorrelationAuditRecordRepository
    from src.database.repositories.degradation import DegradationRecordRepository
    from src.database.repositories.discovery import DiscoveryHistoryRepository
    from src.database.repositories.earnings_calendar import EarningsCalendarRecordRepository
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
    from src.database.repositories.snapshot import PortfolioSnapshotRepository


@dataclass
class RepositoryBundle:
    """Bundle of all repositories for dependency injection."""

    metadata_repository: MetadataRepository
    analysis_repository: AnalysisRecordRepository
    position_repository: PositionRecordRepository
    action_repository: PositionManagementActionRepository
    optimization_repository: OptimizationRecordRepository
    rebalancing_repository: RebalancingRecordRepository
    sector_rotation_repository: SectorRotationRecordRepository
    peer_analysis_repository: PeerAnalysisRecordRepository
    correlation_audit_repository: CorrelationAuditRecordRepository
    risk_report_repository: RiskReportRecordRepository
    monte_carlo_repository: MonteCarloRecordRepository
    prefetch_repository: PrefetchRecordRepository
    screening_repository: ScreeningRecordRepository
    earnings_repository: EarningsCalendarRecordRepository
    profiling_repository: ProfilingRecordRepository
    discovery_repository: DiscoveryHistoryRepository
    active_discovery_repository: ActiveDiscoveryCandidateRepository
    game_plan_repository: GamePlanRecordRepository
    degradation_repository: DegradationRecordRepository
    snapshot_repository: PortfolioSnapshotRepository
