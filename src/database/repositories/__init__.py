"""Repository layer for database access."""

from src.database.repositories.active_discovery import ActiveDiscoveryCandidateRepository
from src.database.repositories.analysis import AnalysisRecordRepository
from src.database.repositories.base import BaseRepository
from src.database.repositories.correlation_audit import CorrelationAuditRecordRepository
from src.database.repositories.degradation import DegradationRecordRepository
from src.database.repositories.discovery import DiscoveryHistoryRecordRepository
from src.database.repositories.earnings_calendar import EarningsCalendarRecordRepository
from src.database.repositories.execution_graph import ExecutionGraphRepository
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

__all__ = [
    "ActiveDiscoveryCandidateRepository",
    "AnalysisRecordRepository",
    "BaseRepository",
    "CorrelationAuditRecordRepository",
    "DegradationRecordRepository",
    "DiscoveryHistoryRecordRepository",
    "EarningsCalendarRecordRepository",
    "ExecutionGraphRepository",
    "GamePlanRecordRepository",
    "MetadataRepository",
    "MonteCarloRecordRepository",
    "OptimizationRecordRepository",
    "PeerAnalysisRecordRepository",
    "PortfolioSnapshotRepository",
    "PositionManagementActionRepository",
    "PositionRecordRepository",
    "PrefetchRecordRepository",
    "ProfilingRecordRepository",
    "RebalancingRecordRepository",
    "RiskReportRecordRepository",
    "ScreeningRecordRepository",
    "SectorRotationRecordRepository",
    "SignalOutcomeRepository",
    "TearSheetRepository",
    "TradeRepository",
]
