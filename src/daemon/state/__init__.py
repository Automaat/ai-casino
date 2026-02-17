"""Daemon state persistence."""

from src.daemon.state.facade import DaemonState
from src.daemon.state.models import (
    AnalysisRecord,
    CorrelationAuditRecord,
    DegradationRecord,
    DiscoveryHistoryRecord,
    EarningsCalendarRecord,
    EarningsEventRecord,
    GamePlanRecord,
    MonteCarloRecord,
    OptimizationRecord,
    PeerAnalysisRecord,
    PortfolioAllocationRecord,
    PortfolioRebalancingRecord,
    PortfolioSnapshot,
    PrefetchRecord,
    ProfilingRecord,
    RiskReportRecord,
    SectorRotationRecord,
)


def _rebuild_daemon_state_model() -> None:
    """Rebuild DaemonState model to resolve forward references.

    Must happen after ExecutionGraphTracker is defined (in src.execution_tracking.tracker).
    """
    try:
        from src.execution_tracking.models import ExecutionGraph
        from src.execution_tracking.tracker import ExecutionGraphTracker

        # Import triggers rebuild - ExecutionGraphTracker and ExecutionGraph are now available
        if ExecutionGraphTracker and ExecutionGraph:
            DaemonState.model_rebuild()
    except ImportError:
        # ExecutionGraphTracker/ExecutionGraph not yet available during import ordering
        pass


_rebuild_daemon_state_model()

__all__ = [
    "AnalysisRecord",
    "CorrelationAuditRecord",
    "DaemonState",
    "DegradationRecord",
    "DiscoveryHistoryRecord",
    "EarningsCalendarRecord",
    "EarningsEventRecord",
    "GamePlanRecord",
    "MonteCarloRecord",
    "OptimizationRecord",
    "PeerAnalysisRecord",
    "PortfolioAllocationRecord",
    "PortfolioRebalancingRecord",
    "PortfolioSnapshot",
    "PrefetchRecord",
    "ProfilingRecord",
    "RiskReportRecord",
    "SectorRotationRecord",
]
