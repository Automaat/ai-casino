"""DaemonState facade maintaining backward compatibility."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from src.execution_tracking.models import ExecutionGraph
    from src.execution_tracking.tracker import ExecutionGraphTracker

from src.daemon.state.managers.data_pipeline import DataPipelineStateManager
from src.daemon.state.managers.discovery import DiscoveryStateManager
from src.daemon.state.managers.portfolio import PortfolioStateManager
from src.daemon.state.managers.positions import PositionStateManager
from src.daemon.state.managers.snapshots import SnapshotStateManager
from src.daemon.state.managers.strategy import StrategyStateManager
from src.daemon.state.managers.trading import TradingStateManager
from src.daemon.state.repositories import RepositoryBundle
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
    ScreeningRecord,
    SectorRotationRecord,
)
from src.discovery.models import DiscoveryCandidate
from src.execution_tracking.models import ExecutionGraph
from src.execution_tracking.tracker import ExecutionGraphTracker
from src.screening.screener import ScreeningResult
from src.strategies.session import TradingSession

if TYPE_CHECKING:
    from src.daemon.degradation import DegradationContext
    from src.daemon.positions import PositionManagementAction, PositionRecord
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


class DaemonState(BaseModel):
    """Persistent state for the trading daemon with domain managers."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Domain managers (internal composition)
    trading: TradingStateManager = Field(default_factory=TradingStateManager)
    positions: PositionStateManager = Field(default_factory=PositionStateManager)
    portfolio: PortfolioStateManager = Field(default_factory=PortfolioStateManager)
    data_pipeline: DataPipelineStateManager = Field(default_factory=DataPipelineStateManager)
    discovery: DiscoveryStateManager = Field(default_factory=DiscoveryStateManager)
    strategy: StrategyStateManager = Field(default_factory=StrategyStateManager)
    snapshots: SnapshotStateManager = Field(default_factory=SnapshotStateManager)

    # Execution tracking (in-memory only, not persisted)
    active_execution_trackers: dict[str, ExecutionGraphTracker] = Field(
        default_factory=dict,
        exclude=True,
        description="Active workflow execution trackers by workflow_id",
    )
    execution_graph_history: deque[ExecutionGraph] = Field(
        default_factory=lambda: deque(maxlen=50),
        exclude=True,
        description="Recent completed execution graphs (last 50)",
    )

    def set_repositories(self, repos: RepositoryBundle) -> None:
        """Inject all database repositories into managers.

        Args:
            repos: Bundle containing all 20 repositories
        """
        # TradingStateManager
        self.trading.set_repositories(
            metadata_repository=repos.metadata_repository,
            analysis_repository=repos.analysis_repository,
        )

        # PositionStateManager
        self.positions.set_repositories(
            position_repository=repos.position_repository,
            position_action_repository=repos.action_repository,
        )

        # PortfolioStateManager
        self.portfolio.set_repositories(repos)

        # DataPipelineStateManager
        self.data_pipeline.set_repositories(
            metadata_repository=repos.metadata_repository,
            prefetch_repository=repos.prefetch_repository,
            screening_repository=repos.screening_repository,
            earnings_repository=repos.earnings_repository,
            profiling_repository=repos.profiling_repository,
        )

        # DiscoveryStateManager
        self.discovery.set_repositories(
            metadata_repository=repos.metadata_repository,
            discovery_repository=repos.discovery_repository,
            active_discovery_repository=repos.active_discovery_repository,
        )

        # StrategyStateManager
        self.strategy.set_repositories(
            metadata_repository=repos.metadata_repository,
            game_plan_repository=repos.game_plan_repository,
            degradation_repository=repos.degradation_repository,
        )

        # SnapshotStateManager
        self.snapshots.set_repository(repos.snapshot_repository)

        logger.debug("All repositories injected into DaemonState managers")

    # ===================
    # Trading Manager API
    # ===================

    async def record_analysis(
        self,
        symbol: str,
        signal: str,
        confidence: float,
        executed: bool = False,
        trading_session: TradingSession = TradingSession.REGULAR,
        is_paper_trade: bool = True,
        rsi: float | None = None,
        macd_hist: float | None = None,
        reasoning: list[str] | None = None,
        technical_analysis_reasoning: str | None = None,
        sentiment_analysis_reasoning: str | None = None,
        news_analysis_reasoning: str | None = None,
    ) -> None:
        """Delegate to trading manager."""
        from src.daemon.state.managers.trading import AnalysisRecordInput

        input_data = AnalysisRecordInput(
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            executed=executed,
            trading_session=trading_session,
            is_paper_trade=is_paper_trade,
            rsi=rsi,
            macd_hist=macd_hist,
            reasoning=reasoning,
            technical_analysis_reasoning=technical_analysis_reasoning,
            sentiment_analysis_reasoning=sentiment_analysis_reasoning,
            news_analysis_reasoning=news_analysis_reasoning,
        )
        await self.trading.record_analysis(input_data)

    async def get_last_run(self) -> datetime | None:
        """Get last run timestamp."""
        return await self.trading.get_last_run()

    async def set_last_run(self, value: datetime | None) -> None:
        """Set last run timestamp."""
        await self.trading.set_last_run(value)

    async def get_analyses(self, limit: int = 1000) -> list[AnalysisRecord]:
        """Get recent analyses."""
        return await self.trading.get_analyses(limit)

    async def get_total_analyses(self) -> int:
        """Get total analyses count."""
        return await self.trading.get_total_analyses()

    async def get_total_trades(self) -> int:
        """Get total trades count."""
        return await self.trading.get_total_trades()

    async def get_paper_trading_start_date(self) -> datetime | None:
        """Get paper trading start date."""
        return await self.trading.get_paper_trading_start_date()

    async def set_paper_trading_start_date(self, value: datetime | None) -> None:
        """Set paper trading start date."""
        await self.trading.set_paper_trading_start_date(value)

    async def get_current_trading_mode(self) -> str:
        """Get current trading mode."""
        return await self.trading.get_current_trading_mode()

    async def set_current_trading_mode(self, value: str) -> None:
        """Set current trading mode."""
        await self.trading.set_current_trading_mode(value)

    async def get_last_journal_date(self) -> str | None:
        """Get last journal date."""
        return await self.trading.get_last_journal_date()

    async def set_last_journal_date(self, value: str | None) -> None:
        """Set last journal date."""
        await self.trading.set_last_journal_date(value)

    async def get_last_signal_tracking(self) -> datetime | None:
        """Get last signal tracking timestamp."""
        return await self.trading.get_last_signal_tracking()

    async def set_last_signal_tracking(self, value: datetime | None) -> None:
        """Set last signal tracking timestamp."""
        await self.trading.set_last_signal_tracking(value)

    # ===================
    # Position Manager API
    # ===================

    async def add_position(self, position: PositionRecord) -> None:
        """Delegate to position manager."""
        await self.positions.add_position(position)

    async def remove_position(self, symbol: str) -> None:
        """Delegate to position manager."""
        await self.positions.remove_position(symbol)

    async def update_position(self, position: PositionRecord) -> None:
        """Delegate to position manager."""
        await self.positions.update_position(position)

    async def record_position_action(self, action: PositionManagementAction) -> None:
        """Delegate to position manager."""
        await self.positions.record_position_action(action)

    async def get_position(self, symbol: str) -> PositionRecord | None:
        """Delegate to position manager."""
        return await self.positions.get_position(symbol)

    async def get_active_positions(self) -> dict[str, dict]:
        """Get active positions."""
        return await self.positions.get_active_positions()

    async def get_position_management_history(self) -> list[dict]:
        """Get position management history."""
        return await self.positions.get_position_management_history()

    # ===================
    # Portfolio Manager API
    # ===================

    async def record_optimization(
        self,
        symbols_optimized: list[str],
        symbols_skipped: list[str],
        total_time_seconds: float,
    ) -> None:
        """Delegate to portfolio manager."""
        await self.portfolio.record_optimization(symbols_optimized, symbols_skipped, total_time_seconds)

    async def record_portfolio_rebalancing(
        self,
        method: str,
        allocations: list[PortfolioAllocationRecord],
        expected_return: float,
        expected_volatility: float,
        sharpe_ratio: float,
        rebalances_executed: int,
        rebalances_pending: int,
    ) -> None:
        """Delegate to portfolio manager."""
        from src.daemon.state.managers.portfolio import PortfolioRebalancingInput

        input_data = PortfolioRebalancingInput(
            method=method,
            allocations=allocations,
            expected_return=expected_return,
            expected_volatility=expected_volatility,
            sharpe_ratio=sharpe_ratio,
            rebalances_executed=rebalances_executed,
            rebalances_pending=rebalances_pending,
        )
        await self.portfolio.record_portfolio_rebalancing(input_data)

    async def record_sector_rotation(
        self,
        leading_sectors: list[str],
        lagging_sectors: list[str],
        sector_strengths: dict[str, float],
        sector_momenta: dict[str, str],
        flagged_positions: list[str] | None = None,
    ) -> None:
        """Delegate to portfolio manager."""
        await self.portfolio.record_sector_rotation(
            leading_sectors, lagging_sectors, sector_strengths, sector_momenta, flagged_positions
        )

    async def record_peer_analysis(
        self,
        symbols_analyzed: list[str],
        rankings: dict[str, int],
        swap_recommendations: list[str],
        total_peers: int,
        total_duration_seconds: float,
    ) -> None:
        """Delegate to portfolio manager."""
        await self.portfolio.record_peer_analysis(
            symbols_analyzed, rankings, swap_recommendations, total_peers, total_duration_seconds
        )

    async def record_correlation_audit(
        self,
        num_positions: int,
        num_correlated_pairs: int,
        max_correlation: float,
        avg_correlation: float,
        diversification_ratio: float,
        num_substitutions: int,
        total_duration_seconds: float,
    ) -> None:
        """Delegate to portfolio manager."""
        from src.daemon.state.managers.portfolio import CorrelationAuditInput

        input_data = CorrelationAuditInput(
            num_positions=num_positions,
            num_correlated_pairs=num_correlated_pairs,
            max_correlation=max_correlation,
            avg_correlation=avg_correlation,
            diversification_ratio=diversification_ratio,
            num_substitutions=num_substitutions,
            total_duration_seconds=total_duration_seconds,
        )
        await self.portfolio.record_correlation_audit(input_data)

    async def record_risk_report(self, report: RiskReportRecord) -> None:
        """Delegate to portfolio manager."""
        await self.portfolio.record_risk_report(report)

    async def record_monte_carlo_test(self, record: MonteCarloRecord) -> None:
        """Delegate to portfolio manager."""
        await self.portfolio.record_monte_carlo_test(record)

    async def record_tearsheet(self, symbol: str, html_path: str) -> None:
        """Delegate to portfolio manager."""
        await self.portfolio.record_tearsheet(symbol, html_path)

    async def get_last_optimization(self) -> datetime | None:
        """Get last optimization timestamp."""
        return await self.portfolio.get_last_optimization()

    async def get_optimization_history(self, limit: int = 10) -> list[OptimizationRecord]:
        """Get optimization history."""
        return await self.portfolio.get_optimization_history(limit)

    async def get_last_portfolio_rebalancing(self) -> datetime | None:
        """Get last rebalancing timestamp."""
        return await self.portfolio.get_last_portfolio_rebalancing()

    async def get_rebalancing_history(self, limit: int = 30) -> list[PortfolioRebalancingRecord]:
        """Get rebalancing history."""
        return await self.portfolio.get_rebalancing_history(limit)

    async def get_active_target_allocations(self) -> dict[str, float] | None:
        """Get active target allocations."""
        return await self.portfolio.get_active_target_allocations()

    async def set_active_target_allocations(self, value: dict[str, float] | None) -> None:
        """Set active target allocations."""
        await self.portfolio.set_active_target_allocations(value)

    async def get_last_sector_rotation(self) -> datetime | None:
        """Get last sector rotation timestamp."""
        return await self.portfolio.get_last_sector_rotation()

    async def set_last_sector_rotation(self, value: datetime | None) -> None:
        """Set last sector rotation timestamp."""
        await self.portfolio.set_last_sector_rotation(value)

    async def get_sector_rotation_history(self, limit: int = 30) -> list[SectorRotationRecord]:
        """Get sector rotation history."""
        return await self.portfolio.get_sector_rotation_history(limit)

    async def get_last_peer_analysis(self) -> datetime | None:
        """Get last peer analysis timestamp."""
        return await self.portfolio.get_last_peer_analysis()

    async def get_peer_analysis_history(self, limit: int = 10) -> list[PeerAnalysisRecord]:
        """Get peer analysis history."""
        return await self.portfolio.get_peer_analysis_history(limit)

    async def get_last_correlation_audit(self) -> datetime | None:
        """Get last correlation audit timestamp."""
        return await self.portfolio.get_last_correlation_audit()

    async def set_last_correlation_audit(self, value: datetime | None) -> None:
        """Set last correlation audit timestamp."""
        await self.portfolio.set_last_correlation_audit(value)

    async def get_correlation_audit_history(self, limit: int = 10) -> list[CorrelationAuditRecord]:
        """Get correlation audit history."""
        return await self.portfolio.get_correlation_audit_history(limit)

    async def get_last_risk_report(self) -> datetime | None:
        """Get last risk report timestamp."""
        return await self.portfolio.get_last_risk_report()

    async def get_risk_report_history(self, limit: int = 30) -> list[RiskReportRecord]:
        """Get risk report history."""
        return await self.portfolio.get_risk_report_history(limit)

    async def get_monte_carlo_tests(self, limit: int = 52) -> list[MonteCarloRecord]:
        """Get Monte Carlo tests."""
        return await self.portfolio.get_monte_carlo_tests(limit)

    async def get_last_tearsheet(self) -> datetime | None:
        """Get last tearsheet timestamp."""
        return await self.portfolio.get_last_tearsheet()

    # ===================
    # Data Pipeline Manager API
    # ===================

    async def record_prefetch(
        self,
        symbols_prefetched: int,
        symbols_failed: int,
        finbert_ready: bool,
        total_duration_seconds: float,
    ) -> None:
        """Delegate to data pipeline manager."""
        await self.data_pipeline.record_prefetch(
            symbols_prefetched, symbols_failed, finbert_ready, total_duration_seconds
        )

    async def record_after_hours_screening(
        self,
        criteria: str,
        universe: str,
        candidates: list[ScreeningResult],
        top_n: int = 10,
        screened_at: datetime | None = None,
    ) -> None:
        """Delegate to data pipeline manager."""
        await self.data_pipeline.record_after_hours_screening(criteria, universe, candidates, top_n, screened_at)

    async def record_earnings_fetch(
        self,
        events: list[EarningsEventRecord],
        symbols_fetched: int,
        symbols_failed: int,
    ) -> None:
        """Delegate to data pipeline manager."""
        await self.data_pipeline.record_earnings_fetch(events, symbols_fetched, symbols_failed)

    async def record_profiling(self, metrics: object) -> None:
        """Delegate to data pipeline manager."""
        await self.data_pipeline.record_profiling(metrics)

    async def get_last_prefetch(self) -> datetime | None:
        """Get last prefetch timestamp."""
        return await self.data_pipeline.get_last_prefetch()

    async def set_last_prefetch(self, value: datetime | None) -> None:
        """Set last prefetch timestamp."""
        await self.data_pipeline.set_last_prefetch(value)

    async def get_last_pre_market_refresh(self) -> datetime | None:
        """Get last pre-market refresh timestamp."""
        return await self.data_pipeline.get_last_pre_market_refresh()

    async def set_last_pre_market_refresh(self, value: datetime | None) -> None:
        """Set last pre-market refresh timestamp."""
        await self.data_pipeline.set_last_pre_market_refresh(value)

    async def get_prefetch_history(self, limit: int = 10) -> list[PrefetchRecord]:
        """Get prefetch history."""
        return await self.data_pipeline.get_prefetch_history(limit)

    async def get_last_after_hours_screening(self) -> datetime | None:
        """Get last after-hours screening timestamp."""
        return await self.data_pipeline.get_last_after_hours_screening()

    async def set_last_after_hours_screening(self, value: datetime | None) -> None:
        """Set last after-hours screening timestamp."""
        await self.data_pipeline.set_last_after_hours_screening(value)

    async def get_screening_history(self, limit: int = 10) -> list[ScreeningRecord]:
        """Get screening history."""
        return await self.data_pipeline.get_screening_history(limit)

    async def get_last_earnings_fetch(self) -> datetime | None:
        """Get last earnings fetch timestamp."""
        return await self.data_pipeline.get_last_earnings_fetch()

    async def get_earnings_calendar_history(self, limit: int = 10) -> list[EarningsCalendarRecord]:
        """Get earnings calendar history."""
        return await self.data_pipeline.get_earnings_calendar_history(limit)

    async def get_profiling_history(self, limit: int = 10) -> list[ProfilingRecord]:
        """Get profiling history."""
        return await self.data_pipeline.get_profiling_history(limit)

    # ===================
    # Discovery Manager API
    # ===================

    async def record_discovery(self, candidates: list[DiscoveryCandidate], added_symbols: list[str]) -> None:
        """Delegate to discovery manager."""
        await self.discovery.record_discovery(candidates, added_symbols)

    async def expire_stale_candidates(self) -> list[str]:
        """Delegate to discovery manager."""
        return await self.discovery.expire_stale_candidates()

    async def get_active_discovery_symbols(self) -> list[str]:
        """Delegate to discovery manager."""
        return await self.discovery.get_active_discovery_symbols()

    async def get_last_discovery(self) -> datetime | None:
        """Get last discovery timestamp."""
        return await self.discovery.get_last_discovery()

    async def set_last_discovery(self, value: datetime | None) -> None:
        """Set last discovery timestamp."""
        await self.discovery.set_last_discovery(value)

    async def get_discovery_history(self, limit: int = 10) -> list[DiscoveryHistoryRecord]:
        """Get discovery history."""
        return await self.discovery.get_discovery_history(limit)

    async def get_active_discovery_candidates(self) -> list[DiscoveryCandidate]:
        """Get active discovery candidates."""
        return await self.discovery.get_active_discovery_candidates()

    async def set_active_discovery_candidates(self, value: list[DiscoveryCandidate]) -> None:
        """Set active discovery candidates."""
        await self.discovery.set_active_discovery_candidates(value)

    # ===================
    # Strategy Manager API
    # ===================

    async def record_game_plan(
        self,
        priority_symbols: list[str],
        risk_stance: str,
        sector_focus: list[str],
    ) -> None:
        """Delegate to strategy manager."""
        await self.strategy.record_game_plan(priority_symbols, risk_stance, sector_focus)

    async def record_degradation(self, context: DegradationContext) -> None:
        """Delegate to strategy manager."""
        await self.strategy.record_degradation(context)

    async def record_error(self, error: str) -> None:
        """Delegate to strategy manager."""
        await self.strategy.record_error(error)

    async def get_last_game_plan(self) -> datetime | None:
        """Get last game plan timestamp."""
        return await self.strategy.get_last_game_plan()

    async def get_game_plan_history(self, limit: int = 10) -> list[GamePlanRecord]:
        """Get game plan history."""
        return await self.strategy.get_game_plan_history(limit)

    async def get_last_degradation(self) -> datetime | None:
        """Get last degradation timestamp."""
        return await self.strategy.get_last_degradation()

    async def get_degradation_history(self, limit: int = 30) -> list[DegradationRecord]:
        """Get degradation history."""
        return await self.strategy.get_degradation_history(limit)

    async def get_market_events(self, limit: int | None = None) -> list[dict]:
        """Get market events.

        Args:
            limit: Max number of events to return (optional)

        Returns:
            List of market events
        """
        return await self.strategy.get_market_events(limit=limit)

    async def get_last_health_check(self) -> datetime | None:
        """Get last health check timestamp."""
        return await self.strategy.get_last_health_check()

    async def set_last_health_check(self, value: datetime | None) -> None:
        """Set last health check timestamp."""
        await self.strategy.set_last_health_check(value)

    async def get_errors(self) -> list[str]:
        """Get errors."""
        return await self.strategy.get_errors()

    # ===================
    # Snapshot Manager API
    # ===================

    def snapshot_portfolio(self, snapshot: PortfolioSnapshot) -> None:
        """Delegate to snapshot manager (fires async task internally)."""
        self.snapshots.snapshot_portfolio(snapshot)

    # ===========================
    # Execution Tracking API
    # ===========================

    def add_execution_tracker(self, workflow_id: str, tracker: ExecutionGraphTracker) -> None:
        """Add active execution tracker.

        Args:
            workflow_id: Workflow ID
            tracker: ExecutionGraphTracker instance
        """
        self.active_execution_trackers[workflow_id] = tracker
        logger.debug(f"Added execution tracker for workflow {workflow_id}")

    def get_execution_tracker(self, workflow_id: str) -> ExecutionGraphTracker | None:
        """Get active execution tracker.

        Args:
            workflow_id: Workflow ID

        Returns:
            ExecutionGraphTracker if active, None otherwise
        """
        return self.active_execution_trackers.get(workflow_id)

    def remove_execution_tracker(self, workflow_id: str) -> None:
        """Remove active execution tracker and archive graph.

        Args:
            workflow_id: Workflow ID
        """
        if tracker := self.active_execution_trackers.pop(workflow_id, None):
            self.execution_graph_history.append(tracker.graph)
            logger.debug(
                f"Archived execution graph for workflow {workflow_id} "
                f"({len(tracker.graph.nodes)} nodes, history size: {len(self.execution_graph_history)})"
            )

    def get_active_execution_graphs(self) -> list[ExecutionGraph]:
        """Get all active execution graphs.

        Returns:
            List of active execution graphs
        """
        return [tracker.graph for tracker in self.active_execution_trackers.values()]

    async def get_active_execution_trackers(self) -> dict[str, ExecutionGraphTracker]:
        """Get active execution trackers.

        Returns:
            Dict of active execution trackers by workflow_id
        """
        return self.active_execution_trackers.copy()

    async def get_execution_graph_history(self, limit: int = 1000) -> list[ExecutionGraph]:
        """Get execution graph history.

        Args:
            limit: Max number of graphs to return

        Returns:
            List of recent execution graphs
        """
        history_list = list(self.execution_graph_history)
        return history_list[-limit:] if limit > 0 else history_list

    def get_execution_graph(self, workflow_id: str) -> ExecutionGraph | None:
        """Get execution graph (active or recent).

        Args:
            workflow_id: Workflow ID

        Returns:
            ExecutionGraph if found, None otherwise
        """
        # Check active first
        if tracker := self.active_execution_trackers.get(workflow_id):
            return tracker.graph

        # Check history
        for graph in self.execution_graph_history:
            if str(graph.workflow_id) == workflow_id:
                return graph

        return None

    async def cleanup_completed_trackers(self) -> None:
        """Persist completed execution graphs and move to history.

        Called periodically by daemon to persist graphs and free memory.
        Non-blocking - database errors logged but don't crash daemon.
        """
        from src.database.connection import get_session
        from src.database.repositories.execution_graph import ExecutionGraphRepository

        completed_ids = []

        # Collect completed graphs first
        completed_graphs = []
        for workflow_id, tracker in self.active_execution_trackers.items():
            if tracker.graph.is_completed():
                completed_graphs.append((workflow_id, tracker.graph))

        # Persist all in single session (reduces connection overhead)
        if completed_graphs:
            try:
                async with get_session() as session:
                    repo = ExecutionGraphRepository(session)
                    for workflow_id, graph in completed_graphs:
                        try:
                            await repo.create(graph)
                            logger.info(f"Persisted execution graph: {workflow_id}")
                            # Move to history only on success
                            self.execution_graph_history.append(graph)
                            completed_ids.append(workflow_id)
                        except Exception as e:
                            logger.opt(exception=True).error(f"Failed to persist graph {workflow_id}: {e}")
            except Exception as e:
                logger.opt(exception=True).error(f"Failed to create database session: {e}")

        # Remove from active trackers
        for workflow_id in completed_ids:
            del self.active_execution_trackers[workflow_id]

        if completed_ids:
            logger.debug(f"Cleaned up {len(completed_ids)} completed trackers")

    def __repr__(self) -> str:
        """Return string representation."""
        return "DaemonState()"
