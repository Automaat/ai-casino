"""DaemonState facade maintaining backward compatibility."""

# ruff: noqa: D102  # Properties delegate to managers for backward compatibility

from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.daemon.state.managers.data_pipeline import DataPipelineStateManager
from src.daemon.state.managers.discovery import DiscoveryStateManager
from src.daemon.state.managers.portfolio import PortfolioStateManager
from src.daemon.state.managers.positions import PositionStateManager
from src.daemon.state.managers.snapshots import SnapshotStateManager
from src.daemon.state.managers.strategy import StrategyStateManager
from src.daemon.state.managers.trading import TradingStateManager
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
from src.screening.screener import ScreeningResult
from src.strategies.session import TradingSession

if TYPE_CHECKING:
    from src.daemon.degradation import DegradationContext
    from src.daemon.positions import PositionManagementAction, PositionRecord
    from src.database.repositories.analysis import AnalysisRecordRepository
    from src.database.repositories.discovery import DiscoveryHistoryRepository
    from src.database.repositories.snapshot import PortfolioSnapshotRepository
    from src.execution_tracking.models import ExecutionGraph
    from src.execution_tracking.tracker import ExecutionGraphTracker


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
    active_execution_trackers: dict[str, "ExecutionGraphTracker"] = Field(
        default_factory=dict,
        exclude=True,
        description="Active workflow execution trackers by workflow_id",
    )
    execution_graph_history: deque["ExecutionGraph"] = Field(
        default_factory=lambda: deque(maxlen=50),
        exclude=True,
        description="Recent completed execution graphs (last 50)",
    )

    @model_validator(mode="wrap")
    @classmethod
    def _distribute_fields_to_managers(  # noqa: C901, PLR0912  # Backward compat, can't simplify
        cls, values: dict, handler: Callable[[dict], DaemonState]
    ) -> DaemonState:
        """Distribute old-style constructor fields to appropriate managers for backward compatibility."""
        if isinstance(values, dict):
            # Extract manager-specific fields
            trading_fields = {}
            positions_fields = {}
            portfolio_fields = {}
            data_pipeline_fields = {}
            discovery_fields = {}
            strategy_fields = {}

            # Trading fields
            for field in [
                "last_run",
                "analyses",
                "total_analyses",
                "total_trades",
                "paper_trading_start_date",
                "current_trading_mode",
                "last_journal_date",
                "last_signal_tracking",
            ]:
                if field in values:
                    trading_fields[field] = values.pop(field)

            # Position fields
            for field in ["active_positions", "position_management_history"]:
                if field in values:
                    positions_fields[field] = values.pop(field)

            # Portfolio fields
            for field in [
                "last_optimization",
                "optimization_history",
                "last_portfolio_rebalancing",
                "portfolio_rebalancing_history",
                "active_target_allocations",
                "last_sector_rotation",
                "sector_rotation_history",
                "last_peer_analysis",
                "peer_analysis_history",
                "last_correlation_audit",
                "correlation_audit_history",
                "last_risk_report",
                "risk_report_history",
                "monte_carlo_tests",
                "last_tearsheet",
            ]:
                if field in values:
                    portfolio_fields[field] = values.pop(field)

            # Data pipeline fields
            for field in [
                "last_prefetch",
                "prefetch_history",
                "last_pre_market_refresh",
                "last_after_hours_screening",
                "screening_history",
                "last_earnings_fetch",
                "earnings_calendar_history",
                "profiling_history",
            ]:
                if field in values:
                    data_pipeline_fields[field] = values.pop(field)

            # Discovery fields
            for field in ["last_discovery", "discovery_history", "active_discovery_candidates"]:
                if field in values:
                    discovery_fields[field] = values.pop(field)

            # Strategy fields
            for field in [
                "last_game_plan",
                "game_plan_history",
                "last_degradation",
                "degradation_history",
                "market_events",
                "last_health_check",
                "errors",
            ]:
                if field in values:
                    strategy_fields[field] = values.pop(field)

            # Create managers with extracted fields
            if trading_fields:
                values["trading"] = TradingStateManager(**trading_fields)
            if positions_fields:
                values["positions"] = PositionStateManager(**positions_fields)
            if portfolio_fields:
                values["portfolio"] = PortfolioStateManager(**portfolio_fields)
            if data_pipeline_fields:
                values["data_pipeline"] = DataPipelineStateManager(**data_pipeline_fields)
            if discovery_fields:
                values["discovery"] = DiscoveryStateManager(**discovery_fields)
            if strategy_fields:
                values["strategy"] = StrategyStateManager(**strategy_fields)

        # Call default handler to create instance
        return handler(values)

    def set_repositories(
        self,
        analysis_repository: AnalysisRecordRepository | None = None,
        discovery_repository: DiscoveryHistoryRepository | None = None,
        snapshot_repository: PortfolioSnapshotRepository | None = None,
    ) -> None:
        """Inject database repositories after loading state.

        Args:
            analysis_repository: Analysis record repository
            discovery_repository: Discovery history repository
            snapshot_repository: Portfolio snapshot repository
        """
        if analysis_repository:
            self.trading.set_repository(analysis_repository)
        if discovery_repository:
            self.discovery.set_repository(discovery_repository)
        if snapshot_repository:
            self.snapshots.set_repository(snapshot_repository)
        logger.debug("Repositories injected into DaemonState")

    @classmethod
    def load(cls, path: str) -> DaemonState:
        """Load state from JSON file.

        Args:
            path: Path to state file (supports ~ expansion)

        Returns:
            DaemonState instance
        """
        expanded_path = Path(path).expanduser()

        if not expanded_path.exists():
            logger.info(f"No existing state at {expanded_path}, starting fresh")
            return cls()

        try:
            with expanded_path.open() as f:
                data = json.load(f)
            logger.info(f"Loaded daemon state from {expanded_path}")
            return cls.model_validate(data)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to load state: {e}, starting fresh")
            return cls()

    def save(self, path: str) -> None:
        """Save state to JSON file.

        Args:
            path: Path to state file (supports ~ expansion)
        """
        expanded_path = Path(path).expanduser()
        expanded_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with expanded_path.open("w") as f:
                json.dump(self.model_dump(mode="json"), f, indent=2, default=str)
            logger.debug(f"Saved daemon state to {expanded_path}")
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to save state: {e}")

    # ===================
    # Trading Manager API
    # ===================

    def record_analysis(  # noqa: PLR0913 - Facade maintains backward compat, delegates to clean manager
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
        self.trading.record_analysis(input_data)

    @property
    def last_run(self) -> datetime | None:
        return self.trading.last_run

    @last_run.setter
    def last_run(self, value: datetime | None) -> None:
        self.trading.last_run = value

    @property
    def analyses(self) -> list[AnalysisRecord]:
        return self.trading.analyses

    @analyses.setter
    def analyses(self, value: list[AnalysisRecord]) -> None:
        self.trading.analyses = value

    @property
    def total_analyses(self) -> int:
        return self.trading.total_analyses

    @property
    def total_trades(self) -> int:
        return self.trading.total_trades

    @property
    def paper_trading_start_date(self) -> datetime | None:
        return self.trading.paper_trading_start_date

    @paper_trading_start_date.setter
    def paper_trading_start_date(self, value: datetime | None) -> None:
        self.trading.paper_trading_start_date = value

    @property
    def current_trading_mode(self) -> str:
        return self.trading.current_trading_mode

    @current_trading_mode.setter
    def current_trading_mode(self, value: str) -> None:
        self.trading.current_trading_mode = value

    @property
    def last_journal_date(self) -> str | None:
        return self.trading.last_journal_date

    @last_journal_date.setter
    def last_journal_date(self, value: str | None) -> None:
        self.trading.last_journal_date = value

    @property
    def last_signal_tracking(self) -> datetime | None:
        return self.trading.last_signal_tracking

    @last_signal_tracking.setter
    def last_signal_tracking(self, value: datetime | None) -> None:
        self.trading.last_signal_tracking = value

    # ===================
    # Position Manager API
    # ===================

    def add_position(self, position: PositionRecord) -> None:
        """Delegate to position manager."""
        self.positions.add_position(position)

    def remove_position(self, symbol: str) -> None:
        """Delegate to position manager."""
        self.positions.remove_position(symbol)

    def update_position(self, position: PositionRecord) -> None:
        """Delegate to position manager."""
        self.positions.update_position(position)

    def record_position_action(self, action: PositionManagementAction) -> None:
        """Delegate to position manager."""
        self.positions.record_position_action(action)

    def get_position(self, symbol: str) -> PositionRecord | None:
        """Delegate to position manager."""
        return self.positions.get_position(symbol)

    @property
    def active_positions(self) -> dict[str, dict]:
        return self.positions.active_positions

    @active_positions.setter
    def active_positions(self, value: dict[str, dict]) -> None:
        self.positions.active_positions = value

    @property
    def position_management_history(self) -> list[dict]:
        return self.positions.position_management_history

    # ===================
    # Portfolio Manager API
    # ===================

    def record_optimization(
        self,
        symbols_optimized: list[str],
        symbols_skipped: list[str],
        total_time_seconds: float,
    ) -> None:
        """Delegate to portfolio manager."""
        self.portfolio.record_optimization(symbols_optimized, symbols_skipped, total_time_seconds)

    def record_portfolio_rebalancing(  # noqa: PLR0913 - Facade maintains backward compat
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
        self.portfolio.record_portfolio_rebalancing(input_data)

    def record_sector_rotation(
        self,
        leading_sectors: list[str],
        lagging_sectors: list[str],
        sector_strengths: dict[str, float],
        sector_momenta: dict[str, str],
        flagged_positions: list[str] | None = None,
    ) -> None:
        """Delegate to portfolio manager."""
        self.portfolio.record_sector_rotation(
            leading_sectors, lagging_sectors, sector_strengths, sector_momenta, flagged_positions
        )

    def record_peer_analysis(
        self,
        symbols_analyzed: list[str],
        rankings: dict[str, int],
        swap_recommendations: list[str],
        total_peers: int,
        total_duration_seconds: float,
    ) -> None:
        """Delegate to portfolio manager."""
        self.portfolio.record_peer_analysis(
            symbols_analyzed, rankings, swap_recommendations, total_peers, total_duration_seconds
        )

    def record_correlation_audit(  # noqa: PLR0913 - Facade maintains backward compat
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
        self.portfolio.record_correlation_audit(input_data)

    def record_risk_report(self, report: RiskReportRecord) -> None:
        """Delegate to portfolio manager."""
        self.portfolio.record_risk_report(report)

    def record_monte_carlo_test(self, record: MonteCarloRecord, max_records: int = 52) -> None:
        """Delegate to portfolio manager."""
        self.portfolio.record_monte_carlo_test(record, max_records)

    def record_tearsheet(self, symbol: str, html_path: str) -> None:
        """Delegate to portfolio manager."""
        self.portfolio.record_tearsheet(symbol, html_path)

    @property
    def last_optimization(self) -> datetime | None:
        return self.portfolio.last_optimization

    @property
    def optimization_history(self) -> list[OptimizationRecord]:
        return self.portfolio.optimization_history

    @property
    def last_portfolio_rebalancing(self) -> datetime | None:
        return self.portfolio.last_portfolio_rebalancing

    @property
    def portfolio_rebalancing_history(self) -> list[PortfolioRebalancingRecord]:
        return self.portfolio.portfolio_rebalancing_history

    @property
    def active_target_allocations(self) -> dict[str, float] | None:
        return self.portfolio.active_target_allocations

    @active_target_allocations.setter
    def active_target_allocations(self, value: dict[str, float] | None) -> None:
        self.portfolio.active_target_allocations = value

    @property
    def last_sector_rotation(self) -> datetime | None:
        return self.portfolio.last_sector_rotation

    @last_sector_rotation.setter
    def last_sector_rotation(self, value: datetime | None) -> None:
        self.portfolio.last_sector_rotation = value

    @property
    def sector_rotation_history(self) -> list[SectorRotationRecord]:
        return self.portfolio.sector_rotation_history

    @sector_rotation_history.setter
    def sector_rotation_history(self, value: list[SectorRotationRecord]) -> None:
        self.portfolio.sector_rotation_history = value

    @property
    def last_peer_analysis(self) -> datetime | None:
        return self.portfolio.last_peer_analysis

    @property
    def peer_analysis_history(self) -> list[PeerAnalysisRecord]:
        return self.portfolio.peer_analysis_history

    @property
    def last_correlation_audit(self) -> datetime | None:
        return self.portfolio.last_correlation_audit

    @last_correlation_audit.setter
    def last_correlation_audit(self, value: datetime | None) -> None:
        self.portfolio.last_correlation_audit = value

    @property
    def correlation_audit_history(self) -> list[CorrelationAuditRecord]:
        return self.portfolio.correlation_audit_history

    @property
    def last_risk_report(self) -> datetime | None:
        return self.portfolio.last_risk_report

    @property
    def risk_report_history(self) -> list[RiskReportRecord]:
        return self.portfolio.risk_report_history

    @risk_report_history.setter
    def risk_report_history(self, value: list[RiskReportRecord]) -> None:
        self.portfolio.risk_report_history = value

    @property
    def monte_carlo_tests(self) -> list[MonteCarloRecord]:
        return self.portfolio.monte_carlo_tests

    @property
    def last_tearsheet(self) -> datetime | None:
        return self.portfolio.last_tearsheet

    # ===================
    # Data Pipeline Manager API
    # ===================

    def record_prefetch(
        self,
        symbols_prefetched: int,
        symbols_failed: int,
        finbert_ready: bool,
        total_duration_seconds: float,
    ) -> None:
        """Delegate to data pipeline manager."""
        self.data_pipeline.record_prefetch(
            symbols_prefetched, symbols_failed, finbert_ready, total_duration_seconds
        )

    def record_after_hours_screening(
        self,
        criteria: str,
        universe: str,
        candidates: list[ScreeningResult],
        top_n: int = 10,
        screened_at: datetime | None = None,
    ) -> None:
        """Delegate to data pipeline manager."""
        self.data_pipeline.record_after_hours_screening(criteria, universe, candidates, top_n, screened_at)

    def record_earnings_fetch(
        self,
        events: list[EarningsEventRecord],
        symbols_fetched: int,
        symbols_failed: int,
    ) -> None:
        """Delegate to data pipeline manager."""
        self.data_pipeline.record_earnings_fetch(events, symbols_fetched, symbols_failed)

    def record_profiling(self, metrics: object) -> None:
        """Delegate to data pipeline manager."""
        self.data_pipeline.record_profiling(metrics)

    @property
    def last_prefetch(self) -> datetime | None:
        return self.data_pipeline.last_prefetch

    @last_prefetch.setter
    def last_prefetch(self, value: datetime | None) -> None:
        self.data_pipeline.last_prefetch = value

    @property
    def last_pre_market_refresh(self) -> datetime | None:
        return self.data_pipeline.last_pre_market_refresh

    @last_pre_market_refresh.setter
    def last_pre_market_refresh(self, value: datetime | None) -> None:
        self.data_pipeline.last_pre_market_refresh = value

    @property
    def prefetch_history(self) -> list[PrefetchRecord]:
        return self.data_pipeline.prefetch_history

    @property
    def last_after_hours_screening(self) -> datetime | None:
        return self.data_pipeline.last_after_hours_screening

    @last_after_hours_screening.setter
    def last_after_hours_screening(self, value: datetime | None) -> None:
        self.data_pipeline.last_after_hours_screening = value

    @property
    def screening_history(self) -> list[ScreeningRecord]:
        return self.data_pipeline.screening_history

    @screening_history.setter
    def screening_history(self, value: list[ScreeningRecord]) -> None:
        self.data_pipeline.screening_history = value

    @property
    def last_earnings_fetch(self) -> datetime | None:
        return self.data_pipeline.last_earnings_fetch

    @property
    def earnings_calendar_history(self) -> list[EarningsCalendarRecord]:
        return self.data_pipeline.earnings_calendar_history

    @property
    def profiling_history(self) -> list[ProfilingRecord]:
        return self.data_pipeline.profiling_history

    # ===================
    # Discovery Manager API
    # ===================

    def record_discovery(self, candidates: list[DiscoveryCandidate], added_symbols: list[str]) -> None:
        """Delegate to discovery manager."""
        self.discovery.record_discovery(candidates, added_symbols)

    def expire_stale_candidates(self) -> list[str]:
        """Delegate to discovery manager."""
        return self.discovery.expire_stale_candidates()

    def get_active_discovery_symbols(self) -> list[str]:
        """Delegate to discovery manager."""
        return self.discovery.get_active_discovery_symbols()

    @property
    def last_discovery(self) -> datetime | None:
        return self.discovery.last_discovery

    @last_discovery.setter
    def last_discovery(self, value: datetime | None) -> None:
        self.discovery.last_discovery = value

    @property
    def discovery_history(self) -> list[DiscoveryHistoryRecord]:
        return self.discovery.discovery_history

    @property
    def active_discovery_candidates(self) -> list[DiscoveryCandidate]:
        return self.discovery.active_discovery_candidates

    @active_discovery_candidates.setter
    def active_discovery_candidates(self, value: list[DiscoveryCandidate]) -> None:
        self.discovery.active_discovery_candidates = value

    # ===================
    # Strategy Manager API
    # ===================

    def record_game_plan(
        self,
        priority_symbols: list[str],
        risk_stance: str,
        sector_focus: list[str],
    ) -> None:
        """Delegate to strategy manager."""
        self.strategy.record_game_plan(priority_symbols, risk_stance, sector_focus)

    def record_degradation(self, context: DegradationContext) -> None:
        """Delegate to strategy manager."""
        self.strategy.record_degradation(context)

    def record_error(self, error: str) -> None:
        """Delegate to strategy manager."""
        self.strategy.record_error(error)

    @property
    def last_game_plan(self) -> datetime | None:
        return self.strategy.last_game_plan

    @property
    def game_plan_history(self) -> list[GamePlanRecord]:
        return self.strategy.game_plan_history

    @game_plan_history.setter
    def game_plan_history(self, value: list[GamePlanRecord]) -> None:
        self.strategy.game_plan_history = value

    @property
    def last_degradation(self) -> datetime | None:
        return self.strategy.last_degradation

    @property
    def degradation_history(self) -> list[DegradationRecord]:
        return self.strategy.degradation_history

    @degradation_history.setter
    def degradation_history(self, value: list[DegradationRecord]) -> None:
        self.strategy.degradation_history = value

    @property
    def market_events(self) -> list[dict]:
        return self.strategy.market_events

    @property
    def last_health_check(self) -> datetime | None:
        return self.strategy.last_health_check

    @last_health_check.setter
    def last_health_check(self, value: datetime | None) -> None:
        self.strategy.last_health_check = value

    @property
    def errors(self) -> list[str]:
        return self.strategy.errors

    @errors.setter
    def errors(self, value: list[str]) -> None:
        self.strategy.errors = value

    # ===================
    # Snapshot Manager API
    # ===================

    def snapshot_portfolio(self, snapshot: PortfolioSnapshot) -> None:
        """Delegate to snapshot manager."""
        self.snapshots.snapshot_portfolio(snapshot)

    # ===========================
    # Execution Tracking API
    # ===========================

    def add_execution_tracker(self, workflow_id: str, tracker: "ExecutionGraphTracker") -> None:
        """Add active execution tracker.

        Args:
            workflow_id: Workflow ID
            tracker: ExecutionGraphTracker instance
        """
        self.active_execution_trackers[workflow_id] = tracker
        logger.debug(f"Added execution tracker for workflow {workflow_id}")

    def get_execution_tracker(self, workflow_id: str) -> "ExecutionGraphTracker | None":
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

    def get_active_execution_graphs(self) -> list["ExecutionGraph"]:
        """Get all active execution graphs.

        Returns:
            List of active execution graphs
        """
        return [tracker.graph for tracker in self.active_execution_trackers.values()]

    def get_execution_graph(self, workflow_id: str) -> "ExecutionGraph | None":
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

    def __repr__(self) -> str:
        """Return string representation."""
        return f"DaemonState(analyses={self.total_analyses}, trades={self.total_trades})"
