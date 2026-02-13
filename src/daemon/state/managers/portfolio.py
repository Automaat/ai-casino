"""Portfolio state manager."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import PrivateAttr

from src.daemon.state.managers.base import StateManager
from src.daemon.state.models import (
    CorrelationAuditRecord,
    MonteCarloRecord,
    OptimizationRecord,
    PeerAnalysisRecord,
    PortfolioAllocationRecord,
    PortfolioRebalancingRecord,
    RiskReportRecord,
    SectorRotationRecord,
)

if TYPE_CHECKING:
    from src.daemon.state.repositories import RepositoryBundle
    from src.database.repositories.correlation_audit import CorrelationAuditRecordRepository
    from src.database.repositories.metadata import MetadataRepository
    from src.database.repositories.monte_carlo import MonteCarloRecordRepository
    from src.database.repositories.optimization import OptimizationRecordRepository
    from src.database.repositories.peer_analysis import PeerAnalysisRecordRepository
    from src.database.repositories.rebalancing import RebalancingRecordRepository
    from src.database.repositories.risk_report import RiskReportRecordRepository
    from src.database.repositories.sector_rotation import SectorRotationRecordRepository


@dataclass
class PortfolioRebalancingInput:
    """Input parameters for recording portfolio rebalancing."""

    method: str
    allocations: list[PortfolioAllocationRecord]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    rebalances_executed: int
    rebalances_pending: int


@dataclass
class CorrelationAuditInput:
    """Input parameters for recording correlation audit."""

    num_positions: int
    num_correlated_pairs: int
    max_correlation: float
    avg_correlation: float
    diversification_ratio: float
    num_substitutions: int
    total_duration_seconds: float


class PortfolioStateManager(StateManager):
    """Advanced portfolio analytics (optimization, risk, correlation, peer, sector)."""

    _metadata_repository: MetadataRepository | None = PrivateAttr(default=None)
    _optimization_repository: OptimizationRecordRepository | None = PrivateAttr(default=None)
    _rebalancing_repository: RebalancingRecordRepository | None = PrivateAttr(default=None)
    _sector_rotation_repository: SectorRotationRecordRepository | None = PrivateAttr(default=None)
    _peer_analysis_repository: PeerAnalysisRecordRepository | None = PrivateAttr(default=None)
    _correlation_audit_repository: CorrelationAuditRecordRepository | None = PrivateAttr(default=None)
    _risk_report_repository: RiskReportRecordRepository | None = PrivateAttr(default=None)
    _monte_carlo_repository: MonteCarloRecordRepository | None = PrivateAttr(default=None)

    _optimization_cache: list[OptimizationRecord] | None = PrivateAttr(default=None)
    _rebalancing_cache: list[PortfolioRebalancingRecord] | None = PrivateAttr(default=None)
    _sector_rotation_cache: list[SectorRotationRecord] | None = PrivateAttr(default=None)
    _peer_analysis_cache: list[PeerAnalysisRecord] | None = PrivateAttr(default=None)
    _correlation_audit_cache: list[CorrelationAuditRecord] | None = PrivateAttr(default=None)
    _risk_report_cache: list[RiskReportRecord] | None = PrivateAttr(default=None)
    _monte_carlo_cache: list[MonteCarloRecord] | None = PrivateAttr(default=None)

    def set_repositories(self, repos: RepositoryBundle) -> None:
        """Inject repositories from bundle."""
        self._metadata_repository = repos.metadata_repository
        self._optimization_repository = repos.optimization_repository
        self._rebalancing_repository = repos.rebalancing_repository
        self._sector_rotation_repository = repos.sector_rotation_repository
        self._peer_analysis_repository = repos.peer_analysis_repository
        self._correlation_audit_repository = repos.correlation_audit_repository
        self._risk_report_repository = repos.risk_report_repository
        self._monte_carlo_repository = repos.monte_carlo_repository
        logger.debug("PortfolioStateManager repositories injected")

    async def get_last_optimization(self) -> datetime | None:
        """Get last optimization timestamp from DB."""
        if not self._metadata_repository:
            return None
        return await self._metadata_repository.get("portfolio.last_optimization")

    async def get_last_portfolio_rebalancing(self) -> datetime | None:
        """Get last rebalancing timestamp from DB."""
        if not self._metadata_repository:
            return None
        return await self._metadata_repository.get("portfolio.last_portfolio_rebalancing")

    async def get_last_sector_rotation(self) -> datetime | None:
        """Get last sector rotation timestamp from DB."""
        if not self._metadata_repository:
            return None
        return await self._metadata_repository.get("portfolio.last_sector_rotation")

    async def get_last_peer_analysis(self) -> datetime | None:
        """Get last peer analysis timestamp from DB."""
        if not self._metadata_repository:
            return None
        return await self._metadata_repository.get("portfolio.last_peer_analysis")

    async def get_last_correlation_audit(self) -> datetime | None:
        """Get last correlation audit timestamp from DB."""
        if not self._metadata_repository:
            return None
        return await self._metadata_repository.get("portfolio.last_correlation_audit")

    async def get_last_risk_report(self) -> datetime | None:
        """Get last risk report timestamp from DB."""
        if not self._metadata_repository:
            return None
        return await self._metadata_repository.get("portfolio.last_risk_report")

    async def get_last_tearsheet(self) -> datetime | None:
        """Get last tearsheet timestamp from DB."""
        if not self._metadata_repository:
            return None
        return await self._metadata_repository.get("portfolio.last_tearsheet")

    async def get_active_target_allocations(self) -> dict[str, float] | None:
        """Get active target allocations from DB."""
        if not self._metadata_repository:
            return None
        value = await self._metadata_repository.get("portfolio.active_target_allocations")
        return value if isinstance(value, dict) else None

    async def get_optimization_history(self, limit: int = 10) -> list[OptimizationRecord]:
        """Get optimization history with lazy loading."""
        if not self._optimization_repository:
            return []
        if self._optimization_cache is None:
            self._optimization_cache = await self._optimization_repository.get_recent(limit)
        return self._optimization_cache

    async def get_rebalancing_history(self, limit: int = 30) -> list[PortfolioRebalancingRecord]:
        """Get rebalancing history with lazy loading."""
        if not self._rebalancing_repository:
            return []
        if self._rebalancing_cache is None:
            self._rebalancing_cache = await self._rebalancing_repository.get_recent(limit)
        return self._rebalancing_cache

    async def get_sector_rotation_history(self, limit: int = 30) -> list[SectorRotationRecord]:
        """Get sector rotation history with lazy loading."""
        if not self._sector_rotation_repository:
            return []
        if self._sector_rotation_cache is None:
            self._sector_rotation_cache = await self._sector_rotation_repository.get_recent(limit)
        return self._sector_rotation_cache

    async def get_peer_analysis_history(self, limit: int = 10) -> list[PeerAnalysisRecord]:
        """Get peer analysis history with lazy loading."""
        if not self._peer_analysis_repository:
            return []
        if self._peer_analysis_cache is None:
            self._peer_analysis_cache = await self._peer_analysis_repository.get_recent(limit)
        return self._peer_analysis_cache

    async def get_correlation_audit_history(self, limit: int = 10) -> list[CorrelationAuditRecord]:
        """Get correlation audit history with lazy loading."""
        if not self._correlation_audit_repository:
            return []
        if self._correlation_audit_cache is None:
            self._correlation_audit_cache = await self._correlation_audit_repository.get_recent(limit)
        return self._correlation_audit_cache

    async def get_risk_report_history(self, limit: int = 30) -> list[RiskReportRecord]:
        """Get risk report history with lazy loading."""
        if not self._risk_report_repository:
            return []
        if self._risk_report_cache is None:
            self._risk_report_cache = await self._risk_report_repository.get_recent(limit)
        return self._risk_report_cache

    async def get_monte_carlo_tests(self, limit: int = 52) -> list[MonteCarloRecord]:
        """Get Monte Carlo tests with lazy loading."""
        if not self._monte_carlo_repository:
            return []
        if self._monte_carlo_cache is None:
            self._monte_carlo_cache = await self._monte_carlo_repository.get_recent(limit)
        return self._monte_carlo_cache

    async def record_optimization(
        self,
        symbols_optimized: list[str],
        symbols_skipped: list[str],
        total_time_seconds: float,
    ) -> None:
        """Record a parameter optimization run."""
        now = datetime.now(UTC)
        record = OptimizationRecord(
            timestamp=now,
            symbols_optimized=symbols_optimized,
            symbols_skipped=symbols_skipped,
            total_time_seconds=total_time_seconds,
        )

        if self._optimization_repository:
            await self._optimization_repository.create(record)
        if self._metadata_repository:
            await self._metadata_repository.set("portfolio.last_optimization", now)

        self._optimization_cache = None

    async def record_portfolio_rebalancing(self, input_data: PortfolioRebalancingInput) -> None:
        """Record portfolio rebalancing run."""
        now = datetime.now(UTC)
        record = PortfolioRebalancingRecord(
            timestamp=now,
            method=input_data.method,
            allocations=input_data.allocations,
            expected_return=input_data.expected_return,
            expected_volatility=input_data.expected_volatility,
            sharpe_ratio=input_data.sharpe_ratio,
            rebalances_executed=input_data.rebalances_executed,
            rebalances_pending=input_data.rebalances_pending,
        )

        if self._rebalancing_repository:
            await self._rebalancing_repository.create(record)
        if self._metadata_repository:
            await self._metadata_repository.set("portfolio.last_portfolio_rebalancing", now)
            # Store active target allocations
            allocations_dict = {a.symbol: a.weight for a in input_data.allocations}
            await self._metadata_repository.set("portfolio.active_target_allocations", allocations_dict)

        self._rebalancing_cache = None

    async def record_sector_rotation(
        self,
        leading_sectors: list[str],
        lagging_sectors: list[str],
        sector_strengths: dict[str, float],
        sector_momenta: dict[str, str],
        flagged_positions: list[str] | None = None,
    ) -> None:
        """Record a sector rotation analysis run."""
        now = datetime.now(UTC)
        record = SectorRotationRecord(
            timestamp=now,
            leading_sectors=leading_sectors,
            lagging_sectors=lagging_sectors,
            sector_strengths=sector_strengths,
            sector_momenta=sector_momenta,
            flagged_positions=flagged_positions or [],
        )

        if self._sector_rotation_repository:
            await self._sector_rotation_repository.create(record)
        if self._metadata_repository:
            await self._metadata_repository.set("portfolio.last_sector_rotation", now)

        self._sector_rotation_cache = None

    async def record_peer_analysis(
        self,
        symbols_analyzed: list[str],
        rankings: dict[str, int],
        swap_recommendations: list[str],
        total_peers: int,
        total_duration_seconds: float,
    ) -> None:
        """Record a deep peer benchmarking analysis run."""
        now = datetime.now(UTC)
        record = PeerAnalysisRecord(
            timestamp=now,
            symbols_analyzed=symbols_analyzed,
            rankings=rankings,
            swap_recommendations=swap_recommendations,
            total_peers=total_peers,
            total_duration_seconds=total_duration_seconds,
        )

        if self._peer_analysis_repository:
            await self._peer_analysis_repository.create(record)
        if self._metadata_repository:
            await self._metadata_repository.set("portfolio.last_peer_analysis", now)

        self._peer_analysis_cache = None

    async def record_correlation_audit(self, input_data: CorrelationAuditInput) -> None:
        """Record a correlation audit run."""
        now = datetime.now(UTC)
        record = CorrelationAuditRecord(
            timestamp=now,
            num_positions=input_data.num_positions,
            num_correlated_pairs=input_data.num_correlated_pairs,
            max_correlation=input_data.max_correlation,
            avg_correlation=input_data.avg_correlation,
            diversification_ratio=input_data.diversification_ratio,
            num_substitutions=input_data.num_substitutions,
            total_duration_seconds=input_data.total_duration_seconds,
        )

        if self._correlation_audit_repository:
            await self._correlation_audit_repository.create(record)
        if self._metadata_repository:
            await self._metadata_repository.set("portfolio.last_correlation_audit", now)

        self._correlation_audit_cache = None

    async def record_risk_report(self, report: RiskReportRecord) -> None:
        """Record a portfolio risk report."""
        if self._risk_report_repository:
            await self._risk_report_repository.create(report)
        if self._metadata_repository:
            await self._metadata_repository.set("portfolio.last_risk_report", report.timestamp)

        self._risk_report_cache = None

    async def record_monte_carlo_test(self, record: MonteCarloRecord) -> None:
        """Add Monte Carlo test record."""
        if self._monte_carlo_repository:
            await self._monte_carlo_repository.create(record)

        self._monte_carlo_cache = None

    async def record_tearsheet(self, symbol: str, html_path: str) -> None:
        """Record a tearsheet generation run."""
        now = datetime.now(UTC)
        if self._metadata_repository:
            await self._metadata_repository.set("portfolio.last_tearsheet", now)

        logger.info(f"Recorded tearsheet generation for {symbol} at {html_path}")

    def __repr__(self) -> str:
        """Return string representation."""
        return "PortfolioStateManager()"
