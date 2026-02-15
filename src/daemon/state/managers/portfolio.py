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
    PeerAnalysisInput,
    PeerAnalysisRecord,
    PortfolioAllocationRecord,
    PortfolioRebalancingRecord,
    RiskReportRecord,
    SectorAttributionRecord,
    SectorRotationRecord,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


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

    _optimization_cache: list[OptimizationRecord] | None = PrivateAttr(default=None)
    _rebalancing_cache: list[PortfolioRebalancingRecord] | None = PrivateAttr(default=None)
    _sector_rotation_cache: list[SectorRotationRecord] | None = PrivateAttr(default=None)
    _sector_attribution_cache: SectorAttributionRecord | None = PrivateAttr(default=None)
    _peer_analysis_cache: list[PeerAnalysisRecord] | None = PrivateAttr(default=None)
    _correlation_audit_cache: list[CorrelationAuditRecord] | None = PrivateAttr(default=None)
    _risk_report_cache: list[RiskReportRecord] | None = PrivateAttr(default=None)
    _monte_carlo_cache: list[MonteCarloRecord] | None = PrivateAttr(default=None)

    async def _get_metadata_datetime(self, key: str, session: AsyncSession | None = None) -> datetime | None:
        """Get datetime metadata value with fresh session.

        Args:
            key: Metadata key to fetch
            session: Optional session for API endpoints

        Returns:
            Datetime value or None if not found/unavailable
        """
        from src.database.repositories.metadata import MetadataRepository

        if session:
            repo = MetadataRepository(session)
            return await repo.get_datetime(key)

        try:
            from src.database.connection import get_session

            async with get_session() as fresh_session:
                repo = MetadataRepository(fresh_session)
                return await repo.get_datetime(key)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to get metadata {key}: {e}")
            return None

    async def get_last_optimization(self, session: AsyncSession | None = None) -> datetime | None:
        """Get last optimization timestamp from DB."""
        return await self._get_metadata_datetime("portfolio.last_optimization", session)

    async def get_last_portfolio_rebalancing(self, session: AsyncSession | None = None) -> datetime | None:
        """Get last rebalancing timestamp from DB."""
        return await self._get_metadata_datetime("portfolio.last_portfolio_rebalancing", session)

    async def get_last_sector_rotation(self, session: AsyncSession | None = None) -> datetime | None:
        """Get last sector rotation timestamp from DB."""
        return await self._get_metadata_datetime("portfolio.last_sector_rotation", session)

    async def get_last_peer_analysis(self, session: AsyncSession | None = None) -> datetime | None:
        """Get last peer analysis timestamp from DB."""
        return await self._get_metadata_datetime("portfolio.last_peer_analysis", session)

    async def get_last_correlation_audit(self, session: AsyncSession | None = None) -> datetime | None:
        """Get last correlation audit timestamp from DB."""
        return await self._get_metadata_datetime("portfolio.last_correlation_audit", session)

    async def get_last_risk_report(self, session: AsyncSession | None = None) -> datetime | None:
        """Get last risk report timestamp from DB."""
        return await self._get_metadata_datetime("portfolio.last_risk_report", session)

    async def get_last_tearsheet(self, session: AsyncSession | None = None) -> datetime | None:
        """Get last tearsheet timestamp from DB."""
        return await self._get_metadata_datetime("portfolio.last_tearsheet", session)

    async def get_active_target_allocations(self) -> dict[str, float] | None:
        """Get active target allocations from DB."""
        try:
            from src.database.connection import get_session
            from src.database.repositories.metadata import MetadataRepository

            async with get_session() as session:
                repo = MetadataRepository(session)
                return await repo.get_dict("portfolio.active_target_allocations")
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to get active target allocations: {e}")
            return None

    async def set_active_target_allocations(self, value: dict[str, float] | None) -> None:
        """Set active target allocations in DB."""
        if value is None:
            return
        try:
            from src.database.connection import get_session
            from src.database.repositories.metadata import MetadataRepository

            async with get_session() as session:
                repo = MetadataRepository(session)
                await repo.set("portfolio.active_target_allocations", value)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to set active target allocations: {e}")

    async def set_last_sector_rotation(self, value: datetime | None) -> None:
        """Set last sector rotation timestamp in DB."""
        if value is None:
            return
        try:
            from src.database.connection import get_session
            from src.database.repositories.metadata import MetadataRepository

            async with get_session() as session:
                repo = MetadataRepository(session)
                await repo.set("portfolio.last_sector_rotation", value)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to set last sector rotation: {e}")

    async def set_last_correlation_audit(self, value: datetime | None) -> None:
        """Set last correlation audit timestamp in DB."""
        if value is None:
            return
        try:
            from src.database.connection import get_session
            from src.database.repositories.metadata import MetadataRepository

            async with get_session() as session:
                repo = MetadataRepository(session)
                await repo.set("portfolio.last_correlation_audit", value)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to set last correlation audit: {e}")

    async def get_optimization_history(
        self, limit: int = 10, session: AsyncSession | None = None
    ) -> list[OptimizationRecord]:
        """Get optimization history with lazy loading."""
        from src.database.repositories.optimization import OptimizationRecordRepository

        if session:
            repo = OptimizationRecordRepository(session)
            return await repo.get_recent(limit)

        if self._optimization_cache is None:
            try:
                from src.database.connection import get_session

                async with get_session() as fresh_session:
                    repo = OptimizationRecordRepository(fresh_session)
                    self._optimization_cache = await repo.get_recent(limit)
            except Exception as e:
                logger.opt(exception=True).warning(f"Failed to get optimization history: {e}")
                return []
        return self._optimization_cache

    async def get_rebalancing_history(
        self, limit: int = 30, session: AsyncSession | None = None
    ) -> list[PortfolioRebalancingRecord]:
        """Get rebalancing history with lazy loading."""
        from src.database.repositories.rebalancing import RebalancingRecordRepository

        if session:
            repo = RebalancingRecordRepository(session)
            return await repo.get_recent(limit)

        if self._rebalancing_cache is None:
            try:
                from src.database.connection import get_session

                async with get_session() as fresh_session:
                    repo = RebalancingRecordRepository(fresh_session)
                    self._rebalancing_cache = await repo.get_recent(limit)
            except Exception as e:
                logger.opt(exception=True).warning(f"Failed to get rebalancing history: {e}")
                return []
        return self._rebalancing_cache

    async def get_sector_rotation_history(
        self, limit: int = 30, session: AsyncSession | None = None
    ) -> list[SectorRotationRecord]:
        """Get sector rotation history with lazy loading."""
        from src.database.repositories.sector_rotation import SectorRotationRecordRepository

        if session:
            repo = SectorRotationRecordRepository(session)
            return await repo.get_recent(limit)

        if self._sector_rotation_cache is None:
            try:
                from src.database.connection import get_session

                async with get_session() as fresh_session:
                    repo = SectorRotationRecordRepository(fresh_session)
                    self._sector_rotation_cache = await repo.get_recent(limit)
            except Exception as e:
                logger.opt(exception=True).warning(f"Failed to get sector rotation history: {e}")
                return []
        return self._sector_rotation_cache

    async def get_peer_analysis_history(
        self, limit: int = 10, session: AsyncSession | None = None
    ) -> list[PeerAnalysisRecord]:
        """Get peer analysis history with lazy loading."""
        from src.database.repositories.peer_analysis import PeerAnalysisRecordRepository

        if session:
            repo = PeerAnalysisRecordRepository(session)
            return await repo.get_recent(limit)

        if self._peer_analysis_cache is None:
            try:
                from src.database.connection import get_session

                async with get_session() as fresh_session:
                    repo = PeerAnalysisRecordRepository(fresh_session)
                    self._peer_analysis_cache = await repo.get_recent(limit)
            except Exception as e:
                logger.opt(exception=True).warning(f"Failed to get peer analysis history: {e}")
                return []
        return self._peer_analysis_cache

    async def get_sector_attribution_latest(
        self, session: AsyncSession | None = None
    ) -> SectorAttributionRecord | None:
        """Get latest sector attribution record with lazy loading."""
        from src.database.repositories.sector_attribution import SectorAttributionRecordRepository

        if session:
            repo = SectorAttributionRecordRepository(session)
            return await repo.get_latest()

        if self._sector_attribution_cache is None:
            try:
                from src.database.connection import get_session

                async with get_session() as fresh_session:
                    repo = SectorAttributionRecordRepository(fresh_session)
                    self._sector_attribution_cache = await repo.get_latest()
            except Exception as e:
                logger.opt(exception=True).warning(f"Failed to get sector attribution: {e}")
                return None
        return self._sector_attribution_cache

    async def get_sector_attribution_history(
        self, limit: int = 30, session: AsyncSession | None = None
    ) -> list[SectorAttributionRecord]:
        """Get sector attribution history."""
        from src.database.repositories.sector_attribution import SectorAttributionRecordRepository

        try:
            if session:
                repo = SectorAttributionRecordRepository(session)
                return await repo.get_history(limit)

            from src.database.connection import get_session

            async with get_session() as fresh_session:
                repo = SectorAttributionRecordRepository(fresh_session)
                return await repo.get_history(limit)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to get sector attribution history: {e}")
            return []

    async def get_correlation_audit_history(
        self, limit: int = 10, session: AsyncSession | None = None
    ) -> list[CorrelationAuditRecord]:
        """Get correlation audit history with lazy loading."""
        from src.database.repositories.correlation_audit import CorrelationAuditRecordRepository

        if session:
            repo = CorrelationAuditRecordRepository(session)
            return await repo.get_recent(limit)

        if self._correlation_audit_cache is None:
            try:
                from src.database.connection import get_session

                async with get_session() as fresh_session:
                    repo = CorrelationAuditRecordRepository(fresh_session)
                    self._correlation_audit_cache = await repo.get_recent(limit)
            except Exception as e:
                logger.opt(exception=True).warning(f"Failed to get correlation audit history: {e}")
                return []
        return self._correlation_audit_cache

    async def get_risk_report_history(
        self, limit: int = 30, session: AsyncSession | None = None
    ) -> list[RiskReportRecord]:
        """Get risk report history with lazy loading."""
        from src.database.repositories.risk_report import RiskReportRecordRepository

        if session:
            repo = RiskReportRecordRepository(session)
            return await repo.get_recent(limit)

        if self._risk_report_cache is None:
            try:
                from src.database.connection import get_session

                async with get_session() as fresh_session:
                    repo = RiskReportRecordRepository(fresh_session)
                    self._risk_report_cache = await repo.get_recent(limit)
            except Exception as e:
                logger.opt(exception=True).warning(f"Failed to get risk report history: {e}")
                return []
        return self._risk_report_cache

    async def get_monte_carlo_tests(
        self, limit: int = 52, session: AsyncSession | None = None
    ) -> list[MonteCarloRecord]:
        """Get Monte Carlo tests with lazy loading."""
        from src.database.repositories.monte_carlo import MonteCarloRecordRepository

        if session:
            repo = MonteCarloRecordRepository(session)
            return await repo.get_recent(limit)

        if self._monte_carlo_cache is None:
            try:
                from src.database.connection import get_session

                async with get_session() as fresh_session:
                    repo = MonteCarloRecordRepository(fresh_session)
                    self._monte_carlo_cache = await repo.get_recent(limit)
            except Exception as e:
                logger.opt(exception=True).warning(f"Failed to get Monte Carlo tests: {e}")
                return []
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

        try:
            from src.database.connection import get_session
            from src.database.repositories.metadata import MetadataRepository
            from src.database.repositories.optimization import OptimizationRecordRepository

            async with get_session() as session:
                await OptimizationRecordRepository(session).create(record)
                await MetadataRepository(session).set("portfolio.last_optimization", now)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to record optimization: {e}")

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

        try:
            from src.database.connection import get_session
            from src.database.repositories.metadata import MetadataRepository
            from src.database.repositories.rebalancing import RebalancingRecordRepository

            async with get_session() as session:
                await RebalancingRecordRepository(session).create(record)
                metadata_repo = MetadataRepository(session)
                await metadata_repo.set("portfolio.last_portfolio_rebalancing", now)
                # Store active target allocations
                allocations_dict = {a.symbol: a.weight for a in input_data.allocations}
                await metadata_repo.set("portfolio.active_target_allocations", allocations_dict)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to record portfolio rebalancing: {e}")

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

        try:
            from src.database.connection import get_session
            from src.database.repositories.metadata import MetadataRepository
            from src.database.repositories.sector_rotation import SectorRotationRecordRepository

            async with get_session() as session:
                await SectorRotationRecordRepository(session).create(record)
                await MetadataRepository(session).set("portfolio.last_sector_rotation", now)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to record sector rotation: {e}")

        self._sector_rotation_cache = None

    async def record_peer_analysis(self, input_data: PeerAnalysisInput) -> None:
        """Record a deep peer benchmarking analysis run."""
        now = datetime.now(UTC)
        record = PeerAnalysisRecord(
            timestamp=now,
            symbols_analyzed=input_data.symbols_analyzed,
            rankings=input_data.rankings,
            swap_recommendations=input_data.swap_recommendations,
            analyses=input_data.analyses or [],
            total_peers=input_data.total_peers,
            total_duration_seconds=input_data.total_duration_seconds,
        )

        try:
            from src.database.connection import get_session
            from src.database.repositories.metadata import MetadataRepository
            from src.database.repositories.peer_analysis import PeerAnalysisRecordRepository

            async with get_session() as session:
                await PeerAnalysisRecordRepository(session).create(record)
                await MetadataRepository(session).set("portfolio.last_peer_analysis", now)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to record peer analysis: {e}")

        self._peer_analysis_cache = None

    async def record_sector_attribution(
        self,
        analysis: SectorAttributionRecord,
        session: AsyncSession | None = None,
    ) -> None:
        """Record a sector attribution analysis run."""
        try:
            from src.database.repositories.sector_attribution import (
                SectorAttributionRecordRepository,
            )

            if session:
                await SectorAttributionRecordRepository(session).create(analysis)
            else:
                from src.database.connection import get_session

                async with get_session() as fresh_session:
                    await SectorAttributionRecordRepository(fresh_session).create(analysis)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to record sector attribution: {e}")

        self._sector_attribution_cache = None

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

        try:
            from src.database.connection import get_session
            from src.database.repositories.correlation_audit import CorrelationAuditRecordRepository
            from src.database.repositories.metadata import MetadataRepository

            async with get_session() as session:
                await CorrelationAuditRecordRepository(session).create(record)
                await MetadataRepository(session).set("portfolio.last_correlation_audit", now)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to record correlation audit: {e}")

        self._correlation_audit_cache = None

    async def record_risk_report(self, report: RiskReportRecord) -> None:
        """Record a portfolio risk report."""
        try:
            from src.database.connection import get_session
            from src.database.repositories.metadata import MetadataRepository
            from src.database.repositories.risk_report import RiskReportRecordRepository

            async with get_session() as session:
                await RiskReportRecordRepository(session).create(report)
                await MetadataRepository(session).set("portfolio.last_risk_report", report.timestamp)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to record risk report: {e}")

        self._risk_report_cache = None

    async def record_monte_carlo_test(self, record: MonteCarloRecord) -> None:
        """Add Monte Carlo test record."""
        try:
            from src.database.connection import get_session
            from src.database.repositories.monte_carlo import MonteCarloRecordRepository

            async with get_session() as session:
                await MonteCarloRecordRepository(session).create(record)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to record Monte Carlo test: {e}")

        self._monte_carlo_cache = None

    async def record_tearsheet(self, symbol: str, html_path: str) -> None:
        """Record a tearsheet generation run."""
        now = datetime.now(UTC)
        try:
            from src.database.connection import get_session
            from src.database.repositories.metadata import MetadataRepository

            async with get_session() as session:
                await MetadataRepository(session).set("portfolio.last_tearsheet", now)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to record tearsheet: {e}")

        logger.info(f"Recorded tearsheet generation for {symbol} at {html_path}")

    def __repr__(self) -> str:
        """Return string representation."""
        return "PortfolioStateManager()"
