"""Portfolio state manager."""

from __future__ import annotations

from datetime import UTC, datetime

from loguru import logger
from pydantic import Field

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


class PortfolioStateManager(StateManager):
    """Advanced portfolio analytics (optimization, risk, correlation, peer, sector)."""

    # Optimization
    last_optimization: datetime | None = None
    optimization_history: list[OptimizationRecord] = Field(default_factory=list)

    # Rebalancing
    last_portfolio_rebalancing: datetime | None = None
    portfolio_rebalancing_history: list[PortfolioRebalancingRecord] = Field(default_factory=list)
    active_target_allocations: dict[str, float] | None = None

    # Sector
    last_sector_rotation: datetime | None = None
    sector_rotation_history: list[SectorRotationRecord] = Field(default_factory=list)

    # Peer
    last_peer_analysis: datetime | None = None
    peer_analysis_history: list[PeerAnalysisRecord] = Field(default_factory=list)

    # Correlation
    last_correlation_audit: datetime | None = None
    correlation_audit_history: list[CorrelationAuditRecord] = Field(default_factory=list)

    # Risk
    last_risk_report: datetime | None = None
    risk_report_history: list[RiskReportRecord] = Field(default_factory=list)
    monte_carlo_tests: list[MonteCarloRecord] = Field(default_factory=list)

    # Reporting
    last_tearsheet: datetime | None = None

    def record_optimization(
        self,
        symbols_optimized: list[str],
        symbols_skipped: list[str],
        total_time_seconds: float,
    ) -> None:
        """Record a parameter optimization run.

        Args:
            symbols_optimized: Symbols that were optimized
            symbols_skipped: Symbols skipped (non-stale)
            total_time_seconds: Total optimization duration
        """
        now = datetime.now(UTC)

        self.optimization_history.append(
            OptimizationRecord(
                timestamp=now,
                symbols_optimized=symbols_optimized,
                symbols_skipped=symbols_skipped,
                total_time_seconds=total_time_seconds,
            )
        )
        self.last_optimization = now
        self.optimization_history = self._cap_history(self.optimization_history, 10, 10)

    def record_portfolio_rebalancing(  # noqa: PLR0913
        self,
        method: str,
        allocations: list[PortfolioAllocationRecord],
        expected_return: float,
        expected_volatility: float,
        sharpe_ratio: float,
        rebalances_executed: int,
        rebalances_pending: int,
    ) -> None:
        """Record portfolio rebalancing run.

        Args:
            method: Optimization method used
            allocations: Asset allocation records
            expected_return: Expected portfolio return
            expected_volatility: Expected portfolio volatility
            sharpe_ratio: Portfolio Sharpe ratio
            rebalances_executed: Number of rebalances executed
            rebalances_pending: Number of rebalances pending
        """
        now = datetime.now(UTC)

        self.portfolio_rebalancing_history.append(
            PortfolioRebalancingRecord(
                timestamp=now,
                method=method,
                allocations=allocations,
                expected_return=expected_return,
                expected_volatility=expected_volatility,
                sharpe_ratio=sharpe_ratio,
                rebalances_executed=rebalances_executed,
                rebalances_pending=rebalances_pending,
            )
        )
        self.last_portfolio_rebalancing = now
        self.portfolio_rebalancing_history = self._cap_history(self.portfolio_rebalancing_history, 30, 30)

        self.active_target_allocations = {a.symbol: a.weight for a in allocations}

    def record_sector_rotation(
        self,
        leading_sectors: list[str],
        lagging_sectors: list[str],
        sector_strengths: dict[str, float],
        sector_momenta: dict[str, str],
        flagged_positions: list[str] | None = None,
    ) -> None:
        """Record a sector rotation analysis run.

        Args:
            leading_sectors: Top 3 sectors by relative strength
            lagging_sectors: Bottom 3 sectors by relative strength
            sector_strengths: Composite strength by sector name
            sector_momenta: Momentum classification by sector name
            flagged_positions: Symbols flagged in weak sectors
        """
        now = datetime.now(UTC)

        self.sector_rotation_history.append(
            SectorRotationRecord(
                timestamp=now,
                leading_sectors=leading_sectors,
                lagging_sectors=lagging_sectors,
                sector_strengths=sector_strengths,
                sector_momenta=sector_momenta,
                flagged_positions=flagged_positions or [],
            )
        )
        self.last_sector_rotation = now
        self.sector_rotation_history = self._cap_history(self.sector_rotation_history, 30, 30)

    def record_peer_analysis(
        self,
        symbols_analyzed: list[str],
        rankings: dict[str, int],
        swap_recommendations: list[str],
        total_peers: int,
        total_duration_seconds: float,
    ) -> None:
        """Record a deep peer benchmarking analysis run.

        Args:
            symbols_analyzed: Symbols that were analyzed
            rankings: Symbol to rank mapping
            swap_recommendations: Generated swap recommendations
            total_peers: Total number of peers analyzed
            total_duration_seconds: Total analysis duration
        """
        now = datetime.now(UTC)

        self.peer_analysis_history.append(
            PeerAnalysisRecord(
                timestamp=now,
                symbols_analyzed=symbols_analyzed,
                rankings=rankings,
                swap_recommendations=swap_recommendations,
                total_peers=total_peers,
                total_duration_seconds=total_duration_seconds,
            )
        )
        self.last_peer_analysis = now
        self.peer_analysis_history = self._cap_history(self.peer_analysis_history, 10, 10)

    def record_correlation_audit(  # noqa: PLR0913
        self,
        num_positions: int,
        num_correlated_pairs: int,
        max_correlation: float,
        avg_correlation: float,
        diversification_ratio: float,
        num_substitutions: int,
        total_duration_seconds: float,
    ) -> None:
        """Record a correlation audit run.

        Args:
            num_positions: Number of positions analyzed
            num_correlated_pairs: Number of highly correlated pairs found
            max_correlation: Maximum correlation found
            avg_correlation: Average portfolio correlation
            diversification_ratio: Portfolio diversification ratio
            num_substitutions: Number of substitution suggestions
            total_duration_seconds: Total audit duration
        """
        now = datetime.now(UTC)

        self.correlation_audit_history.append(
            CorrelationAuditRecord(
                timestamp=now,
                num_positions=num_positions,
                num_correlated_pairs=num_correlated_pairs,
                max_correlation=max_correlation,
                avg_correlation=avg_correlation,
                diversification_ratio=diversification_ratio,
                num_substitutions=num_substitutions,
                total_duration_seconds=total_duration_seconds,
            )
        )
        self.last_correlation_audit = now
        self.correlation_audit_history = self._cap_history(self.correlation_audit_history, 10, 10)

    def record_risk_report(self, report: RiskReportRecord) -> None:
        """Record a portfolio risk report.

        Args:
            report: Risk report record to store
        """
        self.risk_report_history.append(report)
        self.last_risk_report = report.timestamp
        self.risk_report_history = self._cap_history(self.risk_report_history, 30, 30)

    def record_monte_carlo_test(self, record: MonteCarloRecord, max_records: int = 52) -> None:
        """Add Monte Carlo test record.

        Args:
            record: Monte Carlo test record
            max_records: Maximum records to retain (default 52)
        """
        self.monte_carlo_tests.append(record)
        self.monte_carlo_tests = self._cap_history(self.monte_carlo_tests, max_records, max_records)

    def record_tearsheet(self, symbol: str, html_path: str) -> None:
        """Record a tearsheet generation run.

        Args:
            symbol: Stock ticker symbol
            html_path: Path to generated HTML tearsheet
        """
        now = datetime.now(UTC)
        self.last_tearsheet = now
        logger.info(f"Recorded tearsheet generation for {symbol} at {html_path}")

    def __repr__(self) -> str:
        """Return string representation."""
        return f"PortfolioStateManager(optimizations={len(self.optimization_history)})"
