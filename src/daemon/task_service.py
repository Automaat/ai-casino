"""Daemon task service for scheduled task execution."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger
from rich.console import Console

from src.daemon.notification_helper import DaemonNotificationHelper

console = Console()


def _log_event_publish_error(task: asyncio.Task[object]) -> None:
    """Log exceptions from fire-and-forget event publish tasks."""
    if task.cancelled():
        return

    exc = task.exception()
    if exc is not None:
        logger.error(f"Event publish failed: {exc}")


if TYPE_CHECKING:
    from src.daemon.factory import DaemonComponents
    from src.daemon.state import SectorRotationRecord
    from src.di.container import AppContainer
    from src.metrics.correlation import CorrelationAuditResult
    from src.metrics.sector_rotation import SectorRotationAnalysis
else:
    from src.daemon.factory import DaemonComponents
    from src.di.container import AppContainer


class DaemonTaskService:
    """Execute scheduled daemon tasks."""

    def __init__(
        self,
        components: DaemonComponents,
        container: AppContainer,
    ) -> None:
        """Initialize task service.

        Args:
            components: Daemon components
            container: DI container for service access
        """
        self.components = components
        self.container = container

        # Notification helper
        self._notification_helper = DaemonNotificationHelper()

        # Track paper readiness notification state
        self._last_readiness_check: datetime | None = None
        self._notified_paper_ready = False

    def __repr__(self) -> str:
        """Return string representation."""
        return "DaemonTaskService()"

    async def run_optimization(self) -> None:
        """Run parameter optimization task."""
        if self.components.daemon_optimizer is None:
            return

        from src.daemon.tasks.portfolio_tasks import OptimizationTask

        await OptimizationTask(self.components, self.container).run()

    async def run_portfolio_rebalancing(self) -> None:
        """Run portfolio rebalancing task."""
        if self.components.daemon_rebalancer is None:
            return

        from src.daemon.tasks.portfolio_tasks import RebalancingTask

        await RebalancingTask(self.components, self.container).run()

    async def run_game_plan(self) -> None:
        """Run game plan generation task."""
        from src.daemon.tasks.analysis_tasks import GamePlanTask

        await GamePlanTask(self.components, self.container).run()

    async def run_prefetch(self) -> None:
        """Run data prefetch task."""
        if not self.components.config.prefetch.enabled:
            return

        from src.daemon.tasks.data_tasks import PrefetchTask

        await PrefetchTask(self.components, self.container).run()

    async def run_discovery(self) -> None:
        """Run stock discovery task."""
        if not self.components.config.discovery.enabled or not self.components.discovery_engine:
            return

        from src.daemon.tasks.analysis_tasks import DiscoveryTask

        await DiscoveryTask(self.components, self.container).run()

    async def run_discovery_outcome(self) -> None:
        """Run discovery outcome tracking task."""
        if not self.components.config.discovery.enabled:
            return

        from src.daemon.tasks.data_tasks import DiscoveryOutcomeTask

        await DiscoveryOutcomeTask(self.components, self.container).run()

    async def run_health_check(self) -> None:
        """Run health check task."""
        if not self.components.config.health.enabled:
            return

        from src.daemon.tasks.monitoring_tasks import HealthCheckTask

        await HealthCheckTask(self.components, self.container).run()

    async def run_journal(self) -> None:
        """Run trade journal task."""
        if not self.components.config.journal.enabled:
            return

        from src.daemon.tasks.reporting_tasks import JournalTask

        await JournalTask(self.components, self.container).run()

    async def check_paper_readiness(self) -> None:
        """Check paper trading readiness."""
        from src.daemon.config import TradingMode

        if self.components.config.trading_mode != TradingMode.PAPER:
            return

        if not self.components.notification_service:
            return

        now = datetime.now(UTC)

        # Check once per day
        if self._last_readiness_check is not None:
            elapsed_days = (now - self._last_readiness_check).days
            if elapsed_days < 1:
                return

        self._last_readiness_check = now

        try:
            from src.daemon.config import NotificationTrigger
            from src.daemon.notifications import NotificationMessage
            from src.daemon.paper_trading_validator import PaperTradingValidator

            validator = PaperTradingValidator(
                config=self.components.config.paper_trading,
                state=self.components.state,
                metrics_tracker=self.components.metrics_tracker,  # type: ignore[arg-type]
            )
            report = await validator.assess_readiness()

            if report.ready_for_live and not self._notified_paper_ready:
                message = NotificationMessage(
                    trigger=NotificationTrigger.PAPER_TRADING_READY,
                    title="Paper Trading Ready for Live",
                    body=(
                        f"Duration: {report.paper_trading_duration_days} days | "
                        f"Trades: {report.total_paper_trades}"
                    ),
                    metadata={
                        "symbol": "SYSTEM",
                        "duration_days": report.paper_trading_duration_days,
                        "total_trades": report.total_paper_trades,
                        "sharpe": report.metrics.sharpe_ratio,
                        "max_dd": report.metrics.max_drawdown_percent,
                    },
                    timestamp=datetime.now(UTC),
                )
                await self.components.notification_service.notify(
                    NotificationTrigger.PAPER_TRADING_READY, message
                )
                self._notified_paper_ready = True
                logger.info("Sent paper trading readiness notification")
        except Exception as e:
            logger.debug(f"Paper readiness check failed: {e}")

    def _is_discovery_time(self) -> bool:
        """Check if current time matches discovery schedule.

        Returns:
            True if discovery should run
        """
        now = datetime.now(self.components.scheduler.timezone)

        # Check day
        day_name = now.strftime("%a").lower()[:3]  # mon, tue, etc.
        if day_name not in [d.lower()[:3] for d in self.components.config.discovery.discovery_days]:
            return False

        # Check time window (16:00-20:00)
        discovery_hour, discovery_min = map(int, self.components.config.discovery.discovery_time.split(":"))
        current_time = now.hour * 60 + now.minute
        discovery_time_mins = discovery_hour * 60 + discovery_min

        # Within 5-minute window
        return abs(current_time - discovery_time_mins) <= 5

    def _publish_event_sync(self, event_type: str, data: dict[str, object]) -> None:
        """Publish event synchronously (helper for non-async methods).

        Args:
            event_type: Event type string
            data: Event data dictionary
        """
        from src.daemon.event_bus import DashboardEvent, EventType

        if not self.components.event_bus:
            return

        try:
            publish_coro = self.components.event_bus.publish(
                DashboardEvent(event_type=EventType[event_type], data=data)
            )

            try:
                # If we're already inside an event loop (e.g. daemon async runner),
                # schedule the publish as a task instead of calling asyncio.run(...)
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running loop in this thread: safe to use asyncio.run
                asyncio.run(publish_coro)
            else:
                task = loop.create_task(publish_coro)
                task.add_done_callback(_log_event_publish_error)
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to publish {event_type} event: {e}")

    def _reconstruct_rotation_analysis(self, record: SectorRotationRecord) -> SectorRotationAnalysis:
        """Reconstruct SectorRotationAnalysis from state record.

        Args:
            record: Sector rotation state record

        Returns:
            Full SectorRotationAnalysis pydantic model
        """
        from src.data.comparative import Sector
        from src.metrics.sector_rotation import (
            Momentum,
            SectorRotationAnalysis,
            SectorStrength,
        )

        # Reconstruct SectorStrength list from record data
        sectors = []
        sorted_sectors = sorted(record.sector_strengths.items(), key=lambda x: x[1], reverse=True)

        for rank, (sector_name, strength) in enumerate(sorted_sectors, 1):
            momentum_str = record.sector_momenta.get(sector_name, "NEUTRAL")

            # Find ETF for sector (map back from Sector enum)
            try:
                sector_enum = Sector[sector_name]
                etf = sector_enum.value
            except KeyError:
                logger.opt(exception=True).warning(f"Unknown sector {sector_name}, skipping")
                continue

            sectors.append(
                SectorStrength(
                    sector=sector_name,
                    etf=etf,
                    return_1w=0.0,  # Not stored in record
                    return_1m=0.0,
                    return_3m=0.0,
                    relative_strength=strength,
                    momentum=Momentum(momentum_str),
                    rank=rank,
                )
            )

        return SectorRotationAnalysis(
            sectors=sectors,
            leading_sectors=record.leading_sectors,
            lagging_sectors=record.lagging_sectors,
            spy_return_1w=0.0,  # Not stored, not needed for weighting
            spy_return_1m=0.0,
            spy_return_3m=0.0,
            timestamp=record.timestamp,
        )

    async def run_pre_market_refresh(self) -> None:
        """Run pre-market data refresh to update stale cache."""
        if (
            not self.components.config.prefetch.enabled
            or not self.components.config.prefetch.enable_pre_market_refresh
        ):
            return

        from src.daemon.tasks.data_tasks import PreMarketRefreshTask

        await PreMarketRefreshTask(self.components, self.container).run()

    async def run_after_hours_screening(self) -> None:
        """Run after-hours screening for watchlist candidates."""
        from src.daemon.tasks.data_tasks import ScreeningTask

        await ScreeningTask(self.components, self.container).run()

    async def run_sector_rotation(self) -> None:
        """Run sector rotation analysis."""
        from src.daemon.tasks.analysis_tasks import SectorRotationTask

        await SectorRotationTask(self.components, self.container).run()

    async def run_earnings_fetch(self) -> None:
        """Run earnings calendar fetch for watchlist symbols."""
        from src.daemon.tasks.data_tasks import EarningsFetchTask

        await EarningsFetchTask(self.components, self.container).run()

    async def run_peer_analysis(self) -> None:
        """Run weekly deep peer benchmarking analysis."""
        from src.daemon.tasks.analysis_tasks import PeerAnalysisTask

        await PeerAnalysisTask(self.components, self.container).run()

    def _should_skip_correlation_audit(self, now: datetime) -> bool:
        """Check if correlation audit should be skipped (already ran today)."""
        # NOTE: Requires async state access after JSON elimination
        # TODO: Implement using await self.components.state.get_last_correlation_audit()
        return False

    def _print_correlation_audit_results(self, result: CorrelationAuditResult, duration: float) -> None:
        """Print correlation audit results to console."""
        console.print(f"[dim]Positions: {result.num_positions}[/dim]")
        console.print(f"[dim]Diversification ratio: {result.diversification_ratio:.3f}[/dim]")

        if result.highly_correlated_pairs:
            count = len(result.highly_correlated_pairs)
            console.print(f"\n[bold yellow]Correlated Pairs ({count}):[/bold yellow]")
            for pair in result.highly_correlated_pairs[:5]:
                console.print(f"  {pair.symbol_a} ↔ {pair.symbol_b}: {pair.correlation:.3f}")

        if result.substitution_suggestions:
            count = len(result.substitution_suggestions)
            console.print(f"\n[bold yellow]Substitutions ({count}):[/bold yellow]")
            for suggestion in result.substitution_suggestions[:3]:
                alts = ", ".join(suggestion.alternatives)
                console.print(f"  Replace {suggestion.symbol_to_replace}: {suggestion.reason}")
                console.print(f"    → {alts}")

        if result.warnings:
            console.print(f"\n[dim]Warnings: {', '.join(result.warnings)}[/dim]")

        console.print(f"\n[dim]Complete in {duration:.1f}s[/dim]\n")

    async def run_correlation_audit(self) -> None:
        """Run portfolio correlation audit."""
        from src.daemon.tasks.portfolio_tasks import CorrelationAuditTask

        await CorrelationAuditTask(self.components, self.container).run()

    async def run_tearsheet_generation(self) -> None:
        """Generate performance tearsheet from analysis history."""
        if not self.components.tearsheet_generator:
            return

        from src.daemon.tasks.reporting_tasks import TearsheetTask

        await TearsheetTask(self.components, self.container).run()

    def _log_screening_results(self, results: list) -> None:
        """Log screening results to console.

        Args:
            results: List of ScreeningResult objects (top 5)
        """
        for i, result in enumerate(results, 1):
            console.print(
                f"[bold]{i}. {result.symbol}[/bold] ({result.name}) - Score: {result.score:.2f}\n"
                f"   {result.reason}"
            )

    async def run_daily_risk_report(self) -> None:
        """Generate and persist daily portfolio risk report."""
        if not self.components.config.risk_limits.enabled or not self.components.broker:
            return

        from src.daemon.tasks.reporting_tasks import RiskReportTask

        await RiskReportTask(self.components, self.container).run()

    async def run_signal_tracking(self) -> None:
        """Update signal outcomes with T+1d/5d/20d prices."""
        if not self.components.config.signal_tracking.enabled:
            return

        from src.daemon.tasks.monitoring_tasks import SignalTrackingTask

        await SignalTrackingTask(self.components, self.container).run()

    async def run_monte_carlo_stress_testing(self) -> None:
        """Execute Monte Carlo portfolio stress testing (weekly/daily task)."""
        from src.daemon.tasks.monitoring_tasks import MonteCarloTask

        await MonteCarloTask(self.components, self.container).run()
