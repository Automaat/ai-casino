"""Data operation tasks for prefetching, screening, and earnings calendar."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING

from loguru import logger
from rich.console import Console

from src.daemon.tasks.base import TaskExecutor

if TYPE_CHECKING:
    from src.daemon.state import SectorRotationRecord
    from src.metrics.sector_rotation import SectorRotationAnalysis
    from src.screening.screener import ScreeningResult

console = Console()


class PrefetchTask(TaskExecutor):
    """After-hours data prefetch task."""

    @property
    def task_name(self) -> str:
        """Task display name."""
        return "Data Prefetch"

    async def execute(self) -> None:
        """Execute prefetch logic."""
        prefetcher = self.components.prefetcher
        if prefetcher is None:
            logger.warning("Prefetcher unavailable (missing ALPHA_VANTAGE_API_KEY), skipping")
            return

        watchlist = await self.components.broker_manager.get_merged_watchlist()

        console.print(f"[dim]Prefetching {len(watchlist)} symbols...[/dim]")
        report = await asyncio.to_thread(prefetcher.prefetch_watchlist, watchlist)

        # Warm FinBERT if configured
        finbert_ready = False
        if self.components.config.prefetch.warm_finbert:
            console.print("[dim]Warming FinBERT model...[/dim]")
            finbert_ready = await asyncio.to_thread(prefetcher.warm_finbert)
        report.finbert_ready = finbert_ready

        # Check API connectivity if configured
        if self.components.config.prefetch.check_connectivity:
            report.api_connectivity = prefetcher.check_api_key_presence()

        # Count successes/failures
        succeeded = sum(1 for r in report.results if r.market_data or r.news or r.fundamentals)
        failed = len(report.results) - succeeded

        # Store for record_success
        self._succeeded = succeeded
        self._failed = failed
        self._finbert_ready = finbert_ready
        self._duration = report.total_duration_seconds

        console.print(
            f"\n[dim]Prefetch complete: {succeeded} symbols cached, "
            f"{failed} failed ({report.total_duration_seconds:.0f}s)[/dim]"
        )
        logger.info(
            f"Data prefetch completed: {succeeded} cached, {failed} failed "
            f"in {report.total_duration_seconds:.0f}s"
        )

    async def get_last_run(self) -> datetime | None:
        """Get last prefetch timestamp."""
        return await self.components.state.get_last_prefetch()

    async def record_success(self, duration: float) -> None:
        """Record prefetch completion."""
        await self.components.state.record_prefetch(
            symbols_prefetched=self._succeeded,
            symbols_failed=self._failed,
            finbert_ready=self._finbert_ready,
            total_duration_seconds=self._duration,
        )


class PreMarketRefreshTask(TaskExecutor):
    """Pre-market data refresh task to update stale cache."""

    @property
    def task_name(self) -> str:
        """Task display name."""
        return "Pre-Market Data Refresh"

    async def execute(self) -> None:
        """Execute pre-market refresh logic."""
        prefetcher = self.components.prefetcher
        if prefetcher is None:
            logger.warning("Prefetcher unavailable (missing ALPHA_VANTAGE_API_KEY), skipping")
            return

        watchlist = await self.components.broker_manager.get_merged_watchlist()

        console.print(f"[dim]Refreshing {len(watchlist)} symbols...[/dim]")
        report = await asyncio.to_thread(prefetcher.prefetch_watchlist, watchlist)

        succeeded = sum(1 for r in report.results if r.market_data or r.news or r.fundamentals)

        console.print(
            f"\n[dim]Pre-market refresh complete: {succeeded} symbols updated "
            f"({report.total_duration_seconds:.0f}s)[/dim]"
        )
        logger.info(
            f"Pre-market refresh completed: {succeeded} symbols in {report.total_duration_seconds:.0f}s"
        )

    async def get_last_run(self) -> datetime | None:
        """Get last pre-market refresh timestamp."""
        return await self.components.state.get_last_pre_market_refresh()

    async def record_success(self, duration: float) -> None:
        """Record pre-market refresh completion."""
        await self.components.state.set_last_pre_market_refresh(
            datetime.now(self.components.scheduler.timezone)
        )


class ScreeningTask(TaskExecutor):
    """After-hours screening task for watchlist candidates."""

    @property
    def task_name(self) -> str:
        """Task display name."""
        return "After-Hours Screening"

    async def execute(self) -> None:
        """Execute screening logic."""
        from src.screening.exporter import ScreeningExporter
        from src.screening.screener import ScreeningCriteria, StockScreener

        # Initialize screener
        universe_fetcher = self.components.container.stock_universe_fetcher()
        screener = StockScreener(universe_fetcher)

        # Parse criteria
        criteria_map = {
            "momentum": ScreeningCriteria.MOMENTUM,
            "value": ScreeningCriteria.VALUE,
            "breakout": ScreeningCriteria.BREAKOUT,
        }
        criteria = criteria_map.get(
            self.components.config.screening.criteria.lower(), ScreeningCriteria.MOMENTUM
        )

        # Run screening
        console.print(
            f"[dim]{criteria.value.title()} Screening[/dim]\n"
            f"[dim]Universe: {self.components.config.screening.universe}[/dim]"
        )
        output = await asyncio.to_thread(
            screener.screen,
            criteria=criteria,
            universe=self.components.config.screening.universe,
            top_n=self.components.config.screening.top_n,
        )

        # Apply sector rotation weighting if available
        results_to_save = output.results
        sector_history = await self.components.state.get_sector_rotation_history(limit=1)
        if self.components.config.sector_rotation.enabled and sector_history:
            try:
                from src.daemon.sector_rotation import DaemonSectorRotation

                # Reconstruct analysis from latest state record
                latest_record = sector_history[-1]
                rotation_analysis = self._reconstruct_rotation_analysis(latest_record)

                daemon_rotation = DaemonSectorRotation()
                results_to_save = daemon_rotation.weight_candidates(
                    output.results,
                    rotation_analysis,
                    self.components.config.sector_rotation.boost_factor,
                )
                logger.info("Applied sector rotation weighting to screening candidates")
            except Exception as e:
                logger.opt(exception=True).warning(f"Failed to apply sector weighting: {e}")

        # Log top 5 to console
        self._log_screening_results(results_to_save[:5])

        # Save to watchlist file
        exporter = ScreeningExporter()
        await asyncio.to_thread(
            exporter.save_to_watchlist,
            results=results_to_save[: self.components.config.screening.top_n],
            criteria=criteria,
            watchlist_name=self.components.config.screening.watchlist_name,
        )

        # Record in state
        await self.components.state.record_after_hours_screening(
            criteria=criteria.value,
            universe=self.components.config.screening.universe,
            candidates=results_to_save,
            top_n=self.components.config.screening.top_n,
            screened_at=output.screened_at,
        )

        console.print(
            f"\n[dim]Top {self.components.config.screening.top_n} candidates saved to daemon state "
            f"({len(output.results)} total screened)[/dim]"
        )
        logger.info(f"After-hours screening completed: {len(output.results)} candidates")

    async def get_last_run(self) -> datetime | None:
        """Get last screening timestamp."""
        return await self.components.state.get_last_after_hours_screening()

    async def record_success(self, duration: float) -> None:
        """Record screening completion."""
        # State already recorded in execute()

    def _reconstruct_rotation_analysis(self, record: SectorRotationRecord) -> SectorRotationAnalysis:
        """Reconstruct SectorRotationAnalysis from state record."""
        from src.metrics.sector_rotation import Momentum, SectorRotationAnalysis, SectorStrength

        # Reconstruct sectors list
        sectors = [
            SectorStrength(
                sector=sector,
                etf="",  # Not stored in record
                return_1w=0.0,  # Not stored in record
                return_1m=0.0,
                return_3m=0.0,
                relative_strength=record.sector_strengths.get(sector, 0.0),
                momentum=Momentum(record.sector_momenta.get(sector, "DECELERATING")),
                rank=0,  # Not stored in record
            )
            for sector in record.sector_strengths
        ]

        return SectorRotationAnalysis(
            sectors=sectors,
            leading_sectors=record.leading_sectors,
            lagging_sectors=record.lagging_sectors,
            spy_return_1w=0.0,  # Not needed for weighting
            spy_return_1m=0.0,
            spy_return_3m=0.0,
            timestamp=record.timestamp,
        )

    def _log_screening_results(self, results: list[ScreeningResult]) -> None:
        """Log top screening results to console."""
        if results:
            console.print("\n[bold]Top candidates:[/bold]")
            for i, result in enumerate(results[:5], 1):
                console.print(
                    f"  {i}. [bold]{result.symbol}[/bold] "
                    f"({result.sector or 'N/A'}) - "
                    f"Score: {result.score:.2f}"
                )


class EarningsFetchTask(TaskExecutor):
    """Weekly earnings calendar fetch task."""

    @property
    def task_name(self) -> str:
        """Task display name."""
        return "Earnings Calendar Fetch"

    async def execute(self) -> None:
        """Execute earnings fetch logic."""
        from src.daemon.earnings import DaemonEarningsCalendar
        from src.daemon.state import EarningsEventRecord

        daemon_earnings = DaemonEarningsCalendar()
        watchlist = await self.components.broker_manager.get_merged_watchlist()

        console.print(f"[dim]Fetching earnings for {len(watchlist)} symbols...[/dim]")
        calendar = await asyncio.to_thread(daemon_earnings.fetch, watchlist)

        # Build event records
        event_records = [
            EarningsEventRecord(
                symbol=e.symbol,
                earnings_date=e.earnings_date.isoformat(),
                estimate_eps=e.estimate_eps,
            )
            for e in calendar.events
        ]

        symbols_with_earnings = len(calendar.events)
        symbols_without_earnings = max(0, len(watchlist) - symbols_with_earnings)
        if symbols_without_earnings:
            logger.info(
                "Earnings calendar: %d symbols with earnings data, %d symbols with no earnings data",
                symbols_with_earnings,
                symbols_without_earnings,
            )

        # NOTE: Missing earnings data is normal, not a failure
        await self.components.state.record_earnings_fetch(
            events=event_records,
            symbols_fetched=symbols_with_earnings,
            symbols_failed=0,  # Only track known fetch failures
        )

        # Show upcoming earnings
        now = datetime.now(self.components.scheduler.timezone)
        upcoming = daemon_earnings.get_upcoming(
            calendar.events, days_ahead=self.components.config.earnings_calendar.lookahead_days
        )
        if upcoming:
            console.print("[bold yellow]Upcoming earnings:[/bold yellow]")
            for event in upcoming:
                days_until = (event.earnings_date - now.date()).days
                console.print(f"  {event.symbol}: {event.earnings_date} ({days_until}d away)")
        else:
            console.print("[dim]No upcoming earnings within lookahead window[/dim]")

        console.print(
            f"\n[dim]Earnings fetch complete: {len(calendar.events)} symbols with earnings data[/dim]"
        )
        logger.info(f"Earnings calendar fetch completed: {len(calendar.events)} events")

    async def get_last_run(self) -> datetime | None:
        """Get last earnings fetch timestamp."""
        return await self.components.state.get_last_earnings_fetch()

    async def record_success(self, duration: float) -> None:
        """Record earnings fetch completion."""
        # State already recorded in execute()

    async def should_skip_today(self) -> bool:
        """Custom dedup: check weekly schedule.

        Returns:
            True if already fetched today or not fetch time
        """
        # Check if already fetched today
        last_run = await self.get_last_run()
        if last_run:
            now = datetime.now(self.components.scheduler.timezone)
            last_date = last_run.astimezone(self.components.scheduler.timezone).date()
            if last_date == now.date():
                return True

        # Check calendar-aware weekly schedule
        if not self.components.scheduler.is_earnings_fetch_time():
            logger.debug("Not earnings fetch time, skipping")
            return True

        return False
