"""Daemon task service for scheduled task execution."""

from __future__ import annotations

import asyncio
import json
import time as time_mod
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from rich.console import Console

from src.daemon.notification_helper import DaemonNotificationHelper

console = Console()

if TYPE_CHECKING:
    from src.daemon.factory import DaemonComponents
    from src.daemon.state import SectorRotationRecord
    from src.di.container import AppContainer
    from src.metrics.correlation import CorrelationAuditResult
    from src.metrics.sector_rotation import SectorRotationAnalysis


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

    async def run_optimization(self) -> None:
        """Run parameter optimization task."""
        if self.components.daemon_optimizer is None:
            return

        now = datetime.now(self.components.scheduler.timezone)
        if self.components.state.last_optimization:
            last_date = self.components.state.last_optimization.astimezone(
                self.components.scheduler.timezone
            ).date()
            if last_date == now.date():
                logger.debug("Optimization already completed today")
                return

        logger.info("Starting after-hours parameter optimization")
        console.print(f"\n[bold cyan]Parameter Optimization ({now:%H:%M})[/bold cyan]")
        console.print("-" * 50)

        self._publish_event_sync("SCHEDULED_TASK", {"task_name": "optimization", "status": "started"})

        try:
            start_time = time_mod.time()
            watchlist = self.components.broker_manager.get_merged_watchlist()

            optimized, skipped, failed = self.components.daemon_optimizer.optimize_watchlist(
                watchlist=watchlist,
                strategies=self.components.config.optimization.strategies,
                refresh_days=self.components.config.optimization.refresh_days,
            )

            total_time = time_mod.time() - start_time

            self.components.state.record_optimization(
                symbols_optimized=optimized,
                symbols_skipped=skipped,
                total_time_seconds=total_time,
            )
            self.components.state.save(self.components.config.state.state_file)

            if failed:
                for symbol, strategies_str in failed:
                    logger.warning(f"Failed to optimize {symbol}: {strategies_str}")

            console.print(
                f"\n[dim]Optimization complete: {len(optimized)} symbols optimized, "
                f"{len(skipped)} skipped ({total_time:.0f}s)[/dim]\n"
            )
            logger.info(f"Parameter optimization completed in {total_time:.0f}s")

            self._publish_event_sync("SCHEDULED_TASK", {"task_name": "optimization", "status": "completed"})

        except Exception as e:
            error_msg = f"Parameter optimization failed: {e}"
            logger.error(error_msg)
            self.components.state.record_error(error_msg)

    async def run_portfolio_rebalancing(self) -> None:
        """Run portfolio rebalancing task."""
        if self.components.daemon_rebalancer is None:
            return

        # Check if already rebalanced today
        now = datetime.now(self.components.scheduler.timezone)
        if self.components.state.last_portfolio_rebalancing:
            last_date = self.components.state.last_portfolio_rebalancing.astimezone(
                self.components.scheduler.timezone
            ).date()
            if last_date == now.date():
                logger.debug("Portfolio rebalancing already completed today")
                return

        logger.info("Starting portfolio rebalancing")
        console.print(f"\n[bold cyan]Portfolio Rebalancing ({now:%H:%M})[/bold cyan]")
        console.print("-" * 50)

        try:
            watchlist = self.components.broker_manager.get_merged_watchlist()
            method = self.components.config.rebalancing.method
            auto_execute = self.components.config.auto_trade

            console.print(f"[dim]Method: {method}, Universe: {len(watchlist)} symbols[/dim]")

            result = self.components.daemon_rebalancer.run(watchlist, method, auto_execute)

            # Convert to state records
            from src.daemon.state import PortfolioAllocationRecord

            allocations = [
                PortfolioAllocationRecord(symbol=alloc.symbol, weight=alloc.weight, action="HOLD", delta=0.0)
                for alloc in result.optimized_portfolio.allocations
            ]

            # Update allocations with rebalance actions
            rebalance_map = {r.symbol: r for r in result.rebalance_instructions}
            for alloc in allocations:
                if alloc.symbol in rebalance_map:
                    rebalance = rebalance_map[alloc.symbol]
                    alloc.action = rebalance.action
                    alloc.delta = rebalance.delta

            self.components.state.record_portfolio_rebalancing(
                method=method,
                allocations=allocations,
                expected_return=result.optimized_portfolio.expected_return,
                expected_volatility=result.optimized_portfolio.expected_volatility,
                sharpe_ratio=result.optimized_portfolio.sharpe_ratio,
                rebalances_executed=result.executed_count,
                rebalances_pending=result.pending_count,
            )
            self.components.state.save(self.components.config.state.state_file)

            # Display summary
            console.print("\n[bold]Portfolio Metrics:[/bold]")
            console.print(f"  Expected Return: {result.optimized_portfolio.expected_return:.2%}")
            console.print(f"  Volatility: {result.optimized_portfolio.expected_volatility:.2%}")
            console.print(f"  Sharpe Ratio: {result.optimized_portfolio.sharpe_ratio:.2f}")

            if result.rebalance_instructions:
                console.print("\n[bold]Rebalancing Instructions:[/bold]")
                for rebalance in result.rebalance_instructions[:10]:
                    action_color = (
                        "green"
                        if rebalance.action == "BUY"
                        else "red"
                        if rebalance.action == "SELL"
                        else "dim"
                    )
                    console.print(
                        f"  [{action_color}]{rebalance.action:4}[/{action_color}] "
                        f"{rebalance.symbol:6} "
                        f"{rebalance.target_weight:6.2%} "
                        f"({rebalance.delta:+.2%})"
                    )

                if len(result.rebalance_instructions) > 10:
                    console.print(f"  [dim]... and {len(result.rebalance_instructions) - 10} more[/dim]")

            console.print(
                f"\n[dim]Rebalancing complete: {result.executed_count} executed, "
                f"{result.pending_count} pending[/dim]\n"
            )
            logger.info(
                f"Portfolio rebalancing completed: {result.executed_count}/"
                f"{len(result.rebalance_instructions)} executed"
            )

        except Exception as e:
            error_msg = f"Portfolio rebalancing failed: {e}"
            logger.error(error_msg)
            self.components.state.record_error(error_msg)

    async def run_game_plan(self) -> None:
        """Run game plan generation task."""
        now = datetime.now(self.components.scheduler.timezone)
        if self.components.state.last_game_plan:
            last_date = self.components.state.last_game_plan.astimezone(
                self.components.scheduler.timezone
            ).date()
            if last_date == now.date():
                logger.debug("Game plan already generated today")
                return

        logger.info("Generating daily game plan")
        console.print(f"\n[bold cyan]Game Plan Generation ({now:%H:%M})[/bold cyan]")
        console.print("-" * 50)

        try:
            # Get or init game plan agent
            if self.components.game_plan_agent is None:
                agent = self.container.game_plan_agent()
            else:
                agent = self.components.game_plan_agent

            watchlist = self.components.broker_manager.get_merged_watchlist()

            # Build contexts via context builder
            context_builder = self.container.context_builder(
                components=self.components,
                container=self.container,
            )
            sector_context, _, _, _ = context_builder.build_analysis_contexts(
                watchlist[0] if watchlist else ""
            )
            earnings_context = context_builder.build_earnings_context_for_watchlist(watchlist)

            plan = await agent.generate(
                watchlist,
                futures_symbols=self.components.config.game_plan.futures_symbols,
                sector_context=sector_context,
                earnings_context=earnings_context,
                timezone=self.components.scheduler.timezone,
            )

            plan_path = agent.persist(plan, self.components.config.game_plan.plan_dir)

            self.components.state.record_game_plan(
                priority_symbols=plan.priority_symbols,
                risk_stance=plan.risk_stance,
                sector_focus=plan.sector_focus,
            )
            self.components.state.save(self.components.config.state.state_file)

            console.print("[bold green]✓ Game Plan Generated[/bold green]")
            console.print(f"  Risk Stance: {plan.risk_stance}")
            console.print(f"  Priority: {', '.join(plan.priority_symbols)}")
            console.print(f"  Sectors: {', '.join(plan.sector_focus)}")
            console.print(f"  Saved: {plan_path}\n")

        except Exception as e:
            error_msg = f"Game plan generation failed: {e}"
            logger.error(error_msg)
            self.components.state.record_error(error_msg)
            console.print(f"[red]✗ {error_msg}[/red]\n")

    async def run_prefetch(self) -> None:
        """Run data prefetch task."""
        if not self.components.config.prefetch.enabled:
            return

        # Dedup check
        now = datetime.now(self.components.scheduler.timezone)
        if self.components.state.last_prefetch:
            last_date = self.components.state.last_prefetch.astimezone(
                self.components.scheduler.timezone
            ).date()
            if last_date == now.date():
                logger.debug("Prefetch already completed today")
                return

        logger.info("Starting after-hours data prefetching")
        console.print(f"\n[bold cyan]Data Prefetch ({now:%H:%M})[/bold cyan]")
        console.print("-" * 50)

        try:
            prefetcher = self.components.prefetcher
            if prefetcher is None:
                logger.warning("Prefetcher unavailable (missing ALPHA_VANTAGE_API_KEY), skipping")
                return

            watchlist = self.components.broker_manager.get_merged_watchlist()

            console.print(f"[dim]Prefetching {len(watchlist)} symbols...[/dim]")
            report = prefetcher.prefetch_watchlist(watchlist)

            # Warm FinBERT if configured
            finbert_ready = False
            if self.components.config.prefetch.warm_finbert:
                console.print("[dim]Warming FinBERT model...[/dim]")
                finbert_ready = prefetcher.warm_finbert()
            report.finbert_ready = finbert_ready

            # Check API connectivity if configured
            if self.components.config.prefetch.check_connectivity:
                report.api_connectivity = prefetcher.check_api_key_presence()

            # Count successes/failures
            succeeded = sum(1 for r in report.results if r.market_data or r.news or r.fundamentals)
            failed = len(report.results) - succeeded

            self.components.state.record_prefetch(
                symbols_prefetched=succeeded,
                symbols_failed=failed,
                finbert_ready=finbert_ready,
                total_duration_seconds=report.total_duration_seconds,
            )
            self.components.state.save(self.components.config.state.state_file)

            console.print(
                f"\n[dim]Prefetch complete: {succeeded} symbols cached, "
                f"{failed} failed ({report.total_duration_seconds:.0f}s)[/dim]\n"
            )
            logger.info(
                f"Data prefetch completed: {succeeded} cached, {failed} failed "
                f"in {report.total_duration_seconds:.0f}s"
            )

        except Exception as e:
            error_msg = f"Data prefetch failed: {e}"
            logger.error(error_msg)
            self.components.state.record_error(error_msg)

    async def run_discovery(self) -> None:
        """Run stock discovery task."""
        if not self.components.config.discovery.enabled or not self.components.discovery_engine:
            return

        if not self._is_discovery_time():
            return

        # Check if already ran today
        today = datetime.now(self.components.scheduler.timezone).date()
        if (
            self.components.state.last_discovery
            and self.components.state.last_discovery.astimezone(self.components.scheduler.timezone).date()
            == today
        ):
            return

        logger.info("Running stock discovery")
        console.print("\n[bold cyan]🔍 Running Stock Discovery...[/bold cyan]")

        try:
            # Get current state
            current_watchlist = self.components.broker_manager.get_merged_watchlist()
            current_positions = {}
            if self.components.broker:
                try:
                    account_info = self.components.broker.get_account_info()
                    current_positions = account_info.positions  # type: ignore[assignment]
                except Exception as e:
                    logger.warning(f"Failed to fetch positions: {e}")

            sector_context = None
            if self.components.state.sector_rotation_history:
                sector_context = self.components.state.sector_rotation_history[-1]

            # Run discovery
            from typing import cast

            result = await self.components.discovery_engine.discover(
                current_watchlist=current_watchlist,
                current_positions=cast("dict[str, object]", current_positions),
                sector_context=sector_context,
            )

            # Add top N to watchlist
            max_new = self.components.config.discovery.max_discovered_per_cycle
            added_candidates = result.candidates[:max_new]
            added_symbols = [c.symbol for c in added_candidates]

            self.components.state.record_discovery(result.candidates, added_symbols)
            self.components.state.last_discovery = datetime.now(UTC)
            self.components.state.save(self.components.config.state.state_file)

            console.print(
                f"[bold green]✓[/bold green] Discovery: "
                f"{len(result.candidates)} candidates, {len(added_symbols)} added"
            )
            logger.info(
                f"Discovery: {result.total_discovered} discovered, "
                f"{result.filtered_count} filtered, {len(added_symbols)} added"
            )

            # Log source breakdown
            for source, count in result.source_breakdown.items():
                logger.debug(f"  {source}: {count} candidates")

        except Exception as e:
            logger.error(f"Discovery failed: {e}", exc_info=True)
            self.components.state.record_error(f"Discovery failed: {e}")

    async def run_health_check(self) -> None:
        """Run health check task."""
        if not self.components.config.health.enabled:
            return

        now = datetime.now(tz=UTC)

        # Run on first startup or after interval elapsed
        if self.components.state.last_health_check:
            elapsed = (now - self.components.state.last_health_check).total_seconds()
            if elapsed < self.components.config.health.check_interval_seconds:
                return

        logger.info("Starting API health checks")
        console.print(f"\n[bold cyan]Running Health Checks ({datetime.now(tz=UTC):%H:%M})[/bold cyan]")

        try:
            from src.daemon.health import HealthChecker

            checker = HealthChecker(
                self.components.config,
                self.components.state,
                container=self.container,
                notification_service=self.components.notification_service,
            )
            report = await checker.run()

            self.components.state.last_health_check = datetime.now(tz=self.components.scheduler.timezone)
            self.components.state.save(self.components.config.state.state_file)

            console.print(
                f"[bold cyan]Health:[/bold cyan] {report.overall_status} "
                f"({len(report.service_checks)} services, {report.total_duration_ms:.0f}ms)"
            )
            logger.info(f"Health check complete: {report.overall_status}")

            # Publish HEALTH_CHECK event
            if self.components.event_bus:
                try:
                    from src.daemon.event_bus import DashboardEvent, EventType

                    failures = [
                        svc.service_name for svc in report.service_checks if svc.status == "UNHEALTHY"
                    ]
                    await self.components.event_bus.publish(
                        DashboardEvent(
                            event_type=EventType.HEALTH_CHECK,
                            data={
                                "status": report.overall_status.value,
                                "failures": failures,
                                "total_duration_ms": report.total_duration_ms,
                            },
                        )
                    )
                except Exception as e:
                    logger.error(f"Failed to publish HEALTH_CHECK event: {e}")

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self.components.state.record_error(f"Health check failed: {e}")

    async def run_journal(self) -> None:
        """Run trade journal task."""
        if not self.components.config.journal.enabled:
            return

        if not self.components.scheduler.is_journal_window(self.components.config.journal.run_offset_minutes):
            return

        today = datetime.now(self.components.scheduler.timezone).date()
        if self.components.state.last_journal_date == today.isoformat():
            return

        # Filter today's analysis records
        today_records = [r for r in self.components.state.analyses if r.timestamp.date() == today]
        if not today_records:
            logger.info("No analyses today, skipping journal")
            return

        logger.info(f"Generating trade journal for {today} ({len(today_records)} records)")
        console.print(f"\n[bold magenta]Generating trade journal for {today}...[/bold magenta]")

        try:
            journal_agent = self.container.trade_journal_agent()

            journal = await journal_agent.generate(today, today_records)
            file_path = journal_agent.persist(journal, self.components.config.journal.journal_dir)

            self.components.state.last_journal_date = today.isoformat()
            self.components.state.save(self.components.config.state.state_file)

            correct = sum(1 for o in journal.outcomes if o.signal_correct)
            total = len(journal.outcomes)
            console.print(f"[bold magenta]Journal saved:[/bold magenta] {file_path}")
            if total > 0:
                console.print(f"[bold magenta]Signal accuracy:[/bold magenta] {correct}/{total}")
        except Exception as e:
            logger.error(f"Journal generation failed: {e}")
            self.components.state.record_error(f"Journal failed: {e}")
            self.components.state.save(self.components.config.state.state_file)

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
            report = validator.assess_readiness()

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
        import asyncio

        from src.daemon.event_bus import DashboardEvent, EventType

        if not self.components.event_bus:
            return

        try:
            asyncio.run(
                self.components.event_bus.publish(DashboardEvent(event_type=EventType[event_type], data=data))
            )
        except Exception as e:
            logger.error(f"Failed to publish {event_type} event: {e}")

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
                logger.warning(f"Unknown sector {sector_name}, skipping")
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

    def run_pre_market_refresh(self) -> None:
        """Run pre-market data refresh to update stale cache."""
        if (
            not self.components.config.prefetch.enabled
            or not self.components.config.prefetch.enable_pre_market_refresh
        ):
            return

        # Dedup check
        now = datetime.now(self.components.scheduler.timezone)
        if self.components.state.last_pre_market_refresh:
            last_date = self.components.state.last_pre_market_refresh.astimezone(
                self.components.scheduler.timezone
            ).date()
            if last_date == now.date():
                logger.debug("Pre-market refresh already completed today")
                return

        logger.info("Starting pre-market data refresh")
        console.print(f"\n[bold cyan]Pre-Market Data Refresh ({now:%H:%M})[/bold cyan]")
        console.print("-" * 50)

        try:
            prefetcher = self.components.prefetcher
            if prefetcher is None:
                logger.warning("Prefetcher unavailable (missing ALPHA_VANTAGE_API_KEY), skipping")
                return

            watchlist = self.components.broker_manager.get_merged_watchlist()

            console.print(f"[dim]Refreshing {len(watchlist)} symbols...[/dim]")
            report = prefetcher.prefetch_watchlist(watchlist)

            succeeded = sum(1 for r in report.results if r.market_data or r.news or r.fundamentals)

            self.components.state.last_pre_market_refresh = datetime.now(self.components.scheduler.timezone)
            self.components.state.save(self.components.config.state.state_file)

            console.print(
                f"\n[dim]Pre-market refresh complete: {succeeded} symbols updated "
                f"({report.total_duration_seconds:.0f}s)[/dim]\n"
            )
            logger.info(
                f"Pre-market refresh completed: {succeeded} symbols in {report.total_duration_seconds:.0f}s"
            )

        except Exception as e:
            error_msg = f"Pre-market refresh failed: {e}"
            logger.error(error_msg)
            self.components.state.record_error(error_msg)

    def run_after_hours_screening(self) -> None:
        """Run after-hours screening for watchlist candidates."""
        from src.data.universe import StockUniverseFetcher
        from src.screening.exporter import ScreeningExporter
        from src.screening.screener import ScreeningCriteria, StockScreener

        # Check if already screened today
        now = datetime.now(self.components.scheduler.timezone)
        if self.components.state.last_after_hours_screening:
            last_date = self.components.state.last_after_hours_screening.astimezone(
                self.components.scheduler.timezone
            ).date()
            if last_date == now.date():
                logger.debug("After-hours screening already completed today")
                return

        logger.info("Starting after-hours watchlist screening")
        console.print(f"\n[bold cyan]After-Hours Screening ({now:%H:%M})[/bold cyan]")
        console.print("-" * 50)

        try:
            # Initialize screener
            universe_fetcher = StockUniverseFetcher()
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
            output = screener.screen(
                criteria=criteria,
                universe=self.components.config.screening.universe,
                top_n=self.components.config.screening.top_n,
            )

            # Apply sector rotation weighting if available
            results_to_save = output.results
            if (
                self.components.config.sector_rotation.enabled
                and self.components.state.sector_rotation_history
            ):
                try:
                    from src.daemon.sector_rotation import DaemonSectorRotation

                    # Reconstruct analysis from latest state record
                    latest_record = self.components.state.sector_rotation_history[-1]
                    rotation_analysis = self._reconstruct_rotation_analysis(latest_record)

                    daemon_rotation = DaemonSectorRotation()
                    results_to_save = daemon_rotation.weight_candidates(
                        output.results,
                        rotation_analysis,
                        self.components.config.sector_rotation.boost_factor,
                    )
                    logger.info("Applied sector rotation weighting to screening candidates")
                except Exception as e:
                    logger.warning(f"Failed to apply sector weighting: {e}")

            # Log top 5 to console
            self._log_screening_results(results_to_save[:5])

            # Save to watchlist file
            exporter = ScreeningExporter()
            exporter.save_to_watchlist(
                results=results_to_save[: self.components.config.screening.top_n],
                criteria=criteria,
                watchlist_name=self.components.config.screening.watchlist_name,
            )

            # Record in state
            self.components.state.record_after_hours_screening(
                criteria=criteria.value,
                universe=self.components.config.screening.universe,
                candidates=results_to_save,
                top_n=self.components.config.screening.top_n,
                screened_at=output.screened_at,
            )
            self.components.state.save(self.components.config.state.state_file)

            console.print(
                f"\n[dim]Top {self.components.config.screening.top_n} candidates saved to daemon state "
                f"({len(output.results)} total screened)[/dim]\n"
            )
            logger.info(f"After-hours screening completed: {len(output.results)} candidates")

        except Exception as e:
            error_msg = f"After-hours screening failed: {e}"
            logger.error(error_msg)
            self.components.state.record_error(error_msg)

    def run_sector_rotation(self) -> None:
        """Run sector rotation analysis."""
        from src.daemon.sector_rotation import DaemonSectorRotation

        now = datetime.now(self.components.scheduler.timezone)
        if self.components.state.last_sector_rotation:
            last_date = self.components.state.last_sector_rotation.astimezone(
                self.components.scheduler.timezone
            ).date()
            if last_date == now.date():
                logger.debug("Sector rotation already completed today")
                return

        logger.info("Starting sector rotation analysis")
        console.print(f"\n[bold cyan]Sector Rotation Analysis ({now:%H:%M})[/bold cyan]")
        console.print("-" * 50)

        self._publish_event_sync("SCHEDULED_TASK", {"task_name": "sector_rotation", "status": "started"})

        try:
            daemon_rotation = DaemonSectorRotation()
            analysis = daemon_rotation.run()

            flagged: list[str] = []
            if self.components.broker:
                try:
                    account_info = self.components.broker.get_account_info()
                    position_symbols = list(account_info.positions.keys())
                    flagged = daemon_rotation.flag_weak_positions(position_symbols, analysis)
                except Exception as e:
                    logger.warning(f"Failed to flag positions: {e}")

            sector_strengths = {s.sector: s.relative_strength for s in analysis.sectors}
            sector_momenta = {s.sector: s.momentum.value for s in analysis.sectors}

            self.components.state.record_sector_rotation(
                leading_sectors=analysis.leading_sectors,
                lagging_sectors=analysis.lagging_sectors,
                sector_strengths=sector_strengths,
                sector_momenta=sector_momenta,
                flagged_positions=flagged,
            )
            self.components.state.save(self.components.config.state.state_file)

            console.print(f"[dim]Leading: {', '.join(analysis.leading_sectors)}[/dim]")
            console.print(f"[dim]Lagging: {', '.join(analysis.lagging_sectors)}[/dim]")
            if flagged:
                console.print(f"[bold yellow]Flagged positions: {', '.join(flagged)}[/bold yellow]")
            console.print(
                f"\n[dim]Sector rotation complete: {len(analysis.sectors)} sectors analyzed[/dim]\n"
            )
            logger.info("Sector rotation analysis completed")

            self._publish_event_sync(
                "SCHEDULED_TASK", {"task_name": "sector_rotation", "status": "completed"}
            )

        except Exception as e:
            error_msg = f"Sector rotation failed: {e}"
            logger.error(error_msg)
            self.components.state.record_error(error_msg)

    def run_earnings_fetch(self) -> None:
        """Run earnings calendar fetch for watchlist symbols."""
        from src.daemon.earnings import DaemonEarningsCalendar
        from src.daemon.state import EarningsEventRecord

        # Weekly dedup: check if already fetched this week on a configured day
        now = datetime.now(self.components.scheduler.timezone)
        if self.components.state.last_earnings_fetch:
            last_date = self.components.state.last_earnings_fetch.astimezone(
                self.components.scheduler.timezone
            ).date()
            # Skip if already fetched today
            if last_date == now.date():
                logger.debug("Earnings calendar already fetched today")
                return

        # Check calendar-aware weekly schedule
        if not self.components.scheduler.is_earnings_fetch_time():
            logger.debug("Not earnings fetch time, skipping")
            return

        logger.info("Starting earnings calendar fetch")
        console.print(f"\n[bold cyan]Earnings Calendar Fetch ({now:%H:%M})[/bold cyan]")
        console.print("-" * 50)

        try:
            daemon_earnings = DaemonEarningsCalendar()
            watchlist = self.components.broker_manager.get_merged_watchlist()

            console.print(f"[dim]Fetching earnings for {len(watchlist)} symbols...[/dim]")
            calendar = daemon_earnings.fetch(watchlist)

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
            self.components.state.record_earnings_fetch(
                events=event_records,
                symbols_fetched=symbols_with_earnings,
                symbols_failed=0,  # Only track known fetch failures
            )
            self.components.state.save(self.components.config.state.state_file)

            # Show upcoming earnings
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
                f"\n[dim]Earnings fetch complete: {len(calendar.events)} symbols with earnings data[/dim]\n"
            )
            logger.info(f"Earnings calendar fetch completed: {len(calendar.events)} events")

        except Exception as e:
            error_msg = f"Earnings calendar fetch failed: {e}"
            logger.error(error_msg)
            self.components.state.record_error(error_msg)

    def run_peer_analysis(self) -> None:
        """Run weekly deep peer benchmarking analysis."""
        from src.daemon.peer_analysis import DeepPeerAnalyzer

        # Dedup check
        now = datetime.now(self.components.scheduler.timezone)
        if self.components.state.last_peer_analysis:
            last_date = self.components.state.last_peer_analysis.astimezone(
                self.components.scheduler.timezone
            ).date()
            if last_date == now.date():
                logger.debug("Peer analysis already completed today")
                return

        logger.info("Starting deep peer benchmarking analysis")
        console.print(f"\n[bold cyan]Peer Benchmarking Analysis ({now:%H:%M})[/bold cyan]")
        console.print("-" * 50)

        try:
            fundamental_fetcher = self.container.fundamental_fetcher()
            universe_fetcher = self.container.stock_universe_fetcher()
            analyzer = DeepPeerAnalyzer(
                fundamental_fetcher=fundamental_fetcher,
                universe_fetcher=universe_fetcher,
                output_dir=self.components.config.peer_analysis.output_dir,
                max_peers=self.components.config.peer_analysis.max_peers,
                rate_limit_sleep=self.components.config.peer_analysis.rate_limit_sleep,
                historical_cache=self.components.historical_cache,
            )

            watchlist = self.components.broker_manager.get_merged_watchlist()
            console.print(f"[dim]Analyzing {len(watchlist)} positions against peers...[/dim]")

            result = analyzer.analyze_positions(watchlist)

            # Build state record
            rankings = {a.symbol: a.rank for a in result.analyses}
            swaps = [a.swap_recommendation for a in result.analyses if a.swap_recommendation]

            self.components.state.record_peer_analysis(
                symbols_analyzed=[a.symbol for a in result.analyses],
                rankings=rankings,
                swap_recommendations=swaps,
                total_peers=result.total_peers_analyzed,
                total_duration_seconds=result.total_duration_seconds,
            )
            self.components.state.save(self.components.config.state.state_file)

            # Console output
            for analysis in result.analyses:
                rank_color = "green" if analysis.rank <= 3 else "yellow" if analysis.rank <= 5 else "red"
                console.print(
                    f"  [bold]{analysis.symbol}[/bold]: "
                    f"[{rank_color}]#{analysis.rank}[/{rank_color}] of {analysis.peer_count} "
                    f"in {analysis.sector}"
                )
            if swaps:
                console.print(f"[bold yellow]Swap recommendations: {len(swaps)}[/bold yellow]")
                for swap in swaps:
                    console.print(f"  {swap}")

            console.print(
                f"\n[dim]Peer analysis complete: {len(result.analyses)} positions, "
                f"{result.total_peers_analyzed} peers ({result.total_duration_seconds:.0f}s)[/dim]\n"
            )
            logger.info("Deep peer benchmarking analysis completed")

        except Exception as e:
            error_msg = f"Peer benchmarking analysis failed: {e}"
            logger.error(error_msg)
            self.components.state.record_error(error_msg)

    def _should_skip_correlation_audit(self, now: datetime) -> bool:
        """Check if correlation audit should be skipped (already ran today)."""
        if not self.components.state.last_correlation_audit:
            return False
        last_date = self.components.state.last_correlation_audit.astimezone(
            self.components.scheduler.timezone
        ).date()
        return last_date == now.date()

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

    def run_correlation_audit(self) -> None:
        """Run portfolio correlation audit."""
        from src.metrics.correlation import CorrelationAuditor

        now = datetime.now(self.components.scheduler.timezone)
        if self._should_skip_correlation_audit(now):
            logger.debug("Correlation audit already run today")
            return

        logger.info("Starting portfolio correlation audit")
        console.print(f"\n[bold cyan]Portfolio Correlation Audit ({now:%H:%M})[/bold cyan]")
        console.print("-" * 50)

        try:
            if not self.components.broker:
                logger.warning("No broker configured")
                return

            account_info = self.components.broker.get_account_info()
            positions = account_info.positions

            if len(positions) < 2:
                logger.info(f"Insufficient positions ({len(positions)}), need ≥2")
                console.print("[dim]Insufficient positions[/dim]\n")
                self.components.state.last_correlation_audit = now
                return

            screening_results = (
                self.components.state.screening_history[-1].candidates
                if self.components.state.screening_history
                else None
            )

            workflow = self.components.workflow
            if not workflow:
                logger.warning("Workflow not initialized")
                return

            auditor = CorrelationAuditor(
                market_fetcher=workflow.market_fetcher,
                correlation_threshold=self.components.config.correlation_audit.correlation_threshold,
                lookback_days=self.components.config.correlation_audit.lookback_days,
                output_dir=self.components.config.correlation_audit.output_dir,
            )

            start = time_mod.time()
            result = auditor.audit(positions, screening_results)
            duration = time_mod.time() - start

            self.components.state.record_correlation_audit(
                num_positions=result.num_positions,
                num_correlated_pairs=len(result.highly_correlated_pairs),
                max_correlation=result.max_correlation,
                avg_correlation=result.avg_correlation,
                diversification_ratio=result.diversification_ratio,
                num_substitutions=len(result.substitution_suggestions),
                total_duration_seconds=duration,
            )
            self.components.state.save(self.components.config.state.state_file)

            self._print_correlation_audit_results(result, duration)

        except Exception as e:
            error_msg = f"Correlation audit failed: {e}"
            logger.error(error_msg)
            self.components.state.record_error(error_msg)

    def run_tearsheet_generation(self) -> None:
        """Generate performance tearsheet from analysis history."""
        if not self.components.tearsheet_generator:
            return

        # Check if already generated today
        now = datetime.now(self.components.scheduler.timezone)
        if self.components.state.last_tearsheet:
            last_date = self.components.state.last_tearsheet.astimezone(
                self.components.scheduler.timezone
            ).date()
            if last_date == now.date():
                logger.debug("Tearsheet already generated today")
                return

        logger.info("Starting tearsheet generation")
        console.print(f"\n[bold cyan]Performance Tearsheet Generation ({now:%H:%M})[/bold cyan]")
        console.print("-" * 50)

        try:
            today = now.date()
            today_analyses = [
                r
                for r in self.components.state.analyses
                if r.timestamp.astimezone(self.components.scheduler.timezone).date() == today
            ]

            if not today_analyses:
                logger.info("No analyses today, skipping tearsheet")
                return

            console.print(f"[dim]Generating tearsheet from {len(today_analyses)} analyses...[/dim]")

            tearsheet = self.components.tearsheet_generator.generate_portfolio_tearsheet(
                analyses=today_analyses,
                benchmark_symbol=self.components.config.reporting.benchmark,
            )

            if tearsheet:
                self.components.tearsheet_generator.cleanup_old_tearsheets(
                    retention_days=self.components.config.reporting.retention_days
                )

                self.components.state.record_tearsheet(
                    symbol="PORTFOLIO",
                    html_path=tearsheet.html_report_path,
                )
                self.components.state.save(self.components.config.state.state_file)

                console.print(f"[bold cyan]Tearsheet saved:[/bold cyan] {tearsheet.html_report_path}")
                if tearsheet.sharpe_ratio is not None:
                    console.print(f"[bold cyan]Sharpe Ratio:[/bold cyan] {tearsheet.sharpe_ratio:.2f}")
                if tearsheet.cagr is not None:
                    console.print(f"[bold cyan]CAGR:[/bold cyan] {tearsheet.cagr:.2%}")
            else:
                logger.info("Insufficient data for tearsheet generation")

            console.print("\n[dim]Tearsheet generation complete[/dim]\n")

        except Exception as e:
            error_msg = f"Tearsheet generation failed: {e}"
            logger.error(error_msg)
            self.components.state.record_error(error_msg)

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

    def run_daily_risk_report(self) -> None:
        """Generate and persist daily portfolio risk report."""
        if not self.components.config.risk_limits.enabled or not self.components.broker:
            return

        # Dedup: only run once per day
        now = datetime.now(self.components.scheduler.timezone)
        if self.components.state.last_risk_report:
            last_date = self.components.state.last_risk_report.astimezone(
                self.components.scheduler.timezone
            ).date()
            if last_date == now.date():
                logger.debug("Risk report already generated today")
                return

        logger.info("Generating daily portfolio risk report")
        console.print(f"\n[bold cyan]Portfolio Risk Report ({now:%H:%M})[/bold cyan]")
        console.print("-" * 50)

        try:
            from src.daemon.state import RiskReportRecord

            account_info = self.components.broker.get_account_info()
            workflow = self.components.workflow
            if not workflow:
                logger.warning("Workflow not initialized")
                return

            report = workflow.risk_manager.generate_risk_report(
                broker_positions=account_info.positions,
                portfolio_value=account_info.portfolio_value,
                total_exposure=account_info.total_exposure,
                lookback_days=self.components.config.risk_limits.lookback_days,
            )

            # Persist to JSON file
            report_dir = Path(self.components.config.risk_limits.report_dir).expanduser()
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / f"risk-report-{report.date}.json"
            with report_path.open("w") as f:
                json.dump(report.model_dump(), f, indent=2)

            # Record in state
            self.components.state.record_risk_report(
                RiskReportRecord(
                    timestamp=datetime.now(UTC),
                    var_95=report.var_95,
                    var_99=report.var_99,
                    cvar_95=report.cvar_95,
                    cvar_99=report.cvar_99,
                    cdar_95=report.cdar_95,
                    max_drawdown=report.max_drawdown,
                    risk_status=report.risk_status,
                )
            )
            self.components.state.save(self.components.config.state.state_file)

            status_color = {"HEALTHY": "green", "WARNING": "yellow", "BREACH": "red"}.get(
                report.risk_status, "white"
            )
            console.print(f"[{status_color}]Risk status: {report.risk_status}[/{status_color}]")
            console.print(f"[dim]VaR95={report.var_95:.4f}, CVaR99={report.cvar_99:.4f}[/dim]")
            console.print(f"[dim]Report saved: {report_path}[/dim]\n")
            logger.info(f"Risk report generated: {report.risk_status}")

            # Send notification if VaR limits breached
            if (
                report.var_limit_breached or report.cvar_limit_breached
            ) and self.components.notification_service:
                task = asyncio.create_task(
                    self._notification_helper.notify_var_breach(report, self.components)
                )
                _ = task  # Suppress RUF006

        except Exception as e:
            error_msg = f"Risk report generation failed: {e}"
            logger.error(error_msg)
            self.components.state.record_error(error_msg)

    def run_signal_tracking(self) -> None:
        """Update signal outcomes with T+1d/5d/20d prices."""
        if not self.components.config.signal_tracking.enabled:
            return

        # Dedup: check if already ran today
        now = datetime.now(self.components.scheduler.timezone)
        if self.components.state.last_signal_tracking:
            last_date = self.components.state.last_signal_tracking.astimezone(
                self.components.scheduler.timezone
            ).date()
            if last_date == now.date():
                logger.debug("Signal tracking already completed today")
                return

        console.print(f"\n[bold cyan]Running Signal Tracking ({now:%H:%M})[/bold cyan]")

        try:
            from src.daemon.signal_tracker import SignalOutcomeTracker

            market_fetcher = self.container.yfinance_market_fetcher()
            tracker = SignalOutcomeTracker(
                self.components.historical_cache, market_fetcher, self.components.broker
            )
            stats = tracker.update_outcomes()

            self.components.state.last_signal_tracking = datetime.now(UTC)
            self.components.state.save(self.components.config.state.state_file)

            console.print(f"[dim]Signal tracking: {stats}[/dim]\n")
            logger.info(f"Signal tracking completed: {stats}")
        except Exception as e:
            error_msg = f"Signal tracking failed: {e}"
            logger.error(error_msg)
            self.components.state.record_error(error_msg)

    def run_monte_carlo_stress_testing(self) -> None:
        """Execute Monte Carlo portfolio stress testing (weekly/daily task)."""
        logger.info("[MONTE CARLO] Starting stress test")

        # Deduplication (check last run within 6 hours)
        if self.components.state.monte_carlo_tests:
            last_run = self.components.state.monte_carlo_tests[-1].timestamp
            now = datetime.now(UTC)
            if (now - last_run).total_seconds() < 6 * 3600:
                logger.info("[MONTE CARLO] Already ran recently, skipping")
                return

        try:
            from src.daemon.stress_testing import DaemonStressTester

            if self.components.broker is None or self.components.market_fetcher is None:
                logger.warning("[MONTE CARLO] Skipping: broker or market_fetcher not configured")
                return

            executor = DaemonStressTester(
                broker_client=self.components.broker,
                market_fetcher=self.components.market_fetcher,
                config=self.components.config.monte_carlo,
            )
            record = executor.execute()

            self.components.state.record_monte_carlo_test(
                record, self.components.config.monte_carlo.max_history_records
            )
            self.components.state.save(self.components.config.state.state_file)

            if record.exceeds_risk_tolerance:
                logger.warning(f"[MONTE CARLO] ALERT: {record.alert_message}")
            else:
                logger.info(
                    f"[MONTE CARLO] Test passed - P(loss>threshold)={record.prob_loss_gt_threshold:.1%}, "
                    f"VaR95={record.var_95:.1%}"
                )
        except Exception as e:
            logger.error(f"[MONTE CARLO] Stress test failed: {e}")
            self.components.state.record_error(f"Monte Carlo stress test error: {e}")
