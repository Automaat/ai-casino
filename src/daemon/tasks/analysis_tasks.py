"""Analysis tasks for game planning, discovery, sector rotation, and peer analysis."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import cast

from loguru import logger
from rich.console import Console

from src.daemon.tasks.base import TaskExecutor

console = Console()


class GamePlanTask(TaskExecutor):
    """Daily game plan generation task."""

    @property
    def task_name(self) -> str:
        """Task display name."""
        return "Game Plan Generation"

    async def execute(self) -> None:
        """Execute game plan generation logic."""
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
        sector_context, _, _, _ = context_builder.build_analysis_contexts(watchlist[0] if watchlist else "")
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

        console.print("[bold green]✓ Game Plan Generated[/bold green]")
        console.print(f"  Risk Stance: {plan.risk_stance}")
        console.print(f"  Priority: {', '.join(plan.priority_symbols)}")
        console.print(f"  Sectors: {', '.join(plan.sector_focus)}")
        console.print(f"  Saved: {plan_path}")

    def get_last_run(self) -> datetime | None:
        """Get last game plan timestamp."""
        return self.components.state.last_game_plan

    def record_success(self, duration: float) -> None:
        """Record game plan completion."""
        # State already recorded in execute()


class DiscoveryTask(TaskExecutor):
    """Stock discovery task with custom time-based dedup."""

    @property
    def task_name(self) -> str:
        """Task display name."""
        return "Stock Discovery"

    async def execute(self) -> None:
        """Execute stock discovery logic."""
        # Type narrowing: discovery_engine checked in run_discovery()
        if self.components.discovery_engine is None:
            msg = "discovery_engine not initialized"
            raise RuntimeError(msg)

        # Get current state
        current_watchlist = self.components.broker_manager.get_merged_watchlist()
        current_positions = {}
        if self.components.broker:
            try:
                account_info = await asyncio.to_thread(self.components.broker.get_account_info)
                current_positions = account_info.positions  # type: ignore[assignment]
            except Exception as e:
                logger.warning(f"Failed to fetch positions: {e}")

        sector_context = None
        if self.components.state.sector_rotation_history:
            sector_context = self.components.state.sector_rotation_history[-1]

        # Run discovery
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

    def get_last_run(self) -> datetime | None:
        """Get last discovery timestamp."""
        return self.components.state.last_discovery

    def record_success(self, duration: float) -> None:
        """Record discovery completion."""
        # State already recorded in execute()

    def should_skip_today(self) -> bool:
        """Custom dedup: check time window + daily.

        Returns:
            True if not within discovery time window or already ran today
        """
        # Check if within discovery time window
        if not self._is_discovery_time():
            return True

        # Check if already ran today
        last_run = self.get_last_run()
        if not last_run:
            return False

        today = datetime.now(self.components.scheduler.timezone).date()
        last_date = last_run.astimezone(self.components.scheduler.timezone).date()
        return last_date == today

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


class SectorRotationTask(TaskExecutor):
    """Sector rotation analysis task with event publishing."""

    @property
    def task_name(self) -> str:
        """Task display name."""
        return "Sector Rotation Analysis"

    async def execute(self) -> None:
        """Execute sector rotation logic."""
        from src.daemon.sector_rotation import DaemonSectorRotation

        self._publish_event_sync("SCHEDULED_TASK", {"task_name": "sector_rotation", "status": "started"})

        daemon_rotation = DaemonSectorRotation()
        analysis = await asyncio.to_thread(daemon_rotation.run)

        flagged: list[str] = []
        if self.components.broker:
            try:
                account_info = await asyncio.to_thread(self.components.broker.get_account_info)
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

        console.print(f"[dim]Leading: {', '.join(analysis.leading_sectors)}[/dim]")
        console.print(f"[dim]Lagging: {', '.join(analysis.lagging_sectors)}[/dim]")
        if flagged:
            console.print(f"[bold yellow]Flagged positions: {', '.join(flagged)}[/bold yellow]")
        console.print(f"\n[dim]Sector rotation complete: {len(analysis.sectors)} sectors analyzed[/dim]")

        self._publish_event_sync("SCHEDULED_TASK", {"task_name": "sector_rotation", "status": "completed"})

    def get_last_run(self) -> datetime | None:
        """Get last sector rotation timestamp."""
        return self.components.state.last_sector_rotation

    def record_success(self, duration: float) -> None:
        """Record sector rotation completion."""
        # State already recorded in execute()

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
                # If already inside event loop, schedule as task
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running loop: safe to use asyncio.run
                asyncio.run(publish_coro)
            else:
                task = loop.create_task(publish_coro)

                def _log_error(t: asyncio.Task[object]) -> None:
                    if t.cancelled():
                        return
                    exc = t.exception()
                    if exc is not None:
                        logger.error(f"Event publish failed: {exc}")

                task.add_done_callback(_log_error)
        except Exception as e:
            logger.error(f"Failed to publish {event_type} event: {e}")


class PeerAnalysisTask(TaskExecutor):
    """Weekly deep peer benchmarking analysis task."""

    @property
    def task_name(self) -> str:
        """Task display name."""
        return "Peer Benchmarking Analysis"

    async def execute(self) -> None:
        """Execute peer analysis logic."""
        from src.daemon.peer_analysis import DeepPeerAnalyzer, PeerAnalyzerConfig

        fundamental_fetcher = self.container.fundamental_fetcher()
        universe_fetcher = self.container.stock_universe_fetcher()
        config = PeerAnalyzerConfig(
            output_dir=self.components.config.peer_analysis.output_dir,
            max_peers=self.components.config.peer_analysis.max_peers,
            rate_limit_sleep=self.components.config.peer_analysis.rate_limit_sleep,
        )
        analyzer = DeepPeerAnalyzer(
            fundamental_fetcher=fundamental_fetcher,
            universe_fetcher=universe_fetcher,
            config=config,
            historical_cache=self.components.historical_cache,
        )

        watchlist = self.components.broker_manager.get_merged_watchlist()
        console.print(f"[dim]Analyzing {len(watchlist)} positions against peers...[/dim]")

        result = await asyncio.to_thread(analyzer.analyze_positions, watchlist)

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
            f"{result.total_peers_analyzed} peers ({result.total_duration_seconds:.0f}s)[/dim]"
        )

    def get_last_run(self) -> datetime | None:
        """Get last peer analysis timestamp."""
        return self.components.state.last_peer_analysis

    def record_success(self, duration: float) -> None:
        """Record peer analysis completion."""
        # State already recorded in execute()
