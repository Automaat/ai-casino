"""Analysis tasks for discovery, sector rotation, and peer analysis."""

from __future__ import annotations

import asyncio
from datetime import datetime

from loguru import logger
from result import Err
from rich.console import Console

from src.daemon.state.models import PeerAnalysisInput
from src.daemon.tasks.base import TaskExecutor

console = Console()


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
                account_result = await asyncio.to_thread(self.components.broker.get_account_info)
                if isinstance(account_result, Err):
                    raise account_result.err_value
                position_symbols = list(account_result.ok().positions.keys())
                flagged = daemon_rotation.flag_weak_positions(position_symbols, analysis)
            except Exception as e:
                logger.opt(exception=True).warning(f"Failed to flag positions: {e}")

        sector_strengths = {s.sector: s.relative_strength for s in analysis.sectors}
        sector_momenta = {s.sector: s.momentum.value for s in analysis.sectors}

        await self.components.state.record_sector_rotation(
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

    async def get_last_run(self) -> datetime | None:
        """Get last sector rotation timestamp."""
        return await self.components.state.get_last_sector_rotation()

    async def record_success(self, duration: float) -> None:
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
            logger.opt(exception=True).error(f"Failed to publish {event_type} event: {e}")


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

        watchlist = await self.components.broker_manager.get_merged_watchlist()
        console.print(f"[dim]Analyzing {len(watchlist)} positions against peers...[/dim]")

        result = await asyncio.to_thread(analyzer.analyze_positions, watchlist)

        # Build state record
        rankings = {a.symbol: a.rank for a in result.analyses}
        swaps = [a.swap_recommendation for a in result.analyses if a.swap_recommendation]

        await self.components.state.record_peer_analysis(
            PeerAnalysisInput(
                symbols_analyzed=[a.symbol for a in result.analyses],
                rankings=rankings,
                swap_recommendations=swaps,
                analyses=[a.model_dump(mode="json") for a in result.analyses],
                total_peers=result.total_peers_analyzed,
                total_duration_seconds=result.total_duration_seconds,
            )
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

    async def get_last_run(self) -> datetime | None:
        """Get last peer analysis timestamp."""
        return await self.components.state.get_last_peer_analysis()

    async def record_success(self, duration: float) -> None:
        """Record peer analysis completion."""
        # State already recorded in execute()
