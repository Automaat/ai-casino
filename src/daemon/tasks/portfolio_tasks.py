"""Portfolio management tasks for optimization, rebalancing, and correlation audit."""

from __future__ import annotations

import asyncio
import time as time_mod
from datetime import datetime
from typing import TYPE_CHECKING

from loguru import logger
from rich.console import Console

from src.daemon.tasks.base import TaskExecutor

if TYPE_CHECKING:
    from src.daemon.factory import DaemonComponents
    from src.di.container import AppContainer
    from src.metrics.correlation import CorrelationAuditResult

console = Console()


class OptimizationTask(TaskExecutor):
    """Parameter optimization task with event publishing."""

    def __init__(self, components: DaemonComponents, container: AppContainer) -> None:
        """Initialize optimization task.

        Args:
            components: Daemon components
            container: DI container
        """
        super().__init__(components, container)
        self._optimized = []
        self._skipped = []
        self._total_time = 0.0
        self._failed = []

    @property
    def task_name(self) -> str:
        """Task display name."""
        return "Parameter Optimization"

    async def execute(self) -> None:
        """Execute optimization logic."""
        # Type narrowing: daemon_optimizer checked in run_optimization()
        if self.components.daemon_optimizer is None:
            msg = "daemon_optimizer not initialized"
            raise RuntimeError(msg)

        self._publish_event_sync("SCHEDULED_TASK", {"task_name": "optimization", "status": "started"})

        watchlist = self.components.broker_manager.get_merged_watchlist()

        start_time = time_mod.time()
        self._optimized, self._skipped, self._failed = await asyncio.to_thread(
            self.components.daemon_optimizer.optimize_watchlist,
            watchlist=watchlist,
            strategies=self.components.config.optimization.strategies,
            refresh_days=self.components.config.optimization.refresh_days,
        )
        self._total_time = time_mod.time() - start_time

        if self._failed:
            for symbol, strategies_str in self._failed:
                logger.warning(f"Failed to optimize {symbol}: {strategies_str}")

        console.print(
            f"\n[dim]Optimization complete: {len(self._optimized)} symbols optimized, "
            f"{len(self._skipped)} skipped ({self._total_time:.0f}s)[/dim]"
        )

        self._publish_event_sync("SCHEDULED_TASK", {"task_name": "optimization", "status": "completed"})

    def get_last_run(self) -> datetime | None:
        """Get last optimization timestamp."""
        return self.components.state.last_optimization

    def record_success(self, duration: float) -> None:  # noqa: ARG002
        """Record optimization completion."""
        self.components.state.record_optimization(
            symbols_optimized=self._optimized,
            symbols_skipped=self._skipped,
            total_time_seconds=self._total_time,
        )

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


class RebalancingTask(TaskExecutor):
    """Portfolio rebalancing task."""

    @property
    def task_name(self) -> str:
        """Task display name."""
        return "Portfolio Rebalancing"

    async def execute(self) -> None:
        """Execute rebalancing logic."""
        # Type narrowing: daemon_rebalancer checked in run_portfolio_rebalancing()
        if self.components.daemon_rebalancer is None:
            msg = "daemon_rebalancer not initialized"
            raise RuntimeError(msg)

        watchlist = self.components.broker_manager.get_merged_watchlist()
        method = self.components.config.rebalancing.method
        auto_execute = self.components.config.auto_trade

        console.print(f"[dim]Method: {method}, Universe: {len(watchlist)} symbols[/dim]")

        result = await asyncio.to_thread(
            self.components.daemon_rebalancer.run, watchlist, method, auto_execute
        )

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

        # Display summary
        console.print("\n[bold]Portfolio Metrics:[/bold]")
        console.print(f"  Expected Return: {result.optimized_portfolio.expected_return:.2%}")
        console.print(f"  Volatility: {result.optimized_portfolio.expected_volatility:.2%}")
        console.print(f"  Sharpe Ratio: {result.optimized_portfolio.sharpe_ratio:.2f}")

        if result.rebalance_instructions:
            console.print("\n[bold]Rebalancing Instructions:[/bold]")
            for rebalance in result.rebalance_instructions[:10]:
                action_color = (
                    "green" if rebalance.action == "BUY" else "red" if rebalance.action == "SELL" else "dim"
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
            f"{result.pending_count} pending[/dim]"
        )

    def get_last_run(self) -> datetime | None:
        """Get last rebalancing timestamp."""
        return self.components.state.last_portfolio_rebalancing

    def record_success(self, duration: float) -> None:
        """Record rebalancing completion."""
        # State already recorded in execute()


class CorrelationAuditTask(TaskExecutor):
    """Portfolio correlation audit task."""

    def __init__(self, components: DaemonComponents, container: AppContainer) -> None:
        """Initialize correlation audit task.

        Args:
            components: Daemon components
            container: DI container
        """
        super().__init__(components, container)
        self._result: CorrelationAuditResult | None = None
        self._duration = 0.0

    @property
    def task_name(self) -> str:
        """Task display name."""
        return "Portfolio Correlation Audit"

    async def execute(self) -> None:
        """Execute correlation audit logic."""
        from src.metrics.correlation import CorrelationAuditor

        if not self.components.broker:
            logger.warning("No broker configured")
            return

        account_info = await asyncio.to_thread(self.components.broker.get_account_info)
        positions = account_info.positions

        if len(positions) < 2:
            logger.info(f"Insufficient positions ({len(positions)}), need ≥2")
            console.print("[dim]Insufficient positions[/dim]")
            # Mark as run even though skipped
            self.components.state.last_correlation_audit = datetime.now(self.components.scheduler.timezone)
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
        self._result = await asyncio.to_thread(auditor.audit, positions, screening_results)
        self._duration = time_mod.time() - start

        self._print_results(self._result, self._duration)

    def get_last_run(self) -> datetime | None:
        """Get last correlation audit timestamp."""
        return self.components.state.last_correlation_audit

    def record_success(self, duration: float) -> None:  # noqa: ARG002
        """Record correlation audit completion."""
        if self._result:
            self.components.state.record_correlation_audit(
                num_positions=self._result.num_positions,
                num_correlated_pairs=len(self._result.highly_correlated_pairs),
                max_correlation=self._result.max_correlation,
                avg_correlation=self._result.avg_correlation,
                diversification_ratio=self._result.diversification_ratio,
                num_substitutions=len(self._result.substitution_suggestions),
                total_duration_seconds=self._duration,
            )

    def _print_results(self, result: CorrelationAuditResult, duration: float) -> None:
        """Print correlation audit results to console.

        Args:
            result: Correlation audit result
            duration: Task duration in seconds
        """
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

        console.print(f"\n[dim]Complete in {duration:.1f}s[/dim]")
