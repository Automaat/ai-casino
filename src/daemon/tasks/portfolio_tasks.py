"""Portfolio management tasks for optimization, rebalancing, and correlation audit."""

from __future__ import annotations

import asyncio
import time as time_mod
from datetime import datetime
from typing import TYPE_CHECKING, TypedDict

from loguru import logger
from rich.console import Console

from src.daemon.state.managers.portfolio import CorrelationAuditInput, PortfolioRebalancingInput
from src.daemon.state.models import PortfolioHealthRecord
from src.daemon.tasks.base import TaskExecutor

if TYPE_CHECKING:
    from src.daemon.config.portfolio import PortfolioHealthConfig
    from src.daemon.factory import DaemonComponents
    from src.di.container import AppContainer
    from src.metrics.correlation import CorrelationAuditResult


class _PortfolioMetrics(TypedDict):
    total_positions: int
    total_exposure_percent: float
    cash_percent: float
    max_concentration_percent: float
    max_concentration_symbol: str
    total_pnl_percent: float
    biggest_drawdown_percent: float
    biggest_drawdown_symbol: str | None


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

        watchlist = await self.components.broker_manager.get_merged_watchlist()

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

    async def get_last_run(self) -> datetime | None:
        """Get last optimization timestamp."""
        return await self.components.state.get_last_optimization()

    async def record_success(self, duration: float) -> None:
        """Record optimization completion."""
        await self.components.state.record_optimization(
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
            logger.opt(exception=True).error(f"Failed to publish {event_type} event: {e}")


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

        watchlist = await self.components.broker_manager.get_merged_watchlist()
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

        input_data = PortfolioRebalancingInput(
            method=method,
            allocations=allocations,
            expected_return=result.optimized_portfolio.expected_return,
            expected_volatility=result.optimized_portfolio.expected_volatility,
            sharpe_ratio=result.optimized_portfolio.sharpe_ratio,
            rebalances_executed=result.executed_count,
            rebalances_pending=result.pending_count,
        )
        await self.components.state.record_portfolio_rebalancing(input_data)

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

    async def get_last_run(self) -> datetime | None:
        """Get last rebalancing timestamp."""
        return await self.components.state.get_last_portfolio_rebalancing()

    async def record_success(self, duration: float) -> None:
        """Record rebalancing completion."""
        # State already recorded in execute()


class PortfolioHealthCheckTask(TaskExecutor):
    """Portfolio health check task with LLM-powered recommendations."""

    def __init__(self, components: DaemonComponents, container: AppContainer) -> None:
        """Initialize portfolio health check task.

        Args:
            components: Daemon components
            container: DI container
        """
        super().__init__(components, container)
        self._record: PortfolioHealthRecord | None = None

    @property
    def task_name(self) -> str:
        """Task display name."""
        return "Portfolio Health Check"

    async def execute(self) -> None:
        """Execute portfolio health check logic."""
        if not self.components.broker:
            logger.warning("No broker configured for health check")
            return

        account_info = await asyncio.to_thread(self.components.broker.get_account_info)
        positions = account_info.positions
        portfolio_value = float(account_info.portfolio_value)
        cash = float(account_info.available_cash)

        if portfolio_value <= 0:
            logger.warning("Portfolio value is zero, skipping health check")
            return

        config = self.components.config.portfolio_health
        metrics = self._compute_metrics(positions, portfolio_value, cash)
        health_status = self._determine_status(metrics, config)

        # Try LLM analysis, fallback to rule-based
        recommendations, constraints = await self._analyze(
            metrics, health_status, positions, config, portfolio_value
        )

        self._record = PortfolioHealthRecord(
            timestamp=datetime.now(self.components.scheduler.timezone),
            total_positions=metrics["total_positions"],
            portfolio_value=portfolio_value,
            cash_percent=metrics["cash_percent"],
            max_concentration_percent=metrics["max_concentration_percent"],
            max_concentration_symbol=metrics["max_concentration_symbol"],
            total_pnl_percent=metrics["total_pnl_percent"],
            biggest_drawdown_symbol=metrics["biggest_drawdown_symbol"],
            biggest_drawdown_percent=metrics["biggest_drawdown_percent"],
            health_status=health_status,
            recommendations=recommendations,
            constraints=constraints,
        )

        console.print(f"[bold]Status: {health_status}[/bold]")
        console.print(
            f"[dim]Positions: {metrics['total_positions']} | "
            f"Exposure: {metrics['total_exposure_percent']:.1f}% | "
            f"Cash: {metrics['cash_percent']:.1f}%[/dim]"
        )
        if constraints:
            console.print(f"[yellow]Constraints: {', '.join(constraints)}[/yellow]")

    def _compute_metrics(
        self,
        positions: dict,
        portfolio_value: float,
        cash: float,
    ) -> _PortfolioMetrics:
        """Compute portfolio health metrics.

        Args:
            positions: Broker positions dict
            portfolio_value: Total portfolio value
            cash: Available cash

        Returns:
            Dict of computed metrics
        """
        total_exposure = 0.0
        total_unrealized_pnl = 0.0
        max_concentration = 0.0
        max_concentration_symbol = "N/A"
        biggest_drawdown = 0.0
        biggest_drawdown_symbol: str | None = None

        for symbol, pos in positions.items():
            market_value = pos.market_value
            total_exposure += market_value
            total_unrealized_pnl += pos.unrealized_pnl

            concentration = (market_value / portfolio_value) * 100 if portfolio_value > 0 else 0.0
            if concentration > max_concentration:
                max_concentration = concentration
                max_concentration_symbol = symbol

            cost_basis = pos.avg_entry_price * pos.qty
            pnl_pct = (pos.unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0.0
            if pnl_pct < biggest_drawdown:
                biggest_drawdown = pnl_pct
                biggest_drawdown_symbol = symbol

        return {
            "total_positions": len(positions),
            "total_exposure_percent": (
                (total_exposure / portfolio_value * 100) if portfolio_value > 0 else 0.0
            ),
            "cash_percent": (cash / portfolio_value * 100) if portfolio_value > 0 else 100.0,
            "max_concentration_percent": max_concentration,
            "max_concentration_symbol": max_concentration_symbol,
            "total_pnl_percent": (
                (total_unrealized_pnl / portfolio_value * 100) if portfolio_value > 0 else 0.0
            ),
            "biggest_drawdown_percent": biggest_drawdown,
            "biggest_drawdown_symbol": biggest_drawdown_symbol,
        }

    def _determine_status(
        self,
        metrics: _PortfolioMetrics,
        config: PortfolioHealthConfig,
    ) -> str:
        """Determine health status from metrics and config thresholds.

        Args:
            metrics: Computed metrics
            config: PortfolioHealthConfig

        Returns:
            HEALTHY, WARNING, or CRITICAL
        """
        issues = 0

        max_conc = metrics["max_concentration_percent"]
        cash_pct = metrics["cash_percent"]
        drawdown = abs(metrics["biggest_drawdown_percent"])

        if max_conc > config.max_position_concentration * 100:
            issues += 1
        if cash_pct < config.min_cash_percent * 100:
            issues += 1
        if drawdown > config.drawdown_alert_threshold * 100:
            issues += 1

        if issues >= 2:
            return "CRITICAL"
        if issues >= 1:
            return "WARNING"
        return "HEALTHY"

    async def _analyze(
        self,
        metrics: _PortfolioMetrics,
        health_status: str,
        positions: dict,
        config: PortfolioHealthConfig,
        portfolio_value: float,
    ) -> tuple[list[str], list[str]]:
        """Analyze portfolio with LLM, fallback to rule-based.

        Args:
            metrics: Computed metrics
            health_status: HEALTHY/WARNING/CRITICAL
            positions: Broker positions
            config: PortfolioHealthConfig
            portfolio_value: Total portfolio value

        Returns:
            Tuple of (recommendations, constraints)
        """
        try:
            return await self._analyze_with_llm(metrics, health_status, positions, config, portfolio_value)
        except Exception as e:
            logger.opt(exception=True).warning(f"LLM analysis failed, using rules: {e}")
            return self._rule_based_analysis(metrics, health_status, config)

    async def _analyze_with_llm(
        self,
        metrics: _PortfolioMetrics,
        health_status: str,
        positions: dict,
        config: PortfolioHealthConfig,
        portfolio_value: float,
    ) -> tuple[list[str], list[str]]:
        """Use LLM for portfolio health analysis.

        Args:
            metrics: Computed metrics
            health_status: HEALTHY/WARNING/CRITICAL
            positions: Broker positions
            config: PortfolioHealthConfig
            portfolio_value: Total portfolio value

        Returns:
            Tuple of (recommendations, constraints)
        """
        from src.agents.portfolio_health.models import PortfolioHealthLLMResponse
        from src.prompts import PromptLoader

        llm_client = self.container.llm_client()
        prompts = PromptLoader("portfolio_health")

        # Build position details string
        position_lines = []
        for symbol, pos in positions.items():
            cost_basis = pos.avg_entry_price * pos.qty
            pnl_pct = (pos.unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0.0
            position_lines.append(
                f"- {symbol}: ${pos.market_value:,.0f} | P&L: {pnl_pct:+.1f}% | Qty: {pos.qty}"
            )
        position_details = "\n".join(position_lines) if position_lines else "No positions"

        prompt = prompts.load(
            "analyze",
            total_positions=metrics["total_positions"],
            portfolio_value=portfolio_value,
            cash_percent=metrics["cash_percent"],
            max_concentration_symbol=metrics["max_concentration_symbol"],
            max_concentration_percent=metrics["max_concentration_percent"],
            total_pnl_percent=metrics["total_pnl_percent"],
            biggest_drawdown_symbol=metrics["biggest_drawdown_symbol"] or "N/A",
            biggest_drawdown_percent=metrics["biggest_drawdown_percent"],
            max_position_concentration=config.max_position_concentration,
            min_cash_percent=config.min_cash_percent,
            drawdown_alert_threshold=config.drawdown_alert_threshold,
            health_status=health_status,
            position_details=position_details,
        )
        system = prompts.load("system")

        response = await llm_client.astructured(
            prompt, PortfolioHealthLLMResponse, system=system, temperature=0.3, max_tokens=512
        )
        return response.recommendations, response.constraints

    def _rule_based_analysis(
        self,
        metrics: _PortfolioMetrics,
        health_status: str,
        config: PortfolioHealthConfig,
    ) -> tuple[list[str], list[str]]:
        """Rule-based fallback analysis.

        Args:
            metrics: Computed metrics
            health_status: HEALTHY/WARNING/CRITICAL
            config: PortfolioHealthConfig

        Returns:
            Tuple of (recommendations, constraints)
        """
        recommendations: list[str] = []
        constraints: list[str] = []

        max_conc = metrics["max_concentration_percent"]
        max_conc_sym = metrics["max_concentration_symbol"]
        cash_pct = metrics["cash_percent"]
        drawdown = metrics["biggest_drawdown_percent"]
        drawdown_sym = metrics["biggest_drawdown_symbol"]

        if max_conc > config.max_position_concentration * 100:
            recommendations.append(
                f"Reduce {max_conc_sym} concentration from {max_conc:.1f}% "
                f"(threshold: {config.max_position_concentration:.0%})"
            )
            constraints.append(f"reduce:{max_conc_sym}")

        if cash_pct < config.min_cash_percent * 100:
            recommendations.append(
                f"Increase cash reserves from {cash_pct:.1f}% (minimum: {config.min_cash_percent:.0%})"
            )
            constraints.append("block_buy:ALL")

        if drawdown_sym and abs(drawdown) > config.drawdown_alert_threshold * 100:
            recommendations.append(f"Review {drawdown_sym} position ({drawdown:+.1f}% drawdown)")
            constraints.append(f"force_review:{drawdown_sym}")

        if not recommendations:
            recommendations.append("Portfolio health is within all thresholds")

        return recommendations, constraints

    async def get_last_run(self) -> datetime | None:
        """Get last portfolio health check timestamp."""
        return await self.components.state.get_last_portfolio_health()

    async def record_success(self, duration: float) -> None:
        """Record portfolio health check completion."""
        if self._record:
            await self.components.state.record_portfolio_health(self._record)


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
            await self.components.state.set_last_correlation_audit(
                datetime.now(self.components.scheduler.timezone)
            )
            return

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
        self._result = await asyncio.to_thread(auditor.audit, positions, None)
        self._duration = time_mod.time() - start

        self._print_results(self._result, self._duration)

    async def get_last_run(self) -> datetime | None:
        """Get last correlation audit timestamp."""
        return await self.components.state.get_last_correlation_audit()

    async def record_success(self, duration: float) -> None:
        """Record correlation audit completion."""
        if self._result:
            input_data = CorrelationAuditInput(
                num_positions=self._result.num_positions,
                num_correlated_pairs=len(self._result.highly_correlated_pairs),
                max_correlation=self._result.max_correlation,
                avg_correlation=self._result.avg_correlation,
                diversification_ratio=self._result.diversification_ratio,
                num_substitutions=len(self._result.substitution_suggestions),
                total_duration_seconds=self._duration,
            )
            await self.components.state.record_correlation_audit(input_data)

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
