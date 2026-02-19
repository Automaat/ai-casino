"""Reporting tasks for journals, tearsheets, and risk reports."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger
from rich.console import Console

from src.daemon.tasks.base import TaskExecutor

if TYPE_CHECKING:
    from src.agents.risk.models import PortfolioRiskReport

console = Console()


class JournalTask(TaskExecutor):
    """Trade journal generation task."""

    @property
    def task_name(self) -> str:
        """Task display name."""
        return "Trade Journal"

    async def execute(self) -> None:
        """Execute journal generation logic."""
        # Check time window
        if not self.components.scheduler.is_journal_window(self.components.config.journal.run_offset_minutes):
            return

        # Check daily dedup (custom dedup, not default)
        today = datetime.now(self.components.scheduler.timezone).date()
        last_journal_date = await self.components.state.get_last_journal_date()
        if last_journal_date == today.isoformat():
            return

        # Filter today's analysis records
        analyses = await self.components.state.get_analyses()
        today_records = [r for r in analyses if r.timestamp.date() == today]
        if not today_records:
            logger.info("No analyses today, skipping journal")
            return

        logger.info(f"Generating trade journal for {today} ({len(today_records)} records)")
        console.print(f"\n[bold magenta]Generating trade journal for {today}...[/bold magenta]")

        journal_agent = self.container.trade_journal_agent()

        journal = await journal_agent.generate(today, today_records)
        file_path = await journal_agent.persist(
            journal, self.components.state, self.components.config.journal.journal_dir
        )

        await self.components.state.set_last_journal_date(today.isoformat())

        correct = sum(1 for o in journal.outcomes if o.signal_correct)
        total = len(journal.outcomes)
        if file_path:
            console.print(f"[bold magenta]Journal saved:[/bold magenta] {file_path}")
        if total > 0:
            console.print(f"[bold magenta]Signal accuracy:[/bold magenta] {correct}/{total}")

    async def get_last_run(self) -> datetime | None:
        """Get last journal timestamp."""
        # Custom dedup based on date string, not timestamp
        last_journal_date = await self.components.state.get_last_journal_date()
        if not last_journal_date:
            return None

        # Convert date string to datetime for compatibility with base class
        try:
            date = datetime.fromisoformat(last_journal_date).date()
            return datetime.combine(date, datetime.min.time()).replace(tzinfo=UTC)
        except ValueError, TypeError:
            return None

    async def record_success(self, duration: float) -> None:
        """Record journal completion."""
        # State already recorded in execute()

    async def should_skip_today(self) -> bool:
        """Custom dedup: handled in execute via time window + date check."""
        # Always return False - execute() handles dedup
        return False


class TearsheetTask(TaskExecutor):
    """Performance tearsheet generation task."""

    @property
    def task_name(self) -> str:
        """Task display name."""
        return "Performance Tearsheet Generation"

    async def execute(self) -> None:
        """Execute tearsheet generation logic."""
        if not self.components.tearsheet_generator:
            return

        now = datetime.now(self.components.scheduler.timezone)
        today = now.date()
        analyses = await self.components.state.get_analyses()
        today_analyses = [
            r for r in analyses if r.timestamp.astimezone(self.components.scheduler.timezone).date() == today
        ]

        if not today_analyses:
            logger.info("No analyses today, skipping tearsheet")
            return

        console.print(f"[dim]Generating tearsheet from {len(today_analyses)} analyses...[/dim]")

        tearsheet = await asyncio.to_thread(
            self.components.tearsheet_generator.generate_portfolio_tearsheet,
            analyses=today_analyses,
            benchmark_symbol=self.components.config.reporting.benchmark,
        )

        if tearsheet:
            await asyncio.to_thread(
                self.components.tearsheet_generator.cleanup_old_tearsheets,
                retention_days=self.components.config.reporting.retention_days,
            )

            await self.components.state.record_tearsheet(
                symbol="PORTFOLIO",
                html_path=tearsheet.html_report_path,
            )

            console.print(f"[bold cyan]Tearsheet saved:[/bold cyan] {tearsheet.html_report_path}")
            if tearsheet.sharpe_ratio is not None:
                console.print(f"[bold cyan]Sharpe Ratio:[/bold cyan] {tearsheet.sharpe_ratio:.2f}")
            if tearsheet.cagr is not None:
                console.print(f"[bold cyan]CAGR:[/bold cyan] {tearsheet.cagr:.2%}")
        else:
            logger.info("Insufficient data for tearsheet generation")

        console.print("\n[dim]Tearsheet generation complete[/dim]")

    async def get_last_run(self) -> datetime | None:
        """Get last tearsheet timestamp."""
        return await self.components.state.get_last_tearsheet()

    async def record_success(self, duration: float) -> None:
        """Record tearsheet completion."""
        # State already recorded in execute()


class RiskReportTask(TaskExecutor):
    """Daily portfolio risk report task."""

    @property
    def task_name(self) -> str:
        """Task display name."""
        return "Portfolio Risk Report"

    async def execute(self) -> None:
        """Execute risk report generation logic."""
        if not self.components.broker:
            return

        from src.daemon.state import RiskReportRecord

        account_info = await asyncio.to_thread(self.components.broker.get_account_info)
        workflow = self.components.workflow
        if not workflow:
            logger.warning("Workflow not initialized")
            return

        report = await asyncio.to_thread(
            workflow.risk_manager.generate_risk_report,
            broker_positions=account_info.positions,
            portfolio_value=account_info.portfolio_value,
            total_exposure=account_info.total_exposure,
            lookback_days=self.components.config.risk_limits.lookback_days,
        )

        # Record in database
        await self.components.state.record_risk_report(
            RiskReportRecord(
                timestamp=datetime.now(UTC),
                var_95=report.var_95,
                var_99=report.var_99,
                cvar_95=report.cvar_95,
                cvar_99=report.cvar_99,
                cdar_95=report.cdar_95,
                max_drawdown=report.max_drawdown,
                portfolio_volatility=report.portfolio_volatility,
                current_exposure_percent=report.current_exposure_percent,
                num_positions=report.num_positions,
                var_limit_breached=report.var_limit_breached,
                cvar_limit_breached=report.cvar_limit_breached,
                risk_status=report.risk_status,
            )
        )

        status_color = {"HEALTHY": "green", "WARNING": "yellow", "BREACH": "red"}.get(
            report.risk_status, "white"
        )
        console.print(f"[{status_color}]Risk status: {report.risk_status}[/{status_color}]")
        console.print(f"[dim]VaR95={report.var_95:.4f}, CVaR99={report.cvar_99:.4f}[/dim]")
        logger.info(f"Risk report generated: {report.risk_status}")

        # Send notification if VaR limits breached
        if (report.var_limit_breached or report.cvar_limit_breached) and self.components.notification_service:
            task = asyncio.create_task(self._notify_var_breach(report))

            def _log_var_notification_result(t: asyncio.Task[object]) -> None:
                if t.cancelled():
                    return
                exc = t.exception()
                if exc is not None:
                    logger.opt(exception=exc).error("VaR notification failed")

            task.add_done_callback(_log_var_notification_result)

    async def get_last_run(self) -> datetime | None:
        """Get last risk report timestamp."""
        return await self.components.state.get_last_risk_report()

    async def record_success(self, duration: float) -> None:
        """Record risk report completion."""
        # State already recorded in execute()

    async def _notify_var_breach(self, report: PortfolioRiskReport) -> None:
        """Send notification for VaR breach (helper method).

        Args:
            report: Risk report with VaR metrics
        """
        from src.notifications.models import NotificationMessage, NotificationSeverity

        if not self.components.notification_service:
            return

        message = NotificationMessage(
            title="Risk Limit Breach",
            body=f"VaR95={report.var_95:.2%}, CVaR95={report.cvar_95:.2%}",
            severity=NotificationSeverity.ERROR,
            metadata={
                "var_95": report.var_95,
                "var_99": report.var_99,
                "cvar_95": report.cvar_95,
                "cvar_99": report.cvar_99,
                "risk_status": report.risk_status,
                "positions": report.num_positions,
            },
            timestamp=datetime.now(UTC),
        )
        await self.components.notification_service.notify(message)
