"""Risk report task — generates daily portfolio VaR report and publishes events."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, cast

from loguru import logger
from result import Err

from src.daemon.events import RiskReportEvent, Sentiment, TriageResult, Urgency
from src.daemon.state.models import RiskReportRecord
from src.v1.tasks.interface import Task
from src.v1.tasks.models import WEEKDAYS, DedupStrategy, TaskResult, TaskSchedule

if TYPE_CHECKING:
    from src.agents.risk.agent import RiskManagementAgent
    from src.daemon.config.risk import RiskLimitsConfig
    from src.daemon.scheduler import MarketScheduler
    from src.daemon.state import DaemonState
    from src.v1.event_queue.service import MarketEventQueue
    from src.v1.notifications.service import NotificationService
    from src.v1.trades.brokers.protocol import Broker


class RiskReportTask(Task):
    """Daily portfolio risk report with event publishing for BREACH/WARNING."""

    def __init__(
        self,
        risk_manager: RiskManagementAgent,
        broker: Broker,
        queue: MarketEventQueue | None,
        state: DaemonState,
        scheduler: MarketScheduler,
        config: RiskLimitsConfig,
        notification_service: NotificationService | None = None,
    ) -> None:
        """Initialize risk report task.

        Args:
            risk_manager: Risk management agent for VaR calculations
            broker: Broker for account info
            queue: Event queue for publishing BREACH/WARNING events (optional)
            state: Daemon state for persistence
            scheduler: Market scheduler for next open timing
            config: Risk limits configuration
            notification_service: Optional notification service for breach alerts
        """
        self._risk_manager = risk_manager
        self._broker = broker
        self._queue = queue
        self._state = state
        self._scheduler = scheduler
        self._config = config
        self._notification_service = notification_service

    @property
    def name(self) -> str:
        """Task identifier."""
        return "risk_report"

    @property
    def schedule(self) -> TaskSchedule:
        """Schedule from config."""
        return TaskSchedule(
            time=self._config.report_time,
            days=WEEKDAYS,
            enabled=self._config.enabled,
            dedup=DedupStrategy.DAILY,
        )

    async def execute(self) -> TaskResult:
        """Generate risk report, persist, and publish event if BREACH/WARNING.

        Returns:
            TaskResult with outcome
        """
        start = time.monotonic()

        _result = await asyncio.to_thread(self._broker.get_account_info)
        if isinstance(_result, Err):
            msg = f"Broker API unavailable: {_result.err_value}"
            logger.opt(exception=True).error(msg)
            return TaskResult(
                task_name=self.name, success=False, duration_seconds=time.monotonic() - start, message=msg
            )
        account = _result.ok()
        report = await asyncio.to_thread(
            self._risk_manager.generate_risk_report,
            broker_positions=account.positions,
            portfolio_value=account.portfolio_value,
            total_exposure=account.total_exposure,
            lookback_days=self._config.lookback_days,
        )

        await self._state.record_risk_report(
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

        if report.risk_status in ("BREACH", "WARNING") and self._queue is not None:
            await self._publish_risk_event(report)

        if (report.var_limit_breached or report.cvar_limit_breached) and self._notification_service:
            task = asyncio.create_task(self._notify_var_breach(report))

            def _log_notification_error(t: asyncio.Task[object]) -> None:
                if t.cancelled():
                    return
                exc = t.exception()
                if exc is not None:
                    logger.opt(exception=exc).error("VaR notification failed")

            task.add_done_callback(_log_notification_error)

        duration = time.monotonic() - start
        msg = f"status={report.risk_status}"
        logger.info(f"Risk report complete: {msg}")

        return TaskResult(
            task_name=self.name,
            success=True,
            duration_seconds=duration,
            message=msg,
        )

    async def _publish_risk_event(self, report: object) -> None:
        """Publish BREACH/WARNING to event queue, deferred to next market open.

        Args:
            report: PortfolioRiskReport with risk metrics
        """
        from src.agents.risk.models import PortfolioRiskReport

        if not isinstance(report, PortfolioRiskReport):
            return

        event = RiskReportEvent(
            timestamp=datetime.now(UTC),
            risk_status=cast("Literal['BREACH', 'WARNING']", report.risk_status),
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
        )
        triage = TriageResult(
            event_id=event.event_id,
            event_type="risk_report",
            symbols=[],
            urgency=Urgency.IMMEDIATE,
            sentiment=Sentiment.BEARISH,
            confidence=1.0,
            reasoning=f"Portfolio risk {report.risk_status}: VaR95={report.var_95:.2%}",
            relevance=1.0,
        )
        process_after = self._scheduler.next_regular_open()
        if self._queue is None:
            return
        now = datetime.now(UTC)
        delay_seconds = (process_after - now).total_seconds()
        delay_hours = max(0, int((delay_seconds + 3599) // 3600))
        ttl_hours = max(24, delay_hours + 4)
        await self._queue.enqueue(event, triage, ttl_hours=ttl_hours, process_after=process_after)
        logger.info(
            f"Risk event enqueued: {report.risk_status}, "
            f"process_after={process_after.isoformat()}, ttl_hours={ttl_hours}"
        )

    async def _notify_var_breach(self, report: object) -> None:
        """Send notification for VaR breach.

        Args:
            report: PortfolioRiskReport with VaR metrics
        """
        from src.agents.risk.models import PortfolioRiskReport
        from src.v1.notifications.models import NotificationMessage, NotificationSeverity

        if not isinstance(report, PortfolioRiskReport) or not self._notification_service:
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
        await self._notification_service.notify(message)

    async def last_run_at(self) -> datetime | None:
        """Get last risk report timestamp from state."""
        return await self._state.get_last_risk_report()

    def __repr__(self) -> str:
        """Return string representation."""
        return f"RiskReportTask(enabled={self._config.enabled}, time={self._config.report_time})"
