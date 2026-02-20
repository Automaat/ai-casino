"""Notification helper for daemon operations."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from src.agents.risk import PortfolioRiskReport
    from src.daemon.degradation import DegradationContext
    from src.daemon.factory import DaemonComponents
    from src.workflows.types import TradingWorkflowResult


class DaemonNotificationHelper:
    """Helper for sending daemon notifications."""

    def __repr__(self) -> str:
        """Return string representation."""
        return "DaemonNotificationHelper()"

    async def notify_degradation(
        self,
        degradation_context: DegradationContext,
        components: DaemonComponents,
    ) -> None:
        """Send degradation notification.

        Args:
            degradation_context: Degradation context
            components: Daemon components
        """
        from src.daemon.degradation import DegradationTier
        from src.v1.notifications.models import NotificationMessage, NotificationSeverity

        if degradation_context.tier == DegradationTier.HALTED:
            title = "Trading System HALTED"
            body = degradation_context.halt_reason or "Critical services unavailable"
            severity = NotificationSeverity.CRITICAL
        else:
            title = f"Trading System {degradation_context.tier.value}"
            services = (
                ", ".join(degradation_context.unavailable_services)
                if degradation_context.unavailable_services
                else "Unknown"
            )
            body = f"APIs down: {services}"
            severity = NotificationSeverity.ERROR

        message = NotificationMessage(
            title=title,
            body=body,
            severity=severity,
            metadata={
                "tier": degradation_context.tier.value,
                "services": ", ".join(degradation_context.unavailable_services or []),
                "confidence_adjustment": degradation_context.confidence_adjustment,
            },
            timestamp=datetime.now(UTC),
        )

        if components.notification_service:
            await components.notification_service.notify(message)

        # Publish DEGRADATION event
        if components.event_bus:
            try:
                from src.daemon.event_bus import DashboardEvent, EventType

                await components.event_bus.publish(
                    DashboardEvent(
                        event_type=EventType.DEGRADATION,
                        data={
                            "tier": degradation_context.tier.value,
                            "unavailable_services": degradation_context.unavailable_services,
                            "confidence_adjustment": degradation_context.confidence_adjustment,
                        },
                    )
                )
            except Exception as e:
                logger.opt(exception=True).error(f"Failed to publish DEGRADATION event: {e}")

    async def maybe_notify_signal(
        self,
        result: TradingWorkflowResult,
        components: DaemonComponents,
    ) -> None:
        """Send signal notification if conditions met.

        Args:
            result: Trading workflow result
            components: Daemon components
        """
        if not components.notification_service:
            return

        if result.decision.action.value == "HOLD":
            return

        if result.decision.confidence < components.config.notifications.min_confidence:
            return

        from src.v1.notifications.models import NotificationMessage, NotificationSeverity

        signal_emoji = "✅" if result.decision.action.value == "BUY" else "🔴"
        message = NotificationMessage(
            title=f"{signal_emoji} {result.decision.action.value} Signal: {result.symbol}",
            body=" | ".join(result.decision.reasoning),
            severity=NotificationSeverity.WARNING,
            metadata={
                "symbol": result.symbol,
                "signal": result.decision.action.value,
                "confidence": result.decision.confidence,
                "price": result.risk.current_price,
                "risk_level": result.risk.validation.risk_level,
                "rsi": result.technical.rsi if result.technical.rsi is not None else "N/A",
                "macd": result.technical.macd_hist if result.technical.macd_hist is not None else "N/A",
                "session": result.trading_session.value,
            },
            timestamp=datetime.now(UTC),
        )

        await components.notification_service.notify(message)

    async def notify_var_breach(
        self,
        report: PortfolioRiskReport,
        components: DaemonComponents,
    ) -> None:
        """Send VaR breach notification.

        Args:
            report: Portfolio risk report
            components: Daemon components
        """
        from src.v1.notifications.models import NotificationMessage, NotificationSeverity

        message = NotificationMessage(
            title="Portfolio VaR Limit Breached",
            body=f"VaR95: {report.var_95:.1%} | CVaR99: {report.cvar_99:.1%}",
            severity=NotificationSeverity.ERROR,
            metadata={
                "var_95": report.var_95,
                "cvar_99": report.cvar_99,
                "var_breached": report.var_limit_breached,
                "cvar_breached": report.cvar_limit_breached,
                "positions": report.num_positions,
            },
            timestamp=datetime.now(UTC),
        )

        if components.notification_service:
            await components.notification_service.notify(message)
