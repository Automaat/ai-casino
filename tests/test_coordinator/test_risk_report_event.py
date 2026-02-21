"""Tests for RiskReportEvent model and event prompt rendering."""

from datetime import UTC, datetime

import pytest

from src.daemon.events import RiskReportEvent
from src.strategies.session import TradingSession
from src.v1.coordinator.event_prompt import (
    EventCycleContext,
    EventCyclePromptBuilder,
    _format_event_details,
)
from src.v1.coordinator.models import CoordinatorConfig
from src.v1.event_queue.models import QueuedMarketEvent


def _make_risk_event_data(risk_status: str = "BREACH") -> dict:
    """Create risk_report event payload dict."""
    return {
        "event_type": "risk_report",
        "source": "risk_report_task",
        "risk_status": risk_status,
        "var_95": 0.04,
        "var_99": 0.06,
        "cvar_95": 0.05,
        "cvar_99": 0.07,
        "cdar_95": 0.08,
        "max_drawdown": 0.12,
        "portfolio_volatility": 0.18,
        "current_exposure_percent": 75.0,
        "num_positions": 5,
        "var_limit_breached": True,
        "cvar_limit_breached": False,
    }


def _make_queued_event(risk_status: str = "BREACH") -> QueuedMarketEvent:
    """Create a QueuedMarketEvent for a risk_report."""
    event_data = _make_risk_event_data(risk_status)
    return QueuedMarketEvent(
        event_id="risk_report_2026-02-21",
        event_type="risk_report",
        payload={
            "event": event_data,
            "triage": {
                "urgency": "IMMEDIATE",
                "sentiment": "BEARISH",
                "confidence": 1.0,
                "reasoning": f"Portfolio risk {risk_status}: VaR95=4.00%",
                "symbols": [],
            },
        },
        enqueued_at=datetime.now(UTC),
    )


class TestRiskReportEvent:
    """Tests for RiskReportEvent model."""

    @pytest.mark.unit
    def test_create_breach(self) -> None:
        event = RiskReportEvent(
            timestamp=datetime.now(UTC),
            risk_status="BREACH",
            var_95=0.04,
            var_99=0.06,
            cvar_95=0.05,
            cvar_99=0.07,
            cdar_95=0.08,
            max_drawdown=0.12,
            portfolio_volatility=0.18,
            current_exposure_percent=75.0,
            num_positions=5,
            var_limit_breached=True,
            cvar_limit_breached=False,
        )
        assert event.event_type == "risk_report"
        assert event.risk_status == "BREACH"

    @pytest.mark.unit
    def test_create_warning(self) -> None:
        event = RiskReportEvent(
            timestamp=datetime.now(UTC),
            risk_status="WARNING",
            var_95=0.032,
            var_99=0.045,
            cvar_95=0.038,
            cvar_99=0.052,
            cdar_95=0.06,
            max_drawdown=0.08,
            portfolio_volatility=0.14,
            current_exposure_percent=60.0,
            num_positions=3,
            var_limit_breached=False,
            cvar_limit_breached=False,
        )
        assert event.risk_status == "WARNING"

    @pytest.mark.unit
    def test_to_prompt_text_includes_key_metrics(self) -> None:
        event = RiskReportEvent(
            timestamp=datetime.now(UTC),
            risk_status="BREACH",
            var_95=0.04,
            var_99=0.06,
            cvar_95=0.05,
            cvar_99=0.07,
            cdar_95=0.08,
            max_drawdown=0.12,
            portfolio_volatility=0.18,
            current_exposure_percent=75.0,
            num_positions=5,
            var_limit_breached=True,
            cvar_limit_breached=False,
        )
        text = event.to_prompt_text()
        assert "BREACH" in text
        assert "VaR95" in text
        assert "CVaR99" in text
        assert "CDaR95" in text
        assert "Max Drawdown" in text
        assert "Exposure" in text


class TestFormatEventDetails:
    """Tests for _format_event_details with risk_report events."""

    @pytest.mark.unit
    def test_includes_risk_status(self) -> None:
        details = _format_event_details(_make_risk_event_data("BREACH"))
        assert "BREACH" in details

    @pytest.mark.unit
    def test_includes_var_metrics(self) -> None:
        details = _format_event_details(_make_risk_event_data("WARNING"))
        assert "VaR95" in details
        assert "CVaR99" in details
        assert "CDaR95" in details

    @pytest.mark.unit
    def test_includes_drawdown_and_volatility(self) -> None:
        details = _format_event_details(_make_risk_event_data("BREACH"))
        assert "Max Drawdown" in details
        assert "Volatility" in details

    @pytest.mark.unit
    def test_includes_exposure_and_positions(self) -> None:
        details = _format_event_details(_make_risk_event_data("BREACH"))
        assert "Exposure" in details
        assert "Positions" in details


class TestEventPromptRendering:
    """Tests for risk_report event prompt rendering via EventCyclePromptBuilder."""

    @pytest.mark.unit
    def test_prompt_builder_renders_risk_report(self) -> None:
        builder = EventCyclePromptBuilder()
        config = CoordinatorConfig(
            enabled=True,
            max_tool_calls=25,
            event_max_tool_calls=15,
            event_max_dequeue=5,
            max_daily_trades=10,
            max_position_pct=10.0,
            min_confidence_to_trade=0.6,
        )
        event = _make_queued_event("BREACH")
        ctx = EventCycleContext(
            positions_summary="5 positions",
            session=TradingSession.REGULAR,
            market_open=True,
        )
        prompt = builder.build(events=[event], context=ctx, config=config)
        assert "BREACH" in prompt

    @pytest.mark.unit
    def test_prompt_uses_risk_report_template(self) -> None:
        builder = EventCyclePromptBuilder()
        config = CoordinatorConfig(
            enabled=True,
            max_tool_calls=25,
            event_max_tool_calls=15,
            event_max_dequeue=5,
            max_daily_trades=10,
            max_position_pct=10.0,
            min_confidence_to_trade=0.6,
        )
        event = _make_queued_event("WARNING")
        ctx = EventCycleContext(
            positions_summary="3 positions",
            session=TradingSession.REGULAR,
            market_open=True,
        )
        prompt = builder.build(events=[event], context=ctx, config=config)
        # Template-specific content from risk_report.txt
        assert "risk report" in prompt.lower() or "Risk Report" in prompt
        assert "WARNING" in prompt
