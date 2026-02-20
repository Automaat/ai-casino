"""Tests for ExecuteTradeTool delegating to TradingService."""

from unittest.mock import AsyncMock, Mock

import pytest

from src.daemon.config.base import TradingMode
from src.v1.coordinator.tools.execute_trade import ExecuteTradeTool
from src.v1.trades.models import TradeAction, TradeRejection, TradeRejectionReason, TradeResult


def _make_daemon_config(trading_mode: TradingMode = TradingMode.PAPER) -> Mock:
    config = Mock()
    config.trading_mode = trading_mode
    return config


def _make_executed_result() -> TradeResult:
    from datetime import UTC, datetime

    return TradeResult(
        executed=True,
        order_id="order-123",
        symbol="AAPL",
        action=TradeAction.BUY,
        quantity=10,
        status="filled",
        filled_avg_price=150.0,
        submitted_at=datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC),
    )


def _make_rejected_result(reason: TradeRejectionReason = TradeRejectionReason.BELOW_THRESHOLD) -> TradeResult:
    return TradeResult(
        executed=False,
        symbol="AAPL",
        action=TradeAction.BUY,
        quantity=10,
        status="rejected",
        rejection=TradeRejection(reason=reason, message="Confidence 50% below threshold 60%"),
    )


_TRADE_KWARGS = {
    "symbol": "AAPL",
    "action": "BUY",
    "quantity": 10,
    "confidence": 0.8,
    "rationale": "Strong momentum breakout detected",
}


@pytest.fixture
def trading_service() -> AsyncMock:
    svc = AsyncMock()
    svc.execute = AsyncMock(return_value=_make_executed_result())
    return svc


@pytest.fixture
def daemon_config() -> Mock:
    return _make_daemon_config()


@pytest.fixture
def tool(trading_service: AsyncMock, daemon_config: Mock) -> ExecuteTradeTool:
    return ExecuteTradeTool(trading_service, daemon_config)


class TestAexecute:
    """Tests for async trade execution."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_delegates_to_service(self, tool: ExecuteTradeTool, trading_service: AsyncMock) -> None:
        result = await tool.aexecute(**_TRADE_KWARGS)

        assert "Trade Executed" in result
        assert "order-123" in result
        trading_service.execute.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_formats_rejection(self, trading_service: AsyncMock, daemon_config: Mock) -> None:
        trading_service.execute = AsyncMock(return_value=_make_rejected_result())
        tool = ExecuteTradeTool(trading_service, daemon_config)

        result = await tool.aexecute(**_TRADE_KWARGS)

        assert "Skipped" in result
        assert "Confidence" in result

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_invalid_action_returns_error(self, tool: ExecuteTradeTool) -> None:
        kwargs = {**_TRADE_KWARGS, "action": "INVALID"}
        result = await tool.aexecute(**kwargs)

        assert "Error" in result

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_invalid_quantity_returns_error(self, tool: ExecuteTradeTool) -> None:
        kwargs = {**_TRADE_KWARGS, "quantity": 0}
        result = await tool.aexecute(**kwargs)

        assert "Error" in result

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_short_rationale_returns_error(self, tool: ExecuteTradeTool) -> None:
        kwargs = {**_TRADE_KWARGS, "rationale": "short"}
        result = await tool.aexecute(**kwargs)

        assert "Error" in result

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_stop_loss_passed_through(self, tool: ExecuteTradeTool, trading_service: AsyncMock) -> None:
        kwargs = {**_TRADE_KWARGS, "stop_loss_price": 140.0}
        await tool.aexecute(**kwargs)

        request = trading_service.execute.call_args[0][0]
        assert request.stop_loss_price == 140.0


class TestRequiresConfirmation:
    """Tests for confirmation requirement."""

    @pytest.mark.unit
    def test_live_requires_confirmation(self) -> None:
        config = _make_daemon_config(TradingMode.LIVE)
        tool = ExecuteTradeTool(AsyncMock(), config)
        assert tool.requires_confirmation is True

    @pytest.mark.unit
    def test_paper_no_confirmation(self) -> None:
        config = _make_daemon_config(TradingMode.PAPER)
        tool = ExecuteTradeTool(AsyncMock(), config)
        assert tool.requires_confirmation is False


class TestFormatResult:
    """Tests for result formatting."""

    @pytest.mark.unit
    def test_format_executed(self) -> None:
        result = _make_executed_result()
        output = ExecuteTradeTool._format_result(result, "Test rationale")

        assert "# Trade Executed" in output
        assert "order-123" in output
        assert "AAPL" in output
        assert "BUY" in output
        assert "Test rationale" in output

    @pytest.mark.unit
    def test_format_rejected(self) -> None:
        result = _make_rejected_result()
        output = ExecuteTradeTool._format_result(result, "Test rationale")

        assert "Skipped" in output
        assert "Confidence" in output

    @pytest.mark.unit
    def test_format_with_stop_loss(self) -> None:
        result = _make_executed_result()
        result.stop_loss_price = 140.0
        output = ExecuteTradeTool._format_result(result, "Test rationale")

        assert "$140.00" in output
