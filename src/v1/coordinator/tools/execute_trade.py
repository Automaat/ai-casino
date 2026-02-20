"""Execute trade tool for coordinator."""

from typing import TYPE_CHECKING

from pydantic import ValidationError

from src.daemon.config.base import TradingMode
from src.tools.base import BaseTool
from src.tools.models import ToolDefinition, ToolFunction, ToolParameter, ToolParametersSchema
from src.v1.trades.models import TradeAction, TradeRequest, TradeResult

if TYPE_CHECKING:
    from src.daemon.config import DaemonConfig
    from src.v1.trades.service import TradingService


class ExecuteTradeTool(BaseTool):
    """Tool to execute trades via TradingService."""

    def __init__(self, trading_service: TradingService, daemon_config: DaemonConfig) -> None:
        """Initialize tool with trading service.

        Args:
            trading_service: Unified trading service
            daemon_config: Daemon configuration for trading mode
        """
        self._trading_service = trading_service
        self._daemon_config = daemon_config

    @property
    def name(self) -> str:
        """Tool name."""
        return "execute_trade"

    @property
    def requires_confirmation(self) -> bool:
        """Requires confirmation in LIVE mode, auto-execute in PAPER mode."""
        return self._daemon_config.trading_mode == TradingMode.LIVE

    def get_tool_definition(self) -> ToolDefinition:
        """Get tool definition in LiteLLM/OpenAI format."""
        return ToolDefinition(
            function=ToolFunction(
                name=self.name,
                description=(
                    "Execute a market order (BUY or SELL). Risk management automatically validates "
                    "position sizing, stop loss, and portfolio constraints. Quantity will be capped "
                    "to risk-approved limits. Use recommended_shares from analyze_symbol output."
                ),
                parameters=ToolParametersSchema(
                    properties={
                        "symbol": ToolParameter(
                            type="string",
                            description="Stock ticker symbol (e.g., AAPL, TSLA)",
                        ),
                        "action": ToolParameter(
                            type="string",
                            description="Order action: BUY or SELL",
                        ),
                        "quantity": ToolParameter(
                            type="integer",
                            description="Number of shares to trade (minimum: 1)",
                        ),
                        "confidence": ToolParameter(
                            type="number",
                            description="Decision confidence (0.0-1.0) for threshold validation",
                        ),
                        "stop_loss_price": ToolParameter(
                            type="number",
                            description="Optional stop loss price",
                        ),
                        "rationale": ToolParameter(
                            type="string",
                            description="Trading rationale (minimum 10 characters)",
                        ),
                    },
                    required=["symbol", "action", "quantity", "confidence", "rationale"],
                ),
            ),
        )

    def execute(self, **kwargs: str | int | float | bool) -> str:
        """Execute trade order (sync version).

        Args:
            **kwargs: Tool arguments

        Returns:
            Order status with details
        """
        import asyncio

        return asyncio.get_event_loop().run_until_complete(self.aexecute(**kwargs))

    async def aexecute(self, **kwargs: str | int | float | bool) -> str:
        """Execute trade order with async support.

        Args:
            **kwargs: Tool arguments (symbol, action, quantity, confidence, stop_loss_price, rationale)

        Returns:
            Order status with details or rejection message
        """
        try:
            original_qty = int(kwargs["quantity"])
            request = TradeRequest(
                symbol=str(kwargs["symbol"]).upper(),
                action=TradeAction(str(kwargs["action"]).upper()),
                quantity=original_qty,
                confidence=float(kwargs["confidence"]),
                rationale=str(kwargs["rationale"]),
                stop_loss_price=float(kwargs["stop_loss_price"])
                if kwargs.get("stop_loss_price") is not None
                else None,
            )
        except (ValidationError, ValueError, KeyError) as e:
            return f"Error: Invalid trade parameters: {e}"

        result = await self._trading_service.execute(request)

        # Track risk capping: quantity may have been reduced by risk limits
        if result.executed and result.quantity < original_qty:
            result.requested_quantity = original_qty
            result.risk_capped = True

        return self._format_result(result, request.rationale)

    @staticmethod
    def _format_result(result: TradeResult, rationale: str) -> str:
        """Format TradeResult as markdown for LLM consumption."""
        if not result.executed:
            rejection = result.rejection
            if rejection:
                return f"Skipped: {rejection.message}"
            return f"Trade {result.action} {result.symbol} was not executed"

        lines = [
            "# Trade Executed",
            "",
            f"**Order ID:** {result.order_id}",
            f"**Symbol:** {result.symbol}",
            f"**Action:** {result.action.value}",
            f"**Quantity:** {result.quantity}",
            f"**Status:** {result.status}",
        ]

        if result.submitted_at:
            lines.append(f"**Submitted:** {result.submitted_at.strftime('%Y-%m-%d %H:%M:%S')}")

        if result.stop_loss_price is not None:
            lines.append(f"**Stop Loss:** ${result.stop_loss_price:.2f}")

        if result.risk_capped and result.requested_quantity is not None:
            lines.append(
                f"\n**Note:** Quantity capped from {result.requested_quantity} "
                f"to {result.quantity} by risk limits"
            )

        lines.extend(["", "## Rationale", rationale])

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return "ExecuteTradeTool()"
