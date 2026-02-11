"""Execute trade tool for coordinator."""

from typing import TYPE_CHECKING, Final

from loguru import logger

from src.daemon.config.base import TradingMode
from src.tools.base import BaseTool
from src.tools.models import ToolDefinition, ToolFunction, ToolParameter, ToolParametersSchema

if TYPE_CHECKING:
    from src.daemon.config import DaemonConfig
    from src.data.broker import AlpacaBroker

MIN_RATIONALE_LENGTH: Final[int] = 10


class ExecuteTradeTool(BaseTool):
    """Tool to execute trades via broker."""

    def __init__(self, broker: AlpacaBroker, daemon_config: DaemonConfig) -> None:
        """Initialize tool with broker and config.

        Args:
            broker: Alpaca broker instance
            daemon_config: Daemon configuration for trading mode
        """
        self._broker = broker
        self._daemon_config = daemon_config

    @property
    def name(self) -> str:
        """Tool name."""
        return "execute_trade"

    @property
    def requires_confirmation(self) -> bool:
        """Requires confirmation in LIVE mode, auto-execute in PAPER mode.

        Returns:
            True if trading mode is LIVE, False if PAPER
        """
        return self._daemon_config.trading_mode == TradingMode.LIVE

    def get_tool_definition(self) -> ToolDefinition:
        """Get tool definition in LiteLLM/OpenAI format.

        Returns:
            Tool definition for LLM function calling
        """
        return ToolDefinition(
            function=ToolFunction(
                name=self.name,
                description=(
                    "Execute a market order (BUY or SELL) with optional stop loss. "
                    "Requires rationale for trade decision. In LIVE mode, requires user confirmation. "
                    "In PAPER mode, executes automatically."
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
                        "stop_loss_price": ToolParameter(
                            type="number",
                            description="Optional stop loss price",
                        ),
                        "rationale": ToolParameter(
                            type="string",
                            description="Trading rationale (minimum 10 characters)",
                        ),
                    },
                    required=["symbol", "action", "quantity", "rationale"],
                ),
            ),
        )

    def execute(self, **kwargs: str | int | float | bool) -> str:
        """Execute trade order.

        Args:
            **kwargs: Tool arguments (symbol: str, action: str, quantity: int,
                     stop_loss_price: float | None, rationale: str)

        Returns:
            Order status with details
        """
        symbol = str(kwargs["symbol"]).upper()
        action = str(kwargs["action"]).upper()
        quantity = int(kwargs["quantity"])
        stop_loss_price = kwargs.get("stop_loss_price")
        rationale = str(kwargs["rationale"])

        # Validate inputs
        if quantity <= 0:
            return "Error: Quantity must be positive"

        if action not in ["BUY", "SELL"]:
            return f"Error: Invalid action '{action}'. Must be BUY or SELL"

        if len(rationale) < MIN_RATIONALE_LENGTH:
            return f"Error: Rationale must be at least {MIN_RATIONALE_LENGTH} characters"

        # Convert stop loss to float if provided
        stop_loss_float = float(stop_loss_price) if stop_loss_price is not None else None

        logger.info(f"Executing {action} order: {quantity} {symbol} (stop_loss={stop_loss_float})")

        try:
            # Submit order
            order_status = self._broker.submit_order(
                symbol=symbol,
                qty=quantity,
                side=action.lower(),
                stop_loss_price=stop_loss_float,
            )

            # Format output
            lines = [
                "# Trade Executed",
                "",
                f"**Order ID:** {order_status.order_id}",
                f"**Symbol:** {order_status.symbol}",
                f"**Action:** {order_status.side.upper()}",
                f"**Quantity:** {order_status.qty}",
                f"**Status:** {order_status.status}",
                f"**Submitted:** {order_status.submitted_at.strftime('%Y-%m-%d %H:%M:%S')}",
            ]

            if stop_loss_float is not None:
                lines.append(f"**Stop Loss:** ${stop_loss_float:.2f}")

            lines.extend(
                [
                    "",
                    "## Rationale",
                    rationale,
                ]
            )

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return f"Failed to execute trade: {e}"

    def __repr__(self) -> str:
        """String representation."""
        return "ExecuteTradeTool()"
