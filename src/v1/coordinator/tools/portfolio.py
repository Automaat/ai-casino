"""Portfolio status tool for coordinator."""

from typing import TYPE_CHECKING

from loguru import logger

from src.tools.base import BaseTool
from src.tools.models import ToolDefinition, ToolFunction, ToolParametersSchema

if TYPE_CHECKING:
    from src.v1.trades.brokers import Broker


class PortfolioStatusTool(BaseTool):
    """Tool to get current portfolio status."""

    def __init__(self, broker: Broker) -> None:
        """Initialize tool with broker.

        Args:
            broker: Broker instance
        """
        self._broker = broker

    @property
    def name(self) -> str:
        """Tool name."""
        return "portfolio_status"

    def get_tool_definition(self) -> ToolDefinition:
        """Get tool definition in LiteLLM/OpenAI format.

        Returns:
            Tool definition for LLM function calling
        """
        return ToolDefinition(
            function=ToolFunction(
                name=self.name,
                description=(
                    "Get current portfolio status including balance, cash, exposure, "
                    "and open positions with P&L. Provides comprehensive account overview."
                ),
                parameters=ToolParametersSchema(
                    properties={},
                    required=[],
                ),
            ),
        )

    def execute(self, **_kwargs: str | int | float | bool) -> str:
        """Execute portfolio status check.

        Args:
            **_kwargs: Tool arguments (none, unused)

        Returns:
            Formatted portfolio status
        """
        logger.info("Fetching portfolio status")

        try:
            account_info = self._broker.get_account_info()

            lines = [
                "# Portfolio Status",
                "",
                f"**Balance:** ${account_info.balance:,.2f}",
                f"**Available Cash:** ${account_info.available_cash:,.2f}",
                f"**Total Exposure:** ${account_info.total_exposure:,.2f}",
                f"**Portfolio Value:** ${account_info.portfolio_value:,.2f}",
                "",
                "## Open Positions",
            ]

            if not account_info.positions:
                lines.append("No open positions")
            else:
                for pos in account_info.positions.values():
                    # Calculate current price from market value and quantity
                    current_price = pos.market_value / pos.qty if pos.qty > 0 else 0.0
                    pnl_sign = "+" if pos.unrealized_pnl >= 0 else ""
                    pnl_pct_sign = "+" if pos.unrealized_pnl_percent >= 0 else ""
                    lines.append(
                        f"- **{pos.symbol}**: {pos.qty} shares @ ${current_price:.2f} "
                        f"| P&L: {pnl_sign}${pos.unrealized_pnl:.2f} "
                        f"({pnl_pct_sign}{pos.unrealized_pnl_percent:.2f}%)"
                    )

            return "\n".join(lines)

        except Exception as e:
            logger.opt(exception=True).error(f"Portfolio status check failed: {e}")
            return f"Failed to fetch portfolio status: {e}"

    def __repr__(self) -> str:
        """String representation."""
        return "PortfolioStatusTool()"
