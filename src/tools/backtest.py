"""Backtest tool for running momentum strategy backtests."""

from typing import TYPE_CHECKING

from loguru import logger

from src.tools.base import BaseTool
from src.tools.models import ToolDefinition, ToolFunction, ToolParameter, ToolParametersSchema

if TYPE_CHECKING:
    from src.backtesting.runner import BacktestResult
    from src.di.container import AppContainer


class RunBacktestTool(BaseTool):
    """Tool to run backtest on a stock with momentum strategy."""

    def __init__(self, container: AppContainer | None = None) -> None:
        """Initialize tool with optional container.

        Args:
            container: DI container (auto-created if not provided)
        """
        from src.di.container import create_container

        self._container = container or create_container()

    @property
    def name(self) -> str:
        """Tool name."""
        return "run_backtest"

    @property
    def requires_confirmation(self) -> bool:
        """Requires confirmation due to expensive computation."""
        return True

    def get_tool_definition(self) -> ToolDefinition:
        """Get tool definition in LiteLLM/OpenAI format.

        Returns:
            Tool definition for LLM function calling
        """
        return ToolDefinition(
            function=ToolFunction(
                name=self.name,
                description=(
                    "Run a backtest of the momentum trading strategy on historical data. "
                    "Returns total return, Sharpe ratio, max drawdown, win rate, and trade stats. "
                    "This is an expensive operation that fetches data and runs simulations."
                ),
                parameters=ToolParametersSchema(
                    properties={
                        "symbol": ToolParameter(
                            type="string",
                            description="Stock ticker symbol (e.g., AAPL, TSLA, MSFT)",
                        ),
                        "start_date": ToolParameter(
                            type="string",
                            description="Backtest start date in YYYY-MM-DD format",
                        ),
                        "end_date": ToolParameter(
                            type="string",
                            description="Backtest end date in YYYY-MM-DD format",
                        ),
                        "cash": ToolParameter(
                            type="integer",
                            description="Initial cash balance (default: 100000)",
                        ),
                    },
                    required=["symbol", "start_date", "end_date"],
                ),
            ),
        )

    def execute(self, **kwargs: str | int | float | bool) -> str:
        """Run backtest for a stock.

        Args:
            **kwargs: Tool arguments (symbol: str, start_date: str, end_date: str, cash: int = 100000)

        Returns:
            Formatted backtest results
        """
        symbol = str(kwargs["symbol"]).upper()
        start_date = str(kwargs["start_date"])
        end_date = str(kwargs["end_date"])
        cash = int(kwargs.get("cash", 100000))

        logger.info(f"Running backtest for {symbol} ({start_date} to {end_date}, cash=${cash:,})")

        try:
            runner = self._container.backtest_runner(cash=float(cash))
            result = runner.run_backtest(symbol, start_date, end_date)

            return self._format_result(result)
        except Exception as e:
            logger.opt(exception=True).error(f"Backtest failed for {symbol}: {e}")
            return f"Backtest failed for {symbol}: {e}"

    def _format_result(self, result: BacktestResult) -> str:
        """Format backtest result as markdown.

        Args:
            result: BacktestResult

        Returns:
            Formatted markdown string
        """
        lines = [
            f"# {result.symbol} Backtest Results",
            f"*{result.start_date:%Y-%m-%d} to {result.end_date:%Y-%m-%d}*",
            "",
            "## Performance",
            f"- Total Return: {result.total_return:.2%}",
            f"- Sharpe Ratio: {result.sharpe_ratio:.2f}",
            f"- Max Drawdown: {result.max_drawdown:.2%}",
            "",
            "## Trade Stats",
            f"- Total Trades: {result.total_trades}",
            f"- Win Rate: {result.win_rate:.2%}",
            f"- Avg Return/Trade: {result.avg_return_per_trade:.2%}",
        ]

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return "RunBacktestTool()"
