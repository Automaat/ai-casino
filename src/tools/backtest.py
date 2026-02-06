"""Backtest tool for running momentum strategy backtests."""

from typing import TYPE_CHECKING

from loguru import logger

from src.tools.base import BaseTool

if TYPE_CHECKING:
    from src.backtesting.runner import BacktestResult


class RunBacktestTool(BaseTool):
    """Tool to run backtest on a stock with momentum strategy."""

    @property
    def name(self) -> str:
        """Tool name."""
        return "run_backtest"

    @property
    def requires_confirmation(self) -> bool:
        """Requires confirmation due to expensive computation."""
        return True

    def get_tool_definition(self) -> dict:
        """Get tool definition in LiteLLM/OpenAI format.

        Returns:
            Tool definition dict for LLM function calling
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Run a backtest of the momentum trading strategy on historical data. "
                    "Returns total return, Sharpe ratio, max drawdown, win rate, and trade stats. "
                    "This is an expensive operation that fetches data and runs simulations."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol (e.g., AAPL, TSLA, MSFT)",
                        },
                        "start_date": {
                            "type": "string",
                            "description": "Backtest start date in YYYY-MM-DD format",
                        },
                        "end_date": {
                            "type": "string",
                            "description": "Backtest end date in YYYY-MM-DD format",
                        },
                        "cash": {
                            "type": "integer",
                            "description": "Initial cash balance (default: 100000)",
                            "default": 100000,
                        },
                    },
                    "required": ["symbol", "start_date", "end_date"],
                },
            },
        }

    def execute(self, symbol: str, start_date: str, end_date: str, cash: int = 100000) -> str:
        """Run backtest for a stock.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            cash: Initial cash balance

        Returns:
            Formatted backtest results
        """
        symbol = symbol.upper()
        logger.info(f"Running backtest for {symbol} ({start_date} to {end_date}, cash=${cash:,})")

        try:
            from src.backtesting.runner import BacktestRunner

            runner = BacktestRunner(cash=float(cash))
            result = runner.run_backtest(symbol, start_date, end_date)

            return self._format_result(result)
        except Exception as e:
            logger.error(f"Backtest failed for {symbol}: {e}")
            return f"Backtest failed for {symbol}: {e}"

    def _format_result(self, result: "BacktestResult") -> str:
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
