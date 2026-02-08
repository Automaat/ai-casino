"""Portfolio optimization tool using Optuna."""

from typing import TYPE_CHECKING

from loguru import logger

from src.tools.base import BaseTool

if TYPE_CHECKING:
    from src.optimization.results import OptimizationResult


class OptimizePortfolioTool(BaseTool):
    """Tool to optimize trading strategy parameters with Optuna."""

    @property
    def name(self) -> str:
        """Tool name."""
        return "optimize_portfolio"

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
                    "Optimize trading strategy parameters using Optuna hyperparameter search. "
                    "Tests different parameter combinations to find optimal Sharpe ratio. "
                    "This is an expensive operation that runs many backtests."
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
                            "description": "Optimization start date in YYYY-MM-DD format",
                        },
                        "end_date": {
                            "type": "string",
                            "description": "Optimization end date in YYYY-MM-DD format",
                        },
                        "strategy": {
                            "type": "string",
                            "description": "Strategy to optimize (default: momentum)",
                            "enum": ["momentum", "trend_following", "mean_reversion", "ensemble"],
                            "default": "momentum",
                        },
                        "n_trials": {
                            "type": "integer",
                            "description": "Number of optimization trials (default: 50)",
                            "default": 50,
                        },
                    },
                    "required": ["symbol", "start_date", "end_date"],
                },
            },
        }

    def execute(self, **kwargs: str | int | float | bool) -> str:
        """Run portfolio optimization.

        Args:
            **kwargs: Tool arguments (symbol: str, start_date: str, end_date: str,
                     strategy: str = "momentum", n_trials: int = 50)
            end_date: End date (YYYY-MM-DD)
            strategy: Strategy name
            n_trials: Number of trials

        Returns:
            Formatted optimization results
        """
        symbol = str(kwargs["symbol"]).upper()
        start_date = str(kwargs["start_date"])
        end_date = str(kwargs["end_date"])
        strategy = str(kwargs.get("strategy", "momentum"))
        n_trials = int(kwargs.get("n_trials", 50))

        logger.info(
            f"Optimizing {strategy} strategy for {symbol} ({start_date} to {end_date}, {n_trials} trials)"
        )

        try:
            from src.optimization.optimizer import OptunaOptimizer

            optimizer = OptunaOptimizer(n_trials=n_trials)
            result = optimizer.optimize(symbol, start_date, end_date, strategy)

            return self._format_result(result)
        except Exception as e:
            logger.error(f"Optimization failed for {symbol}: {e}")
            return f"Optimization failed for {symbol}: {e}"

    def _format_result(self, result: "OptimizationResult") -> str:
        """Format optimization result as markdown.

        Args:
            result: OptimizationResult

        Returns:
            Formatted markdown string
        """
        lines = [
            f"# {result.symbol} Strategy Optimization",
            f"*Strategy: {result.strategy_name}*",
            "",
            "## Best Parameters",
        ]

        for param, value in result.best_params.items():
            if isinstance(value, float):
                lines.append(f"- {param}: {value:.4f}")
            else:
                lines.append(f"- {param}: {value}")

        lines.extend(
            [
                "",
                "## Best Metrics",
                f"- Sharpe Ratio: {result.best_metrics.get('sharpe_ratio', 0):.2f}",
                f"- Total Return: {result.best_metrics.get('total_return', 0):.2%}",
                f"- Max Drawdown: {result.best_metrics.get('max_drawdown', 0):.2%}",
                "",
                "## Optimization Stats",
                f"- Total Trials: {result.total_trials}",
                f"- Time: {result.optimization_time_seconds:.1f}s",
            ]
        )

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return "OptimizePortfolioTool()"
