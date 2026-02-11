"""Risk metrics tool for VaR, CVaR, and drawdown analysis."""

from typing import TYPE_CHECKING

from loguru import logger

from src.tools.base import BaseTool
from src.tools.models import ToolDefinition, ToolFunction, ToolParameter, ToolParametersSchema

if TYPE_CHECKING:
    from src.di.container import AppContainer
    from src.metrics.risk import RiskMetrics


class GetRiskMetricsTool(BaseTool):
    """Tool to calculate institutional-grade risk metrics."""

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
        return "get_risk_metrics"

    def get_tool_definition(self) -> ToolDefinition:
        """Get tool definition in LiteLLM/OpenAI format.

        Returns:
            Tool definition for LLM function calling
        """
        return ToolDefinition(
            function=ToolFunction(
                name=self.name,
                description=(
                    "Calculate risk metrics for a stock including Value at Risk (VaR), "
                    "Conditional VaR (CVaR), maximum drawdown, CDaR, volatility, "
                    "and downside deviation."
                ),
                parameters=ToolParametersSchema(
                    properties={
                        "symbol": ToolParameter(
                            type="string",
                            description="Stock ticker symbol (e.g., AAPL, TSLA, MSFT)",
                        ),
                        "days": ToolParameter(
                            type="integer",
                            description="Number of days of historical data (default: 90)",
                        ),
                    },
                    required=["symbol"],
                ),
            ),
        )

    def execute(self, **kwargs: str | int | float | bool) -> str:
        """Calculate risk metrics for a stock.

        Args:
            **kwargs: Tool arguments (symbol: str, days: int = 90)

        Returns:
            Formatted risk metrics summary
        """
        symbol = str(kwargs["symbol"]).upper()
        days = int(kwargs.get("days", 90))

        logger.info(f"Calculating risk metrics for {symbol} ({days} days)")

        try:
            fetcher = self._container.market_fetcher()
            market_data = fetcher.fetch_daily(symbol, period_days=days)

            close = market_data.data.get("close", market_data.data.get("Close"))
            if close is None or close.empty:
                return f"No price data available for {symbol}"

            returns = close.pct_change().dropna().tolist()

            calculator = self._container.risk_metrics_calculator()
            metrics = calculator.calculate_all(returns)

            return self._format_result(symbol, days, metrics)
        except Exception as e:
            logger.error(f"Risk metrics calculation failed for {symbol}: {e}")
            return f"Risk metrics calculation failed for {symbol}: {e}"

    def _format_result(self, symbol: str, days: int, metrics: RiskMetrics) -> str:
        """Format risk metrics as markdown.

        Args:
            symbol: Stock ticker symbol
            days: Days of data used
            metrics: RiskMetrics result

        Returns:
            Formatted markdown string
        """
        var = metrics.var_metrics
        dd = metrics.drawdown_metrics

        lines = [
            f"# {symbol} Risk Metrics ({days}d)",
            "",
            "## Value at Risk",
            f"- VaR (95%): {var.var_95:.4f}",
            f"- VaR (99%): {var.var_99:.4f}",
            f"- CVaR (95%): {var.cvar_95:.4f}",
            f"- CVaR (99%): {var.cvar_99:.4f}",
            "",
            "## Drawdown",
            f"- Max Drawdown: {dd.max_drawdown:.4f}",
            f"- CDaR (95%): {dd.cdar_95:.4f}",
            f"- Avg Drawdown: {dd.avg_drawdown:.4f}",
            f"- Max DD Duration: {dd.max_drawdown_duration_days} days",
            "",
            "## Volatility",
            f"- Annual Volatility: {metrics.volatility_annual:.4f}",
            f"- Downside Deviation: {metrics.downside_deviation:.4f}",
        ]

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return "GetRiskMetricsTool()"
