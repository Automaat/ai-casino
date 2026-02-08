"""Tearsheet tool for generating QuantStats performance reports."""

import asyncio
import concurrent.futures
from typing import TYPE_CHECKING

from loguru import logger

from src.tools.base import BaseTool

if TYPE_CHECKING:
    import pandas as pd

    from src.metrics.tracker import TearSheet


class GenerateTearsheetTool(BaseTool):
    """Tool to generate QuantStats performance tearsheet."""

    @property
    def name(self) -> str:
        """Tool name."""
        return "generate_tearsheet"

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
                    "Generate a QuantStats performance tearsheet for a stock's trading history. "
                    "Returns CAGR, Sharpe, Sortino, max drawdown, win rate, profit factor, "
                    "and benchmark comparison. Requires existing trade history."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol (e.g., AAPL, TSLA, MSFT)",
                        },
                        "period": {
                            "type": "string",
                            "description": "Time period: 1m, 3m, 6m, 1y, all (default: 1y)",
                            "default": "1y",
                        },
                        "benchmark": {
                            "type": "string",
                            "description": "Benchmark symbol for comparison (default: SPY)",
                            "default": "SPY",
                        },
                    },
                    "required": ["symbol"],
                },
            },
        }

    def execute(self, **kwargs: str | int | float | bool) -> str:
        """Generate performance tearsheet.

        Args:
            **kwargs: Tool arguments (symbol: str, period: str = "1y", benchmark: str = "SPY")

        Returns:
            Formatted tearsheet summary
        """
        symbol = str(kwargs["symbol"]).upper()
        period = str(kwargs.get("period", "1y"))
        benchmark = str(kwargs.get("benchmark", "SPY"))

        logger.info(f"Generating tearsheet for {symbol} (period={period}, benchmark={benchmark})")

        def run_in_thread() -> str:
            return asyncio.run(self._run_tearsheet(symbol, period, benchmark))

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_in_thread)
                return future.result()
        except Exception as e:
            logger.error(f"Tearsheet generation failed for {symbol}: {e}")
            return f"Tearsheet generation failed for {symbol}: {e}"

    async def _run_tearsheet(self, symbol: str, period: str, benchmark: str) -> str:
        """Run tearsheet generation asynchronously.

        Args:
            symbol: Stock ticker symbol
            period: Time period
            benchmark: Benchmark symbol

        Returns:
            Formatted tearsheet summary
        """
        from datetime import UTC, datetime, timedelta

        from src.data.market import MarketDataFetcher
        from src.metrics.quantstats_reporter import QuantStatsReporter
        from src.metrics.tracker import create_metrics_tracker

        period_days = self._parse_period(period)
        tracker = create_metrics_tracker()

        if hasattr(tracker, "trades"):
            all_trades = tracker.trades
        else:
            from src.database.connection import get_session
            from src.database.repositories.trade import TradeRepository

            async with get_session() as session:
                repo = TradeRepository(session)
                all_trades = await repo.get_all()

        symbol_trades = [t for t in all_trades if t.symbol == symbol and t.is_closed()]

        if not symbol_trades:
            return f"No closed trades found for {symbol}"

        if period_days != -1:
            cutoff = datetime.now(UTC) - timedelta(days=period_days)
            filtered_trades = [t for t in symbol_trades if t.timestamp >= cutoff]
        else:
            filtered_trades = symbol_trades

        if not filtered_trades:
            return f"No trades in period {period} for {symbol}"

        benchmark_returns = self._fetch_benchmark(benchmark, period_days, MarketDataFetcher)

        reporter = QuantStatsReporter()
        tearsheet = reporter.generate_tearsheet(
            symbol=symbol,
            trades=filtered_trades,
            benchmark_symbol=benchmark or None,
            benchmark_returns=benchmark_returns,
        )

        return self._format_result(tearsheet)

    def _fetch_benchmark(self, benchmark: str, period_days: int, fetcher_cls: type) -> "pd.Series | None":
        """Fetch benchmark returns data.

        Args:
            benchmark: Benchmark symbol
            period_days: Number of days
            fetcher_cls: MarketDataFetcher class

        Returns:
            Benchmark returns series or None
        """
        if not benchmark:
            return None
        try:
            fetcher = fetcher_cls(use_alpha_vantage=False)
            fetch_days = period_days if period_days != -1 else 365 * 5
            market_data = fetcher.fetch_daily(benchmark, period_days=fetch_days)
            close = market_data.data.get("close", market_data.data.get("Close"))
            if close is not None and not close.empty:
                return close.pct_change().fillna(0.0)
        except Exception as e:
            logger.warning(f"Failed to fetch benchmark data: {e}")
        return None

    def _parse_period(self, period: str) -> int:
        """Parse period string to days.

        Args:
            period: Period specification

        Returns:
            Number of days (-1 for "all")
        """
        period_map = {"1m": 30, "3m": 90, "6m": 180, "1y": 365, "all": -1}
        if period in period_map:
            return period_map[period]
        try:
            return int(period)
        except ValueError:
            logger.warning(f"Unknown period '{period}', using 1y")
            return 365

    def _format_result(self, tearsheet: "TearSheet") -> str:
        """Format tearsheet as markdown.

        Args:
            tearsheet: TearSheet result

        Returns:
            Formatted markdown string
        """
        from pathlib import Path

        lines = [
            f"# {tearsheet.symbol} Performance Tearsheet",
            f"*{tearsheet.start_date:%Y-%m-%d} to {tearsheet.end_date:%Y-%m-%d}*",
            "",
        ]

        self._add_performance_section(lines, tearsheet)
        self._add_risk_section(lines, tearsheet)
        self._add_win_loss_section(lines, tearsheet)
        self._add_benchmark_section(lines, tearsheet)

        filename = Path(tearsheet.html_report_path).name
        lines.extend(["", f"**HTML Report:** {filename}"])

        return "\n".join(lines)

    def _add_performance_section(self, lines: list[str], tearsheet: "TearSheet") -> None:
        """Add performance metrics to lines."""
        lines.append("## Performance")
        metrics = [
            (tearsheet.cagr, "CAGR", ".2%"),
            (tearsheet.sharpe_ratio, "Sharpe Ratio", ".2f"),
            (tearsheet.sortino_ratio, "Sortino Ratio", ".2f"),
            (tearsheet.calmar_ratio, "Calmar Ratio", ".2f"),
        ]
        for value, label, fmt in metrics:
            if value is not None:
                lines.append(f"- {label}: {value:{fmt}}")

    def _add_risk_section(self, lines: list[str], tearsheet: "TearSheet") -> None:
        """Add risk metrics to lines."""
        lines.extend(["", "## Risk"])
        if tearsheet.max_drawdown is not None:
            lines.append(f"- Max Drawdown: {tearsheet.max_drawdown:.2%}")
        if tearsheet.volatility_annual is not None:
            lines.append(f"- Annual Volatility: {tearsheet.volatility_annual:.2%}")

    def _add_win_loss_section(self, lines: list[str], tearsheet: "TearSheet") -> None:
        """Add win/loss metrics to lines."""
        lines.extend(["", "## Win/Loss"])
        if tearsheet.win_rate is not None:
            lines.append(f"- Win Rate: {tearsheet.win_rate:.2%}")
        if tearsheet.profit_factor is not None:
            lines.append(f"- Profit Factor: {tearsheet.profit_factor:.2f}")

    def _add_benchmark_section(self, lines: list[str], tearsheet: "TearSheet") -> None:
        """Add benchmark comparison metrics to lines."""
        if not tearsheet.benchmark_symbol:
            return

        lines.extend(["", f"## Benchmark ({tearsheet.benchmark_symbol})"])
        metrics = [
            (tearsheet.benchmark_cagr, "Benchmark CAGR", ".2%"),
            (tearsheet.benchmark_sharpe, "Benchmark Sharpe", ".2f"),
            (tearsheet.alpha, "Alpha", ".4f"),
            (tearsheet.beta, "Beta", ".2f"),
        ]
        for value, label, fmt in metrics:
            if value is not None:
                lines.append(f"- {label}: {value:{fmt}}")

    def __repr__(self) -> str:
        """String representation."""
        return "GenerateTearsheetTool()"
