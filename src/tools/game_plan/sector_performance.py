"""Fetch sector performance tool for game plan agent."""

import asyncio

from loguru import logger

from src.tools.base import BaseTool
from src.tools.models import ToolDefinition, ToolFunction, ToolParametersSchema

_MIN_HISTORY_DAYS = 2

SECTOR_ETFS = {
    "XLK": "Technology",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLB": "Materials",
}


class FetchSectorPerformanceTool(BaseTool):
    """Fetch sector ETF performance for rotation analysis."""

    @property
    def name(self) -> str:
        """Tool name."""
        return "fetch_sector_performance"

    def get_tool_definition(self) -> ToolDefinition:
        """Get tool definition for LLM.

        Returns:
            Tool definition
        """
        return ToolDefinition(
            function=ToolFunction(
                name=self.name,
                description=(
                    "Fetch sector ETF performance (XLK, XLE, XLF, XLV, etc.) "
                    "to identify sector rotation and leading/lagging sectors."
                ),
                parameters=ToolParametersSchema(properties={}, required=[]),
            ),
        )

    def execute(self, **_kwargs: str | int | float | bool) -> str:
        """Fetch sector ETF performance.

        Returns:
            Formatted sector performance summary
        """
        try:
            import yfinance as yf
        except ImportError:
            return "yfinance not available for sector scan"

        results: list[tuple[str, str, float]] = []
        for etf, sector in SECTOR_ETFS.items():
            try:
                ticker = yf.Ticker(etf)
                data = ticker.history(period="2d")
                if data.empty or len(data) < _MIN_HISTORY_DAYS:
                    continue

                prev_close = data["Close"].iloc[-2]
                current = data["Close"].iloc[-1]
                pct_change = ((current - prev_close) / prev_close) * 100
                results.append((etf, sector, pct_change))
            except Exception as e:
                logger.debug(f"Sector scan failed for {etf}: {e}")

        if not results:
            return "Sector performance data unavailable"

        results.sort(key=lambda x: x[2], reverse=True)

        lines = ["## Sector Performance (1-Day)"]
        for etf, sector, pct in results:
            direction = "+" if pct > 0 else ""
            lines.append(f"- {sector} ({etf}): {direction}{pct:.1f}%")

        leading = [sector for _, sector, pct in results[:3] if pct > 0]
        lagging = [sector for _, sector, pct in results[-3:] if pct < 0]

        if leading:
            lines.append(f"\n**Leading:** {', '.join(leading)}")
        if lagging:
            lines.append(f"**Lagging:** {', '.join(lagging)}")

        return "\n".join(lines)

    async def aexecute(self, **kwargs: str | int | float | bool) -> str:
        """Async execution via thread pool.

        Returns:
            Formatted sector performance summary
        """
        return await asyncio.to_thread(self.execute, **kwargs)

    def __repr__(self) -> str:
        """String representation."""
        return "FetchSectorPerformanceTool()"
