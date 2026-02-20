"""Fetch pre-market movers tool for game plan agent."""

import asyncio

from loguru import logger

from src.tools.base import BaseTool
from src.tools.models import ToolDefinition, ToolFunction, ToolParameter, ToolParametersSchema

_MIN_HISTORY_DAYS = 2


class FetchPremarketMoversTool(BaseTool):
    """Scan watchlist for biggest pre-market movers."""

    @property
    def name(self) -> str:
        """Tool name."""
        return "fetch_premarket_movers"

    def get_tool_definition(self) -> ToolDefinition:
        """Get tool definition for LLM.

        Returns:
            Tool definition
        """
        return ToolDefinition(
            function=ToolFunction(
                name=self.name,
                description=(
                    "Scan watchlist symbols for biggest pre-market movers "
                    "(gainers/losers by % change from previous close)."
                ),
                parameters=ToolParametersSchema(
                    properties={
                        "symbols": ToolParameter(
                            type="string",
                            description="Comma-separated symbols to scan (e.g., AAPL,TSLA,NVDA)",
                        ),
                    },
                    required=["symbols"],
                ),
            ),
        )

    def execute(self, **kwargs: str | int | float | bool) -> str:
        """Fetch pre-market movers for given symbols.

        Args:
            **kwargs: symbols parameter (comma-separated)

        Returns:
            Formatted movers summary
        """
        symbols_str = str(kwargs["symbols"])
        symbols = [s.strip().upper() for s in symbols_str.split(",")]
        limited = symbols[:15]

        movers = self._scan_movers(limited)

        if not movers:
            return "No pre-market data available"

        movers.sort(key=lambda x: abs(x[1]), reverse=True)
        top = movers[:5]

        lines = ["## Pre-Market Movers"]
        for symbol, pct in top:
            direction = "+" if pct > 0 else ""
            lines.append(f"- {symbol}: {direction}{pct:.1f}%")

        return "\n".join(lines)

    def _scan_movers(self, symbols: list[str]) -> list[tuple[str, float]]:
        """Scan symbols for price changes.

        Args:
            symbols: List of ticker symbols

        Returns:
            List of (symbol, pct_change) tuples
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance not available for premarket scan")
            return []

        movers: list[tuple[str, float]] = []
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="2d")
                if data.empty or len(data) < _MIN_HISTORY_DAYS:
                    continue

                prev_close = data["Close"].iloc[-2]
                current = data["Close"].iloc[-1]
                pct_change = ((current - prev_close) / prev_close) * 100
                movers.append((symbol, pct_change))
            except Exception as e:
                logger.debug(f"Premarket scan failed for {symbol}: {e}")

        return movers

    async def aexecute(self, **kwargs: str | int | float | bool) -> str:
        """Async execution via thread pool.

        Args:
            **kwargs: Tool arguments

        Returns:
            Formatted movers summary
        """
        return await asyncio.to_thread(self.execute, **kwargs)

    def __repr__(self) -> str:
        """String representation."""
        return "FetchPremarketMoversTool()"
