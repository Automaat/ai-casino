"""Options flow coordinator tool."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from src.tools.base import BaseTool
from src.tools.models import ToolDefinition, ToolFunction, ToolParameter, ToolParametersSchema

if TYPE_CHECKING:
    from src.daemon.events import OptionsFlowSignal
    from src.v1.watchers.options_flow_watcher import OptionsFlowWatcher


class GetOptionsFlowTool(BaseTool):
    """Read institutional options flow signals from background watcher."""

    def __init__(self, watcher: OptionsFlowWatcher | None) -> None:
        """Initialize tool with optional options flow watcher.

        Args:
            watcher: Options flow watcher instance, or None if disabled
        """
        self._watcher = watcher

    @property
    def name(self) -> str:
        """Tool name."""
        return "get_options_flow"

    def get_tool_definition(self) -> ToolDefinition:
        """Get tool definition for LLM function calling."""
        return ToolDefinition(
            function=ToolFunction(
                name=self.name,
                description=(
                    "Get institutional options flow signals: put/call ratio, volume spikes vs "
                    "average, block trades (>=100k premium), and net premium direction "
                    "(BULLISH/BEARISH/NEUTRAL). Data refreshes every 15 minutes. "
                    "Call without symbol to scan all watchlist symbols for unusual activity. "
                    "Useful before or after analyze_symbol to confirm/contradict technical signals."
                ),
                parameters=ToolParametersSchema(
                    properties={
                        "symbol": ToolParameter(
                            type="string",
                            description=(
                                "Ticker symbol (e.g. AAPL). Omit to return all symbols with unusual activity."
                            ),
                        ),
                    },
                    required=[],
                ),
            ),
        )

    def execute(self, **kwargs: str | int | float | bool) -> str:
        """Execute options flow lookup.

        Args:
            **kwargs: symbol (optional)

        Returns:
            Formatted options flow data
        """
        if self._watcher is None:
            return "Options flow watcher disabled. Enable `options_flow_watcher.enabled: true` in config."

        watcher = self._watcher
        symbol = kwargs.get("symbol")
        if symbol:
            return self._format_single(watcher, str(symbol).upper())
        return self._format_all(watcher)

    def _format_single(self, watcher: OptionsFlowWatcher, symbol: str) -> str:
        signal = watcher.get_signal(symbol)
        if signal is None:
            return f"No options flow data for {symbol} (not in watchlist or first poll pending)."
        return self._format_signal(signal)

    def _format_all(self, watcher: OptionsFlowWatcher) -> str:
        all_signals = watcher.get_all_signals()
        if not all_signals:
            return "No signals cached yet — first poll may not have completed."

        unusual = [s for s in all_signals if s.has_unusual_activity]
        lines = [
            "# Options Flow Scan",
            f"**Tracked:** {len(all_signals)} | **Unusual activity:** {len(unusual)}",
            "",
        ]
        if unusual:
            lines.append("## Symbols With Unusual Activity")
            for sig in sorted(unusual, key=lambda s: s.significance_score, reverse=True):
                lines += [
                    f"### {sig.symbol}",
                    f"- **Direction:** {sig.net_premium_direction} | P/C: {sig.put_call_ratio:.2f} | "
                    f"Vol: {sig.volume_vs_avg:.1f}x | Score: {sig.significance_score:.2f}",
                    f"- {sig.reason}",
                    "",
                ]
        else:
            lines.append("No unusual options activity across watchlist.")
        return "\n".join(lines)

    def _format_signal(self, signal: OptionsFlowSignal) -> str:
        age_min = int((datetime.now(UTC) - signal.computed_at).total_seconds() // 60)
        lines = [
            f"# Options Flow: {signal.symbol}",
            f"**Direction:** {signal.net_premium_direction} | **Score:** {signal.significance_score:.2f}",
            f"**P/C Ratio:** {signal.put_call_ratio:.2f} | **Volume:** {signal.volume_vs_avg:.1f}x avg"
            f" | **Age:** {age_min}m",
            f"**Unusual:** {'YES' if signal.has_unusual_activity else 'no'}",
            f"**Reason:** {signal.reason}",
        ]
        if signal.block_trades:
            lines += ["", f"## Block Trades ({len(signal.block_trades)})"]
            for bt in signal.block_trades[:5]:
                lines.append(
                    f"- {bt.option_type.upper()} ${bt.strike:.0f} exp {bt.expiry} | "
                    f"Vol {bt.volume:,} | ${bt.premium:,.0f}" + (" ITM" if bt.is_itm else "")
                )
        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return f"GetOptionsFlowTool(enabled={self._watcher is not None})"
