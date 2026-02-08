"""Command handlers for TUI slash commands."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from src.screening.exporter import ScreeningExporter, Watchlist
    from src.workflows.types import TradingWorkflowResult

ProgressCallback = Callable[[str, str, str], None]
CancelledCallback = Callable[[], bool]


@dataclass
class CommandResult:
    """Result of a command execution."""

    success: bool
    message: str
    data: dict | None = None
    workflow_result: object | None = field(default=None, repr=False)


class CommandHandler:
    """Handler for TUI slash commands."""

    def __init__(self) -> None:
        """Initialize command handler."""
        self._commands: dict[str, Callable[..., Coroutine]] = {
            "analyze": self._cmd_analyze,
            "technical": self._cmd_technical,
            "sentiment": self._cmd_sentiment,
            "news": self._cmd_news,
            "screen": self._cmd_screen,
            "discover": self._cmd_screen,  # alias
            "export": self._cmd_export,
            "watchlist": self._cmd_watchlist,
            "candidates": self._cmd_candidates,
            "trump": self._cmd_trump,
            "casino": self._cmd_casino,
            "help": self._cmd_help,
        }
        self._last_screening_output = None  # Store last screening for export
        self._app = None  # Will be set by app
        logger.info("CommandHandler initialized")

    @property
    def command_names(self) -> list[str]:
        """Get list of available command names."""
        return list(self._commands.keys())

    def set_app(self, app: object) -> None:
        """Set the app instance for personality switching.

        Args:
            app: The TradingChatApp instance
        """
        self._app = app

    def is_command(self, text: str) -> bool:
        """Check if text is a slash command.

        Args:
            text: Input text

        Returns:
            True if text starts with /
        """
        return text.strip().startswith("/")

    def parse_command(self, text: str) -> tuple[str, list[str]]:
        """Parse a command into name and arguments.

        Args:
            text: Command text (e.g., "/analyze AAPL")

        Returns:
            Tuple of (command_name, args_list)
        """
        parts = text.strip().split()
        cmd = parts[0][1:].lower() if parts else ""
        args = parts[1:] if len(parts) > 1 else []
        return cmd, args

    async def execute(
        self,
        text: str,
        progress_callback: ProgressCallback | None = None,
        is_cancelled: CancelledCallback | None = None,
    ) -> CommandResult:
        """Execute a slash command.

        Args:
            text: Command text
            progress_callback: Optional callback for progress updates (step_id, status, detail)
            is_cancelled: Optional callback returning True if execution should stop

        Returns:
            CommandResult with execution result
        """
        cmd, args = self.parse_command(text)

        if cmd not in self._commands:
            return CommandResult(
                success=False,
                message=f"Unknown command: /{cmd}. Type /help for available commands.",
            )

        try:
            if cmd in ("analyze", "technical", "sentiment", "news"):
                return await self._commands[cmd](args, progress_callback, is_cancelled)
            if cmd in ("screen", "discover"):
                return await self._cmd_screen(args, progress_callback, is_cancelled)
            return await self._commands[cmd](args)
        except Exception as e:
            logger.exception(f"Command /{cmd} failed")
            return CommandResult(success=False, message=f"Command failed: {e}")

    async def _cmd_help(self, _args: list[str]) -> CommandResult:
        """Show help for available commands."""
        help_text = """**Available Commands:**

- **/analyze SYMBOL** - Full trading analysis for a stock
- **/technical SYMBOL** - Technical analysis only
- **/sentiment SYMBOL** - Sentiment analysis only
- **/news SYMBOL** - News analysis only
- **/screen [criteria] [universe] [top_n]** - Screen stocks for opportunities
- **/discover** - Alias for /screen
- **/export [format] [filename]** - Export last screening results
- **/watchlist [action]** - Manage watchlists
- **/candidates** - Show after-hours screening candidates
- **/candidates add SYMBOLS** - Add candidates to watchlist
- **/candidates clear** - Clear old candidates
- **/trump** - Switch to Trump personality mode 🇺🇸
- **/casino** - Switch to AI Casino personality mode 🎰
- **/help** - Show this help message

**Screening Examples:**
- `/screen momentum` - Find momentum stocks in combined universe
- `/screen value SP500 20` - Find 20 value stocks from S&P 500
- `/screen breakout NASDAQ100 --save` - Find breakouts and save to watchlist

**Export Examples:**
- `/export csv` - Export last results as CSV
- `/export json my_picks` - Export as JSON with custom name

**Watchlist Examples:**
- `/watchlist` - Show default watchlist
- `/watchlist list` - List all watchlists
- `/watchlist show picks` - Show specific watchlist

Type freely to chat about markets or ask questions."""
        return CommandResult(success=True, message=help_text)

    async def _cmd_analyze(
        self,
        args: list[str],
        progress_callback: ProgressCallback | None = None,
        is_cancelled: CancelledCallback | None = None,
    ) -> CommandResult:
        """Run full trading analysis in subprocess to avoid Textual fd conflicts."""
        if not args:
            return CommandResult(success=False, message="Usage: /analyze SYMBOL")

        symbol = args[0].upper()

        from src.tui.worker import run_analysis_in_process

        result_dict = await run_analysis_in_process(
            symbol, period_days=90, progress_callback=progress_callback, is_cancelled=is_cancelled
        )
        from src.workflows.types import TradingWorkflowResult

        result = TradingWorkflowResult.model_validate(result_dict)

        if progress_callback:
            progress_callback("decision", "complete", "")

        cmd_result = self._format_analysis_result(result)
        cmd_result.workflow_result = result
        return cmd_result

    async def _cmd_technical(
        self,
        args: list[str],
        progress_callback: ProgressCallback | None = None,
        is_cancelled: CancelledCallback | None = None,
    ) -> CommandResult:
        """Run technical analysis only."""
        if not args:
            return CommandResult(success=False, message="Usage: /technical SYMBOL")

        symbol = args[0].upper()
        from src.tui.worker import run_analysis_in_process

        result_dict = await run_analysis_in_process(
            symbol,
            period_days=90,
            progress_callback=progress_callback,
            is_cancelled=is_cancelled,
        )
        from src.workflows.types import TradingWorkflowResult

        result = TradingWorkflowResult.model_validate(result_dict)

        if progress_callback:
            progress_callback("decision", "complete", "")

        msg = self._format_technical(result)
        return CommandResult(success=True, message=msg, data={"symbol": symbol})

    async def _cmd_sentiment(
        self,
        args: list[str],
        progress_callback: ProgressCallback | None = None,
        is_cancelled: CancelledCallback | None = None,
    ) -> CommandResult:
        """Run sentiment analysis only."""
        if not args:
            return CommandResult(success=False, message="Usage: /sentiment SYMBOL")

        symbol = args[0].upper()
        from src.tui.worker import run_analysis_in_process

        result_dict = await run_analysis_in_process(
            symbol,
            period_days=90,
            progress_callback=progress_callback,
            is_cancelled=is_cancelled,
        )
        from src.workflows.types import TradingWorkflowResult

        result = TradingWorkflowResult.model_validate(result_dict)

        if progress_callback:
            progress_callback("decision", "complete", "")

        msg = self._format_sentiment(result)
        return CommandResult(success=True, message=msg, data={"symbol": symbol})

    async def _cmd_news(
        self,
        args: list[str],
        progress_callback: ProgressCallback | None = None,
        is_cancelled: CancelledCallback | None = None,
    ) -> CommandResult:
        """Run news analysis only."""
        if not args:
            return CommandResult(success=False, message="Usage: /news SYMBOL")

        symbol = args[0].upper()
        from src.tui.worker import run_analysis_in_process

        result_dict = await run_analysis_in_process(
            symbol,
            period_days=90,
            progress_callback=progress_callback,
            is_cancelled=is_cancelled,
        )
        from src.workflows.types import TradingWorkflowResult

        result = TradingWorkflowResult.model_validate(result_dict)

        if progress_callback:
            progress_callback("decision", "complete", "")

        msg = self._format_news(result)
        return CommandResult(success=True, message=msg, data={"symbol": symbol})

    def _format_analysis_result(self, result: TradingWorkflowResult) -> CommandResult:
        """Format full analysis result."""
        signal = result.decision.action.value
        confidence = result.decision.confidence
        rsi_str = f"{result.technical.rsi:.2f}" if result.technical.rsi is not None else "N/A"

        msg = f"""## Analysis for {result.symbol}

**Decision: {signal}** (confidence: {confidence:.2f})
**Risk Level:** {result.decision.risk_level}

### Technical
- Signal: {result.technical.signal.value}
- RSI: {rsi_str}
- Confidence: {result.technical.confidence:.2f}

### Sentiment
- Overall: {result.sentiment.overall_sentiment}
- Score: {result.sentiment.sentiment_score:.2f}
- Articles analyzed: {result.sentiment.article_count}

### News
- Key themes: {", ".join(result.news.key_themes[:3])}
- Impact: {result.news.impact_assessment[:100]}

### Reasoning
{result.decision.reasoning}"""

        return CommandResult(
            success=True,
            message=msg,
            data={"symbol": result.symbol, "signal": signal, "confidence": confidence},
        )

    def _format_technical(self, result: TradingWorkflowResult) -> str:
        """Format technical analysis."""
        rsi_str = f"{result.technical.rsi:.2f}" if result.technical.rsi is not None else "N/A"
        macd_str = f"{result.technical.macd_hist:.4f}" if result.technical.macd_hist is not None else "N/A"
        return f"""## Technical Analysis for {result.symbol}

- **Signal:** {result.technical.signal.value}
- **RSI:** {rsi_str}
- **MACD Histogram:** {macd_str}
- **Confidence:** {result.technical.confidence:.2f}

**Interpretation:**
{result.technical.interpretation}"""

    def _format_sentiment(self, result: TradingWorkflowResult) -> str:
        """Format sentiment analysis."""
        return f"""## Sentiment Analysis for {result.symbol}

- **Overall:** {result.sentiment.overall_sentiment}
- **Score:** {result.sentiment.sentiment_score:.2f}
- **Articles:** {result.sentiment.article_count}
- **Positive:** {result.sentiment.positive_ratio * 100:.1f}%
- **Negative:** {result.sentiment.negative_ratio * 100:.1f}%"""

    def _format_news(self, result: TradingWorkflowResult) -> str:
        """Format news analysis."""
        themes = ", ".join(result.news.key_themes[:5]) if result.news.key_themes else "None"
        return f"""## News Analysis for {result.symbol}

**Key Themes:** {themes}

**Impact Assessment:**
{result.news.impact_assessment}

**Recommendation:**
{result.news.recommendation}"""

    async def _cmd_trump(self, _args: list[str]) -> CommandResult:
        """Switch to Trump personality mode."""
        if not self._app:
            return CommandResult(success=False, message="App not initialized")
        self._app.set_personality("trump")
        return CommandResult(
            success=True,
            message=(
                "🇺🇸 **Switched to TRUMP MODE!** 🇺🇸\n\n"
                "I'm Donald J. Trump, and we're going to Make Portfolio Great Again! "
                "The markets are FANTASTIC! Let's talk stocks - BELIEVE ME! 🚀💰"
            ),
        )

    async def _cmd_casino(self, _args: list[str]) -> CommandResult:
        """Switch to AI Casino personality mode."""
        if not self._app:
            return CommandResult(success=False, message="App not initialized")
        self._app.set_personality("casino")
        return CommandResult(
            success=True,
            message=(
                "🎰 **Switched to AI CASINO MODE!** 🎰\n\n"
                "Back to finding alpha and calling out BS! Let's dissect some markets! 📊🔥"
            ),
        )

    async def _cmd_screen(
        self,
        args: list[str],
        progress_callback: ProgressCallback | None = None,
        is_cancelled: CancelledCallback | None = None,
    ) -> CommandResult:
        """Run stock screening.

        Usage: /screen [criteria] [universe] [top_n] [--save]

        Examples:
            /screen momentum
            /screen value SP500 20
            /screen breakout NASDAQ100 --save
        """
        from src.tui.worker import run_screening_in_process

        criteria = "momentum"
        universe = "COMBINED"
        top_n = 10
        save_to_watchlist = False

        for arg in args:
            if arg == "--save":
                save_to_watchlist = True
            elif arg.lower() in ("momentum", "value", "breakout"):
                criteria = arg.lower()
            elif arg.upper() in ("SP500", "NASDAQ100", "COMBINED"):
                universe = arg.upper()
            elif arg.isdigit():
                top_n = max(1, min(int(arg), 50))

        result_dict = await run_screening_in_process(
            criteria=criteria,
            universe=universe,
            top_n=top_n,
            save_to_watchlist=save_to_watchlist,
            progress_callback=progress_callback,
            is_cancelled=is_cancelled,
        )

        if progress_callback:
            progress_callback("analyzing", "complete", "")

        self._last_screening_output = result_dict.get("screening_output")

        message = result_dict.get("formatted_output", "Screening complete")
        return CommandResult(
            success=True,
            message=message,
            data={
                "criteria": criteria,
                "universe": universe,
                "count": len(result_dict.get("screening_output", {}).get("results", [])),
            },
        )

    async def _cmd_export(self, args: list[str]) -> CommandResult:
        """Export last screening results.

        Usage: /export [format] [filename]

        Examples:
            /export csv
            /export json my_picks
        """
        if not self._last_screening_output:
            return CommandResult(
                success=False,
                message="No screening results to export. Run /screen first.",
            )

        from src.screening.exporter import ScreeningExporter
        from src.screening.screener import ScreeningOutput

        export_format = "csv"
        filename = None

        for arg in args:
            if arg.lower() in ("csv", "json"):
                export_format = arg.lower()
            else:
                filename = arg

        exporter = ScreeningExporter()
        output = ScreeningOutput.model_validate(self._last_screening_output)

        if export_format == "csv":
            filepath = exporter.export_to_csv(output, filename)
        else:
            filepath = exporter.export_to_json(output, filename)

        return CommandResult(
            success=True,
            message=f"Exported {len(output.results)} results to:\n`{filepath}`",
            data={"filepath": str(filepath), "format": export_format},
        )

    async def _cmd_watchlist(self, args: list[str]) -> CommandResult:
        """Manage watchlists.

        Usage:
            /watchlist              - show default watchlist
            /watchlist list         - list all watchlists
            /watchlist show NAME    - show specific watchlist
            /watchlist remove SYM   - remove symbol from default
        """
        from src.screening.exporter import ScreeningExporter

        exporter = ScreeningExporter()

        if not args:
            return self._handle_watchlist_default(exporter)

        action = args[0].lower()
        handlers = {
            "list": lambda: self._handle_watchlist_list(exporter),
            "show": lambda: self._handle_watchlist_show(exporter, args),
            "remove": lambda: self._handle_watchlist_remove(exporter, args),
        }

        handler = handlers.get(action)
        if handler:
            return handler()

        return CommandResult(
            success=False,
            message="Unknown watchlist action. Use: list, show NAME, remove SYMBOL",
        )

    def _handle_watchlist_default(self, exporter: ScreeningExporter) -> CommandResult:
        """Handle showing default watchlist."""
        watchlist = exporter.load_watchlist("default")
        if not watchlist:
            return CommandResult(
                success=True,
                message="Default watchlist is empty. Use `/screen ... --save` to add stocks.",
            )
        return CommandResult(
            success=True,
            message=self._format_watchlist(watchlist),
            data={"name": "default", "count": len(watchlist.entries)},
        )

    def _handle_watchlist_list(self, exporter: ScreeningExporter) -> CommandResult:
        """Handle listing all watchlists."""
        watchlists = exporter.list_watchlists()
        if not watchlists:
            return CommandResult(success=True, message="No watchlists found.")
        return CommandResult(
            success=True,
            message="**Available Watchlists:**\n" + "\n".join(f"- {w}" for w in watchlists),
        )

    def _handle_watchlist_show(self, exporter: ScreeningExporter, args: list[str]) -> CommandResult:
        """Handle showing a specific watchlist."""
        if len(args) <= 1:
            return CommandResult(success=False, message="Usage: /watchlist show NAME")
        name = args[1]
        watchlist = exporter.load_watchlist(name)
        if not watchlist:
            return CommandResult(success=False, message=f"Watchlist '{name}' not found.")
        return CommandResult(
            success=True,
            message=self._format_watchlist(watchlist),
            data={"name": name, "count": len(watchlist.entries)},
        )

    def _handle_watchlist_remove(self, exporter: ScreeningExporter, args: list[str]) -> CommandResult:
        """Handle removing a symbol from watchlist."""
        if len(args) <= 1:
            return CommandResult(success=False, message="Usage: /watchlist remove SYMBOL")
        symbol = args[1].upper()
        watchlist_name = args[2] if len(args) > 2 else "default"
        success = exporter.remove_from_watchlist(symbol, watchlist_name)
        if success:
            return CommandResult(
                success=True,
                message=f"Removed {symbol} from watchlist '{watchlist_name}'.",
            )
        return CommandResult(
            success=False,
            message=f"{symbol} not found in watchlist '{watchlist_name}'.",
        )

    async def _cmd_candidates(self, args: list[str]) -> CommandResult:
        """Manage after-hours screening candidates.

        Usage:
            /candidates              - show latest screening candidates
            /candidates add SYM...   - add candidates to watchlist
            /candidates clear        - clear old candidates
        """
        from pathlib import Path

        from src.daemon.config import DaemonConfig
        from src.daemon.state import DaemonState

        # Load daemon state
        config_path = Path("~/.ai-casino/daemon.yaml").expanduser()
        if config_path.exists():
            config = DaemonConfig.from_yaml(config_path)
            state_file = config.state.state_file
        else:
            state_file = "~/.ai-casino/daemon-state.json"

        state = DaemonState.load(state_file)

        # Handle subcommands
        if args and args[0].lower() == "add":
            return self._handle_candidates_add(args[1:], state, state_file)
        if args and args[0].lower() == "clear":
            return self._handle_candidates_clear(state, state_file)

        # Show latest candidates
        if not state.screening_history:
            return CommandResult(
                success=True,
                message="No screening candidates yet. Enable after-hours screening in daemon.yaml.",
            )

        latest = state.screening_history[-1]
        return CommandResult(
            success=True,
            message=self._format_candidates(latest),
            data={"count": len(latest.candidates)},
        )

    def _handle_candidates_add(self, symbols: list[str], state: object, _state_file: str) -> CommandResult:
        """Add candidates to watchlist.

        Args:
            symbols: List of stock symbols to add
            state: DaemonState instance
            _state_file: Path to state file (unused but kept for consistency)
        """
        from src.screening.exporter import ScreeningExporter
        from src.screening.screener import ScreeningCriteria

        if not symbols:
            return CommandResult(success=False, message="Usage: /candidates add SYMBOL [SYMBOL...]")

        if not state.screening_history:
            return CommandResult(success=False, message="No screening candidates available.")

        latest = state.screening_history[-1]
        exporter = ScreeningExporter()

        # Find matching candidates and add to watchlist
        selected = []
        added = []
        for sym in symbols:
            sym_upper = sym.upper()
            candidate = next((c for c in latest.candidates if c.symbol == sym_upper), None)
            if candidate:
                selected.append(candidate)
                added.append(sym_upper)

        if selected:
            # Use save_to_watchlist with criteria from ScreeningRecord
            criteria = ScreeningCriteria(latest.criteria)
            exporter.save_to_watchlist(selected, criteria, "default")

            return CommandResult(
                success=True,
                message=f"Added to watchlist: {', '.join(added)}",
                data={"added": added},
            )

        return CommandResult(
            success=False,
            message=f"No matching candidates found for: {', '.join(symbols)}",
        )

    def _handle_candidates_clear(self, state: object, state_file: str) -> CommandResult:
        """Clear old screening candidates."""
        cleared = len(state.screening_history)
        state.screening_history = []
        state.last_after_hours_screening = None
        state.save(state_file)

        return CommandResult(
            success=True,
            message=f"Cleared {cleared} screening record(s).",
            data={"cleared": cleared},
        )

    def _format_candidates(self, record: object) -> str:
        """Format screening candidates for display.

        Args:
            record: ScreeningRecord instance or dict
        """
        from src.daemon.state import ScreeningRecord

        if isinstance(record, dict):
            record = ScreeningRecord.model_validate(record)

        lines = [
            "## After-Hours Screening Candidates",
            f"*{record.criteria.title()} | {record.universe} | "
            f"{record.screened_at.strftime('%Y-%m-%d %H:%M')}*",
            "",
        ]

        for i, candidate in enumerate(record.candidates, 1):
            lines.append(
                f"{i}. **{candidate.symbol}** ({candidate.name}) - Score: {candidate.score:.2f}\n"
                f"   {candidate.reason}"
            )

        lines.append("\n*Use `/candidates add SYMBOL [SYMBOL...]` to add to watchlist*")
        return "\n".join(lines)

    def _format_watchlist(self, watchlist: Watchlist) -> str:
        """Format watchlist for display."""
        lines = [
            f"## Watchlist: {watchlist.name}",
            f"*Updated: {watchlist.updated_at.strftime('%Y-%m-%d %H:%M')}*",
            "",
        ]

        if not watchlist.entries:
            lines.append("*No entries*")
        else:
            for entry in watchlist.entries:
                notes_str = f" - {entry.notes}" if entry.notes else ""
                lines.append(
                    f"- **{entry.symbol}** ({entry.name}) | "
                    f"Score: {entry.score:.2f} | {entry.criteria.value}{notes_str}"
                )

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return string representation."""
        return "CommandHandler()"
