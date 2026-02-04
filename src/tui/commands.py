"""Command handlers for TUI slash commands."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field

from loguru import logger

from src.data.fundamental import FundamentalDataFetcher
from src.data.market import MarketDataFetcher
from src.data.news import NewsFetcher
from src.models.llm import LLMClient
from src.models.sentiment import FinBERTSentiment
from src.workflows.trading import TradingWorkflow, TradingWorkflowResult

ProgressCallback = Callable[[str, str], None]


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
        self._workflow: TradingWorkflow | None = None
        self._commands: dict[str, Callable[..., Coroutine]] = {
            "analyze": self._cmd_analyze,
            "technical": self._cmd_technical,
            "sentiment": self._cmd_sentiment,
            "news": self._cmd_news,
            "help": self._cmd_help,
        }
        self._progress_callback: ProgressCallback | None = None
        logger.info("CommandHandler initialized")

    def _init_workflow(self) -> TradingWorkflow:
        """Initialize trading workflow lazily."""
        if self._workflow is None:
            llm_client = LLMClient()
            market_fetcher = MarketDataFetcher(use_alpha_vantage=False)
            news_fetcher = NewsFetcher()
            finbert = FinBERTSentiment()
            fundamental_fetcher = FundamentalDataFetcher()

            self._workflow = TradingWorkflow(
                llm_client,
                market_fetcher,
                news_fetcher,
                finbert,
                fundamental_fetcher,
                broker=None,
                metrics_tracker=None,
                use_meta_agent=True,
            )
        return self._workflow

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

    async def execute(self, text: str, progress_callback: ProgressCallback | None = None) -> CommandResult:
        """Execute a slash command.

        Args:
            text: Command text
            progress_callback: Optional callback for progress updates (step_id, status)

        Returns:
            CommandResult with execution result
        """
        cmd, args = self.parse_command(text)
        self._progress_callback = progress_callback

        if cmd not in self._commands:
            return CommandResult(
                success=False,
                message=f"Unknown command: /{cmd}. Type /help for available commands.",
            )

        try:
            return await self._commands[cmd](args)
        except Exception as e:
            logger.exception(f"Command /{cmd} failed")
            return CommandResult(success=False, message=f"Command failed: {e}")
        finally:
            self._progress_callback = None

    def _report_progress(self, step_id: str, status: str = "active") -> None:
        """Report progress to callback if set.

        Args:
            step_id: Step identifier
            status: Step status
        """
        if self._progress_callback:
            self._progress_callback(step_id, status)

    async def _cmd_help(self, _args: list[str]) -> CommandResult:
        """Show help for available commands."""
        help_text = """**Available Commands:**

- **/analyze SYMBOL** - Full trading analysis for a stock
- **/technical SYMBOL** - Technical analysis only
- **/sentiment SYMBOL** - Sentiment analysis only
- **/news SYMBOL** - News analysis only
- **/help** - Show this help message

**Examples:**
- `/analyze AAPL` - Full analysis for Apple
- `/technical TSLA` - Technical analysis for Tesla

Type freely to chat about markets or ask questions."""
        return CommandResult(success=True, message=help_text)

    async def _cmd_analyze(self, args: list[str]) -> CommandResult:
        """Run full trading analysis."""
        if not args:
            return CommandResult(success=False, message="Usage: /analyze SYMBOL")

        symbol = args[0].upper()
        workflow = self._init_workflow()

        self._report_progress("fetch_data", "active")
        result = await workflow.analyze(symbol, period_days=90)
        self._report_progress("decision", "complete")

        cmd_result = self._format_analysis_result(result)
        cmd_result.workflow_result = result
        return cmd_result

    async def _cmd_technical(self, args: list[str]) -> CommandResult:
        """Run technical analysis only."""
        if not args:
            return CommandResult(success=False, message="Usage: /technical SYMBOL")

        symbol = args[0].upper()
        workflow = self._init_workflow()

        result = await workflow.analyze(symbol, period_days=90)
        msg = self._format_technical(result)
        return CommandResult(success=True, message=msg, data={"symbol": symbol})

    async def _cmd_sentiment(self, args: list[str]) -> CommandResult:
        """Run sentiment analysis only."""
        if not args:
            return CommandResult(success=False, message="Usage: /sentiment SYMBOL")

        symbol = args[0].upper()
        workflow = self._init_workflow()

        result = await workflow.analyze(symbol, period_days=90)
        msg = self._format_sentiment(result)
        return CommandResult(success=True, message=msg, data={"symbol": symbol})

    async def _cmd_news(self, args: list[str]) -> CommandResult:
        """Run news analysis only."""
        if not args:
            return CommandResult(success=False, message="Usage: /news SYMBOL")

        symbol = args[0].upper()
        workflow = self._init_workflow()

        result = await workflow.analyze(symbol, period_days=90)
        msg = self._format_news(result)
        return CommandResult(success=True, message=msg, data={"symbol": symbol})

    def _format_analysis_result(self, result: TradingWorkflowResult) -> CommandResult:
        """Format full analysis result."""
        signal = result.decision.action.value
        confidence = result.decision.confidence
        rsi_str = f"{result.technical.rsi:.2f}" if result.technical.rsi else "N/A"

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
        rsi_str = f"{result.technical.rsi:.2f}" if result.technical.rsi else "N/A"
        macd_str = f"{result.technical.macd_hist:.4f}" if result.technical.macd_hist else "N/A"
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

    def __repr__(self) -> str:
        """Return string representation."""
        return "CommandHandler()"
