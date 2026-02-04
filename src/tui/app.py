"""Main TUI application for interactive chat."""

import json
import os
from pathlib import Path

from loguru import logger
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.events import Click
from textual.worker import Worker, get_current_worker

from src.models.llm import LLMClient
from src.tools import (
    AnalyzeStockTool,
    GetMarketDataTool,
    GetNewsTool,
    ScreenStocksTool,
    ToolRegistry,
    WebSearchTool,
)
from src.tui.commands import CommandHandler
from src.tui.events import AnalysisComplete, AnalysisProgress
from src.tui.themes import NORD_LIGHT_THEME, detect_dark_mode
from src.tui.widgets.autocomplete_input import AutocompleteInput
from src.tui.widgets.chat_view import ChatView
from src.tui.widgets.status_bar import StatusBar
from src.workflows.trading import TradingWorkflowResult

HISTORY_FILE = Path("~/.ai-casino/chat-history.json").expanduser()

AGENTIC_SYSTEM_PROMPT = """You are AI Casino, an expert assistant for stock trading and market analysis.
You have access to the following tools to help users:

- get_market_data: Fetch current stock prices and market data
- get_news: Fetch recent news articles for a company
- web_search: Search the web for information
- analyze_stock: Run comprehensive trading analysis (EXPENSIVE - requires confirmation)
- screen_stocks: Screen stocks for investment opportunities (EXPENSIVE - requires confirmation)

Use tools when appropriate to answer user questions. Be concise but informative.
Use markdown formatting for readability."""

STREAMING_SYSTEM_PROMPT = """You are AI Casino, an expert assistant for stock trading and market analysis.
You help users understand markets, trading strategies, and financial concepts.
Be concise but informative. Use markdown formatting for readability."""


class TradingChatApp(App):
    """Interactive TUI for AI Casino."""

    TITLE = "AI Casino"
    CSS_PATH = "app.tcss"

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear", "Clear"),
        Binding("ctrl+t", "toggle_theme", "Toggle Theme"),
        Binding("escape", "focus_input", "Cancel/Focus"),
    ]

    def __init__(self) -> None:
        """Initialize the app."""
        super().__init__()
        self._command_handler = CommandHandler()
        self._llm: LLMClient | None = None
        self._history: list[dict[str, str]] = []
        self._model_name = self._get_model_name()
        self._analysis_worker: Worker | None = None
        self._tool_registry = self._create_tool_registry()
        self._pending_tool_confirmation: dict | None = None
        self._load_history()

    def _create_tool_registry(self) -> ToolRegistry:
        """Create and populate tool registry."""
        registry = ToolRegistry()
        registry.register(WebSearchTool())
        registry.register(GetMarketDataTool())
        registry.register(GetNewsTool())
        registry.register(AnalyzeStockTool())
        registry.register(ScreenStocksTool())
        return registry

    def _get_model_name(self) -> str:
        """Get current LLM model name from env."""
        provider = os.getenv("LLM_PROVIDER", "ollama")
        model = os.getenv("LLM_MODEL", "qwen3:14b")
        return f"{provider}/{model}"

    def _load_history(self) -> None:
        """Load chat history from file."""
        if HISTORY_FILE.exists():
            try:
                with HISTORY_FILE.open() as f:
                    self._history = json.load(f)
                    self._history = self._history[-50:]
            except Exception as e:
                logger.warning(f"Failed to load history: {e}")
                self._history = []

    def _save_history(self) -> None:
        """Save chat history to file."""
        try:
            HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            with HISTORY_FILE.open("w") as f:
                json.dump(self._history[-100:], f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save history: {e}")

    def compose(self) -> ComposeResult:
        """Compose the app layout."""
        yield ChatView(id="chat-container")
        yield AutocompleteInput(
            placeholder="> Type a message or /help...",
            commands=self._command_handler.command_names,
            widget_id="input-box",
        )
        yield StatusBar()

    def on_mount(self) -> None:
        """Handle app mount."""
        self.register_theme(NORD_LIGHT_THEME)
        self.theme = "nord" if detect_dark_mode() else "nord-light"
        chat = self.query_one(ChatView)
        chat.show_welcome(self._model_name)
        self.query_one(AutocompleteInput).focus()

    async def on_autocomplete_input_submitted(self, event: AutocompleteInput.Submitted) -> None:
        """Handle input submission."""
        text = event.value.strip()
        if not text:
            return

        input_widget = self.query_one(AutocompleteInput)
        input_widget.value = ""

        if self._pending_tool_confirmation:
            await self._handle_tool_confirmation(text)
            return

        chat = self.query_one(ChatView)
        chat.add_user_message(text)
        self._history.append({"role": "user", "content": text})

        if self._command_handler.is_command(text):
            await self._handle_command(text)
        else:
            await self._handle_chat(text)

        self._save_history()

    async def _handle_tool_confirmation(self, text: str) -> None:
        """Handle user response to tool confirmation prompt."""
        chat = self.query_one(ChatView)
        pending = self._pending_tool_confirmation
        self._pending_tool_confirmation = None

        chat.add_user_message(text)

        tool_widgets = chat.query("ToolCallWidget")
        tool_widget = tool_widgets.last() if tool_widgets else None

        if text.lower() in ("yes", "y"):
            result = self._tool_registry.execute(pending["name"], pending["args"])
            result_preview = result[:100] + "..." if len(result) > 100 else result
            if tool_widget:
                tool_widget.set_complete(result_preview)
            chat.add_assistant_message(f"Tool result:\n\n{result}")
        else:
            if tool_widget:
                tool_widget.set_complete("Skipped")
            chat.add_assistant_message("Tool execution skipped.")

        self._save_history()

    async def _handle_command(self, text: str) -> None:
        """Handle slash command."""
        chat = self.query_one(ChatView)
        status_bar = self.query_one(StatusBar)

        cmd, args = self._command_handler.parse_command(text)
        symbol = args[0].upper() if args else ""

        if cmd == "analyze" and symbol:
            status_bar.set_working(f"Analyzing {symbol}...")
            chat.show_tool_call("Trading Analysis", f"{symbol} full analysis")
            chat.show_progress(symbol)
            self._analysis_worker = self._run_analysis_worker(text, symbol)
        else:
            result = await self._command_handler.execute(text)
            chat.add_assistant_message(result.message)
            self._history.append({"role": "assistant", "content": result.message})

    @work(thread=True, exclusive=True)
    def _run_analysis_worker(self, text: str, symbol: str) -> None:
        """Run analysis in background thread.

        Args:
            text: Full command text
            symbol: Stock ticker symbol
        """
        import asyncio

        worker = get_current_worker()

        def progress_callback(step_id: str, status: str, detail: str) -> None:
            if not worker.is_cancelled:
                self.post_message(AnalysisProgress(step_id, status, detail))

        def is_cancelled() -> bool:
            return worker.is_cancelled

        try:
            result = asyncio.run(self._command_handler.execute(text, progress_callback, is_cancelled))
            if not worker.is_cancelled:
                self.post_message(AnalysisComplete(result, symbol))
        except asyncio.CancelledError:
            logger.info("Analysis worker cancelled for %s", symbol)

    def on_analysis_progress(self, event: AnalysisProgress) -> None:
        """Handle progress update from worker."""
        chat = self.query_one(ChatView)
        chat.update_progress(event.step_id, event.status, event.detail)

    def on_analysis_complete(self, event: AnalysisComplete) -> None:
        """Handle analysis completion from worker."""
        chat = self.query_one(ChatView)
        status_bar = self.query_one(StatusBar)

        tool_widgets = chat.query("ToolCallWidget")
        tool_widget = tool_widgets.last() if tool_widgets else None

        if event.result.success:
            chat.complete_progress()
            if tool_widget:
                tool_widget.set_complete("Analysis complete")
            if event.result.workflow_result and isinstance(
                event.result.workflow_result, TradingWorkflowResult
            ):
                chat.show_result_box(event.result.workflow_result)
            else:
                chat.add_assistant_message(event.result.message)
        else:
            chat.complete_progress()
            if tool_widget:
                tool_widget.set_complete("Analysis failed")
            chat.add_assistant_message(event.result.message)

        status_bar.clear_working()
        self._history.append({"role": "assistant", "content": event.result.message})
        self._analysis_worker = None
        self._save_history()

    async def _handle_chat(self, text: str) -> None:
        """Handle free-form chat - dispatch to agentic or streaming mode."""
        if self._llm is None:
            self._llm = LLMClient()

        if self._llm.supports_tools:
            await self._handle_agentic_chat(text)
        else:
            await self._handle_streaming_chat(text)

    async def _handle_streaming_chat(self, text: str) -> None:
        """Handle chat with streaming (Ollama fallback)."""
        chat = self.query_one(ChatView)
        status_bar = self.query_one(StatusBar)

        status_bar.set_working("Thinking...")
        chat.show_thinking()
        chat.start_streaming_message()
        response_text = ""
        first_token = True

        try:
            async for token in self._llm.astream(text, system=STREAMING_SYSTEM_PROMPT, temperature=0.7):
                if first_token:
                    chat.hide_thinking()
                    first_token = False
                response_text += token
                chat.append_token(token)

            chat.finish_streaming()
            self._history.append({"role": "assistant", "content": response_text})
        except Exception as e:
            logger.exception("Chat failed")
            chat.hide_thinking()
            chat.finish_streaming()
            chat.add_assistant_message(f"Error: {e}")
        finally:
            status_bar.clear_working()

    async def _handle_agentic_chat(self, text: str) -> None:
        """Handle chat with tool calling (Anthropic/OpenAI)."""
        chat = self.query_one(ChatView)
        status_bar = self.query_one(StatusBar)

        status_bar.set_working("Thinking...")
        chat.show_thinking()

        confirmed_tools: set[str] = set()
        tool_history: list[dict[str, str]] = []

        def on_tool_call(name: str, args: dict, result: str) -> None:
            """Callback for tool execution updates."""
            tool_history.append(
                {"role": "assistant", "content": f"[Used tool {name} with {args}]\n\nResult: {result[:500]}"}
            )
            if name in confirmed_tools:
                return
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
            widget = chat.show_tool_call(name, args_str)
            result_preview = result[:100] + "..." if len(result) > 100 else result
            widget.set_complete(result_preview)

        def tool_executor(name: str, args: dict) -> str:
            """Execute tool with confirmation check."""
            if self._tool_registry.requires_confirmation(name):
                confirmed_tools.add(name)
                args_str = ", ".join(f"{k}={v}" for k, v in args.items())
                chat.show_tool_call(name, args_str)
                chat.add_assistant_message(
                    f"Tool `{name}` requires confirmation. Type 'yes' to proceed or anything else to skip."
                )
                self._pending_tool_confirmation = {"name": name, "args": args}
                return f"[Awaiting user confirmation for {name}]"
            return self._tool_registry.execute(name, args)

        try:
            import asyncio

            response = await asyncio.to_thread(
                self._llm.complete_with_tools,
                text,
                self._tool_registry.get_definitions(),
                tool_executor,
                AGENTIC_SYSTEM_PROMPT,
                0.7,
                5,
                on_tool_call,
            )

            chat.hide_thinking()
            chat.add_assistant_message(response)
            self._history.extend(tool_history)
            self._history.append({"role": "assistant", "content": response})
        except Exception as e:
            logger.exception("Agentic chat failed")
            chat.hide_thinking()
            chat.add_assistant_message(f"Error: {e}")
        finally:
            status_bar.clear_working()

    def action_clear(self) -> None:
        """Clear chat messages."""
        chat = self.query_one(ChatView)
        chat.clear_messages()
        chat.show_welcome(self._model_name)
        self._history = []
        self._save_history()

    def action_focus_input(self) -> None:
        """Focus input or cancel running analysis."""
        if self._analysis_worker and self._analysis_worker.is_running:
            self._analysis_worker.cancel()
            self._on_analysis_cancelled()
        self.query_one(AutocompleteInput).focus()

    def _on_analysis_cancelled(self) -> None:
        """Handle analysis cancellation."""
        chat = self.query_one(ChatView)
        status_bar = self.query_one(StatusBar)

        tool_widgets = chat.query("ToolCallWidget")
        tool_widget = tool_widgets.last() if tool_widgets else None

        chat.complete_progress()
        if tool_widget:
            tool_widget.set_complete("Cancelled")
        chat.add_assistant_message("Analysis cancelled.")
        self._history.append({"role": "assistant", "content": "Analysis cancelled."})
        self._save_history()
        status_bar.clear_working()
        self._analysis_worker = None

    def action_toggle_theme(self) -> None:
        """Toggle between Nord dark and light themes."""
        self.theme = "nord-light" if self.theme == "nord" else "nord"

    def on_click(self, _event: Click) -> None:
        """Keep focus on input after any click."""
        self.query_one(AutocompleteInput).focus()
