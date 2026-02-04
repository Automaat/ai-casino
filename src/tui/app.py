"""Main TUI application for interactive chat."""

import json
import os
from pathlib import Path

from loguru import logger
from textual.app import App, ComposeResult
from textual.binding import Binding

from src.models.llm import LLMClient
from src.tui.commands import CommandHandler
from src.tui.themes import NORD_LIGHT_THEME, detect_dark_mode
from src.tui.widgets.autocomplete_input import AutocompleteInput
from src.tui.widgets.chat_view import ChatView
from src.tui.widgets.status_bar import StatusBar
from src.workflows.trading import TradingWorkflowResult

HISTORY_FILE = Path("~/.ai-casino/chat-history.json").expanduser()


class TradingChatApp(App):
    """Interactive TUI for AI Casino."""

    TITLE = "AI Casino"
    CSS_PATH = "app.tcss"

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear", "Clear"),
        Binding("ctrl+t", "toggle_theme", "Toggle Theme"),
        Binding("escape", "focus_input", "Focus Input"),
    ]

    def __init__(self) -> None:
        """Initialize the app."""
        super().__init__()
        self._command_handler = CommandHandler()
        self._llm: LLMClient | None = None
        self._history: list[dict[str, str]] = []
        self._model_name = self._get_model_name()
        self._load_history()

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
        yield AutocompleteInput(placeholder="> Type a message or /help...", widget_id="input-box")
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

        chat = self.query_one(ChatView)
        chat.add_user_message(text)
        self._history.append({"role": "user", "content": text})

        if self._command_handler.is_command(text):
            await self._handle_command(text)
        else:
            await self._handle_chat(text)

        self._save_history()

    async def _handle_command(self, text: str) -> None:
        """Handle slash command."""
        chat = self.query_one(ChatView)
        status_bar = self.query_one(StatusBar)

        cmd, args = self._command_handler.parse_command(text)
        symbol = args[0].upper() if args else ""

        if cmd == "analyze" and symbol:
            status_bar.set_working(f"Analyzing {symbol}...")
            tool_widget = chat.show_tool_call("Trading Analysis", f"{symbol} full analysis")
            chat.show_progress(symbol)

            def progress_callback(step_id: str, status: str) -> None:
                chat.update_progress(step_id, status)

            result = await self._command_handler.execute(text, progress_callback)

            if result.success:
                chat.complete_progress()
                tool_widget.set_complete("Analysis complete")
                if result.workflow_result and isinstance(result.workflow_result, TradingWorkflowResult):
                    chat.show_result_box(result.workflow_result)
                else:
                    chat.add_assistant_message(result.message)
            else:
                tool_widget.set_complete("Analysis failed")
                chat.add_assistant_message(result.message)

            status_bar.clear_working()
            self._history.append({"role": "assistant", "content": result.message})
        else:
            result = await self._command_handler.execute(text)
            chat.add_assistant_message(result.message)
            self._history.append({"role": "assistant", "content": result.message})

    async def _handle_chat(self, text: str) -> None:
        """Handle free-form chat with streaming."""
        chat = self.query_one(ChatView)
        status_bar = self.query_one(StatusBar)

        if self._llm is None:
            self._llm = LLMClient()

        system_prompt = """You are AI Casino, an expert assistant for stock trading and market analysis.
You help users understand markets, trading strategies, and financial concepts.
Be concise but informative. Use markdown formatting for readability."""

        status_bar.set_working("Thinking...")
        chat.start_streaming_message()
        response_text = ""

        try:
            async for token in self._llm.astream(text, system=system_prompt, temperature=0.7):
                response_text += token
                chat.append_token(token)

            chat.finish_streaming()
            self._history.append({"role": "assistant", "content": response_text})
        except Exception as e:
            logger.exception("Chat failed")
            chat.finish_streaming()
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
        """Focus the input box."""
        self.query_one(AutocompleteInput).focus()

    def action_toggle_theme(self) -> None:
        """Toggle between Nord dark and light themes."""
        self.theme = "nord-light" if self.theme == "nord" else "nord"
