"""Main TUI application for interactive chat."""

import json
from pathlib import Path

from loguru import logger
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Input

from src.models.llm import LLMClient
from src.tui.commands import CommandHandler
from src.tui.widgets.chat_view import ChatView
from src.tui.widgets.status_bar import StatusBar

HISTORY_FILE = Path("~/.ai-casino/chat-history.json").expanduser()


class TradingChatApp(App):
    """Interactive TUI for AI Casino."""

    TITLE = "AI Casino"
    CSS = """
    Screen {
        layout: vertical;
    }

    #chat-container {
        height: 1fr;
    }

    #input-box {
        dock: bottom;
        margin: 1 1 0 1;
    }

    Input {
        dock: bottom;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear", "Clear"),
        Binding("escape", "focus_input", "Focus Input"),
    ]

    def __init__(self) -> None:
        """Initialize the app."""
        super().__init__()
        self._command_handler = CommandHandler()
        self._llm: LLMClient | None = None
        self._history: list[dict[str, str]] = []
        self._load_history()

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
        yield Header()
        yield ChatView(id="chat-container")
        yield Input(placeholder="Type a message or /help for commands...", id="input-box")
        yield StatusBar()
        yield Footer()

    def on_mount(self) -> None:
        """Handle app mount."""
        chat = self.query_one(ChatView)
        chat.add_message(
            "Welcome to AI Casino! Type /help for available commands or chat freely about markets.",
            "assistant",
        )
        self.query_one(Input).focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        text = event.value.strip()
        if not text:
            return

        event.input.value = ""

        chat = self.query_one(ChatView)
        chat.add_message(text, "user")
        self._history.append({"role": "user", "content": text})

        if self._command_handler.is_command(text):
            await self._handle_command(text)
        else:
            await self._handle_chat(text)

        self._save_history()

    async def _handle_command(self, text: str) -> None:
        """Handle slash command."""
        chat = self.query_one(ChatView)

        result = await self._command_handler.execute(text)

        chat.add_message(result.message, "assistant")
        self._history.append({"role": "assistant", "content": result.message})

    async def _handle_chat(self, text: str) -> None:
        """Handle free-form chat."""
        chat = self.query_one(ChatView)

        if self._llm is None:
            self._llm = LLMClient()

        system_prompt = """You are AI Casino, an expert assistant for stock trading and market analysis.
You help users understand markets, trading strategies, and financial concepts.
Be concise but informative. Use markdown formatting for readability."""

        response_text = ""

        try:
            async for token in self._llm.astream(text, system=system_prompt, temperature=0.7):
                response_text += token

            chat.add_message(response_text, "assistant")
            self._history.append({"role": "assistant", "content": response_text})
        except Exception as e:
            logger.exception("Chat failed")
            chat.add_message(f"Error: {e}", "assistant")

    def action_clear(self) -> None:
        """Clear chat messages."""
        chat = self.query_one(ChatView)
        chat.clear_messages()
        chat.add_message("Chat cleared. Type /help for commands.", "assistant")
        self._history = []
        self._save_history()

    def action_focus_input(self) -> None:
        """Focus the input box."""
        self.query_one(Input).focus()
