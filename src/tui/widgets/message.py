"""Message widgets for chat display."""

from loguru import logger
from rich.markdown import Markdown as RichMarkdown
from textual.reactive import reactive
from textual.selection import Selection
from textual.widgets import Static


class SelectionSafeMixin:
    """Mixin to handle Textual selection bug with coordinate mismatch."""

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        """Override to catch IndexError from Textual selection bug.

        Textual 7.5.0 has a bug where selection coordinates can be screen-relative
        but text extraction expects widget-relative coordinates, causing IndexError.
        """
        try:
            return super().get_selection(selection)  # type: ignore[misc]
        except IndexError:
            return None


class UserMessage(SelectionSafeMixin, Static):
    """User message with prompt indicator."""

    DEFAULT_CSS = """
    UserMessage {
        background: transparent;
        color: #C5CDD9;
        padding: 0;
        margin: 1 0 0 0;
        height: auto;
    }
    """

    def __init__(self, content: str) -> None:
        """Initialize user message."""
        super().__init__(f"> {content}")


class AssistantMessage(SelectionSafeMixin, Static):
    """Assistant message - clean text with bullet."""

    DEFAULT_CSS = """
    AssistantMessage {
        background: transparent;
        color: #C5CDD9;
        padding: 0;
        margin: 0;
        height: auto;
    }

    AssistantMessage.streaming {
        color: #8899A6;
    }
    """

    message_content: reactive[str] = reactive("")
    is_streaming: reactive[bool] = reactive(default=False)

    def __init__(self, content: str = "", streaming: bool = False) -> None:
        """Initialize assistant message."""
        super().__init__()
        self.message_content = content
        self.is_streaming = streaming
        if streaming:
            self.add_class("streaming")

    def on_mount(self) -> None:
        """Update display on mount."""
        self._update_display()

    def watch_message_content(self, _content: str) -> None:
        """React to content changes."""
        self._update_display()

    def watch_is_streaming(self, streaming: bool) -> None:
        """React to streaming state changes."""
        if streaming:
            self.add_class("streaming")
        else:
            self.remove_class("streaming")

    def _update_display(self) -> None:
        """Update displayed content with markdown rendering."""
        try:
            content = self.message_content or "..."
            # Render markdown content with Rich
            rendered = RichMarkdown(content)
            self.update(rendered)
            if self.parent and hasattr(self.parent, "scroll_end"):
                self.parent.scroll_end(animate=False)
        except Exception as e:
            logger.debug(f"Message update skipped: {e}")

    def append_token(self, token: str) -> None:
        """Append a token to the message."""
        self.message_content += token

    def finish_streaming(self) -> None:
        """Mark streaming as complete."""
        self.is_streaming = False


class ToolCallWidget(SelectionSafeMixin, Static):
    """Widget showing tool/agent activity."""

    DEFAULT_CSS = """
    ToolCallWidget {
        background: transparent;
        color: #5DADE2;
        padding: 0;
        margin: 1 0 0 0;
        height: auto;
    }
    """

    def __init__(self, tool_name: str, args: str = "", status: str = "running") -> None:
        """Initialize tool call widget."""
        display = f"● {tool_name}"
        if args:
            display += f'("{args[:40]}...")'
        super().__init__(display)
        self._tool_name = tool_name
        self._status = status

    def set_complete(self, timing: str) -> None:
        """Mark tool call as complete with timing."""
        self._status = "complete"
        self.update(f"● {self._tool_name}\n  └ {timing}")

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ToolCallWidget(tool={self._tool_name}, status={self._status})"


class WelcomeWidget(Static):
    """Welcome screen with ASCII logo."""

    ASCII_LOGO = """ █████╗ ██╗     ██████╗ █████╗ ███████╗██╗███╗   ██╗ ██████╗
██╔══██╗██║    ██╔════╝██╔══██╗██╔════╝██║████╗  ██║██╔═══██╗
███████║██║    ██║     ███████║███████╗██║██╔██╗ ██║██║   ██║
██╔══██║██║    ██║     ██╔══██║╚════██║██║██║╚██╗██║██║   ██║
██║  ██║██║    ╚██████╗██║  ██║███████║██║██║ ╚████║╚██████╔╝
╚═╝  ╚═╝╚═╝     ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝╚═╝  ╚═══╝ ╚═════╝"""

    DEFAULT_CSS = """
    WelcomeWidget {
        background: transparent;
        color: #5DADE2;
        padding: 0;
        margin: 2 0 1 0;
        height: auto;
    }
    """

    def __init__(self, model_name: str = "ollama/qwen3:14b") -> None:
        """Initialize welcome widget."""
        text = f"""{self.ASCII_LOGO}

Your AI assistant for stock trading analysis.
Current model: {model_name}
Type /help for commands or ask about any stock."""
        super().__init__(text)

    def __repr__(self) -> str:
        """Return string representation."""
        return "WelcomeWidget()"
