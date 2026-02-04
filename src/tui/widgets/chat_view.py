"""Chat view widget for message display."""

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Markdown, Static


class MessageWidget(Static):
    """A single chat message."""

    def __init__(self, content: str, role: str = "user") -> None:
        """Initialize message widget.

        Args:
            content: Message content (markdown supported)
            role: Message role (user/assistant)
        """
        super().__init__()
        self.content = content
        self.role = role

    def compose(self) -> ComposeResult:
        """Compose the message widget."""
        prefix = "**You:**" if self.role == "user" else "**AI Casino:**"
        yield Markdown(f"{prefix}\n\n{self.content}")


class ChatView(VerticalScroll):
    """Scrollable chat message view."""

    DEFAULT_CSS = """
    ChatView {
        height: 1fr;
        border: solid $primary;
        padding: 1;
    }

    MessageWidget {
        margin-bottom: 1;
        padding: 1;
    }

    MessageWidget.user {
        background: $surface;
    }

    MessageWidget.assistant {
        background: $panel;
    }
    """

    def add_message(self, content: str, role: str = "user") -> None:
        """Add a message to the chat.

        Args:
            content: Message content
            role: Message role (user/assistant)
        """
        msg = MessageWidget(content, role)
        msg.add_class(role)
        self.mount(msg)
        self.scroll_end(animate=False)

    def add_streaming_message(self) -> MessageWidget:
        """Add a placeholder for streaming message.

        Returns:
            The message widget for updating
        """
        msg = MessageWidget("", "assistant")
        msg.add_class("assistant")
        self.mount(msg)
        self.scroll_end(animate=False)
        return msg

    def clear_messages(self) -> None:
        """Clear all messages."""
        for child in self.query(MessageWidget):
            child.remove()
