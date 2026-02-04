"""Chat view widget for message display."""

from textual.containers import VerticalScroll

from src.tui.widgets.message import AssistantMessage, ToolCallWidget, UserMessage, WelcomeWidget
from src.tui.widgets.progress import ProgressPanel
from src.tui.widgets.result_box import ResultBox
from src.workflows.trading import TradingWorkflowResult


class ChatView(VerticalScroll):
    """Scrollable chat message view."""

    DEFAULT_CSS = """
    ChatView {
        height: 1fr;
        background: #1E2530;
        padding: 1;
        border: none;
    }
    """

    def __init__(self, **kwargs: object) -> None:
        """Initialize chat view."""
        super().__init__(**kwargs)
        self._streaming_message: AssistantMessage | None = None
        self._progress_panel: ProgressPanel | None = None

    def show_welcome(self, model_name: str = "ollama/qwen3:14b") -> None:
        """Show welcome screen with logo.

        Args:
            model_name: Current LLM model name
        """
        welcome = WelcomeWidget(model_name)
        self.mount(welcome)

    def add_user_message(self, content: str) -> None:
        """Add a user message.

        Args:
            content: Message content
        """
        msg = UserMessage(content)
        self.mount(msg)
        self.scroll_end(animate=False)

    def add_assistant_message(self, content: str) -> None:
        """Add a complete assistant message.

        Args:
            content: Message content
        """
        msg = AssistantMessage(content, streaming=False)
        self.mount(msg)
        self.scroll_end(animate=False)

    def start_streaming_message(self) -> AssistantMessage:
        """Start a new streaming message.

        Returns:
            The streaming message widget for updates
        """
        self._streaming_message = AssistantMessage("", streaming=True)
        self.mount(self._streaming_message)
        self.scroll_end(animate=False)
        return self._streaming_message

    def append_token(self, token: str) -> None:
        """Append token to current streaming message.

        Args:
            token: Token to append
        """
        if self._streaming_message:
            self._streaming_message.append_token(token)

    def finish_streaming(self) -> None:
        """Finish the current streaming message."""
        if self._streaming_message:
            self._streaming_message.finish_streaming()
            self._streaming_message = None

    def show_tool_call(self, tool_name: str, args: str = "") -> ToolCallWidget:
        """Show a tool call indicator.

        Args:
            tool_name: Name of tool being called
            args: Tool arguments

        Returns:
            The tool call widget
        """
        widget = ToolCallWidget(tool_name, args, "running")
        self.mount(widget)
        self.scroll_end(animate=False)
        return widget

    def show_progress(self, symbol: str) -> ProgressPanel:
        """Show analysis progress panel.

        Args:
            symbol: Stock symbol being analyzed

        Returns:
            The progress panel widget
        """
        self._progress_panel = ProgressPanel(symbol)
        self.mount(self._progress_panel)
        self.scroll_end(animate=False)
        return self._progress_panel

    def update_progress(self, step_id: str, status: str = "active") -> None:
        """Update progress panel step.

        Args:
            step_id: Step identifier
            status: Step status
        """
        if self._progress_panel:
            if status == "active":
                self._progress_panel.set_step_active(step_id)
            elif status == "complete":
                self._progress_panel.set_step_complete(step_id)
            elif status == "error":
                self._progress_panel.set_step_error(step_id)

    def complete_progress(self) -> None:
        """Complete the progress panel."""
        if self._progress_panel:
            self._progress_panel.complete()
            self._progress_panel = None

    def show_result_box(self, result: TradingWorkflowResult) -> ResultBox:
        """Show a boxed analysis result.

        Args:
            result: Trading workflow result

        Returns:
            The result box widget
        """
        box = ResultBox(result)
        self.mount(box)
        self.scroll_end(animate=False)
        return box

    def add_message(self, content: str, role: str = "user") -> None:
        """Add a message (backward compatibility).

        Args:
            content: Message content
            role: Message role (user/assistant)
        """
        if role == "user":
            self.add_user_message(content)
        else:
            self.add_assistant_message(content)

    def clear_messages(self) -> None:
        """Clear all messages."""
        self._streaming_message = None
        self._progress_panel = None
        for child in list(self.children):
            child.remove()

    def __repr__(self) -> str:
        """Return string representation."""
        return "ChatView()"
