"""Progress indicator widgets."""

import contextlib

from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widgets import Static


class TaskStep(Static):
    """Single task step in progress panel."""

    DEFAULT_CSS = """
    TaskStep {
        padding: 0;
        height: auto;
        margin-left: 2;
    }

    TaskStep.pending {
        color: #5A6778;
    }

    TaskStep.active {
        color: #5DADE2;
    }

    TaskStep.complete {
        color: #5DADE2;
    }

    TaskStep.error {
        color: #E74C3C;
    }
    """

    status: reactive[str] = reactive("pending")

    def __init__(self, label: str, status: str = "pending") -> None:
        """Initialize task step."""
        super().__init__()
        self._label = label
        self._detail = ""
        self.status = status

    def on_mount(self) -> None:
        """Apply initial status class."""
        self.add_class(self.status)

    def watch_status(self, old_status: str, new_status: str) -> None:
        """React to status changes."""
        self.remove_class(old_status)
        self.add_class(new_status)
        if new_status != "active":
            self._detail = ""

    def set_detail(self, detail: str) -> None:
        """Set detail text and schedule refresh."""
        if detail != self._detail:
            self._detail = detail
            self.call_later(self.refresh)

    def render(self) -> str:
        """Render the step with optional detail line."""
        icons = {"pending": "○", "active": "◉", "complete": "✓", "error": "✗"}
        icon = icons.get(self.status, "○")
        line = f"{icon} {self._label}"
        if self.status == "active" and self._detail:
            truncated = self._detail[:60] + "..." if len(self._detail) > 60 else self._detail
            line += f"\n    [dim]{escape(truncated)}[/dim]"
        return line

    def __repr__(self) -> str:
        """Return string representation."""
        return f"TaskStep(label={self._label}, status={self.status})"


class ProgressPanel(Static):
    """Panel showing workflow progress - minimal style."""

    DEFAULT_CSS = """
    ProgressPanel {
        background: transparent;
        padding: 0;
        margin: 0;
        height: auto;
    }

    .progress-steps {
        height: auto;
    }
    """

    is_loading: reactive[bool] = reactive(default=True)

    WORKFLOW_STEPS = [
        ("fetch_data", "Fetching market data"),
        ("technical", "Running technical analysis"),
        ("sentiment", "Running sentiment analysis"),
        ("news", "Running news analysis"),
        ("decision", "Making trading decision"),
    ]

    def __init__(self, symbol: str) -> None:
        """Initialize progress panel."""
        super().__init__()
        self._symbol = symbol
        self._steps: dict[str, TaskStep] = {}
        self._current_step: str | None = None

    def compose(self) -> ComposeResult:
        """Compose the progress panel."""
        with Vertical(classes="progress-steps"):
            for step_id, label in self.WORKFLOW_STEPS:
                step = TaskStep(label, "pending")
                self._steps[step_id] = step
                yield step

    def set_step_active(self, step_id: str, detail: str = "") -> None:
        """Mark a step as active with optional detail."""
        if self._current_step and self._current_step in self._steps:
            prev_step = self._steps[self._current_step]
            if prev_step.status in ("pending", "active") and step_id != self._current_step:
                prev_step.status = "complete"
        if step_id in self._steps:
            self._steps[step_id].status = "active"
            self._steps[step_id].set_detail(detail)
            self._current_step = step_id

    def set_step_complete(self, step_id: str) -> None:
        """Mark a step as complete."""
        if step_id in self._steps:
            self._steps[step_id].status = "complete"

    def set_step_error(self, step_id: str) -> None:
        """Mark a step as error."""
        if step_id in self._steps:
            self._steps[step_id].status = "error"

    def complete(self) -> None:
        """Mark all steps as complete."""
        self.is_loading = False
        for step in self._steps.values():
            if step.status != "error":
                step.status = "complete"
        with contextlib.suppress(Exception):
            self.display = False

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ProgressPanel(symbol={self._symbol})"
