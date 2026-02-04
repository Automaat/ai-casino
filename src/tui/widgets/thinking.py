"""Thinking indicator widget for chat."""

import time

from textual.widget import Widget


class ThinkingIndicator(Widget):
    """Animated thinking indicator shown while LLM processes."""

    DEFAULT_CSS = """
    ThinkingIndicator {
        height: 1;
    }
    """

    def __init__(self, **kwargs: object) -> None:
        """Initialize thinking indicator."""
        super().__init__(**kwargs)
        self.auto_refresh = 1 / 8  # 8 FPS animation
        self._start_time = time.monotonic()

    def render(self) -> str:
        """Render animated thinking indicator."""
        elapsed = time.monotonic() - self._start_time
        frame = int(elapsed * 3)  # 3 states per second
        dots = "." * ((frame % 3) + 1)
        return f"● Thinking{dots}"

    def __repr__(self) -> str:
        """Return string representation."""
        return "ThinkingIndicator()"
