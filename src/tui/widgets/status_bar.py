"""Status bar widget for market status and info."""

from datetime import datetime
from zoneinfo import ZoneInfo

from textual.reactive import reactive
from textual.widgets import Static


class StatusBar(Static):
    """Status bar showing market status, time, and working indicator."""

    DEFAULT_CSS = """
    StatusBar {
        dock: bottom;
        height: 1;
        background: #1E293B;
        color: #F1F5F9;
        padding: 0 1;
    }
    """

    market_status: reactive[str] = reactive("Checking...")
    current_time: reactive[str] = reactive("")
    working_status: reactive[str] = reactive("")

    def __init__(self) -> None:
        """Initialize status bar."""
        super().__init__()
        self._timezone = ZoneInfo("America/New_York")

    def on_mount(self) -> None:
        """Set up timer when mounted."""
        self.set_interval(1, self._update_time)
        self._update_time()

    def _update_time(self) -> None:
        """Update current time."""
        now = datetime.now(self._timezone)
        self.current_time = now.strftime("%H:%M:%S ET")

        weekday = now.weekday()
        hour = now.hour
        minute = now.minute

        if weekday >= 5:
            self.market_status = "CLOSED"
        elif hour < 9 or (hour == 9 and minute < 30):
            self.market_status = "PRE-MKT"
        elif hour >= 16:
            self.market_status = "AFTER-HRS"
        else:
            self.market_status = "OPEN"

    def set_working(self, status: str) -> None:
        """Set working indicator.

        Args:
            status: Working status text (empty to clear)
        """
        self.working_status = status

    def clear_working(self) -> None:
        """Clear working indicator."""
        self.working_status = ""

    def render(self) -> str:
        """Render the status bar."""
        parts = ["AI Casino"]

        market_display = f"Market: {self.market_status}"
        parts.append(market_display)

        parts.append(self.current_time)

        if self.working_status:
            parts.append(f"[{self.working_status}]")

        return " | ".join(parts)

    def __repr__(self) -> str:
        """Return string representation."""
        return "StatusBar()"
