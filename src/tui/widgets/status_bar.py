"""Status bar widget for market status and info."""

from datetime import datetime
from zoneinfo import ZoneInfo

from textual.reactive import reactive
from textual.widgets import Static


class StatusBar(Static):
    """Status bar showing market status and time."""

    DEFAULT_CSS = """
    StatusBar {
        dock: bottom;
        height: 1;
        background: $primary;
        color: $text;
        padding: 0 1;
    }
    """

    market_status: reactive[str] = reactive("Checking...")
    current_time: reactive[str] = reactive("")

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
            self.market_status = "CLOSED (Weekend)"
        elif hour < 9 or (hour == 9 and minute < 30):
            self.market_status = "PRE-MARKET"
        elif hour >= 16:
            self.market_status = "AFTER-HOURS"
        else:
            self.market_status = "OPEN"

    def render(self) -> str:
        """Render the status bar."""
        return f"AI Casino | Market: {self.market_status} | {self.current_time} | /help for commands"
