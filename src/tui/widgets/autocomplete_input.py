"""Input widget with autocomplete dropdown."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Input, OptionList
from textual.widgets.option_list import Option


class ChatInput(Input):
    """Input that doesn't consume ctrl+c (allows app to handle quit)."""

    BINDINGS = [
        Binding("ctrl+a", "cursor_left", "cursor left", show=False),
        Binding("ctrl+b", "cursor_left", "cursor left", show=False),
        Binding("ctrl+d", "delete_right", "delete right", show=False),
        Binding("ctrl+e", "cursor_right", "cursor right", show=False),
        Binding("ctrl+f", "cursor_right", "cursor right", show=False),
        Binding("ctrl+h", "delete_left", "delete left", show=False),
        Binding("ctrl+k", "delete_right_all", "delete all to the right", show=False),
        Binding("ctrl+u", "delete_left_all", "delete all to the left", show=False),
        Binding("ctrl+w", "delete_word_left", "delete word left", show=False),
        Binding("delete", "delete_right", "delete right", show=False),
        Binding("end", "cursor_right_word", "cursor word right", show=False),
        Binding("enter", "submit", "submit", show=False),
        Binding("home", "cursor_left_word", "cursor word left", show=False),
        Binding("left", "cursor_left", "cursor left", show=False),
        Binding("right", "cursor_right", "cursor right", show=False),
        Binding("shift+end", "cursor_right_word(True)", "cursor word right select", show=False),
        Binding("shift+home", "cursor_left_word(True)", "cursor word left select", show=False),
        Binding("shift+left", "cursor_left(True)", "cursor left select", show=False),
        Binding("shift+right", "cursor_right(True)", "cursor right select", show=False),
        # Removed ctrl+c (copy) to allow app-level quit handling
    ]


DEFAULT_SYMBOLS = [
    "AAPL",
    "ABBV",
    "ABNB",
    "ADBE",
    "AMD",
    "AMZN",
    "AVGO",
    "BA",
    "BABA",
    "BAC",
    "BRK.B",
    "C",
    "CRM",
    "CSCO",
    "CVX",
    "DIS",
    "GOOG",
    "GOOGL",
    "GS",
    "HD",
    "INTC",
    "JNJ",
    "JPM",
    "KO",
    "LLY",
    "MA",
    "MCD",
    "META",
    "MSFT",
    "NFLX",
    "NKE",
    "NVDA",
    "ORCL",
    "PEP",
    "PFE",
    "PG",
    "PLTR",
    "PYPL",
    "QCOM",
    "SBUX",
    "SHOP",
    "SQ",
    "T",
    "TGT",
    "TSLA",
    "UNH",
    "V",
    "VZ",
    "WMT",
    "XOM",
]

# Commands that don't require arguments - submit immediately on Enter
NO_ARG_COMMANDS = {"help"}


class AutocompleteInput(Widget):
    """Input with autocomplete dropdown showing matching options."""

    BINDINGS = [
        Binding("down", "next_option", "Next", show=False),
        Binding("up", "prev_option", "Previous", show=False),
        Binding("tab", "select_option", "Select", show=False),
        Binding("escape", "hide_dropdown", "Hide", show=False),
    ]

    show_dropdown: reactive[bool] = reactive(default=False)

    class Submitted(Message):
        """Posted when input is submitted."""

        def __init__(self, value: str) -> None:
            """Initialize submitted message.

            Args:
                value: The submitted input value
            """
            super().__init__()
            self.value = value

    def __init__(
        self,
        placeholder: str = "",
        symbols: list[str] | None = None,
        commands: list[str] | None = None,
        widget_id: str | None = None,
    ) -> None:
        """Initialize autocomplete input.

        Args:
            placeholder: Placeholder text for input
            symbols: List of stock symbols to suggest
            commands: List of slash commands to suggest
            widget_id: Widget ID
        """
        super().__init__(id=widget_id)
        self._placeholder = placeholder
        self._symbols = symbols or DEFAULT_SYMBOLS
        self._commands = commands or []
        self._matches: list[str] = []
        self._input_history: list[str] = []
        self._history_index: int = -1
        self._draft: str = ""

    def compose(self) -> ComposeResult:
        """Compose the widget."""
        with Vertical(id="autocomplete-container"):
            yield OptionList(id="autocomplete-dropdown")
            yield ChatInput(placeholder=self._placeholder, id="autocomplete-input")

    def on_mount(self) -> None:
        """Handle mount."""
        dropdown = self.query_one("#autocomplete-dropdown", OptionList)
        dropdown.display = False

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes to update suggestions."""
        if self._history_index != -1:
            self._history_index = -1
        value = event.value
        self._update_matches(value)
        self._refresh_dropdown()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        self._history_index = -1
        # If dropdown visible with matches, select highlighted option instead
        if self.show_dropdown and self._matches:
            event.stop()
            dropdown = self.query_one("#autocomplete-dropdown", OptionList)
            index = dropdown.highlighted
            if index is None:
                index = 0  # Default to first option
            option = dropdown.get_option_at_index(index)
            self._apply_selection(str(option.prompt), submit=True)
            return

        self._hide_dropdown()
        self.post_message(self.Submitted(event.value))

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle option selection from dropdown (Enter/click)."""
        self._apply_selection(str(event.option.prompt), submit=True)

    def _update_matches(self, value: str) -> None:
        """Update matches based on current input."""
        self._matches = []

        if not value:
            return

        # Command suggestions
        if value.startswith("/") and " " not in value:
            prefix = value[1:].lower()
            self._matches = [f"/{c}" for c in self._commands if c.startswith(prefix)]

        # Symbol suggestions after command
        elif value.startswith("/") and " " in value:
            parts = value.rsplit(" ", 1)
            if len(parts) == 2:
                prefix = parts[1].upper()
                if prefix:
                    self._matches = [s for s in self._symbols if s.startswith(prefix)]

    def _refresh_dropdown(self) -> None:
        """Refresh dropdown with current matches."""
        dropdown = self.query_one("#autocomplete-dropdown", OptionList)
        dropdown.clear_options()

        if not self._matches:
            dropdown.display = False
            self.show_dropdown = False
            return

        for match in self._matches[:10]:  # Limit to 10 options
            dropdown.add_option(Option(match))

        dropdown.display = True
        dropdown.highlighted = 0
        self.show_dropdown = True

    def _apply_selection(self, selected: str, submit: bool = False) -> None:
        """Apply selected option to input.

        Args:
            selected: The selected option text
            submit: If True and selecting a symbol, submit the command
        """
        input_widget = self.query_one("#autocomplete-input", ChatInput)
        value = input_widget.value

        # Command selection - submit no-arg commands immediately, else wait for symbol
        if selected.startswith("/"):
            cmd_name = selected[1:]
            if cmd_name in NO_ARG_COMMANDS and submit:
                self._hide_dropdown()
                self.post_message(self.Submitted(selected))
                return
            input_widget.value = selected + " "
            input_widget.cursor_position = len(input_widget.value)
        # Symbol selection - complete the command
        elif value.startswith("/") and " " in value:
            parts = value.rsplit(" ", 1)
            final_value = parts[0] + " " + selected
            input_widget.value = final_value
            input_widget.cursor_position = len(input_widget.value)

            # Submit if requested (Enter on dropdown)
            if submit:
                self._hide_dropdown()
                self.post_message(self.Submitted(final_value))
                return

        self._hide_dropdown()
        input_widget.focus()

    def _hide_dropdown(self) -> None:
        """Hide the dropdown."""
        dropdown = self.query_one("#autocomplete-dropdown", OptionList)
        dropdown.display = False
        self.show_dropdown = False

    def action_next_option(self) -> None:
        """Move to next option in dropdown or newer history."""
        if self.show_dropdown:
            dropdown = self.query_one("#autocomplete-dropdown", OptionList)
            dropdown.action_cursor_down()
        else:
            self._navigate_history(-1)

    def action_prev_option(self) -> None:
        """Move to previous option in dropdown or older history."""
        if self.show_dropdown:
            dropdown = self.query_one("#autocomplete-dropdown", OptionList)
            dropdown.action_cursor_up()
        else:
            self._navigate_history(1)

    def _navigate_history(self, direction: int) -> None:
        """Navigate through input history.

        Args:
            direction: 1 for older messages, -1 for newer
        """
        if not self._input_history:
            return

        input_widget = self.query_one("#autocomplete-input", ChatInput)

        if self._history_index == -1:
            self._draft = input_widget.value

        new_index = self._history_index + direction

        if new_index < -1:
            new_index = -1
        elif new_index >= len(self._input_history):
            return

        self._history_index = new_index

        if new_index == -1:
            input_widget.value = self._draft
        else:
            input_widget.value = self._input_history[new_index]

        input_widget.cursor_position = len(input_widget.value)

    def set_input_history(self, messages: list[str]) -> None:
        """Set input history for arrow key navigation.

        Args:
            messages: User messages, most recent first
        """
        self._input_history = messages
        self._history_index = -1

    def action_select_option(self) -> None:
        """Select current option."""
        if not self.show_dropdown or not self._matches:
            return
        dropdown = self.query_one("#autocomplete-dropdown", OptionList)
        if dropdown.highlighted is not None:
            option = dropdown.get_option_at_index(dropdown.highlighted)
            self._apply_selection(str(option.prompt))

    def action_hide_dropdown(self) -> None:
        """Hide dropdown or delegate to app for cancel."""
        if self.show_dropdown:
            self._hide_dropdown()
        else:
            app = getattr(self, "app", None)
            if app is not None and hasattr(app, "action_focus_input"):
                app.action_focus_input()
            else:
                self.focus()

    def focus(self, scroll_visible: bool = True) -> AutocompleteInput:
        """Focus the input."""
        self.query_one("#autocomplete-input", ChatInput).focus(scroll_visible)
        return self

    @property
    def value(self) -> str:
        """Get current input value."""
        return self.query_one("#autocomplete-input", ChatInput).value

    @value.setter
    def value(self, new_value: str) -> None:
        """Set input value."""
        self.query_one("#autocomplete-input", ChatInput).value = new_value

    def __repr__(self) -> str:
        """Return string representation."""
        return f"AutocompleteInput(symbols={len(self._symbols)}, commands={len(self._commands)})"
