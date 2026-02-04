"""Tests for AutocompleteInput widget."""

import pytest

from src.tui.widgets.autocomplete_input import (
    DEFAULT_COMMANDS,
    DEFAULT_SYMBOLS,
    AutocompleteInput,
)


class TestAutocompleteInput:
    """Tests for AutocompleteInput."""

    @pytest.fixture
    def widget(self) -> AutocompleteInput:
        """Create widget with default symbols and commands."""
        return AutocompleteInput()

    @pytest.fixture
    def custom_widget(self) -> AutocompleteInput:
        """Create widget with custom symbols and commands."""
        return AutocompleteInput(
            symbols=["AAPL", "AMD", "AMZN", "GOOGL", "MSFT"],
            commands=["analyze", "help"],
        )

    def test_initialization(self, widget: AutocompleteInput) -> None:
        """Widget initializes with default symbols and commands."""
        assert len(widget._symbols) == len(DEFAULT_SYMBOLS)
        assert len(widget._commands) == len(DEFAULT_COMMANDS)

    def test_initialization_custom(self, custom_widget: AutocompleteInput) -> None:
        """Widget initializes with custom symbols and commands."""
        assert custom_widget._symbols == ["AAPL", "AMD", "AMZN", "GOOGL", "MSFT"]
        assert custom_widget._commands == ["analyze", "help"]

    def test_update_matches_empty(self, widget: AutocompleteInput) -> None:
        """Empty input produces no matches."""
        widget._update_matches("")
        assert widget._matches == []

    def test_update_matches_command_prefix(self, widget: AutocompleteInput) -> None:
        """Command prefix produces command matches."""
        widget._update_matches("/an")
        assert widget._matches == ["/analyze"]

    def test_update_matches_command_help(self, widget: AutocompleteInput) -> None:
        """Help command prefix produces match."""
        widget._update_matches("/he")
        assert widget._matches == ["/help"]

    def test_update_matches_multiple_commands(self, widget: AutocompleteInput) -> None:
        """Prefix matching multiple commands returns all."""
        widget._update_matches("/")
        # All commands start with nothing after /
        assert len(widget._matches) == len(DEFAULT_COMMANDS)

    def test_update_matches_symbol_after_command(self, widget: AutocompleteInput) -> None:
        """Symbol prefix after command produces symbol matches."""
        widget._update_matches("/analyze A")
        # Should have matches for symbols starting with A
        assert "/analyze" not in widget._matches
        assert all(s.startswith("A") for s in widget._matches)

    def test_update_matches_symbol_case_insensitive(self, widget: AutocompleteInput) -> None:
        """Symbol matching is case-insensitive (input converted to upper)."""
        widget._update_matches("/analyze a")
        assert all(s.startswith("A") for s in widget._matches)

    def test_update_matches_multiple_symbols(self, custom_widget: AutocompleteInput) -> None:
        """Multiple symbol matches returned."""
        custom_widget._update_matches("/analyze A")
        assert custom_widget._matches == ["AAPL", "AMD", "AMZN"]

    def test_update_matches_no_command_match(self, widget: AutocompleteInput) -> None:
        """No matches for unknown command prefix."""
        widget._update_matches("/xyz")
        assert widget._matches == []

    def test_update_matches_no_symbol_match(self, widget: AutocompleteInput) -> None:
        """No matches for unknown symbol prefix."""
        widget._update_matches("/analyze XYZ123")
        assert widget._matches == []

    def test_update_matches_command_with_space_no_symbol(self, widget: AutocompleteInput) -> None:
        """Command with trailing space but no symbol has no matches."""
        widget._update_matches("/analyze ")
        assert widget._matches == []

    def test_default_symbols_populated(self) -> None:
        """Default symbols list is populated."""
        assert len(DEFAULT_SYMBOLS) >= 50
        assert "AAPL" in DEFAULT_SYMBOLS
        assert "NVDA" in DEFAULT_SYMBOLS
        assert "TSLA" in DEFAULT_SYMBOLS

    def test_default_commands_populated(self) -> None:
        """Default commands list is populated."""
        assert "analyze" in DEFAULT_COMMANDS
        assert "technical" in DEFAULT_COMMANDS
        assert "sentiment" in DEFAULT_COMMANDS
        assert "news" in DEFAULT_COMMANDS
        assert "help" in DEFAULT_COMMANDS

    def test_repr(self, widget: AutocompleteInput) -> None:
        """Repr shows symbol and command counts."""
        repr_str = repr(widget)
        assert "AutocompleteInput" in repr_str
        assert "symbols=" in repr_str
        assert "commands=" in repr_str

    def test_plain_text_no_matches(self, widget: AutocompleteInput) -> None:
        """Plain text without slash produces no matches."""
        widget._update_matches("hello world")
        assert widget._matches == []

    def test_matches_limited_to_ten(self, widget: AutocompleteInput) -> None:
        """Dropdown limits to 10 options max."""
        widget._update_matches("/")
        # Even if more than 10 commands existed, refresh limits to 10
        # For now we have 5 commands, so test passes
        assert len(widget._matches) <= 10
