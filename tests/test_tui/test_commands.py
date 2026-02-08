"""Tests for TUI commands."""

import pytest

from src.tui.commands import CommandHandler, CommandResult


class TestCommandHandler:
    def test_initialization(self):
        handler = CommandHandler()

        assert "analyze" in handler._commands
        assert "technical" in handler._commands
        assert "sentiment" in handler._commands
        assert "news" in handler._commands
        assert "help" in handler._commands

    def test_is_command(self):
        handler = CommandHandler()

        assert handler.is_command("/analyze AAPL") is True
        assert handler.is_command("/help") is True
        assert handler.is_command("analyze AAPL") is False
        assert handler.is_command("Hello") is False
        assert handler.is_command("  /test") is True

    def test_parse_command_with_args(self):
        handler = CommandHandler()

        cmd, args = handler.parse_command("/analyze AAPL")

        assert cmd == "analyze"
        assert args == ["AAPL"]

    def test_parse_command_no_args(self):
        handler = CommandHandler()

        cmd, args = handler.parse_command("/help")

        assert cmd == "help"
        assert args == []

    def test_parse_command_multiple_args(self):
        handler = CommandHandler()

        cmd, args = handler.parse_command("/test arg1 arg2 arg3")

        assert cmd == "test"
        assert args == ["arg1", "arg2", "arg3"]

    async def test_execute_unknown_command(self):
        handler = CommandHandler()

        result = await handler.execute("/unknown")

        assert result.success is False
        assert "Unknown command" in result.message

    async def test_execute_help(self):
        handler = CommandHandler()

        result = await handler.execute("/help")

        assert result.success is True
        assert "Available Commands" in result.message
        assert "/analyze" in result.message

    async def test_execute_analyze_no_symbol(self):
        handler = CommandHandler()

        result = await handler.execute("/analyze")

        assert result.success is False
        assert "Usage" in result.message

    async def test_execute_technical_no_symbol(self):
        handler = CommandHandler()

        result = await handler.execute("/technical")

        assert result.success is False
        assert "Usage" in result.message

    async def test_execute_sentiment_no_symbol(self):
        handler = CommandHandler()

        result = await handler.execute("/sentiment")

        assert result.success is False
        assert "Usage" in result.message

    async def test_execute_news_no_symbol(self):
        handler = CommandHandler()

        result = await handler.execute("/news")

        assert result.success is False
        assert "Usage" in result.message

    def test_repr(self):
        handler = CommandHandler()

        assert repr(handler) == "CommandHandler()"


class TestCommandResult:
    def test_success_result(self):
        result = CommandResult(success=True, message="OK")

        assert result.success is True
        assert result.message == "OK"
        assert result.data is None

    def test_failure_result(self):
        result = CommandResult(success=False, message="Error")

        assert result.success is False
        assert result.message == "Error"

    def test_result_with_data(self):
        result = CommandResult(success=True, message="OK", data={"symbol": "AAPL"})

        assert result.data == {"symbol": "AAPL"}


class TestCancellationCallback:
    """Tests for cancellation callback propagation."""

    async def test_execute_analyze_passes_is_cancelled(self, monkeypatch):
        """is_cancelled callback is passed to run_analysis_in_process."""
        import asyncio

        from src.tui import worker

        captured_is_cancelled = None

        async def mock_run_analysis(symbol, period_days=90, progress_callback=None, is_cancelled=None):
            nonlocal captured_is_cancelled
            captured_is_cancelled = is_cancelled
            if is_cancelled and is_cancelled():
                raise asyncio.CancelledError
            return {
                "symbol": symbol,
                "decision": {
                    "action": "HOLD",
                    "confidence": 0.5,
                    "risk_level": "MEDIUM",
                    "reasoning": "test",
                },
                "technical": {
                    "signal": "HOLD",
                    "rsi": 50.0,
                    "macd_hist": 0.0,
                    "confidence": 0.5,
                    "interpretation": "",
                },
                "sentiment": {
                    "overall_sentiment": "neutral",
                    "sentiment_score": 0.0,
                    "article_count": 0,
                    "positive_ratio": 0.0,
                    "negative_ratio": 0.0,
                },
                "news": {"key_themes": [], "impact_assessment": "", "recommendation": ""},
            }

        monkeypatch.setattr(worker, "run_analysis_in_process", mock_run_analysis)

        handler = CommandHandler()

        def is_cancelled_fn():
            return False

        await handler.execute("/analyze AAPL", is_cancelled=is_cancelled_fn)

        assert captured_is_cancelled is is_cancelled_fn

    async def test_execute_analyze_cancellation_raises(self, monkeypatch):
        """is_cancelled=True raises CancelledError."""
        import asyncio

        from src.tui import worker

        async def mock_run_analysis(symbol, period_days=90, progress_callback=None, is_cancelled=None):
            if is_cancelled and is_cancelled():
                raise asyncio.CancelledError
            pytest.fail("Should have been cancelled")

        monkeypatch.setattr(worker, "run_analysis_in_process", mock_run_analysis)

        handler = CommandHandler()

        def always_cancelled():
            return True

        with pytest.raises(asyncio.CancelledError):
            await handler.execute("/analyze AAPL", is_cancelled=always_cancelled)


class TestPersonalityCommands:
    """Tests for personality switching commands."""

    async def test_execute_trump(self):
        """Test /trump command switches to Trump mode."""
        handler = CommandHandler()

        # Mock app
        class MockApp:
            personality = None

            def set_personality(self, p):
                self.personality = p

        app = MockApp()
        handler.set_app(app)

        result = await handler.execute("/trump")

        assert result.success is True
        assert "TRUMP MODE" in result.message
        assert app.personality == "trump"

    async def test_execute_casino(self):
        """Test /casino command switches to AI Casino mode."""
        handler = CommandHandler()

        # Mock app
        class MockApp:
            personality = None

            def set_personality(self, p):
                self.personality = p

        app = MockApp()
        handler.set_app(app)

        result = await handler.execute("/casino")

        assert result.success is True
        assert "AI CASINO MODE" in result.message
        assert app.personality == "casino"


class TestCandidatesCommands:
    """Tests for /candidates command."""

    def test_candidates_add_no_symbols(self):
        """Test /candidates add without symbols."""
        from unittest.mock import MagicMock

        handler = CommandHandler()
        state = MagicMock()

        result = handler._handle_candidates_add([], state, "")

        assert result.success is False
        assert "Usage" in result.message

    def test_candidates_add_no_history(self):
        """Test /candidates add with no screening history."""
        from unittest.mock import MagicMock

        handler = CommandHandler()
        state = MagicMock()
        state.screening_history = []

        result = handler._handle_candidates_add(["AAPL"], state, "")

        assert result.success is False
        assert "No screening candidates" in result.message

    def test_candidates_add_success(self):
        """Test /candidates add SYMBOL."""
        from datetime import UTC, datetime
        from unittest.mock import MagicMock, patch

        from src.daemon.state import ScreeningRecord
        from src.screening.screener import ScreeningResult
        from src.strategies.signal import Signal

        handler = CommandHandler()
        state = MagicMock()

        candidates = [
            ScreeningResult(
                symbol="AAPL",
                name="Apple Inc",
                sector="Technology",
                score=85.5,
                signal=Signal.BUY,
                metrics={},
                reason="Strong",
            )
        ]
        state.screening_history = [
            ScreeningRecord(
                timestamp=datetime.now(UTC),
                criteria="momentum",
                universe="SP500",
                top_symbols=["AAPL"],
                candidates=candidates,
                screened_at=datetime.now(UTC),
            )
        ]

        with (
            patch("src.screening.exporter.ScreeningExporter") as mock_exporter_class,
            patch("src.screening.screener.ScreeningCriteria") as mock_criteria_class,
        ):
            mock_exporter = MagicMock()
            mock_exporter_class.return_value = mock_exporter
            mock_criteria_class.return_value = MagicMock()

            result = handler._handle_candidates_add(["AAPL"], state, "")

            assert result.success is True
            assert "AAPL" in result.data["added"]
            mock_exporter.save_to_watchlist.assert_called_once()

    def test_candidates_add_not_found(self):
        """Test /candidates add with symbol not in candidates."""
        from datetime import UTC, datetime
        from unittest.mock import MagicMock

        from src.daemon.state import ScreeningRecord
        from src.screening.screener import ScreeningResult
        from src.strategies.signal import Signal

        handler = CommandHandler()
        state = MagicMock()

        candidates = [
            ScreeningResult(
                symbol="AAPL",
                name="Apple Inc",
                sector="Technology",
                score=85.5,
                signal=Signal.BUY,
                metrics={},
                reason="Strong",
            )
        ]
        state.screening_history = [
            ScreeningRecord(
                timestamp=datetime.now(UTC),
                criteria="momentum",
                universe="SP500",
                top_symbols=["AAPL"],
                candidates=candidates,
                screened_at=datetime.now(UTC),
            )
        ]

        result = handler._handle_candidates_add(["TSLA"], state, "")

        assert result.success is False
        assert "No matching candidates" in result.message

    def test_candidates_clear(self):
        """Test /candidates clear."""
        from datetime import UTC, datetime
        from unittest.mock import MagicMock

        from src.daemon.state import ScreeningRecord

        handler = CommandHandler()
        state = MagicMock()
        state.screening_history = [
            ScreeningRecord(
                timestamp=datetime.now(UTC),
                criteria="momentum",
                universe="SP500",
                top_symbols=[],
                candidates=[],
                screened_at=datetime.now(UTC),
            )
        ]

        result = handler._handle_candidates_clear(state, "~/.ai-casino/state.json")

        assert result.success is True
        assert state.screening_history == []
        assert state.last_after_hours_screening is None
        state.save.assert_called_once_with("~/.ai-casino/state.json")
