"""Tests for TUI commands."""

import pytest

from src.tui.commands import CommandHandler, CommandResult


class TestCommandHandler:
    def test_initialization(self):
        handler = CommandHandler()

        assert handler._workflow is None
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

    @pytest.mark.asyncio
    async def test_execute_unknown_command(self):
        handler = CommandHandler()

        result = await handler.execute("/unknown")

        assert result.success is False
        assert "Unknown command" in result.message

    @pytest.mark.asyncio
    async def test_execute_help(self):
        handler = CommandHandler()

        result = await handler.execute("/help")

        assert result.success is True
        assert "Available Commands" in result.message
        assert "/analyze" in result.message

    @pytest.mark.asyncio
    async def test_execute_analyze_no_symbol(self):
        handler = CommandHandler()

        result = await handler.execute("/analyze")

        assert result.success is False
        assert "Usage" in result.message

    @pytest.mark.asyncio
    async def test_execute_technical_no_symbol(self):
        handler = CommandHandler()

        result = await handler.execute("/technical")

        assert result.success is False
        assert "Usage" in result.message

    @pytest.mark.asyncio
    async def test_execute_sentiment_no_symbol(self):
        handler = CommandHandler()

        result = await handler.execute("/sentiment")

        assert result.success is False
        assert "Usage" in result.message

    @pytest.mark.asyncio
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
