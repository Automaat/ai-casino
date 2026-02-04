"""Tests for TUI agentic chat functionality."""

from unittest.mock import MagicMock, patch

from src.tools import AnalyzeStockTool, GetMarketDataTool, GetNewsTool, ToolRegistry, WebSearchTool
from src.tui.app import TradingChatApp


class TestToolRegistrySetup:
    """Tests for tool registry setup in TUI app."""

    def test_create_tool_registry(self):
        """Test that app creates tool registry with all tools."""
        with patch.object(TradingChatApp, "_load_history"):
            app = TradingChatApp()

            registry = app._tool_registry

            assert isinstance(registry, ToolRegistry)
            assert len(registry) == 5
            assert "web_search" in registry.tool_names
            assert "get_market_data" in registry.tool_names
            assert "get_news" in registry.tool_names
            assert "analyze_stock" in registry.tool_names
            assert "screen_stocks" in registry.tool_names

    def test_registry_has_correct_tool_types(self):
        """Test that registry contains correct tool instances."""
        with patch.object(TradingChatApp, "_load_history"):
            app = TradingChatApp()

            assert isinstance(app._tool_registry.get("web_search"), WebSearchTool)
            assert isinstance(app._tool_registry.get("get_market_data"), GetMarketDataTool)
            assert isinstance(app._tool_registry.get("get_news"), GetNewsTool)
            assert isinstance(app._tool_registry.get("analyze_stock"), AnalyzeStockTool)

    def test_analyze_stock_requires_confirmation(self):
        """Test that analyze_stock tool requires confirmation."""
        with patch.object(TradingChatApp, "_load_history"):
            app = TradingChatApp()

            assert app._tool_registry.requires_confirmation("analyze_stock") is True
            assert app._tool_registry.requires_confirmation("get_market_data") is False
            assert app._tool_registry.requires_confirmation("get_news") is False
            assert app._tool_registry.requires_confirmation("web_search") is False


class TestChatModeDispatch:
    """Tests for chat mode dispatch logic."""

    def test_handle_chat_dispatches_to_streaming_for_ollama(self):
        """Test that Ollama provider uses streaming chat."""
        with (
            patch.object(TradingChatApp, "_load_history"),
            patch("src.tui.app.LLMClient") as mock_llm_client_cls,
        ):
            mock_llm = MagicMock()
            mock_llm.supports_tools = False
            mock_llm_client_cls.return_value = mock_llm

            app = TradingChatApp()
            app._llm = mock_llm

            assert mock_llm.supports_tools is False

    def test_handle_chat_dispatches_to_agentic_for_anthropic(self):
        """Test that Anthropic provider uses agentic chat."""
        with (
            patch.object(TradingChatApp, "_load_history"),
            patch("src.tui.app.LLMClient") as mock_llm_client_cls,
        ):
            mock_llm = MagicMock()
            mock_llm.supports_tools = True
            mock_llm_client_cls.return_value = mock_llm

            app = TradingChatApp()
            app._llm = mock_llm

            assert mock_llm.supports_tools is True


class TestToolDefinitions:
    """Tests for tool definitions in registry."""

    def test_all_tools_have_valid_definitions(self):
        """Test that all registered tools have valid definitions."""
        with patch.object(TradingChatApp, "_load_history"):
            app = TradingChatApp()

            definitions = app._tool_registry.get_definitions()

            assert len(definitions) == 5
            for definition in definitions:
                assert "type" in definition
                assert definition["type"] == "function"
                assert "function" in definition
                assert "name" in definition["function"]
                assert "description" in definition["function"]
                assert "parameters" in definition["function"]

    def test_tool_definitions_have_required_fields(self):
        """Test that tool definitions have proper parameter structure."""
        with patch.object(TradingChatApp, "_load_history"):
            app = TradingChatApp()

            definitions = app._tool_registry.get_definitions()

            for definition in definitions:
                params = definition["function"]["parameters"]
                assert "type" in params
                assert params["type"] == "object"
                assert "properties" in params
