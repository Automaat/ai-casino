"""Tests for LLM client."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.models.llm import LLMClient


@pytest.fixture
def mock_completion():
    with patch("src.models.llm.completion") as mock:
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "Mocked response"
        mock.return_value = response
        yield mock


def test_llm_client_init_ollama(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "qwen3:14b")

    client = LLMClient()

    assert client.provider == "ollama"
    assert client.model == "qwen3:14b"
    assert client._model_id == "ollama/qwen3:14b"


def test_llm_client_init_anthropic(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-4-20250514")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    client = LLMClient()

    assert client.provider == "anthropic"
    assert client.model == "claude-sonnet-4-20250514"
    assert client._model_id == "anthropic/claude-sonnet-4-20250514"


def test_llm_client_init_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    client = LLMClient()

    assert client.provider == "openai"
    assert client.model == "gpt-4o"
    assert client._model_id == "openai/gpt-4o"
    assert client._api_base is None


def test_llm_client_init_openai_custom_api_base(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "hf:moonshotai/Kimi-K2.5")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_BASE", "https://api.synthetic.new/openai/v1")

    client = LLMClient()

    assert client.provider == "openai"
    assert client.model == "hf:moonshotai/Kimi-K2.5"
    assert client._model_id == "openai/hf:moonshotai/Kimi-K2.5"
    assert client._api_base == "https://api.synthetic.new/openai/v1"


def test_llm_client_unsupported_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "invalid")

    with pytest.raises(ValueError, match="Unsupported provider: invalid"):
        LLMClient()


def test_complete_with_system_prompt(mock_completion, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    client = LLMClient()

    result = client.complete("Test prompt", system="System message", temperature=0.5)

    assert result == "Mocked response"
    mock_completion.assert_called_once()
    call_args = mock_completion.call_args
    assert call_args.kwargs["messages"] == [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "Test prompt"},
    ]
    assert call_args.kwargs["temperature"] == 0.5


def test_complete_with_api_base(mock_completion, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "hf:moonshotai/Kimi-K2.5")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_BASE", "https://api.synthetic.new/openai/v1")

    client = LLMClient()
    result = client.complete("Test prompt")

    assert result == "Mocked response"
    call_args = mock_completion.call_args
    assert call_args.kwargs["api_base"] == "https://api.synthetic.new/openai/v1"


def test_complete_without_system_prompt(mock_completion, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    client = LLMClient()

    result = client.complete("Test prompt")

    assert result == "Mocked response"
    call_args = mock_completion.call_args
    assert call_args.kwargs["messages"] == [
        {"role": "user", "content": "Test prompt"},
    ]


def test_chat(mock_completion, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    client = LLMClient()

    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "How are you?"},
    ]

    result = client.chat(messages, temperature=0.3)

    assert result == "Mocked response"
    mock_completion.assert_called_once()
    call_args = mock_completion.call_args
    assert call_args.kwargs["messages"] == messages
    assert call_args.kwargs["temperature"] == 0.3


def test_complete_handles_exception(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    client = LLMClient()

    with patch("src.models.llm.completion", side_effect=Exception("API Error")):
        with pytest.raises(Exception, match="API Error"):
            client.complete("Test prompt")


def test_repr(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "qwen3:14b")

    client = LLMClient()

    assert repr(client) == "LLMClient(provider=ollama, model=qwen3:14b)"


class TestCompleteWithTools:
    """Tests for complete_with_tools method."""

    @pytest.fixture
    def sample_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ]

    @pytest.fixture
    def mock_tool_executor(self):
        def executor(name: str, args: dict) -> str:
            if name == "get_weather":
                return f"Weather in {args['location']}: Sunny, 72°F"
            return "Unknown tool"

        return executor

    def test_complete_with_tools_no_tool_calls(self, monkeypatch, sample_tools, mock_tool_executor):
        """Test when LLM returns without tool calls."""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_BASE", raising=False)

        with patch("src.models.llm.completion") as mock:
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = "I don't need tools for this"
            response.choices[0].message.tool_calls = None
            mock.return_value = response

            client = LLMClient()
            result = client.complete_with_tools("Hello", sample_tools, mock_tool_executor)

            assert result == "I don't need tools for this"
            assert mock.call_count == 1

    def test_complete_with_tools_executes_tool(self, monkeypatch, sample_tools, mock_tool_executor):
        """Test tool execution and final response."""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_BASE", raising=False)

        with patch("src.models.llm.completion") as mock:
            # First call: LLM requests tool
            tool_call_msg = MagicMock()
            tool_call_msg.content = None
            tool_call = MagicMock()
            tool_call.id = "call_123"
            tool_call.function.name = "get_weather"
            tool_call.function.arguments = '{"location": "NYC"}'
            tool_call_msg.tool_calls = [tool_call]
            tool_call_msg.model_dump.return_value = {"role": "assistant", "tool_calls": []}

            # Second call: LLM returns final answer
            final_msg = MagicMock()
            final_msg.content = "The weather in NYC is sunny and 72°F"
            final_msg.tool_calls = None

            mock.side_effect = [
                MagicMock(choices=[MagicMock(message=tool_call_msg)]),
                MagicMock(choices=[MagicMock(message=final_msg)]),
            ]

            client = LLMClient()
            result = client.complete_with_tools(
                "What's the weather in NYC?", sample_tools, mock_tool_executor
            )

            assert result == "The weather in NYC is sunny and 72°F"
            assert mock.call_count == 2

    def test_complete_with_tools_max_calls_limit(self, monkeypatch, sample_tools, mock_tool_executor):
        """Test max_tool_calls limit is respected."""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_BASE", raising=False)

        with patch("src.models.llm.completion") as mock:
            # Always return tool call
            tool_call_msg = MagicMock()
            tool_call_msg.content = None
            tool_call = MagicMock()
            tool_call.id = "call_123"
            tool_call.function.name = "get_weather"
            tool_call.function.arguments = '{"location": "NYC"}'
            tool_call_msg.tool_calls = [tool_call]
            tool_call_msg.model_dump.return_value = {"role": "assistant", "tool_calls": []}

            final_msg = MagicMock()
            final_msg.content = "Final response after max calls"
            final_msg.tool_calls = None

            # Return tool calls until limit, then final
            mock.side_effect = [
                MagicMock(choices=[MagicMock(message=tool_call_msg)]),
                MagicMock(choices=[MagicMock(message=tool_call_msg)]),
                MagicMock(choices=[MagicMock(message=final_msg)]),
            ]

            client = LLMClient()
            result = client.complete_with_tools("prompt", sample_tools, mock_tool_executor, max_tool_calls=2)

            assert result == "Final response after max calls"

    def test_complete_with_tools_malformed_json(self, monkeypatch, sample_tools, mock_tool_executor):
        """Test handling of malformed JSON in tool arguments."""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_BASE", raising=False)

        with patch("src.models.llm.completion") as mock:
            tool_call_msg = MagicMock()
            tool_call_msg.content = None
            tool_call = MagicMock()
            tool_call.id = "call_123"
            tool_call.function.name = "get_weather"
            tool_call.function.arguments = "invalid json{"
            tool_call_msg.tool_calls = [tool_call]

            mock.return_value = MagicMock(choices=[MagicMock(message=tool_call_msg)])

            client = LLMClient()
            with pytest.raises(json.JSONDecodeError):
                client.complete_with_tools("prompt", sample_tools, mock_tool_executor)


class TestAcompleteWithTools:
    """Tests for acomplete_with_tools method."""

    @pytest.fixture
    def sample_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]

    @pytest.fixture
    def mock_tool_executor(self):
        def executor(name: str, args: dict) -> str:
            if name == "search":
                return f"Results for: {args['query']}"
            return "Unknown tool"

        return executor

    @pytest.mark.asyncio
    async def test_acomplete_with_tools_no_tool_calls(self, monkeypatch, sample_tools, mock_tool_executor):
        """Test async when LLM returns without tool calls."""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_BASE", raising=False)

        with patch("src.models.llm.acompletion") as mock:
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = "No tools needed"
            response.choices[0].message.tool_calls = None
            mock.return_value = response

            client = LLMClient()
            result = await client.acomplete_with_tools("Hello", sample_tools, mock_tool_executor)

            assert result == "No tools needed"

    @pytest.mark.asyncio
    async def test_acomplete_with_tools_executes_tool(self, monkeypatch, sample_tools, mock_tool_executor):
        """Test async tool execution."""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_BASE", raising=False)

        with patch("src.models.llm.acompletion") as mock:
            # First call: LLM requests tool
            tool_call_msg = MagicMock()
            tool_call_msg.content = None
            tool_call = MagicMock()
            tool_call.id = "call_456"
            tool_call.function.name = "search"
            tool_call.function.arguments = '{"query": "python testing"}'
            tool_call_msg.tool_calls = [tool_call]
            tool_call_msg.model_dump.return_value = {"role": "assistant", "tool_calls": []}

            # Second call: final answer
            final_msg = MagicMock()
            final_msg.content = "Found results about Python testing"
            final_msg.tool_calls = None

            mock.side_effect = [
                MagicMock(choices=[MagicMock(message=tool_call_msg)]),
                MagicMock(choices=[MagicMock(message=final_msg)]),
            ]

            client = LLMClient()
            result = await client.acomplete_with_tools(
                "Search for python testing", sample_tools, mock_tool_executor
            )

            assert result == "Found results about Python testing"


class TestSupportsTools:
    """Tests for supports_tools property."""

    def test_supports_tools_anthropic(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        client = LLMClient()
        assert client.supports_tools is True

    def test_supports_tools_openai(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_BASE", raising=False)

        client = LLMClient()
        assert client.supports_tools is True

    def test_supports_tools_ollama(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")

        client = LLMClient()
        assert client.supports_tools is False
