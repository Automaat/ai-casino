"""Tests for LLM client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from src.models.llm import LLMClient
from src.models.providers.base import StructuredOutputError, ToolCall


@pytest.fixture
def mock_ollama_provider():
    """Mock Ollama provider."""
    with patch("src.models.llm.OllamaProvider") as mock:
        provider = MagicMock()
        provider.acomplete = AsyncMock(return_value="Mocked response")
        provider.supports_tools = False
        mock.return_value = provider
        yield mock, provider


@pytest.fixture
def mock_openai_provider():
    """Mock OpenAI provider."""
    with patch("src.models.llm.OpenAIProvider") as mock:
        provider = MagicMock()
        provider.acomplete = AsyncMock(return_value="Mocked response")
        provider.acomplete_with_tools = AsyncMock(return_value=("No tools needed", None))
        provider.supports_tools = True
        mock.return_value = provider
        yield mock, provider


@pytest.fixture
def mock_anthropic_provider():
    """Mock Anthropic provider."""
    with patch("src.models.llm.AnthropicProvider") as mock:
        provider = MagicMock()
        provider.acomplete = AsyncMock(return_value="Mocked response")
        provider.acomplete_with_tools = AsyncMock(return_value=("No tools needed", None))
        provider.supports_tools = True
        mock.return_value = provider
        yield mock, provider


def test_llm_client_init_ollama(mock_ollama_provider, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "qwen3:14b")

    client = LLMClient()

    assert client.provider == "ollama"
    assert client.model == "qwen3:14b"
    mock_ollama_provider[0].assert_called_once_with(model="qwen3:14b", base_url="http://localhost:11434")


def test_llm_client_init_anthropic(mock_anthropic_provider, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-4-20250514")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    client = LLMClient()

    assert client.provider == "anthropic"
    assert client.model == "claude-sonnet-4-20250514"
    mock_anthropic_provider[0].assert_called_once_with(model="claude-sonnet-4-20250514", api_key=None)


def test_llm_client_init_openai(mock_openai_provider, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    client = LLMClient()

    assert client.provider == "openai"
    assert client.model == "gpt-4o"
    mock_openai_provider[0].assert_called_once_with(model="gpt-4o", api_key=None, base_url=None)


def test_llm_client_init_openai_custom_api_base(mock_openai_provider, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "hf:moonshotai/Kimi-K2.5")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_BASE", "https://api.synthetic.new/openai/v1")

    client = LLMClient()

    assert client.provider == "openai"
    assert client.model == "hf:moonshotai/Kimi-K2.5"
    mock_openai_provider[0].assert_called_once_with(
        model="hf:moonshotai/Kimi-K2.5", api_key=None, base_url="https://api.synthetic.new/openai/v1"
    )


def test_llm_client_unsupported_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "invalid")

    with pytest.raises(ValueError, match="Unsupported provider: invalid"):
        LLMClient()


def test_complete_with_system_prompt(mock_ollama_provider, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    _, provider = mock_ollama_provider

    client = LLMClient()
    result = client.complete("Test prompt", system="System message", temperature=0.5)

    assert result == "Mocked response"
    provider.acomplete.assert_called_once()
    call_args = provider.acomplete.call_args
    assert call_args[0][0] == [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "Test prompt"},
    ]
    assert call_args[0][1] == 0.5


def test_complete_without_system_prompt(mock_ollama_provider, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    _, provider = mock_ollama_provider

    client = LLMClient()
    result = client.complete("Test prompt")

    assert result == "Mocked response"
    call_args = provider.acomplete.call_args
    assert call_args[0][0] == [{"role": "user", "content": "Test prompt"}]


def test_chat(mock_ollama_provider, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    _, provider = mock_ollama_provider

    client = LLMClient()

    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "How are you?"},
    ]

    result = client.chat(messages, temperature=0.3)

    assert result == "Mocked response"
    provider.acomplete.assert_called_once()
    call_args = provider.acomplete.call_args
    assert call_args[0][0] == messages
    assert call_args[0][1] == 0.3


def test_complete_handles_exception(mock_ollama_provider, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    _, provider = mock_ollama_provider
    provider.acomplete = AsyncMock(side_effect=Exception("API Error"))

    client = LLMClient()

    with pytest.raises(Exception, match="API Error"):
        client.complete("Test prompt")


def test_repr(mock_ollama_provider, monkeypatch):
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

    def test_complete_with_tools_no_tool_calls(
        self, mock_openai_provider, monkeypatch, sample_tools, mock_tool_executor
    ):
        """Test when LLM returns without tool calls."""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_BASE", raising=False)
        _, provider = mock_openai_provider
        provider.acomplete_with_tools = AsyncMock(return_value=("I don't need tools for this", None))

        client = LLMClient()
        result = client.complete_with_tools("Hello", sample_tools, mock_tool_executor)

        assert result == "I don't need tools for this"
        assert provider.acomplete_with_tools.call_count == 1

    def test_complete_with_tools_executes_tool(
        self, mock_openai_provider, monkeypatch, sample_tools, mock_tool_executor
    ):
        """Test tool execution and final response."""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_BASE", raising=False)
        _, provider = mock_openai_provider

        tool_call = ToolCall(id="call_123", name="get_weather", arguments={"location": "NYC"})
        provider.acomplete_with_tools = AsyncMock(
            side_effect=[
                (None, [tool_call]),
                ("The weather in NYC is sunny and 72°F", None),
            ]
        )

        client = LLMClient()
        result = client.complete_with_tools("What's the weather in NYC?", sample_tools, mock_tool_executor)

        assert result == "The weather in NYC is sunny and 72°F"
        assert provider.acomplete_with_tools.call_count == 2

    def test_complete_with_tools_max_calls_limit(
        self, mock_openai_provider, monkeypatch, sample_tools, mock_tool_executor
    ):
        """Test max_tool_calls limit is respected."""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_BASE", raising=False)
        _, provider = mock_openai_provider

        tool_call = ToolCall(id="call_123", name="get_weather", arguments={"location": "NYC"})
        # Return tool calls for first 2, then final completion is called without tools
        provider.acomplete_with_tools = AsyncMock(
            side_effect=[
                (None, [tool_call]),
                (None, [tool_call]),
            ]
        )
        provider.acomplete = AsyncMock(return_value="Final response after max calls")

        client = LLMClient()
        result = client.complete_with_tools("prompt", sample_tools, mock_tool_executor, max_tool_calls=2)

        assert result == "Final response after max calls"
        assert provider.acomplete_with_tools.call_count == 2
        assert provider.acomplete.call_count == 1


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

    async def test_acomplete_with_tools_no_tool_calls(
        self, mock_openai_provider, monkeypatch, sample_tools, mock_tool_executor
    ):
        """Test async when LLM returns without tool calls."""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_BASE", raising=False)
        _, provider = mock_openai_provider
        provider.acomplete_with_tools = AsyncMock(return_value=("No tools needed", None))

        client = LLMClient()
        result = await client.acomplete_with_tools("Hello", sample_tools, mock_tool_executor)

        assert result == "No tools needed"

    async def test_acomplete_with_tools_executes_tool(
        self, mock_openai_provider, monkeypatch, sample_tools, mock_tool_executor
    ):
        """Test async tool execution."""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_BASE", raising=False)
        _, provider = mock_openai_provider

        tool_call = ToolCall(id="call_456", name="search", arguments={"query": "python testing"})
        provider.acomplete_with_tools = AsyncMock(
            side_effect=[
                (None, [tool_call]),
                ("Found results about Python testing", None),
            ]
        )

        client = LLMClient()
        result = await client.acomplete_with_tools(
            "Search for python testing", sample_tools, mock_tool_executor
        )

        assert result == "Found results about Python testing"


class TestSupportsTools:
    """Tests for supports_tools property."""

    def test_supports_tools_anthropic(self, mock_anthropic_provider, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        _ = mock_anthropic_provider  # Fixture required to mock provider creation

        client = LLMClient()
        assert client.supports_tools is True

    def test_supports_tools_openai(self, mock_openai_provider, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_BASE", raising=False)
        _ = mock_openai_provider  # Fixture required to mock provider creation

        client = LLMClient()
        assert client.supports_tools is True

    def test_supports_tools_ollama(self, mock_ollama_provider, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        _ = mock_ollama_provider  # Fixture required to mock provider creation

        client = LLMClient()
        assert client.supports_tools is False


class SampleResponseModel(BaseModel):
    """Sample response model for structured output tests."""

    answer: str = Field(description="The answer")
    confidence: float = Field(description="Confidence score", ge=0.0, le=1.0)


class TestStructuredOutput:
    """Tests for structured output methods."""

    @pytest.fixture
    def mock_ollama_provider_structured(self):
        """Mock Ollama provider with structured output."""
        with patch("src.models.llm.OllamaProvider") as mock:
            provider = MagicMock()
            provider.acomplete = AsyncMock(return_value="Mocked response")
            provider.supports_tools = False
            provider.supports_structured_output = True
            mock.return_value = provider
            yield mock, provider

    @pytest.fixture
    def mock_openai_provider_structured(self):
        """Mock OpenAI provider with structured output."""
        with patch("src.models.llm.OpenAIProvider") as mock:
            provider = MagicMock()
            provider.acomplete = AsyncMock(return_value="Mocked response")
            provider.supports_tools = True
            provider.supports_structured_output = True
            mock.return_value = provider
            yield mock, provider

    def test_supports_structured_output_ollama(self, mock_ollama_provider_structured, monkeypatch):
        """Test supports_structured_output for Ollama."""
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        _ = mock_ollama_provider_structured

        client = LLMClient()
        assert client.supports_structured_output is True

    def test_supports_structured_output_openai(self, mock_openai_provider_structured, monkeypatch):
        """Test supports_structured_output for OpenAI."""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_BASE", raising=False)
        _ = mock_openai_provider_structured

        client = LLMClient()
        assert client.supports_structured_output is True

    def test_structured_returns_validated_model(self, mock_ollama_provider_structured, monkeypatch):
        """Test structured output returns validated Pydantic model."""
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        _, provider = mock_ollama_provider_structured

        expected = SampleResponseModel(answer="test answer", confidence=0.85)
        provider.astructured = AsyncMock(return_value=expected)

        client = LLMClient()
        result = client.structured("What is 2+2?", SampleResponseModel, temperature=0.5)

        assert result == expected
        assert result.answer == "test answer"
        assert result.confidence == 0.85
        provider.astructured.assert_called_once()

    async def test_astructured_returns_validated_model(self, mock_ollama_provider_structured, monkeypatch):
        """Test async structured output returns validated Pydantic model."""
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        _, provider = mock_ollama_provider_structured

        expected = SampleResponseModel(answer="async answer", confidence=0.9)
        provider.astructured = AsyncMock(return_value=expected)

        client = LLMClient()
        result = await client.astructured("prompt", SampleResponseModel, system="system msg")

        assert result == expected
        provider.astructured.assert_called_once()

    def test_structured_raises_on_error(self, mock_ollama_provider_structured, monkeypatch):
        """Test structured output raises StructuredOutputError on failure."""
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        _, provider = mock_ollama_provider_structured

        provider.astructured = AsyncMock(
            side_effect=StructuredOutputError("Validation failed", raw_response='{"invalid": "json"}')
        )

        client = LLMClient()
        with pytest.raises(StructuredOutputError) as exc_info:
            client.structured("prompt", SampleResponseModel)

        assert "Validation failed" in str(exc_info.value)
        assert exc_info.value.raw_response == '{"invalid": "json"}'

    def test_structured_passes_system_prompt(self, mock_ollama_provider_structured, monkeypatch):
        """Test structured output passes system prompt correctly."""
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        _, provider = mock_ollama_provider_structured

        expected = SampleResponseModel(answer="with system", confidence=0.7)
        provider.astructured = AsyncMock(return_value=expected)

        client = LLMClient()
        client.structured("prompt", SampleResponseModel, system="You are a helpful assistant")

        call_args = provider.astructured.call_args
        messages = call_args[0][0]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant"
        assert messages[1]["role"] == "user"
