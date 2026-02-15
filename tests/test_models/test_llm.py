"""Tests for LLM client."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from src.models.llm import LLMClient, ToolCallingParams
from src.models.providers.base import StructuredOutputError, ToolCall
from src.tools.models import ToolDefinition, ToolFunction, ToolParameter, ToolParametersSchema


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


def test_llm_client_init_ollama(mock_ollama_provider):
    client = LLMClient(provider="ollama", model="qwen3:14b")

    assert client.provider == "ollama"
    assert client.model == "qwen3:14b"
    mock_ollama_provider[0].assert_called_once_with(model="qwen3:14b", base_url="http://localhost:11434")


def test_llm_client_init_anthropic(mock_anthropic_provider):
    client = LLMClient(provider="anthropic", model="claude-sonnet-4-20250514")

    assert client.provider == "anthropic"
    assert client.model == "claude-sonnet-4-20250514"
    mock_anthropic_provider[0].assert_called_once_with(model="claude-sonnet-4-20250514", api_key=None)


def test_llm_client_init_openai(mock_openai_provider):
    client = LLMClient(provider="openai", model="gpt-4o")

    assert client.provider == "openai"
    assert client.model == "gpt-4o"
    mock_openai_provider[0].assert_called_once_with(model="gpt-4o", api_key=None, base_url=None)


def test_llm_client_init_openai_custom_api_base(mock_openai_provider):
    client = LLMClient(
        provider="openai",
        model="hf:moonshotai/Kimi-K2.5",
        openai_base_url="https://api.synthetic.new/openai/v1",
    )

    assert client.provider == "openai"
    assert client.model == "hf:moonshotai/Kimi-K2.5"
    mock_openai_provider[0].assert_called_once_with(
        model="hf:moonshotai/Kimi-K2.5", api_key=None, base_url="https://api.synthetic.new/openai/v1"
    )


def test_llm_client_unsupported_provider(monkeypatch):
    with pytest.raises(ValueError, match="Unsupported provider: invalid"):
        LLMClient(provider="invalid", model="test-model")


def test_complete_with_system_prompt(mock_ollama_provider):
    _, provider = mock_ollama_provider

    client = LLMClient(provider="ollama", model="qwen3:14b")
    result = client.complete("Test prompt", system="System message", temperature=0.5)

    assert result == "Mocked response"
    provider.acomplete.assert_called_once()
    call_args = provider.acomplete.call_args
    assert call_args[0][0] == [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "Test prompt"},
    ]
    assert call_args[0][1] == 0.5


def test_complete_without_system_prompt(mock_ollama_provider):
    _, provider = mock_ollama_provider

    client = LLMClient(provider="ollama", model="qwen3:14b")
    result = client.complete("Test prompt")

    assert result == "Mocked response"
    call_args = provider.acomplete.call_args
    assert call_args[0][0] == [{"role": "user", "content": "Test prompt"}]


def test_chat(mock_ollama_provider):
    _, provider = mock_ollama_provider

    client = LLMClient(provider="ollama", model="qwen3:14b")

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


def test_complete_handles_exception(mock_ollama_provider):
    _, provider = mock_ollama_provider
    provider.acomplete = AsyncMock(side_effect=Exception("API Error"))

    client = LLMClient(provider="ollama", model="qwen3:14b")

    with pytest.raises(Exception, match="API Error"):
        client.complete("Test prompt")


def test_repr(mock_ollama_provider):
    client = LLMClient(provider="ollama", model="qwen3:14b")

    assert repr(client) == "LLMClient(provider=ollama, model=qwen3:14b)"


class TestCompleteWithTools:
    """Tests for complete_with_tools method."""

    @pytest.fixture
    def sample_tools(self):
        return [
            ToolDefinition(
                function=ToolFunction(
                    name="get_weather",
                    description="Get weather for a location",
                    parameters=ToolParametersSchema(
                        properties={"location": ToolParameter(type="string", description="Location name")},
                        required=["location"],
                    ),
                ),
            )
        ]

    @pytest.fixture
    def mock_tool_executor(self):
        def executor(name: str, args: dict) -> str:
            if name == "get_weather":
                return f"Weather in {args['location']}: Sunny, 72°F"
            return "Unknown tool"

        return executor

    def test_complete_with_tools_no_tool_calls(self, mock_openai_provider, sample_tools, mock_tool_executor):
        """Test when LLM returns without tool calls."""
        _, provider = mock_openai_provider
        provider.acomplete_with_tools = AsyncMock(return_value=("I don't need tools for this", None))

        from src.models.llm import ToolCallingParams

        client = LLMClient(provider="openai", model="gpt-4o")
        params = ToolCallingParams(
            prompt="Hello",
            tools=sample_tools,
            tool_executor=mock_tool_executor,
        )
        result = client.complete_with_tools(params)

        assert result == "I don't need tools for this"
        assert provider.acomplete_with_tools.call_count == 1

    def test_complete_with_tools_executes_tool(self, mock_openai_provider, sample_tools, mock_tool_executor):
        """Test tool execution and final response."""
        _, provider = mock_openai_provider

        tool_call = ToolCall(id="call_123", name="get_weather", arguments={"location": "NYC"})
        provider.acomplete_with_tools = AsyncMock(
            side_effect=[
                (None, [tool_call]),
                ("The weather in NYC is sunny and 72°F", None),
            ]
        )

        client = LLMClient(provider="openai", model="gpt-4o")
        params = ToolCallingParams(
            prompt="What's the weather in NYC?", tools=sample_tools, tool_executor=mock_tool_executor
        )
        result = client.complete_with_tools(params)

        assert result == "The weather in NYC is sunny and 72°F"
        assert provider.acomplete_with_tools.call_count == 2

    def test_complete_with_tools_max_calls_limit(
        self, mock_openai_provider, sample_tools, mock_tool_executor
    ):
        """Test max_tool_calls limit is respected."""
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

        client = LLMClient(provider="openai", model="gpt-4o")
        params = ToolCallingParams(
            prompt="prompt", tools=sample_tools, tool_executor=mock_tool_executor, max_tool_calls=2
        )
        result = client.complete_with_tools(params)

        assert result == "Final response after max calls"
        assert provider.acomplete_with_tools.call_count == 2
        assert provider.acomplete.call_count == 1


class TestAcompleteWithTools:
    """Tests for acomplete_with_tools method."""

    @pytest.fixture
    def sample_tools(self):
        return [
            ToolDefinition(
                function=ToolFunction(
                    name="search",
                    description="Search the web",
                    parameters=ToolParametersSchema(
                        properties={"query": ToolParameter(type="string", description="Search query")},
                        required=["query"],
                    ),
                ),
            )
        ]

    @pytest.fixture
    def mock_tool_executor(self):
        def executor(name: str, args: dict) -> str:
            if name == "search":
                return f"Results for: {args['query']}"
            return "Unknown tool"

        return executor

    async def test_acomplete_with_tools_no_tool_calls(
        self, mock_openai_provider, sample_tools, mock_tool_executor
    ):
        """Test async when LLM returns without tool calls."""
        _, provider = mock_openai_provider
        provider.acomplete_with_tools = AsyncMock(return_value=("No tools needed", None))

        client = LLMClient(provider="openai", model="gpt-4o")
        params = ToolCallingParams(prompt="Hello", tools=sample_tools, tool_executor=mock_tool_executor)
        result = await client.acomplete_with_tools(params)

        assert result == "No tools needed"

    async def test_acomplete_with_tools_executes_tool(
        self, mock_openai_provider, sample_tools, mock_tool_executor
    ):
        """Test async tool execution."""
        _, provider = mock_openai_provider

        tool_call = ToolCall(id="call_456", name="search", arguments={"query": "python testing"})
        provider.acomplete_with_tools = AsyncMock(
            side_effect=[
                (None, [tool_call]),
                ("Found results about Python testing", None),
            ]
        )

        client = LLMClient(provider="openai", model="gpt-4o")
        params = ToolCallingParams(
            prompt="Search for python testing", tools=sample_tools, tool_executor=mock_tool_executor
        )
        result = await client.acomplete_with_tools(params)

        assert result == "Found results about Python testing"


class TestAsyncToolExecutor:
    """Tests for async tool executor support in acomplete_with_tools."""

    @pytest.fixture
    def sample_tools(self):
        return [
            ToolDefinition(
                function=ToolFunction(
                    name="analyze",
                    description="Analyze stock",
                    parameters=ToolParametersSchema(
                        properties={"symbol": ToolParameter(type="string", description="Stock symbol")},
                        required=["symbol"],
                    ),
                ),
            )
        ]

    @pytest.mark.asyncio
    async def test_acomplete_with_tools_sync_executor(self, mock_openai_provider, sample_tools):
        """Test backward compatibility with sync executor."""
        _, provider = mock_openai_provider

        def sync_executor(name: str, args: dict) -> str:
            return f"Sync result: {name} for {args.get('symbol', 'N/A')}"

        tool_call = ToolCall(id="call_123", name="analyze", arguments={"symbol": "AAPL"})
        provider.acomplete_with_tools = AsyncMock(
            side_effect=[
                (None, [tool_call]),
                ("Analysis complete", None),
            ]
        )

        client = LLMClient(provider="openai", model="gpt-4o")
        params = ToolCallingParams(prompt="Analyze AAPL", tools=sample_tools, tool_executor=sync_executor)
        result = await client.acomplete_with_tools(params)

        assert result == "Analysis complete"

    @pytest.mark.asyncio
    async def test_acomplete_with_tools_async_executor(self, mock_openai_provider, sample_tools):
        """Test new async executor support."""
        _, provider = mock_openai_provider

        async def async_executor(name: str, args: dict) -> str:
            await asyncio.sleep(0.01)
            return f"Async result: {name} for {args.get('symbol', 'N/A')}"

        tool_call = ToolCall(id="call_456", name="analyze", arguments={"symbol": "TSLA"})
        provider.acomplete_with_tools = AsyncMock(
            side_effect=[
                (None, [tool_call]),
                ("Async analysis complete", None),
            ]
        )

        client = LLMClient(provider="openai", model="gpt-4o")
        params = ToolCallingParams(prompt="Analyze TSLA", tools=sample_tools, tool_executor=async_executor)
        result = await client.acomplete_with_tools(params)

        assert result == "Async analysis complete"

    @pytest.mark.asyncio
    async def test_acomplete_with_tools_multiple_async_calls(self, mock_openai_provider, sample_tools):
        """Test multiple tool calls with async executor."""
        _, provider = mock_openai_provider

        call_count = 0

        async def async_executor(name: str, args: dict) -> str:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return f"Call {call_count}: {args.get('symbol', 'N/A')}"

        tool_call_1 = ToolCall(id="call_1", name="analyze", arguments={"symbol": "AAPL"})
        tool_call_2 = ToolCall(id="call_2", name="analyze", arguments={"symbol": "TSLA"})
        provider.acomplete_with_tools = AsyncMock(
            side_effect=[
                (None, [tool_call_1, tool_call_2]),
                ("Multiple analyses complete", None),
            ]
        )

        client = LLMClient(provider="openai", model="gpt-4o")
        params = ToolCallingParams(prompt="Analyze stocks", tools=sample_tools, tool_executor=async_executor)
        result = await client.acomplete_with_tools(params)

        assert result == "Multiple analyses complete"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_acomplete_with_tools_sync_executor_error(self, mock_openai_provider, sample_tools):
        """Test error handling with failing sync executor."""
        _, provider = mock_openai_provider

        def failing_sync_executor(name: str, args: dict) -> str:
            msg = "Sync execution failed"
            raise ValueError(msg)

        tool_call = ToolCall(id="call_789", name="analyze", arguments={"symbol": "MSFT"})
        provider.acomplete_with_tools = AsyncMock(
            side_effect=[
                (None, [tool_call]),
                ("Error handled", None),
            ]
        )

        client = LLMClient(provider="openai", model="gpt-4o")
        params = ToolCallingParams(
            prompt="Analyze MSFT", tools=sample_tools, tool_executor=failing_sync_executor
        )
        result = await client.acomplete_with_tools(params)

        assert result == "Error handled"

    @pytest.mark.asyncio
    async def test_acomplete_with_tools_async_executor_error(self, mock_openai_provider, sample_tools):
        """Test error handling with failing async executor."""
        _, provider = mock_openai_provider

        async def failing_async_executor(name: str, args: dict) -> str:
            await asyncio.sleep(0.01)
            msg = "Async execution failed"
            raise RuntimeError(msg)

        tool_call = ToolCall(id="call_abc", name="analyze", arguments={"symbol": "GOOGL"})
        provider.acomplete_with_tools = AsyncMock(
            side_effect=[
                (None, [tool_call]),
                ("Async error handled", None),
            ]
        )

        client = LLMClient(provider="openai", model="gpt-4o")
        params = ToolCallingParams(
            prompt="Analyze GOOGL", tools=sample_tools, tool_executor=failing_async_executor
        )
        result = await client.acomplete_with_tools(params)

        assert result == "Async error handled"

    @pytest.mark.asyncio
    async def test_acomplete_with_tools_sync_returning_awaitable(self, mock_openai_provider, sample_tools):
        """Test sync function that returns awaitable (edge case)."""
        _, provider = mock_openai_provider

        async def inner_async(symbol: str) -> str:
            await asyncio.sleep(0.01)
            return f"Wrapped result: {symbol}"

        def sync_returning_awaitable(name: str, args: dict):
            return inner_async(args.get("symbol", "N/A"))

        tool_call = ToolCall(id="call_xyz", name="analyze", arguments={"symbol": "AMZN"})
        provider.acomplete_with_tools = AsyncMock(
            side_effect=[
                (None, [tool_call]),
                ("Awaitable handled", None),
            ]
        )

        client = LLMClient(provider="openai", model="gpt-4o")
        params = ToolCallingParams(
            prompt="Analyze AMZN", tools=sample_tools, tool_executor=sync_returning_awaitable
        )
        result = await client.acomplete_with_tools(params)

        assert result == "Awaitable handled"

    @pytest.mark.asyncio
    async def test_acomplete_with_tools_callback_with_async_executor(
        self, mock_openai_provider, sample_tools
    ):
        """Test on_tool_call callback invoked correctly with async executor."""
        _, provider = mock_openai_provider

        async def async_executor(name: str, args: dict) -> str:
            await asyncio.sleep(0.01)
            return f"Result for {args.get('symbol', 'N/A')}"

        callback_calls = []

        def on_tool_call(name: str, args: dict, result: str) -> None:
            callback_calls.append((name, args, result))

        tool_call = ToolCall(id="call_callback", name="analyze", arguments={"symbol": "NVDA"})
        provider.acomplete_with_tools = AsyncMock(
            side_effect=[
                (None, [tool_call]),
                ("Callback test complete", None),
            ]
        )

        client = LLMClient(provider="openai", model="gpt-4o")
        params = ToolCallingParams(
            prompt="Analyze NVDA", tools=sample_tools, tool_executor=async_executor, on_tool_call=on_tool_call
        )
        result = await client.acomplete_with_tools(params)

        assert result == "Callback test complete"
        assert len(callback_calls) == 1
        assert callback_calls[0][0] == "analyze"
        assert callback_calls[0][1] == {"symbol": "NVDA"}
        assert "Result for NVDA" in callback_calls[0][2]


class TestParallelToolExecution:
    """Tests for parallel tool execution in acomplete_with_tools."""

    @pytest.fixture
    def sample_tools(self):
        return [
            ToolDefinition(
                function=ToolFunction(
                    name="search",
                    description="Search the web",
                    parameters=ToolParametersSchema(
                        properties={"query": ToolParameter(type="string", description="Search query")},
                        required=["query"],
                    ),
                ),
            )
        ]

    @pytest.mark.asyncio
    async def test_tools_execute_concurrently(self, mock_openai_provider, sample_tools):
        """Verify multiple tools execute concurrently, not sequentially."""
        _, provider = mock_openai_provider

        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def async_executor(name: str, args: dict) -> str:
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)

            await asyncio.sleep(0.05)  # Simulate work

            async with lock:
                current_concurrent -= 1
            return f"Result for {args.get('query', 'N/A')}"

        tool_calls = [
            ToolCall(id=f"call_{i}", name="search", arguments={"query": f"query{i}"}) for i in range(3)
        ]
        provider.acomplete_with_tools = AsyncMock(
            side_effect=[
                (None, tool_calls),
                ("All searches complete", None),
            ]
        )

        client = LLMClient(provider="openai", model="gpt-4o")
        params = ToolCallingParams(prompt="Search", tools=sample_tools, tool_executor=async_executor)

        import time

        start = time.perf_counter()
        result = await client.acomplete_with_tools(params)

        assert result == "All searches complete"
        assert max_concurrent > 1, "Tools should execute concurrently"

    @pytest.mark.asyncio
    async def test_callback_ordering_preserved(self, mock_openai_provider, sample_tools):
        """Verify callbacks invoked in tool_calls order despite random delays."""
        _, provider = mock_openai_provider

        import random

        async def async_executor(name: str, args: dict) -> str:
            await asyncio.sleep(random.uniform(0.001, 0.01))  # Random delay
            return f"Result for {args.get('query', 'N/A')}"

        callback_order = []

        def on_tool_call(name: str, args: dict, result: str) -> None:
            callback_order.append(args["query"])

        tool_calls = [
            ToolCall(id=f"call_{i}", name="search", arguments={"query": f"query{i}"}) for i in range(5)
        ]
        provider.acomplete_with_tools = AsyncMock(
            side_effect=[
                (None, tool_calls),
                ("Complete", None),
            ]
        )

        client = LLMClient(provider="openai", model="gpt-4o")
        params = ToolCallingParams(
            prompt="Search", tools=sample_tools, tool_executor=async_executor, on_tool_call=on_tool_call
        )
        await client.acomplete_with_tools(params)

        # Callbacks should be in original tool_calls order
        assert callback_order == ["query0", "query1", "query2", "query3", "query4"]

    @pytest.mark.asyncio
    async def test_message_ordering_preserved(self, mock_openai_provider, sample_tools):
        """Verify messages formatted in deterministic order."""
        _, provider = mock_openai_provider

        import random

        async def async_executor(name: str, args: dict) -> str:
            await asyncio.sleep(random.uniform(0.001, 0.01))
            return f"Result {args['query']}"

        tool_calls = [ToolCall(id=f"call_{i}", name="search", arguments={"query": str(i)}) for i in range(3)]

        messages_log = []

        # Capture messages passed to provider
        async def capture_messages(msgs, tools, temp):
            messages_log.append(list(msgs))
            if len(messages_log) == 1:
                return (None, tool_calls)
            return ("Complete", None)

        provider.acomplete_with_tools = AsyncMock(side_effect=capture_messages)

        client = LLMClient(provider="openai", model="gpt-4o")
        params = ToolCallingParams(prompt="Search", tools=sample_tools, tool_executor=async_executor)
        await client.acomplete_with_tools(params)

        # Second call should have tool results in order
        assert len(messages_log) == 2
        tool_results = [m for m in messages_log[1] if m.get("role") == "tool"]
        assert len(tool_results) == 3
        # Results should match tool_calls order
        for i, msg in enumerate(tool_results):
            assert f"Result {i}" in msg["content"]

    @pytest.mark.asyncio
    async def test_partial_failures_handled(self, mock_openai_provider, sample_tools):
        """Verify mixed success/failure results handled gracefully."""
        _, provider = mock_openai_provider

        async def async_executor(name: str, args: dict) -> str:
            query = args.get("query", "")
            if "fail" in query:
                msg = f"Simulated failure for {query}"
                raise ValueError(msg)
            return f"Success for {query}"

        tool_calls = [
            ToolCall(id="call_0", name="search", arguments={"query": "success1"}),
            ToolCall(id="call_1", name="search", arguments={"query": "fail1"}),
            ToolCall(id="call_2", name="search", arguments={"query": "success2"}),
        ]
        provider.acomplete_with_tools = AsyncMock(
            side_effect=[
                (None, tool_calls),
                ("Handled partial failures", None),
            ]
        )

        client = LLMClient(provider="openai", model="gpt-4o")
        params = ToolCallingParams(prompt="Search", tools=sample_tools, tool_executor=async_executor)
        result = await client.acomplete_with_tools(params)

        assert result == "Handled partial failures"

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self, mock_openai_provider, sample_tools, monkeypatch):
        """Verify semaphore enforces concurrency limit."""
        monkeypatch.setenv("TOOL_EXECUTION_MAX_CONCURRENT", "2")

        # Recreate module-level constants with monkeypatch for automatic cleanup
        from src.models import llm

        monkeypatch.setattr(llm, "MAX_CONCURRENT_TOOL_EXECUTIONS", 2)
        monkeypatch.setattr(llm, "_tool_semaphore_holder", {})

        _, provider = mock_openai_provider

        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def async_executor(name: str, args: dict) -> str:
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)

            await asyncio.sleep(0.02)

            async with lock:
                current_concurrent -= 1
            return f"Result {args.get('query', 'N/A')}"

        tool_calls = [ToolCall(id=f"call_{i}", name="search", arguments={"query": str(i)}) for i in range(5)]
        provider.acomplete_with_tools = AsyncMock(
            side_effect=[
                (None, tool_calls),
                ("Complete", None),
            ]
        )

        client = LLMClient(provider="openai", model="gpt-4o")
        params = ToolCallingParams(prompt="Search", tools=sample_tools, tool_executor=async_executor)
        await client.acomplete_with_tools(params)

        assert max_concurrent == 2, f"Expected max 2 concurrent, got {max_concurrent}"

    @pytest.mark.asyncio
    async def test_control_flow_exception_propagates(self, mock_openai_provider, sample_tools):
        """Verify CancelledError raises immediately."""
        _, provider = mock_openai_provider

        async def async_executor(name: str, args: dict) -> str:
            raise asyncio.CancelledError

        tool_calls = [ToolCall(id="call_0", name="search", arguments={"query": "test"})]
        provider.acomplete_with_tools = AsyncMock(return_value=(None, tool_calls))

        client = LLMClient(provider="openai", model="gpt-4o")
        params = ToolCallingParams(prompt="Search", tools=sample_tools, tool_executor=async_executor)

        with pytest.raises(asyncio.CancelledError):
            await client.acomplete_with_tools(params)

    @pytest.mark.asyncio
    async def test_single_tool_call(self, mock_openai_provider, sample_tools):
        """Verify single tool still works (edge case)."""
        _, provider = mock_openai_provider

        async def async_executor(name: str, args: dict) -> str:
            return f"Single result for {args.get('query', 'N/A')}"

        tool_call = ToolCall(id="call_0", name="search", arguments={"query": "single"})
        provider.acomplete_with_tools = AsyncMock(
            side_effect=[
                (None, [tool_call]),
                ("Single tool complete", None),
            ]
        )

        client = LLMClient(provider="openai", model="gpt-4o")
        params = ToolCallingParams(prompt="Search", tools=sample_tools, tool_executor=async_executor)
        result = await client.acomplete_with_tools(params)

        assert result == "Single tool complete"


class TestSupportsTools:
    """Tests for supports_tools property."""

    def test_supports_tools_anthropic(self, mock_anthropic_provider):
        _ = mock_anthropic_provider  # Fixture required to mock provider creation

        client = LLMClient(provider="anthropic", model="claude-sonnet-4-20250514")
        assert client.supports_tools is True

    def test_supports_tools_openai(self, mock_openai_provider):
        _ = mock_openai_provider  # Fixture required to mock provider creation

        client = LLMClient(provider="openai", model="gpt-4o")
        assert client.supports_tools is True

    def test_supports_tools_ollama(self, mock_ollama_provider):
        _ = mock_ollama_provider  # Fixture required to mock provider creation

        client = LLMClient(provider="ollama", model="qwen3:14b")
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

    def test_supports_structured_output_ollama(self, mock_ollama_provider_structured):
        """Test supports_structured_output for Ollama."""
        _ = mock_ollama_provider_structured

        client = LLMClient(provider="ollama", model="qwen3:14b")
        assert client.supports_structured_output is True

    def test_supports_structured_output_openai(self, mock_openai_provider_structured):
        """Test supports_structured_output for OpenAI."""
        _ = mock_openai_provider_structured

        client = LLMClient(provider="openai", model="gpt-4o")
        assert client.supports_structured_output is True

    def test_structured_returns_validated_model(self, mock_ollama_provider_structured):
        """Test structured output returns validated Pydantic model."""
        _, provider = mock_ollama_provider_structured

        expected = SampleResponseModel(answer="test answer", confidence=0.85)
        provider.astructured = AsyncMock(return_value=expected)

        client = LLMClient(provider="ollama", model="qwen3:14b")
        result = client.structured("What is 2+2?", SampleResponseModel, temperature=0.5)

        assert result == expected
        assert result.answer == "test answer"
        assert result.confidence == 0.85
        provider.astructured.assert_called_once()

    async def test_astructured_returns_validated_model(self, mock_ollama_provider_structured):
        """Test async structured output returns validated Pydantic model."""
        _, provider = mock_ollama_provider_structured

        expected = SampleResponseModel(answer="async answer", confidence=0.9)
        provider.astructured = AsyncMock(return_value=expected)

        client = LLMClient(provider="ollama", model="qwen3:14b")
        result = await client.astructured("prompt", SampleResponseModel, system="system msg")

        assert result == expected
        provider.astructured.assert_called_once()

    def test_structured_raises_on_error(self, mock_ollama_provider_structured):
        """Test structured output raises StructuredOutputError on failure."""
        _, provider = mock_ollama_provider_structured

        provider.astructured = AsyncMock(
            side_effect=StructuredOutputError("Validation failed", raw_response='{"invalid": "json"}')
        )

        client = LLMClient(provider="ollama", model="qwen3:14b")
        with pytest.raises(StructuredOutputError) as exc_info:
            client.structured("prompt", SampleResponseModel)

        assert "Validation failed" in str(exc_info.value)
        assert exc_info.value.raw_response == '{"invalid": "json"}'

    def test_structured_passes_system_prompt(self, mock_ollama_provider_structured):
        """Test structured output passes system prompt correctly."""
        _, provider = mock_ollama_provider_structured

        expected = SampleResponseModel(answer="with system", confidence=0.7)
        provider.astructured = AsyncMock(return_value=expected)

        client = LLMClient(provider="ollama", model="qwen3:14b")
        client.structured("prompt", SampleResponseModel, system="You are a helpful assistant")

        call_args = provider.astructured.call_args
        messages = call_args[0][0]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant"
        assert messages[1]["role"] == "user"
