"""LLM abstraction using custom provider implementations."""

import asyncio
import contextlib
import json
import os
import time
from collections.abc import AsyncIterator, Callable
from types import TracebackType
from typing import TypeVar, cast

import sniffio
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

from src.metrics.execution import ExecutionMetricsCollector
from src.models.providers import AnthropicProvider, BaseLLMProvider, OllamaProvider, OpenAIProvider
from src.models.providers.base import ToolCall

T = TypeVar("T", bound=BaseModel)

load_dotenv()


def _set_asyncio_context() -> None:
    """Set sniffio context to asyncio for httpx/httpcore compatibility.

    httpx/httpcore uses sniffio to detect the async library. This explicitly
    sets it to asyncio to ensure proper detection in CLI and daemon contexts.
    """
    sniffio.current_async_library_cvar.set("asyncio")  # type: ignore[bad-argument-type]


_DEFAULT_CONCURRENT_REQUESTS = 5
_MIN_CONCURRENT_REQUESTS = 1
_MAX_CONCURRENT_REQUESTS = 20


def _parse_max_concurrent_requests() -> int:
    """Parse and validate LLM_MAX_CONCURRENT environment variable.

    Valid range is 1-20. Falls back to default (5) on invalid values.

    Returns:
        Validated concurrency limit (1-20)
    """
    raw_value = os.getenv("LLM_MAX_CONCURRENT")

    if raw_value is None:
        return _DEFAULT_CONCURRENT_REQUESTS

    try:
        value = int(raw_value)
    except ValueError:
        logger.warning(
            "Invalid LLM_MAX_CONCURRENT value %r; using default %d",
            raw_value,
            _DEFAULT_CONCURRENT_REQUESTS,
        )
        return _DEFAULT_CONCURRENT_REQUESTS

    if value < _MIN_CONCURRENT_REQUESTS:
        logger.warning(
            "LLM_MAX_CONCURRENT value %d is below minimum %d; clamping to %d",
            value,
            _MIN_CONCURRENT_REQUESTS,
            _MIN_CONCURRENT_REQUESTS,
        )
        return _MIN_CONCURRENT_REQUESTS

    if value > _MAX_CONCURRENT_REQUESTS:
        logger.warning(
            "LLM_MAX_CONCURRENT value %d exceeds maximum %d; clamping to %d",
            value,
            _MAX_CONCURRENT_REQUESTS,
            _MAX_CONCURRENT_REQUESTS,
        )
        return _MAX_CONCURRENT_REQUESTS

    return value


# Limit concurrent async requests for multi-agent workflows (env: LLM_MAX_CONCURRENT, default 5)
# With concurrency=5, analyses stage: ~80-100s (vs ~287s serialized)
# OpenAI/Anthropic allow ~8-10 req/sec, Ollama (local) has no limits
MAX_CONCURRENT_REQUESTS = _parse_max_concurrent_requests()
_semaphore_holder: dict[str, asyncio.Semaphore | int | None] = {}


def _get_semaphore() -> asyncio.Semaphore:
    """Get or create the global request semaphore for current event loop."""
    try:
        current_loop = asyncio.get_running_loop()
        current_loop_id = id(current_loop)
    except RuntimeError:
        current_loop_id = None

    # Recreate semaphore if it doesn't exist or is bound to different loop
    stored_loop_id = _semaphore_holder.get("loop_id")
    if "semaphore" not in _semaphore_holder or stored_loop_id != current_loop_id:
        _semaphore_holder["semaphore"] = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        _semaphore_holder["loop_id"] = current_loop_id

    return cast("asyncio.Semaphore", _semaphore_holder["semaphore"])


class ToolResult(BaseModel):
    """Result of executing a tool."""

    tool_call_id: str
    content: str


class LLMClient:
    """Unified LLM client supporting Ollama (dev) and API providers (prod)."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        openai_base_url: str | None = None,
    ) -> None:
        """Initialize LLM client.

        Args:
            provider: LLM provider (ollama, anthropic, openai). Defaults to env.
            model: Model name. Defaults to env.
            base_url: Base URL for Ollama. Defaults to env.
            api_key: API key for provider (optional, falls back to env var)
            openai_base_url: Custom base URL for OpenAI (optional)
        """
        self.provider = provider or os.getenv("LLM_PROVIDER", "ollama")
        self.model = model or os.getenv("LLM_MODEL", "qwen3:14b")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self._api_key = api_key
        self._openai_base_url = openai_base_url

        self._provider: BaseLLMProvider = self._create_provider()
        self._metrics_collector: ExecutionMetricsCollector | None = None
        logger.info(f"Initialized LLM client: provider={self.provider}, model={self.model}")

    def set_metrics_collector(self, collector: ExecutionMetricsCollector | None) -> None:
        """Set or clear the execution metrics collector.

        Args:
            collector: Collector instance, or None to disable
        """
        self._metrics_collector = collector

    def _create_provider(self) -> BaseLLMProvider:
        """Create provider instance based on configuration."""
        if self.provider == "ollama":
            return OllamaProvider(model=self.model, base_url=self.base_url)
        if self.provider == "anthropic":
            return AnthropicProvider(model=self.model, api_key=self._api_key)
        if self.provider == "openai":
            return OpenAIProvider(
                model=self.model,
                api_key=self._api_key,
                base_url=self._openai_base_url or os.getenv("OPENAI_API_BASE"),
            )
        msg = f"Unsupported provider: {self.provider}"
        raise ValueError(msg)

    def _build_messages(self, prompt: str, system: str | None = None) -> list[dict[str, str]]:
        """Build messages list from prompt and optional system message."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.7) -> str:
        """Generate completion from prompt (sync wrapper).

        Args:
            prompt: User prompt
            system: System prompt (optional)
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            Generated text response
        """
        return asyncio.run(self.acomplete(prompt, system, temperature))

    async def acomplete(self, prompt: str, system: str | None = None, temperature: float = 0.7) -> str:
        """Generate completion from prompt asynchronously.

        Args:
            prompt: User prompt
            system: System prompt (optional)
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            Generated text response
        """
        _set_asyncio_context()
        messages = self._build_messages(prompt, system)
        start = time.perf_counter() if self._metrics_collector else None
        error_msg = None
        try:
            async with _get_semaphore():
                return await self._provider.acomplete(messages, temperature)
        except Exception as e:
            error_msg = str(e)
            raise
        finally:
            if self._metrics_collector and start is not None:
                self._metrics_collector.record_llm_call(
                    method="acomplete",
                    latency_ms=(time.perf_counter() - start) * 1000,
                    usage=self._provider.last_usage,
                    success=error_msg is None,
                    error=error_msg,
                )

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.7) -> str:
        """Multi-turn chat completion (sync wrapper).

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            Generated text response
        """
        return asyncio.run(self.achat(messages, temperature))

    async def achat(self, messages: list[dict[str, str]], temperature: float = 0.7) -> str:
        """Multi-turn chat completion asynchronously.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            Generated text response
        """
        _set_asyncio_context()
        start = time.perf_counter() if self._metrics_collector else None
        error_msg = None
        try:
            async with _get_semaphore():
                return await self._provider.acomplete(messages, temperature)
        except Exception as e:
            error_msg = str(e)
            raise
        finally:
            if self._metrics_collector and start is not None:
                self._metrics_collector.record_llm_call(
                    method="achat",
                    latency_ms=(time.perf_counter() - start) * 1000,
                    usage=self._provider.last_usage,
                    success=error_msg is None,
                    error=error_msg,
                )

    async def astream(
        self, prompt: str, system: str | None = None, temperature: float = 0.7
    ) -> AsyncIterator[str]:
        """Stream completion tokens asynchronously.

        Args:
            prompt: User prompt
            system: System prompt (optional)
            temperature: Sampling temperature (0.0-1.0)

        Yields:
            Individual tokens as they're generated
        """
        _set_asyncio_context()
        messages = self._build_messages(prompt, system)
        async with _get_semaphore():
            async for token in self._provider.astream(messages, temperature):
                yield token

    @property
    def supports_tools(self) -> bool:
        """Check if current provider supports tool calling.

        Returns:
            True if provider supports tool calling (anthropic, openai)
        """
        return self._provider.supports_tools

    @property
    def supports_structured_output(self) -> bool:
        """Check if current provider supports structured output.

        Returns:
            True if provider supports structured output
        """
        return self._provider.supports_structured_output

    def structured(
        self,
        prompt: str,
        response_model: type[T],
        system: str | None = None,
        temperature: float = 0.7,
    ) -> T:
        """Generate structured output validated against Pydantic model (sync wrapper).

        Args:
            prompt: User prompt
            response_model: Pydantic model class to validate response against
            system: System prompt (optional)
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            Validated instance of response_model

        Raises:
            StructuredOutputError: If response cannot be parsed or validated
        """
        return asyncio.run(self.astructured(prompt, response_model, system, temperature))

    async def astructured(
        self,
        prompt: str,
        response_model: type[T],
        system: str | None = None,
        temperature: float = 0.7,
    ) -> T:
        """Generate structured output validated against Pydantic model (async).

        Args:
            prompt: User prompt
            response_model: Pydantic model class to validate response against
            system: System prompt (optional)
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            Validated instance of response_model

        Raises:
            StructuredOutputError: If response cannot be parsed or validated
        """
        _set_asyncio_context()
        messages = self._build_messages(prompt, system)
        start = time.perf_counter() if self._metrics_collector else None
        error_msg = None
        try:
            async with _get_semaphore():
                return await self._provider.astructured(messages, response_model, temperature)
        except Exception as e:
            error_msg = str(e)
            raise
        finally:
            if self._metrics_collector and start is not None:
                self._metrics_collector.record_llm_call(
                    method="astructured",
                    latency_ms=(time.perf_counter() - start) * 1000,
                    usage=self._provider.last_usage,
                    success=error_msg is None,
                    error=error_msg,
                )

    def complete_with_tools(  # noqa: PLR0913
        self,
        prompt: str,
        tools: list[dict],
        tool_executor: Callable[[str, dict], str],
        system: str | None = None,
        temperature: float = 0.7,
        max_tool_calls: int = 5,
        on_tool_call: Callable[[str, dict, str], None] | None = None,
    ) -> str:
        """Generate completion with tool calling support (sync wrapper).

        Args:
            prompt: User prompt
            tools: List of tool definitions in OpenAI format
            tool_executor: Function to execute tools (name, args) -> result
            system: System prompt (optional)
            temperature: Sampling temperature (0.0-1.0)
            max_tool_calls: Maximum tool calls per completion
            on_tool_call: Callback invoked after each tool execution (name, args, result)

        Returns:
            Final text response after tool execution
        """
        return asyncio.run(
            self.acomplete_with_tools(
                prompt, tools, tool_executor, system, temperature, max_tool_calls, on_tool_call
            )
        )

    async def acomplete_with_tools(  # noqa: PLR0913
        self,
        prompt: str,
        tools: list[dict],
        tool_executor: Callable[[str, dict], str],
        system: str | None = None,
        temperature: float = 0.7,
        max_tool_calls: int = 5,
        on_tool_call: Callable[[str, dict, str], None] | None = None,
    ) -> str:
        """Generate completion with tool calling support (async).

        Args:
            prompt: User prompt
            tools: List of tool definitions in OpenAI format
            tool_executor: Function to execute tools (name, args) -> result
            system: System prompt (optional)
            temperature: Sampling temperature (0.0-1.0)
            max_tool_calls: Maximum tool calls per completion
            on_tool_call: Callback invoked after each tool execution (name, args, result)

        Returns:
            Final text response after tool execution
        """
        _set_asyncio_context()
        messages: list[dict] = self._build_messages(prompt, system)
        tool_calls_made = 0

        while tool_calls_made < max_tool_calls:
            start = time.perf_counter() if self._metrics_collector else None
            error_msg = None
            try:
                async with _get_semaphore():
                    text_response, tool_calls = await self._provider.acomplete_with_tools(
                        messages, tools, temperature
                    )
            except Exception as e:
                error_msg = str(e)
                raise
            finally:
                if self._metrics_collector and start is not None:
                    self._metrics_collector.record_llm_call(
                        method="acomplete_with_tools",
                        latency_ms=(time.perf_counter() - start) * 1000,
                        usage=self._provider.last_usage,
                        success=error_msg is None,
                        error=error_msg,
                    )

            if not tool_calls:
                return text_response or ""

            # Add assistant message with tool calls
            messages.append(self._format_tool_call_message(tool_calls))

            # Execute tools and add results
            for tool_call in tool_calls:
                tool_calls_made += 1
                result = self._execute_tool(tool_call, tool_executor)

                if on_tool_call:
                    on_tool_call(tool_call.name, tool_call.arguments, result)

                messages.append(self._format_tool_result_message(tool_call, result))

        # Final completion without tools
        async with _get_semaphore():
            return await self._provider.acomplete(messages, temperature)

    def _format_tool_call_message(self, tool_calls: list[ToolCall]) -> dict:
        """Format tool calls for assistant message."""
        if self.provider == "anthropic":
            content = []
            for tc in tool_calls:
                content.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
            return {"role": "assistant", "content": content}
        # OpenAI format
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in tool_calls
            ],
        }

    def _format_tool_result_message(self, tool_call: ToolCall, result: str) -> dict:
        """Format tool result for message."""
        if self.provider == "anthropic":
            return {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tool_call.id, "content": result}],
            }
        # OpenAI format
        return {"role": "tool", "tool_call_id": tool_call.id, "content": result}

    def _execute_tool(self, tool_call: ToolCall, executor: Callable[[str, dict], str]) -> str:
        """Execute a tool call and handle errors."""
        logger.debug(f"Executing tool: {tool_call.name} with args: {tool_call.arguments}")
        try:
            return executor(tool_call.name, tool_call.arguments)
        except Exception as e:
            logger.error(f"Tool '{tool_call.name}' execution failed: {e}")
            return f"Tool '{tool_call.name}' failed: {e}"

    async def close(self) -> None:
        """Close provider HTTP client."""
        await self._provider.close()

    def _schedule_close(self) -> None:
        """Best-effort scheduling of async close for sync contexts."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            with contextlib.suppress(RuntimeError):
                asyncio.run(self.close())
        else:
            task = loop.create_task(self.close())
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

    def __enter__(self) -> LLMClient:
        """Enter sync context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Exit sync context manager and ensure cleanup."""
        self._schedule_close()

    async def __aenter__(self) -> LLMClient:
        """Enter async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Exit async context manager and ensure cleanup."""
        await self.close()

    def __del__(self) -> None:
        """Best-effort cleanup on garbage collection."""
        with contextlib.suppress(Exception):
            if hasattr(self, "_provider"):
                self._schedule_close()

    def __repr__(self) -> str:
        """String representation."""
        return f"LLMClient(provider={self.provider}, model={self.model})"
