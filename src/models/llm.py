"""LLM abstraction using custom provider implementations."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, TypeVar, cast

from loguru import logger
from pydantic import BaseModel

from src.metrics.execution import ExecutionMetricsCollector
from src.models.message_models import (
    AnthropicAssistantMessage,
    AnthropicToolResultMessage,
    OpenAIAssistantMessage,
    OpenAIToolCall,
    OpenAIToolFunction,
    OpenAIToolResultMessage,
    ToolResultContent,
    ToolUseContent,
)
from src.models.providers import AnthropicProvider, BaseLLMProvider, OllamaProvider, OpenAIProvider
from src.models.providers.base import ToolCall

if TYPE_CHECKING:
    from src.tools.models import ToolDefinition


@dataclass
class ToolCallingParams:
    """Parameters for tool calling methods."""

    prompt: str
    tools: list[ToolDefinition]
    tool_executor: Callable[[str, dict], str] | Callable[[str, dict], Awaitable[str]]
    system: str | None = None
    temperature: float = 0.7
    max_tool_calls: int = 5
    on_tool_call: Callable[[str, dict, str], None] | None = None


T = TypeVar("T", bound=BaseModel)


_MIN_CONCURRENT_REQUESTS = 1
_MAX_CONCURRENT_REQUESTS = 20
_DEFAULT_CONCURRENT_REQUESTS = 5


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
        logger.opt(exception=True).warning(
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
        provider: str,
        model: str,
        base_url: str = "http://localhost:11434",
        api_key: str | None = None,
        openai_base_url: str | None = None,
    ) -> None:
        """Initialize LLM client.

        Args:
            provider: LLM provider (ollama, anthropic, openai)
            model: Model name
            base_url: Base URL for Ollama
            api_key: API key for provider (optional)
            openai_base_url: Custom base URL for OpenAI (optional)
        """
        self.provider = provider
        self.model = model
        self.base_url = base_url
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
                base_url=self._openai_base_url,
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

    def complete_with_tools(self, params: ToolCallingParams) -> str:
        """Generate completion with tool calling support (sync wrapper).

        Args:
            params: Tool calling parameters

        Returns:
            Final text response after tool execution
        """
        return asyncio.run(self.acomplete_with_tools(params))

    async def acomplete_with_tools(self, params: ToolCallingParams) -> str:
        """Generate completion with tool calling support (async).

        Args:
            params: Tool calling parameters

        Returns:
            Final text response after tool execution
        """
        messages: list[dict] = self._build_messages(params.prompt, params.system)
        tool_calls_made = 0

        # Convert ToolDefinition models to dicts for provider
        tools_dict = [tool.model_dump(mode="json", by_alias=True, exclude_none=True) for tool in params.tools]

        while tool_calls_made < params.max_tool_calls:
            start = time.perf_counter() if self._metrics_collector else None
            error_msg = None
            try:
                async with _get_semaphore():
                    text_response, tool_calls = await self._provider.acomplete_with_tools(
                        messages, tools_dict, params.temperature
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
                result = await self._execute_tool_async(tool_call, params.tool_executor)

                if params.on_tool_call:
                    params.on_tool_call(tool_call.name, tool_call.arguments, result)

                messages.append(self._format_tool_result_message(tool_call, result))

        # Final completion without tools
        async with _get_semaphore():
            return await self._provider.acomplete(messages, params.temperature)

    def _format_tool_call_message(self, tool_calls: list[ToolCall]) -> dict:
        """Format tool calls for assistant message."""
        if self.provider == "anthropic":
            content = [ToolUseContent(id=tc.id, name=tc.name, input=tc.arguments) for tc in tool_calls]
            message = AnthropicAssistantMessage(content=content)
            return message.model_dump(mode="json", exclude_none=True)
        # OpenAI format
        openai_tool_calls = [
            OpenAIToolCall(
                id=tc.id,
                function=OpenAIToolFunction(name=tc.name, arguments=json.dumps(tc.arguments)),
            )
            for tc in tool_calls
        ]
        message = OpenAIAssistantMessage(tool_calls=openai_tool_calls)
        message_dict = message.model_dump(mode="json", exclude_none=True)
        # Ensure OpenAI assistant tool-call messages always include an explicit content field
        if "content" not in message_dict:
            message_dict["content"] = None
        return message_dict

    def _format_tool_result_message(self, tool_call: ToolCall, result: str) -> dict:
        """Format tool result for message."""
        if self.provider == "anthropic":
            content = [ToolResultContent(tool_use_id=tool_call.id, content=result)]
            message = AnthropicToolResultMessage(content=content)
            return message.model_dump(mode="json", exclude_none=True)
        # OpenAI format
        message = OpenAIToolResultMessage(tool_call_id=tool_call.id, content=result)
        return message.model_dump(mode="json", exclude_none=True)

    def _execute_tool(self, tool_call: ToolCall, executor: Callable[[str, dict], str]) -> str:
        """Execute a tool call and handle errors."""
        logger.debug(f"Executing tool: {tool_call.name} with args: {tool_call.arguments}")
        try:
            return executor(tool_call.name, tool_call.arguments)
        except Exception as e:
            logger.opt(exception=True).error(f"Tool '{tool_call.name}' execution failed: {e}")
            return f"Tool '{tool_call.name}' failed: {e}"

    async def _execute_tool_async(
        self,
        tool_call: ToolCall,
        executor: Callable[[str, dict], str] | Callable[[str, dict], Awaitable[str]],
    ) -> str:
        """Execute a tool call (sync or async) and handle errors.

        Args:
            tool_call: Tool call details (name, args, id)
            executor: Sync or async executor function

        Returns:
            Tool execution result as string
        """
        logger.debug(f"Executing tool: {tool_call.name} with args: {tool_call.arguments}")
        try:
            # Check if executor is async function
            if inspect.iscoroutinefunction(executor):
                async_executor = cast("Callable[[str, dict], Awaitable[str]]", executor)
                return await async_executor(tool_call.name, tool_call.arguments)

            # Sync executor - offload to thread to avoid blocking event loop
            sync_executor = cast("Callable[[str, dict], str]", executor)
            result = await asyncio.to_thread(sync_executor, tool_call.name, tool_call.arguments)

            # Handle edge case: sync function returns awaitable (e.g., coroutine, Future, Task)
            if inspect.isawaitable(result):
                return await result

            return result
        except Exception as e:
            logger.opt(exception=True).error(f"Tool '{tool_call.name}' execution failed: {e}")
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
