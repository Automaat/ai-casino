"""Anthropic LLM provider using official SDK."""

from collections.abc import AsyncIterator
from typing import TypeVar, cast

from anthropic import AsyncAnthropic
from anthropic.types import Message, TextBlock
from loguru import logger
from pydantic import BaseModel, ValidationError

from src.metrics.execution import LLMUsageStats
from src.models.providers.base import BaseLLMProvider, StructuredOutputError, ToolCall
from src.models.providers.retry import retry

T = TypeVar("T", bound=BaseModel)


def _convert_tools_to_anthropic(tools: list[dict]) -> list[dict]:
    """Convert OpenAI-style tools to Anthropic format.

    Args:
        tools: List of tool definitions in OpenAI format

    Returns:
        List of tool definitions in Anthropic format
    """
    anthropic_tools = []
    for tool in tools:
        if tool.get("type") == "function":
            func = tool["function"]
            anthropic_tools.append(
                {
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                }
            )
    return anthropic_tools


class AnthropicProvider(BaseLLMProvider):
    """Anthropic provider using official SDK."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        max_tokens: int = 1024,
        enable_caching: bool = False,
    ) -> None:
        """Initialize Anthropic provider.

        Args:
            model: Model name (e.g., "claude-sonnet-4-20250514")
            api_key: API key (defaults to ANTHROPIC_API_KEY env var)
            max_tokens: Maximum tokens in response (default: 1024)
            enable_caching: Enable prompt caching via cache_control blocks

        Raises:
            ValueError: If API key is not provided and ANTHROPIC_API_KEY env var is empty
        """
        super().__init__(enable_caching=enable_caching)

        if not api_key:
            msg = "Anthropic API key required in config (api_keys.anthropic_api_key)"
            raise ValueError(msg)

        self._model = model
        self._max_tokens = max_tokens
        self._client = AsyncAnthropic(api_key=api_key)
        logger.debug(f"Initialized AnthropicProvider: model={model}, max_tokens={max_tokens}")

    async def close(self) -> None:
        """Close HTTP client."""
        await self._client.close()

    def _extract_system(self, messages: list[dict]) -> tuple[str | None, list[dict]]:
        """Extract system message and return remaining messages.

        Args:
            messages: List of messages

        Returns:
            Tuple of (system_prompt, remaining_messages)
        """
        system_messages: list[str] = []
        remaining: list[dict] = []
        for msg in messages:
            if msg.get("role") == "system":
                system_messages.append(msg.get("content", ""))
            else:
                remaining.append(msg)

        system: str | None = None
        if system_messages:
            if len(system_messages) > 1:
                logger.warning(
                    f"Multiple system messages detected ({len(system_messages)}). Concatenating in order."
                )
            system = "\n\n".join(system_messages)

        return system, remaining

    def _build_system_param(self, system: str | None) -> list[dict] | str | None:
        """Build system parameter with optional cache_control.

        Args:
            system: System prompt text

        Returns:
            System param with cache_control block when caching enabled, plain string otherwise
        """
        if not system:
            return None
        if not self._enable_caching:
            return system
        return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]

    def _extract_usage(self, response: Message) -> LLMUsageStats:
        """Extract usage stats including cache fields from response.

        Args:
            response: Anthropic API response

        Returns:
            LLMUsageStats with cache token counts
        """
        usage = response.usage
        return LLMUsageStats(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", None),
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", None),
        )

    @retry(max_attempts=3, delay=1.0)
    async def acomplete(
        self, messages: list[dict], temperature: float = 0.7, max_tokens: int | None = None
    ) -> str:
        """Generate completion from messages."""
        system, chat_messages = self._extract_system(messages)

        kwargs: dict = {
            "model": self._model,
            "messages": chat_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self._max_tokens,
        }
        system_param = self._build_system_param(system)
        if system_param:
            kwargs["system"] = system_param

        response = cast("Message", await self._client.messages.create(**kwargs))
        self._last_usage = self._extract_usage(response)
        content = next((b.text for b in response.content if isinstance(b, TextBlock)), "")
        logger.debug(f"Anthropic response length: {len(content)} chars")
        return content

    async def astream(
        self, messages: list[dict], temperature: float = 0.7, max_tokens: int | None = None
    ) -> AsyncIterator[str]:
        """Stream completion tokens."""
        system, chat_messages = self._extract_system(messages)

        kwargs: dict = {
            "model": self._model,
            "messages": chat_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self._max_tokens,
        }
        system_param = self._build_system_param(system)
        if system_param:
            kwargs["system"] = system_param

        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    @retry(max_attempts=3, delay=1.0)
    async def acomplete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> tuple[str | None, list[ToolCall] | None]:
        """Generate completion with tool calling support."""
        system, chat_messages = self._extract_system(messages)
        converted_tools = _convert_tools_to_anthropic(tools)

        if self._enable_caching and converted_tools:
            converted_tools[-1]["cache_control"] = {"type": "ephemeral"}

        kwargs: dict = {
            "model": self._model,
            "messages": chat_messages,
            "tools": converted_tools,
            "temperature": temperature,
            "max_tokens": max_tokens or self._max_tokens,
        }
        system_param = self._build_system_param(system)
        if system_param:
            kwargs["system"] = system_param

        response = cast("Message", await self._client.messages.create(**kwargs))
        self._last_usage = self._extract_usage(response)

        text_content = None
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_content = block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )
                )

        if tool_calls:
            return None, tool_calls
        return text_content, None

    @property
    def supports_tools(self) -> bool:
        """Anthropic supports tool calling."""
        return True

    @retry(max_attempts=3, delay=1.0)
    async def astructured(
        self,
        messages: list[dict],
        response_model: type[T],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> T:
        """Generate structured output using tool use pattern."""
        system, chat_messages = self._extract_system(messages)

        schema = response_model.model_json_schema()
        tool: dict = {
            "name": "respond",
            "description": "Provide the structured response",
            "input_schema": schema,
        }
        if self._enable_caching:
            tool["cache_control"] = {"type": "ephemeral"}

        kwargs: dict = {
            "model": self._model,
            "messages": chat_messages,
            "tools": [tool],
            "tool_choice": {"type": "tool", "name": "respond"},
            "temperature": temperature,
            "max_tokens": max_tokens or self._max_tokens,
        }
        system_param = self._build_system_param(system)
        if system_param:
            kwargs["system"] = system_param

        response = cast("Message", await self._client.messages.create(**kwargs))
        self._last_usage = self._extract_usage(response)

        for block in response.content:
            if block.type == "tool_use" and block.name == "respond":
                try:
                    return response_model.model_validate(block.input)
                except ValidationError as e:
                    msg = f"Validation failed: {e}"
                    raise StructuredOutputError(msg, raw_response=str(block.input)) from e

        msg = "No tool_use block in response"
        raise StructuredOutputError(msg, raw_response=str(response.content))

    @property
    def supports_structured_output(self) -> bool:
        """Anthropic supports structured output via tool use."""
        return True

    def __repr__(self) -> str:
        """String representation."""
        return f"AnthropicProvider(model={self._model})"
