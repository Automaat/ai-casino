"""Anthropic LLM provider using official SDK."""

import os
from collections.abc import AsyncIterator
from typing import TypeVar

from anthropic import AsyncAnthropic
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

    def __init__(self, model: str, api_key: str | None = None, max_tokens: int = 4096) -> None:
        """Initialize Anthropic provider.

        Args:
            model: Model name (e.g., "claude-sonnet-4-20250514")
            api_key: API key (defaults to ANTHROPIC_API_KEY env var)
            max_tokens: Maximum tokens in response (default: 4096)

        Raises:
            ValueError: If API key is not provided and ANTHROPIC_API_KEY env var is empty
        """
        resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not resolved_key:
            msg = "Anthropic API key required: set ANTHROPIC_API_KEY env var or pass api_key"
            raise ValueError(msg)

        self._model = model
        self._max_tokens = max_tokens
        self._client = AsyncAnthropic(api_key=resolved_key)
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

    @retry(max_attempts=3, delay=1.0)
    async def acomplete(self, messages: list[dict], temperature: float = 0.7) -> str:
        """Generate completion from messages."""
        system, chat_messages = self._extract_system(messages)

        kwargs: dict = {
            "model": self._model,
            "messages": chat_messages,
            "temperature": temperature,
            "max_tokens": self._max_tokens,
        }
        if system:
            kwargs["system"] = system

        response = await self._client.messages.create(**kwargs)
        self._last_usage = LLMUsageStats(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        content = response.content[0].text if response.content else ""
        logger.debug(f"Anthropic response length: {len(content)} chars")
        return content

    @retry(max_attempts=3, delay=1.0)
    async def astream(self, messages: list[dict], temperature: float = 0.7) -> AsyncIterator[str]:
        """Stream completion tokens."""
        system, chat_messages = self._extract_system(messages)

        kwargs: dict = {
            "model": self._model,
            "messages": chat_messages,
            "temperature": temperature,
            "max_tokens": self._max_tokens,
        }
        if system:
            kwargs["system"] = system

        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    @retry(max_attempts=3, delay=1.0)
    async def acomplete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float = 0.7,
    ) -> tuple[str | None, list[ToolCall] | None]:
        """Generate completion with tool calling support."""
        system, chat_messages = self._extract_system(messages)

        kwargs: dict = {
            "model": self._model,
            "messages": chat_messages,
            "tools": _convert_tools_to_anthropic(tools),
            "temperature": temperature,
            "max_tokens": self._max_tokens,
        }
        if system:
            kwargs["system"] = system

        response = await self._client.messages.create(**kwargs)
        self._last_usage = LLMUsageStats(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

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
    ) -> T:
        """Generate structured output using tool use pattern."""
        system, chat_messages = self._extract_system(messages)

        schema = response_model.model_json_schema()
        tool = {
            "name": "respond",
            "description": "Provide the structured response",
            "input_schema": schema,
        }

        kwargs: dict = {
            "model": self._model,
            "messages": chat_messages,
            "tools": [tool],
            "tool_choice": {"type": "tool", "name": "respond"},
            "temperature": temperature,
            "max_tokens": self._max_tokens,
        }
        if system:
            kwargs["system"] = system

        response = await self._client.messages.create(**kwargs)
        self._last_usage = LLMUsageStats(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

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
