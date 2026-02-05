"""Anthropic LLM provider using official SDK."""

import os
from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic
from loguru import logger

from src.models.providers.base import BaseLLMProvider, ToolCall, retry


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

    def __init__(self, model: str, api_key: str | None = None) -> None:
        """Initialize Anthropic provider.

        Args:
            model: Model name (e.g., "claude-sonnet-4-20250514")
            api_key: API key (defaults to ANTHROPIC_API_KEY env var)
        """
        self._model = model
        self._client = AsyncAnthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        logger.debug(f"Initialized AnthropicProvider: model={model}")

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
        system = None
        remaining = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                remaining.append(msg)
        return system, remaining

    @retry(max_attempts=3, delay=1.0)
    async def acomplete(self, messages: list[dict], temperature: float = 0.7) -> str:
        """Generate completion from messages."""
        system, chat_messages = self._extract_system(messages)

        kwargs: dict = {
            "model": self._model,
            "messages": chat_messages,
            "temperature": temperature,
            "max_tokens": 4096,
        }
        if system:
            kwargs["system"] = system

        response = await self._client.messages.create(**kwargs)
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
            "max_tokens": 4096,
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
            "max_tokens": 4096,
        }
        if system:
            kwargs["system"] = system

        response = await self._client.messages.create(**kwargs)

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

    def __repr__(self) -> str:
        """String representation."""
        return f"AnthropicProvider(model={self._model})"
