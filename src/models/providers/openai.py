"""OpenAI LLM provider using official SDK."""

import json
import os
from collections.abc import AsyncIterator

from loguru import logger
from openai import AsyncOpenAI

from src.models.providers.base import BaseLLMProvider, ToolCall, retry


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider using official SDK."""

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None) -> None:
        """Initialize OpenAI provider.

        Args:
            model: Model name (e.g., "gpt-4o")
            api_key: API key (defaults to OPENAI_API_KEY env var)
            base_url: Custom base URL (defaults to OPENAI_API_BASE env var)
        """
        self._model = model
        self._is_gpt5 = model.startswith("gpt-5")
        self._client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_API_BASE"),
        )
        logger.debug(f"Initialized OpenAIProvider: model={model}")

    def _effective_temperature(self, temperature: float) -> float:
        """Get effective temperature, forcing 1.0 for GPT-5 models."""
        if self._is_gpt5 and temperature != 1.0:
            logger.debug(f"GPT-5 requires temperature=1, ignoring requested {temperature}")
            return 1.0
        return temperature

    async def close(self) -> None:
        """Close HTTP client."""
        await self._client.close()

    @retry(max_attempts=3, delay=1.0)
    async def acomplete(self, messages: list[dict], temperature: float = 0.7) -> str:
        """Generate completion from messages."""
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._effective_temperature(temperature),
        )
        content = response.choices[0].message.content or ""
        logger.debug(f"OpenAI response length: {len(content)} chars")
        return content

    @retry(max_attempts=3, delay=1.0)
    async def astream(self, messages: list[dict], temperature: float = 0.7) -> AsyncIterator[str]:
        """Stream completion tokens."""
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._effective_temperature(temperature),
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    @retry(max_attempts=3, delay=1.0)
    async def acomplete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float = 0.7,
    ) -> tuple[str | None, list[ToolCall] | None]:
        """Generate completion with tool calling support."""
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools,
            temperature=self._effective_temperature(temperature),
        )
        message = response.choices[0].message

        if message.tool_calls:
            tool_calls: list[ToolCall] = []
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError as exc:
                    logger.error(f"Failed to parse tool call arguments for tool '{tc.function.name}': {exc}")
                    return None, None

                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                    )
                )
            return None, tool_calls

        return message.content, None

    @property
    def supports_tools(self) -> bool:
        """OpenAI supports tool calling."""
        return True

    def __repr__(self) -> str:
        """String representation."""
        return f"OpenAIProvider(model={self._model})"
