"""OpenAI LLM provider using official SDK."""

import json
import os
from collections.abc import AsyncIterator
from typing import TypeVar

from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from src.models.providers.base import BaseLLMProvider, StructuredOutputError, ToolCall, retry

T = TypeVar("T", bound=BaseModel)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider using official SDK."""

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None) -> None:
        """Initialize OpenAI provider.

        Args:
            model: Model name (e.g., "gpt-4o")
            api_key: API key (defaults to OPENAI_API_KEY env var)
            base_url: Custom base URL (defaults to OPENAI_API_BASE env var)

        Raises:
            ValueError: If API key is not provided and OPENAI_API_KEY env var is empty
        """
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            msg = "OpenAI API key required: set OPENAI_API_KEY env var or pass api_key"
            raise ValueError(msg)

        self._model = model
        self._is_gpt5 = model.startswith("gpt-5")
        self._client = AsyncOpenAI(
            api_key=resolved_key,
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

    def _ensure_additional_properties_false(self, schema: dict) -> dict:
        """Recursively ensure additionalProperties is false in schema (required by OpenAI strict mode).

        Args:
            schema: JSON schema dictionary

        Returns:
            Modified schema with additionalProperties: false
        """
        if isinstance(schema, dict):
            # Set additionalProperties: false for objects
            if schema.get("type") == "object" or "properties" in schema:
                schema["additionalProperties"] = False

            # Recursively process nested schemas
            for key in ["properties", "items", "additionalProperties", "allOf", "anyOf", "oneOf"]:
                if key in schema:
                    if key == "properties" and isinstance(schema[key], dict):
                        for prop_schema in schema[key].values():
                            self._ensure_additional_properties_false(prop_schema)
                    elif key in ("allOf", "anyOf", "oneOf") and isinstance(schema[key], list):
                        for sub_schema in schema[key]:
                            self._ensure_additional_properties_false(sub_schema)
                    elif isinstance(schema[key], dict):
                        self._ensure_additional_properties_false(schema[key])

        return schema

    @retry(max_attempts=3, delay=1.0)
    async def astructured(
        self,
        messages: list[dict],
        response_model: type[T],
        temperature: float = 0.7,
    ) -> T:
        """Generate structured output using native JSON schema."""
        schema = response_model.model_json_schema()
        # OpenAI strict mode requires additionalProperties: false
        schema = self._ensure_additional_properties_false(schema)

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._effective_temperature(temperature),
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": schema,
                },
            },
        )
        content = response.choices[0].message.content or ""
        logger.debug(f"OpenAI structured response: {len(content)} chars")

        try:
            return response_model.model_validate_json(content)
        except ValidationError as e:
            msg = f"Validation failed: {e}"
            raise StructuredOutputError(msg, raw_response=content) from e

    @property
    def supports_structured_output(self) -> bool:
        """OpenAI supports structured output via JSON schema."""
        return True

    def __repr__(self) -> str:
        """String representation."""
        return f"OpenAIProvider(model={self._model})"
