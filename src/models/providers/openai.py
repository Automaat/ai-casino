"""OpenAI LLM provider using official SDK."""

import copy
import json
from collections.abc import AsyncIterator
from typing import TypeVar

from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from src.metrics.execution import LLMUsageStats
from src.models.providers.base import BaseLLMProvider, StructuredOutputError, ToolCall
from src.models.providers.retry import retry

T = TypeVar("T", bound=BaseModel)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider using official SDK."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        enable_caching: bool = False,
        max_tokens: int = 2048,
    ) -> None:
        """Initialize OpenAI provider.

        Args:
            model: Model name (e.g., "gpt-4o")
            api_key: API key (defaults to OPENAI_API_KEY env var)
            base_url: Custom base URL (defaults to OPENAI_API_BASE env var)
            enable_caching: Enable cache metrics extraction from responses
            max_tokens: Default max output tokens (default: 2048)

        Raises:
            ValueError: If API key is not provided and OPENAI_API_KEY env var is empty
        """
        super().__init__(enable_caching=enable_caching)

        if not api_key:
            msg = "OpenAI API key required in config (api_keys.openai_api_key)"
            raise ValueError(msg)

        self._model = model
        self._max_tokens = max_tokens
        self._is_gpt5 = model.startswith("gpt-5")
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        logger.debug(f"Initialized OpenAIProvider: model={model}, max_tokens={max_tokens}")

    def _effective_temperature(self, temperature: float) -> float:
        """Get effective temperature, forcing 1.0 for GPT-5 models."""
        if self._is_gpt5 and temperature != 1.0:
            logger.debug(f"GPT-5 requires temperature=1, ignoring requested {temperature}")
            return 1.0
        return temperature

    def _extract_usage(self, usage: object) -> LLMUsageStats:
        """Extract usage stats with cache metrics from OpenAI/OpenRouter response.

        Args:
            usage: OpenAI usage object from response

        Returns:
            LLMUsageStats with cache_read_input_tokens from prompt_tokens_details
        """
        cached = None
        details = getattr(usage, "prompt_tokens_details", None)
        if details:
            cached = getattr(details, "cached_tokens", None)
        return LLMUsageStats(
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            cache_read_input_tokens=cached,
        )

    def _repair_json(self, json_str: str) -> str:
        """Attempt to repair common JSON errors from unreliable LLMs.

        Args:
            json_str: Potentially malformed JSON string

        Returns:
            Repaired JSON string (best-effort; the result may still be invalid JSON)
        """
        import re

        original = json_str

        # Fix 1: Remove duplicate keys (keep last occurrence)
        # Use JSONDecoder with object_pairs_hook to reliably handle all JSON structures
        try:

            def _keep_last_object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
                result: dict[str, object] = {}
                for k, v in pairs:
                    # Later occurrences overwrite earlier ones
                    result[k] = v
                return result

            decoder = json.JSONDecoder(object_pairs_hook=_keep_last_object_pairs)
            parsed = decoder.decode(json_str)
            json_str = json.dumps(parsed)
        except Exception as e:
            logger.debug(f"Duplicate key removal failed: {e}")

        # Fix 2: Double quotes before enum values (""VALUE" -> "VALUE")
        json_str = re.sub(r':\s*""([^"]+)"', r': "\1"', json_str)

        # Fix 3: Concatenated values without delimiters (valueCOMBINED -> value)
        # Look for patterns like "momentum|value|breakout" followed by uppercase word
        json_str = re.sub(
            r'"(momentum|value|breakout|SP500|NASDAQ100|COMBINED)([A-Z][A-Z]+)"', r'"\1"', json_str
        )

        # Fix 4: Clean up whitespace
        json_str = re.sub(r"\s+", " ", json_str)

        if json_str != original:
            logger.debug(f"JSON repair applied: {original!r} -> {json_str!r}")

        return json_str

    async def close(self) -> None:
        """Close HTTP client."""
        await self._client.close()

    @retry(max_attempts=3, delay=1.0)
    async def acomplete(
        self, messages: list[dict], temperature: float = 0.7, max_tokens: int | None = None
    ) -> str:
        """Generate completion from messages."""
        response = await self._client.chat.completions.create(  # type: ignore[arg-type]
            model=self._model,
            messages=messages,
            temperature=self._effective_temperature(temperature),
            max_tokens=max_tokens or self._max_tokens,
        )
        if response.usage:
            self._last_usage = self._extract_usage(response.usage)
        content = response.choices[0].message.content or ""
        logger.debug(f"OpenAI response length: {len(content)} chars")
        return content

    async def astream(
        self, messages: list[dict], temperature: float = 0.7, max_tokens: int | None = None
    ) -> AsyncIterator[str]:
        """Stream completion tokens."""
        stream = await self._client.chat.completions.create(  # type: ignore[arg-type]
            model=self._model,
            messages=messages,
            temperature=self._effective_temperature(temperature),
            max_tokens=max_tokens or self._max_tokens,
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
        max_tokens: int | None = None,
    ) -> tuple[str | None, list[ToolCall] | None]:
        """Generate completion with tool calling support."""
        response = await self._client.chat.completions.create(  # type: ignore[arg-type]
            model=self._model,
            messages=messages,
            tools=tools,
            temperature=self._effective_temperature(temperature),
            max_tokens=max_tokens or self._max_tokens,
        )
        if response.usage:
            self._last_usage = self._extract_usage(response.usage)
        message = response.choices[0].message

        if message.tool_calls:
            tool_calls: list[ToolCall] = []
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError as exc:
                    logger.error(
                        f"Failed to parse tool call arguments for '{tc.function.name}': {exc}. "
                        f"Raw arguments: {tc.function.arguments!r}"
                    )
                    # Try to repair common JSON errors from unreliable LLMs
                    try:
                        repaired = self._repair_json(tc.function.arguments)
                        arguments = json.loads(repaired)
                        logger.info(f"Successfully repaired malformed JSON for '{tc.function.name}'")
                    except (json.JSONDecodeError, ValueError) as repair_exc:
                        logger.opt(exception=True).error(
                            f"JSON repair failed for '{tc.function.name}': {repair_exc}. "
                            "Skipping this malformed tool call but continuing with others."
                        )
                        # Skip this malformed tool call but continue processing others
                        continue

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

    def _process_schema_properties(self, properties: dict, visited: set[int]) -> None:
        """Process properties dict recursively."""
        for prop_schema in properties.values():
            self._ensure_additional_properties_false(prop_schema, visited)

    def _process_schema_combinators(self, schemas: list, visited: set[int]) -> None:
        """Process allOf/anyOf/oneOf combinator schemas."""
        for sub_schema in schemas:
            self._ensure_additional_properties_false(sub_schema, visited)

    def _process_schema_items(self, items: dict | list, visited: set[int]) -> None:
        """Process items (array or tuple validation)."""
        if isinstance(items, dict):
            self._ensure_additional_properties_false(items, visited)
        elif isinstance(items, list):  # Tuple validation
            for item_schema in items:
                self._ensure_additional_properties_false(item_schema, visited)

    def _ensure_all_properties_required(self, schema: dict) -> dict:
        """Ensure all properties are in the required array (OpenAI strict mode requirement).

        Args:
            schema: JSON schema dictionary

        Returns:
            Modified schema with all properties in required array
        """
        # If schema has properties, ensure all are in required array
        if "properties" in schema and isinstance(schema["properties"], dict):
            all_props = list(schema["properties"].keys())
            if all_props:
                schema["required"] = all_props

        return schema

    def _ensure_additional_properties_false(self, schema: dict, visited: set[int] | None = None) -> dict:
        """Recursively ensure additionalProperties is false in schema (required by OpenAI strict mode).

        Args:
            schema: JSON schema dictionary
            visited: Set of visited schema IDs to detect circular references

        Returns:
            Modified schema with additionalProperties: false
        """
        if visited is None:
            visited = set()

        # Detect circular references
        schema_id = id(schema)
        if schema_id in visited:
            return schema
        visited.add(schema_id)

        if not isinstance(schema, dict):
            return schema

        # Set additionalProperties: false for objects
        if schema.get("type") == "object" or "properties" in schema:
            schema["additionalProperties"] = False
            # Also ensure all properties are required (OpenAI strict mode)
            self._ensure_all_properties_required(schema)

        # Process nested schemas
        if "properties" in schema and isinstance(schema["properties"], dict):
            self._process_schema_properties(schema["properties"], visited)

        if "items" in schema:
            self._process_schema_items(schema["items"], visited)

        for key in ("allOf", "anyOf", "oneOf"):
            if key in schema and isinstance(schema[key], list):
                self._process_schema_combinators(schema[key], visited)

        if "additionalProperties" in schema and isinstance(schema["additionalProperties"], dict):
            self._ensure_additional_properties_false(schema["additionalProperties"], visited)

        return schema

    @retry(max_attempts=3, delay=1.0)
    async def astructured(
        self,
        messages: list[dict],
        response_model: type[T],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> T:
        """Generate structured output using native JSON schema."""
        # Deep copy to avoid mutating cached schema
        schema = copy.deepcopy(response_model.model_json_schema())
        # OpenAI strict mode requires additionalProperties: false
        schema = self._ensure_additional_properties_false(schema)

        response = await self._client.chat.completions.create(  # type: ignore[arg-type]
            model=self._model,
            messages=messages,
            temperature=self._effective_temperature(temperature),
            max_tokens=max_tokens or self._max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": schema,
                },
            },
        )
        if response.usage:
            self._last_usage = self._extract_usage(response.usage)
        choice = response.choices[0]
        if choice.finish_reason == "length":
            content = choice.message.content or ""
            limit = max_tokens or self._max_tokens
            msg = f"Response truncated by max_tokens limit ({limit}): {len(content)} chars"
            raise StructuredOutputError(msg, raw_response=content)
        content = choice.message.content or ""
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
