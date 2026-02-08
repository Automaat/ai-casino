"""Ollama LLM provider using httpx.

Uses synchronous httpx wrapped in asyncio.to_thread to avoid anyio/httpcore
issues with Python 3.14's asyncio changes. The TypeError "cannot create weak
reference to 'NoneType' object" happens when httpcore's async connection pool
tries to use anyio's CancelScope during cleanup - sync httpx avoids this.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import TypeVar

import httpx
from loguru import logger
from pydantic import BaseModel, ValidationError

from src.metrics.execution import LLMUsageStats
from src.models.providers.base import BaseLLMProvider, StructuredOutputError, ToolCall
from src.models.providers.retry import retry

T = TypeVar("T", bound=BaseModel)


class OllamaProvider(BaseLLMProvider):
    """Ollama provider using direct HTTP API."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434") -> None:
        """Initialize Ollama provider.

        Args:
            model: Model name (e.g., "qwen3:14b")
            base_url: Ollama server URL
        """
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._client: httpx.Client | None = None
        logger.debug(f"Initialized OllamaProvider: model={model}, base_url={base_url}")

    def _get_client(self) -> httpx.Client:
        """Get or create httpx client for Ollama requests."""
        if self._client is None:
            self._client = httpx.Client(base_url=self._base_url, timeout=120.0)
        return self._client

    def _sync_complete(self, messages: list[dict], temperature: float) -> str:
        """Synchronous completion - runs in thread to avoid anyio issues."""
        client = self._get_client()
        response = client.post(
            "/api/chat",
            json={
                "model": self._model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature},
            },
        )
        response.raise_for_status()
        data = response.json()
        self._last_usage = LLMUsageStats(
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
        )
        content = data["message"]["content"]
        logger.debug(f"Ollama response length: {len(content)} chars")
        return content

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    @retry(max_attempts=3, delay=1.0)
    async def acomplete(self, messages: list[dict], temperature: float = 0.7) -> str:
        """Generate completion from messages."""
        return await asyncio.to_thread(self._sync_complete, messages, temperature)

    def _sync_stream(self, messages: list[dict], temperature: float) -> list[str]:
        """Synchronous streaming - collects all tokens then returns.

        For true streaming in async context, we'd need a more complex queue-based
        approach. This simplified version buffers all tokens for compatibility.
        """
        tokens = []
        client = self._get_client()
        with client.stream(
            "POST",
            "/api/chat",
            json={
                "model": self._model,
                "messages": messages,
                "stream": True,
                "options": {"temperature": temperature},
            },
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if content := data.get("message", {}).get("content"):
                        tokens.append(content)
        return tokens

    async def astream(self, messages: list[dict], temperature: float = 0.7) -> AsyncIterator[str]:
        """Stream completion tokens."""
        # Run sync streaming in thread, then yield tokens
        tokens = await asyncio.to_thread(self._sync_stream, messages, temperature)
        for token in tokens:
            yield token

    async def acomplete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float = 0.7,
    ) -> tuple[str | None, list[ToolCall] | None]:
        """Tool calling not supported by Ollama."""
        msg = "Ollama does not support tool calling"
        raise NotImplementedError(msg)

    @property
    def supports_tools(self) -> bool:
        """Ollama does not support tool calling."""
        return False

    def _sync_structured(
        self,
        messages: list[dict],
        response_model: type[T],
        temperature: float,
    ) -> T:
        """Synchronous structured output with retry on validation failure."""
        schema = response_model.model_json_schema()
        schema_prompt = f"\n\nRespond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"

        augmented_messages = messages.copy()
        if augmented_messages and augmented_messages[-1]["role"] == "user":
            augmented_messages[-1] = {
                "role": "user",
                "content": augmented_messages[-1]["content"] + schema_prompt,
            }
        else:
            augmented_messages.append({"role": "user", "content": schema_prompt})

        client = self._get_client()
        last_error: Exception | None = None
        last_response: str = ""

        for attempt in range(2):  # 1 initial + 1 retry
            response = client.post(
                "/api/chat",
                json={
                    "model": self._model,
                    "messages": augmented_messages,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": temperature},
                },
            )
            response.raise_for_status()
            resp_data = response.json()
            self._last_usage = LLMUsageStats(
                input_tokens=resp_data.get("prompt_eval_count"),
                output_tokens=resp_data.get("eval_count"),
            )
            content = resp_data["message"]["content"]
            last_response = content
            logger.debug(f"Ollama structured response (attempt {attempt + 1}): {len(content)} chars")

            try:
                data = json.loads(content)
                return response_model.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as e:
                last_error = e
                if attempt == 0:
                    logger.warning(f"Structured output validation failed, retrying: {e}")
                    error_feedback = f"\n\nPrevious response was invalid: {e}\nPlease provide valid JSON."
                    augmented_messages[-1] = {
                        "role": "user",
                        "content": augmented_messages[-1]["content"] + error_feedback,
                    }

        msg = f"Validation failed after 2 attempts: {last_error}"
        raise StructuredOutputError(msg, raw_response=last_response)

    @retry(max_attempts=3, delay=1.0)
    async def astructured(
        self,
        messages: list[dict],
        response_model: type[T],
        temperature: float = 0.7,
    ) -> T:
        """Generate structured output using JSON mode with schema in prompt."""
        return await asyncio.to_thread(self._sync_structured, messages, response_model, temperature)

    @property
    def supports_structured_output(self) -> bool:
        """Ollama supports structured output via JSON mode."""
        return True

    def __repr__(self) -> str:
        """String representation."""
        return f"OllamaProvider(model={self._model})"
