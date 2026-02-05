"""Ollama LLM provider using httpx.

Uses synchronous httpx wrapped in asyncio.to_thread to avoid anyio/httpcore
issues with Python 3.14's asyncio changes. The TypeError "cannot create weak
reference to 'NoneType' object" happens when httpcore's async connection pool
tries to use anyio's CancelScope during cleanup - sync httpx avoids this.
"""

import asyncio
import json
from collections.abc import AsyncIterator

import httpx
from loguru import logger

from src.models.providers.base import BaseLLMProvider, ToolCall, retry


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

    @retry(max_attempts=3, delay=1.0)
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

    def __repr__(self) -> str:
        """String representation."""
        return f"OllamaProvider(model={self._model})"
