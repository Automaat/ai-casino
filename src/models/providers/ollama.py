"""Ollama LLM provider using httpx.

Uses synchronous httpx wrapped in asyncio.to_thread to avoid anyio/httpcore
issues with Python 3.14's asyncio changes. The TypeError "cannot create weak
reference to 'NoneType' object" happens when httpcore's async connection pool
tries to use anyio's CancelScope during cleanup - sync httpx avoids this.
"""

import asyncio
import json
import threading
from collections.abc import AsyncIterator

import httpx
from loguru import logger

from src.models.providers.base import BaseLLMProvider, ToolCall, retry

_CLIENT: httpx.Client | None = None
_CLIENT_LOCK = threading.Lock()


def _get_client(base_url: str, timeout: float = 120.0) -> httpx.Client:
    """Get or create shared httpx client for Ollama requests."""
    global _CLIENT
    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:
                _CLIENT = httpx.Client(base_url=base_url, timeout=timeout)
    return _CLIENT


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
        logger.debug(f"Initialized OllamaProvider: model={model}, base_url={base_url}")

    def _sync_complete(self, messages: list[dict], temperature: float) -> str:
        """Synchronous completion - runs in thread to avoid anyio issues."""
        client = _get_client(self._base_url)
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
        """Close shared HTTP client."""
        global _CLIENT
        if _CLIENT is not None:
            with _CLIENT_LOCK:
                if _CLIENT is not None:
                    _CLIENT.close()
                    _CLIENT = None

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
        client = _get_client(self._base_url)
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
