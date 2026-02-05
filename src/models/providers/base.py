"""Base LLM provider interface and retry decorator."""

import asyncio
import functools
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import ParamSpec, TypeVar

import httpx
from loguru import logger
from pydantic import BaseModel

P = ParamSpec("P")
T = TypeVar("T")


class ToolCall(BaseModel):
    """Represents a tool call from the LLM."""

    id: str
    name: str
    arguments: dict


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    exceptions: tuple[type[Exception], ...] = (httpx.ConnectError, httpx.TimeoutException),
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator for retrying async functions on transient errors.

    Args:
        max_attempts: Maximum retry attempts
        delay: Base delay between retries (exponential backoff)
        exceptions: Exception types to retry on
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_error: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        wait_time = delay * (attempt + 1)
                        logger.warning(f"Retry {attempt + 1}/{max_attempts} after {wait_time}s: {e}")
                        await asyncio.sleep(wait_time)
            raise last_error  # type: ignore[misc]

        return wrapper

    return decorator


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def acomplete(self, messages: list[dict], temperature: float = 0.7) -> str:
        """Generate completion from messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            Generated text response
        """

    @abstractmethod
    async def astream(self, messages: list[dict], temperature: float = 0.7) -> AsyncIterator[str]:
        """Stream completion tokens.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-1.0)

        Yields:
            Individual tokens as they're generated
        """

    @abstractmethod
    async def acomplete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float = 0.7,
    ) -> tuple[str | None, list[ToolCall] | None]:
        """Generate completion with tool calling support.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: List of tool definitions in OpenAI format
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            Tuple of (text_response, tool_calls). One will be None.
        """

    @property
    @abstractmethod
    def supports_tools(self) -> bool:
        """Check if provider supports tool calling."""
