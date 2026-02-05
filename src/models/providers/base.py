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


class StructuredOutputError(Exception):
    """Error during structured output parsing or validation."""

    def __init__(self, message: str, raw_response: str | None = None) -> None:
        """Initialize error with message and optional raw response.

        Args:
            message: Error description
            raw_response: Original LLM response that failed parsing/validation
        """
        super().__init__(message)
        self.raw_response = raw_response


class ToolCall(BaseModel):
    """Represents a tool call from the LLM."""

    id: str
    name: str
    arguments: dict


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    exceptions: tuple[type[Exception], ...] = (
        httpx.ConnectError,
        httpx.TimeoutException,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.PoolTimeout,
        httpx.HTTPStatusError,
    ),
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
            if last_error is not None:
                raise last_error
            msg = "Retry decorator invoked with max_attempts <= 0 or no exceptions captured"
            raise RuntimeError(msg)

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

    @abstractmethod
    async def close(self) -> None:
        """Release resources held by provider.

        Implementations without resources can provide empty implementation.
        """

    @property
    @abstractmethod
    def supports_tools(self) -> bool:
        """Check if provider supports tool calling."""

    @abstractmethod
    async def astructured(
        self,
        messages: list[dict],
        response_model: type[T],
        temperature: float = 0.7,
    ) -> T:
        """Generate structured output validated against a Pydantic model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            response_model: Pydantic model class to validate response against
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            Validated instance of response_model

        Raises:
            StructuredOutputError: If response cannot be parsed or validated
        """

    @property
    @abstractmethod
    def supports_structured_output(self) -> bool:
        """Check if provider supports structured output."""
