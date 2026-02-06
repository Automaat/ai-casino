"""Base LLM provider interface."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TypeVar

from pydantic import BaseModel

from src.metrics.execution import LLMUsageStats

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


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    _last_usage: LLMUsageStats | None = None

    @property
    def last_usage(self) -> LLMUsageStats | None:
        """Get usage stats from the last API call."""
        return self._last_usage

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
