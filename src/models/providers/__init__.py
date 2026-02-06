"""LLM provider implementations."""

from typing import TYPE_CHECKING

# Eager: lightweight base infrastructure
from src.models.providers.base import BaseLLMProvider, ToolCall

# Lazy: heavy SDK dependencies deferred until needed
if TYPE_CHECKING:
    from src.models.providers.anthropic import AnthropicProvider
    from src.models.providers.ollama import OllamaProvider
    from src.models.providers.openai import OpenAIProvider

__all__ = [
    "AnthropicProvider",
    "BaseLLMProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "ToolCall",
]


def __getattr__(name: str) -> type:
    """Lazy import provider classes to defer SDK imports until needed.

    Args:
        name: Attribute name to import

    Returns:
        The requested provider class

    Raises:
        AttributeError: If the requested attribute doesn't exist
    """
    if name == "AnthropicProvider":
        from src.models.providers.anthropic import AnthropicProvider

        return AnthropicProvider

    if name == "OpenAIProvider":
        from src.models.providers.openai import OpenAIProvider

        return OpenAIProvider

    if name == "OllamaProvider":
        from src.models.providers.ollama import OllamaProvider

        return OllamaProvider

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
