"""LLM abstraction using LiteLLM for flexible provider switching."""

import asyncio
import json
import os
from collections.abc import AsyncIterator, Callable

import litellm
import sniffio
from dotenv import load_dotenv
from litellm import acompletion, completion
from loguru import logger
from pydantic import BaseModel

load_dotenv()

# Disable aiohttp transport to avoid Python 3.14 asyncio.Timeout issues
# Use httpx instead which is compatible with worker thread context
litellm.disable_aiohttp_transport = True


def _set_asyncio_context() -> None:
    """Set sniffio context to asyncio to fix detection issues in Textual."""
    sniffio.current_async_library_cvar.set("asyncio")


# Default retry settings for transient errors
DEFAULT_NUM_RETRIES = 3
DEFAULT_TIMEOUT = 120

# Limit concurrent async requests (OpenAI allows 500 req/min = ~8 req/sec)
# Set conservatively below limit to allow headroom
MAX_CONCURRENT_REQUESTS = 6
_semaphore_holder: dict[str, asyncio.Semaphore] = {}


def _get_semaphore() -> asyncio.Semaphore:
    """Get or create the global request semaphore."""
    if "semaphore" not in _semaphore_holder:
        _semaphore_holder["semaphore"] = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    return _semaphore_holder["semaphore"]


class ToolCall(BaseModel):
    """Represents a tool call from the LLM."""

    id: str
    name: str
    arguments: dict


class ToolResult(BaseModel):
    """Result of executing a tool."""

    tool_call_id: str
    content: str


class LLMClient:
    """Unified LLM client supporting Ollama (dev) and API providers (prod)."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Initialize LLM client.

        Args:
            provider: LLM provider (ollama, anthropic, openai). Defaults to env.
            model: Model name. Defaults to env.
            base_url: Base URL for Ollama. Defaults to env.
        """
        self.provider = provider or os.getenv("LLM_PROVIDER", "ollama")
        self.model = model or os.getenv("LLM_MODEL", "qwen3:14b")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        self._api_base: str | None = None

        if self.provider == "ollama":
            os.environ["OLLAMA_API_BASE"] = self.base_url
            self._model_id = f"ollama/{self.model}"
        elif self.provider == "anthropic":
            self._model_id = f"anthropic/{self.model}"
        elif self.provider == "openai":
            self._model_id = f"openai/{self.model}"
            self._api_base = os.getenv("OPENAI_API_BASE")
        else:
            msg = f"Unsupported provider: {self.provider}"
            raise ValueError(msg)

        logger.info(f"Initialized LLM client: provider={self.provider}, model={self.model}")

    @property
    def _is_gpt5(self) -> bool:
        """Check if model is GPT-5 (temperature restricted)."""
        return self.provider == "openai" and self.model.startswith("gpt-5")

    def _effective_temperature(self, temperature: float) -> float:
        """Get effective temperature, forcing 1.0 for GPT-5 models."""
        if self._is_gpt5 and temperature != 1.0:
            logger.debug(f"GPT-5 requires temperature=1, ignoring requested {temperature}")
            return 1.0
        return temperature

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.7) -> str:
        """Generate completion from prompt.

        Args:
            prompt: User prompt
            system: System prompt (optional)
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            Generated text response
        """
        messages: list[dict[str, str]] = []

        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})

        try:
            kwargs: dict = {
                "model": self._model_id,
                "messages": messages,
                "temperature": self._effective_temperature(temperature),
                "num_retries": DEFAULT_NUM_RETRIES,
                "timeout": DEFAULT_TIMEOUT,
            }
            if self._api_base:
                kwargs["api_base"] = self._api_base
            response = completion(**kwargs)
            content = response.choices[0].message.content
            logger.debug(f"LLM response length: {len(content)} chars")
            return content
        except Exception as e:
            logger.error(f"LLM completion failed: {e}")
            raise

    async def acomplete(self, prompt: str, system: str | None = None, temperature: float = 0.7) -> str:
        """Generate completion from prompt asynchronously.

        Args:
            prompt: User prompt
            system: System prompt (optional)
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            Generated text response
        """
        messages: list[dict[str, str]] = []

        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})

        _set_asyncio_context()
        async with _get_semaphore():
            try:
                kwargs: dict = {
                    "model": self._model_id,
                    "messages": messages,
                    "temperature": self._effective_temperature(temperature),
                    "num_retries": DEFAULT_NUM_RETRIES,
                    "timeout": DEFAULT_TIMEOUT,
                }
                if self._api_base:
                    kwargs["api_base"] = self._api_base
                response = await acompletion(**kwargs)
                content = response.choices[0].message.content
                logger.debug(f"LLM async response length: {len(content)} chars")
                return content
            except Exception as e:
                logger.error(f"LLM async completion failed: {e}")
                logger.error(f"Exception type: {type(e).__name__}")
                logger.error(f"Exception details: {e!r}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.7) -> str:
        """Multi-turn chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            Generated text response
        """
        try:
            kwargs: dict = {
                "model": self._model_id,
                "messages": messages,
                "temperature": self._effective_temperature(temperature),
                "num_retries": DEFAULT_NUM_RETRIES,
                "timeout": DEFAULT_TIMEOUT,
            }
            if self._api_base:
                kwargs["api_base"] = self._api_base
            response = completion(**kwargs)
            content = response.choices[0].message.content
            logger.debug(f"LLM chat response length: {len(content)} chars")
            return content
        except Exception as e:
            logger.error(f"LLM chat failed: {e}")
            raise

    async def astream(
        self, prompt: str, system: str | None = None, temperature: float = 0.7
    ) -> AsyncIterator[str]:
        """Stream completion tokens asynchronously.

        Args:
            prompt: User prompt
            system: System prompt (optional)
            temperature: Sampling temperature (0.0-1.0)

        Yields:
            Individual tokens as they're generated
        """
        messages: list[dict[str, str]] = []

        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})

        _set_asyncio_context()
        async with _get_semaphore():
            try:
                kwargs: dict = {
                    "model": self._model_id,
                    "messages": messages,
                    "temperature": self._effective_temperature(temperature),
                    "stream": True,
                    "num_retries": DEFAULT_NUM_RETRIES,
                    "timeout": DEFAULT_TIMEOUT,
                }
                if self._api_base:
                    kwargs["api_base"] = self._api_base

                response = await acompletion(**kwargs)

                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content

            except Exception as e:
                logger.error(f"LLM streaming failed: {e}")
                raise

    @property
    def supports_tools(self) -> bool:
        """Check if current provider supports tool calling.

        Returns:
            True if provider supports tool calling (anthropic, openai)
        """
        return self.provider in ("anthropic", "openai")

    def complete_with_tools(  # noqa: PLR0913
        self,
        prompt: str,
        tools: list[dict],
        tool_executor: Callable[[str, dict], str],
        system: str | None = None,
        temperature: float = 0.7,
        max_tool_calls: int = 5,
        on_tool_call: Callable[[str, dict, str], None] | None = None,
    ) -> str:
        """Generate completion with tool calling support.

        Args:
            prompt: User prompt
            tools: List of tool definitions in OpenAI format
            tool_executor: Function to execute tools (name, args) -> result
            system: System prompt (optional)
            temperature: Sampling temperature (0.0-1.0)
            max_tool_calls: Maximum tool calls per completion
            on_tool_call: Callback invoked after each tool execution (name, args, result)

        Returns:
            Final text response after tool execution
        """
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        tool_calls_made = 0

        effective_temp = self._effective_temperature(temperature)
        try:
            while tool_calls_made < max_tool_calls:
                kwargs: dict = {
                    "model": self._model_id,
                    "messages": messages,
                    "temperature": effective_temp,
                    "tools": tools,
                    "num_retries": DEFAULT_NUM_RETRIES,
                    "timeout": DEFAULT_TIMEOUT,
                }
                if self._api_base:
                    kwargs["api_base"] = self._api_base

                response = completion(**kwargs)
                message = response.choices[0].message

                if not message.tool_calls:
                    return message.content or ""

                messages.append(message.model_dump())

                for tool_call in message.tool_calls:
                    tool_calls_made += 1
                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)

                    logger.debug(f"Executing tool: {name} with args: {args}")
                    try:
                        result = tool_executor(name, args)
                    except Exception as tool_error:
                        logger.error(f"Tool '{name}' execution failed: {tool_error}")
                        result = f"Tool '{name}' failed: {tool_error}"

                    if on_tool_call:
                        on_tool_call(name, args, result)

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        }
                    )

            kwargs = {
                "model": self._model_id,
                "messages": messages,
                "temperature": effective_temp,
                "num_retries": DEFAULT_NUM_RETRIES,
                "timeout": DEFAULT_TIMEOUT,
            }
            if self._api_base:
                kwargs["api_base"] = self._api_base
            final_response = completion(**kwargs)
            return final_response.choices[0].message.content or ""

        except Exception as e:
            logger.error(f"Tool calling failed: {e}")
            raise

    async def acomplete_with_tools(  # noqa: PLR0913
        self,
        prompt: str,
        tools: list[dict],
        tool_executor: Callable[[str, dict], str],
        system: str | None = None,
        temperature: float = 0.7,
        max_tool_calls: int = 5,
        on_tool_call: Callable[[str, dict, str], None] | None = None,
    ) -> str:
        """Generate completion with tool calling support (async).

        Args:
            prompt: User prompt
            tools: List of tool definitions in OpenAI format
            tool_executor: Function to execute tools (name, args) -> result
            system: System prompt (optional)
            temperature: Sampling temperature (0.0-1.0)
            max_tool_calls: Maximum tool calls per completion
            on_tool_call: Callback invoked after each tool execution (name, args, result)

        Returns:
            Final text response after tool execution
        """
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        tool_calls_made = 0
        effective_temp = self._effective_temperature(temperature)

        _set_asyncio_context()
        try:
            while tool_calls_made < max_tool_calls:
                kwargs: dict = {
                    "model": self._model_id,
                    "messages": messages,
                    "temperature": effective_temp,
                    "tools": tools,
                    "num_retries": DEFAULT_NUM_RETRIES,
                    "timeout": DEFAULT_TIMEOUT,
                }
                if self._api_base:
                    kwargs["api_base"] = self._api_base

                async with _get_semaphore():
                    response = await acompletion(**kwargs)
                message = response.choices[0].message

                if not message.tool_calls:
                    return message.content or ""

                messages.append(message.model_dump())

                for tool_call in message.tool_calls:
                    tool_calls_made += 1
                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)

                    logger.debug(f"Executing tool: {name} with args: {args}")
                    try:
                        result = tool_executor(name, args)
                    except Exception as tool_error:
                        logger.error(f"Tool '{name}' execution failed: {tool_error}")
                        result = f"Tool '{name}' failed: {tool_error}"

                    if on_tool_call:
                        on_tool_call(name, args, result)

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        }
                    )

            kwargs = {
                "model": self._model_id,
                "messages": messages,
                "temperature": effective_temp,
                "num_retries": DEFAULT_NUM_RETRIES,
                "timeout": DEFAULT_TIMEOUT,
            }
            if self._api_base:
                kwargs["api_base"] = self._api_base
            async with _get_semaphore():
                final_response = await acompletion(**kwargs)
            return final_response.choices[0].message.content or ""

        except Exception as e:
            logger.error(f"Async tool calling failed: {e}")
            raise

    def __repr__(self) -> str:
        """String representation."""
        return f"LLMClient(provider={self.provider}, model={self.model})"
