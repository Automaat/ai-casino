"""LLM abstraction using LiteLLM for flexible provider switching."""

import os
from collections.abc import AsyncIterator

from dotenv import load_dotenv
from litellm import acompletion, completion
from loguru import logger

load_dotenv()


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
                "temperature": temperature,
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

        try:
            kwargs: dict = {
                "model": self._model_id,
                "messages": messages,
                "temperature": temperature,
            }
            if self._api_base:
                kwargs["api_base"] = self._api_base
            response = await acompletion(**kwargs)
            content = response.choices[0].message.content
            logger.debug(f"LLM async response length: {len(content)} chars")
            return content
        except Exception as e:
            logger.error(f"LLM async completion failed: {e}")
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
                "temperature": temperature,
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

        try:
            kwargs: dict = {
                "model": self._model_id,
                "messages": messages,
                "temperature": temperature,
                "stream": True,
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

    def __repr__(self) -> str:
        """String representation."""
        return f"LLMClient(provider={self.provider}, model={self.model})"
