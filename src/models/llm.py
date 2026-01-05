"""LLM abstraction using LiteLLM for flexible provider switching."""

import os

from dotenv import load_dotenv
from litellm import completion
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

        if self.provider == "ollama":
            os.environ["OLLAMA_API_BASE"] = self.base_url
            self._model_id = f"ollama/{self.model}"
        elif self.provider == "anthropic":
            self._model_id = f"anthropic/{self.model}"
        elif self.provider == "openai":
            self._model_id = f"openai/{self.model}"
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
            response = completion(
                model=self._model_id,
                messages=messages,
                temperature=temperature,
            )
            content = response.choices[0].message.content
            logger.debug(f"LLM response length: {len(content)} chars")
            return content
        except Exception as e:
            logger.error(f"LLM completion failed: {e}")
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
            response = completion(
                model=self._model_id,
                messages=messages,
                temperature=temperature,
            )
            content = response.choices[0].message.content
            logger.debug(f"LLM chat response length: {len(content)} chars")
            return content
        except Exception as e:
            logger.error(f"LLM chat failed: {e}")
            raise

    def __repr__(self) -> str:
        """String representation."""
        return f"LLMClient(provider={self.provider}, model={self.model})"
