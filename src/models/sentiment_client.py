"""HTTP client for FinBERT microservice (backward-compatible with FinBERTSentiment)."""

import asyncio

import httpx
from loguru import logger

from src.models.sentiment import FinBERTProtocol, SentimentScore

__all__ = ["FinBERTClient", "SentimentScore"]


class FinBERTClient(FinBERTProtocol):
    """HTTP client for FinBERT microservice (backward-compatible interface)."""

    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        """Initialize FinBERT HTTP client.

        Args:
            base_url: Service base URL (e.g., http://localhost:8485)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.device = "remote"
        self._client = httpx.Client(timeout=timeout, base_url=self.base_url)
        logger.info(f"Initialized FinBERTClient (url={base_url}, timeout={timeout}s)")

    def analyze_batch(self, texts: list[str]) -> list[SentimentScore]:
        """Analyze sentiment of multiple texts (synchronous).

        Args:
            texts: List of financial texts to analyze

        Returns:
            List of SentimentScore objects

        Raises:
            httpx.HTTPError: If request fails
        """
        if not texts:
            return []

        try:
            response = self._client.post("/analyze", json={"texts": texts})
            response.raise_for_status()
            data = response.json()
            return [SentimentScore(**s) for s in data["scores"]]
        except httpx.HTTPError as e:
            logger.error(f"FinBERT service request failed: {e}")
            raise

    async def analyze_batch_async(self, texts: list[str]) -> list[SentimentScore]:
        """Analyze sentiment asynchronously using thread offload (Python 3.14 compat).

        Args:
            texts: List of financial texts to analyze

        Returns:
            List of SentimentScore objects
        """
        return await asyncio.to_thread(self.analyze_batch, texts)

    def close(self) -> None:
        """Explicitly close HTTP client."""
        self._client.close()
        logger.debug("FinBERTClient closed")

    def __del__(self) -> None:
        """Cleanup on garbage collection."""
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        return f"FinBERTClient(base_url={self.base_url})"
