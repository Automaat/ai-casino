"""Worker providers for DI container."""

from typing import TYPE_CHECKING

from src.models.llm import LLMClient

if TYPE_CHECKING:
    from src.workers.technical import TechnicalWorker


def create_technical_worker(llm_client: LLMClient) -> TechnicalWorker:
    """Create TechnicalWorker with LLM client.

    Args:
        llm_client: LLM client for generating interpretations

    Returns:
        Configured TechnicalWorker
    """
    from src.workers.technical import TechnicalWorker

    return TechnicalWorker(llm_client)
