"""Model providers for DI container."""

import os
from typing import TYPE_CHECKING

from src.daemon.config import DaemonConfig
from src.di.config import resolve_config_or_env
from src.models.llm import LLMClient

if TYPE_CHECKING:
    from src.metrics.execution import ExecutionMetricsCollector
    from src.models.sentiment import FinBERTSentiment


def create_llm_client(
    daemon_config: DaemonConfig,
    metrics_collector: "ExecutionMetricsCollector | None" = None,
) -> LLMClient:
    """Create LLMClient with resolved config.

    Resolves provider/model from daemon_config.llm.* with env fallbacks.
    API key resolution by provider type (anthropic/openai/ollama).

    Args:
        daemon_config: Daemon configuration
        metrics_collector: Optional metrics collector for instrumentation

    Returns:
        Configured LLMClient
    """
    provider = daemon_config.llm.provider or os.getenv("LLM_PROVIDER", "ollama")

    if provider == "anthropic":
        api_key = resolve_config_or_env(
            daemon_config.api_keys.anthropic_api_key,
            "ANTHROPIC_API_KEY",
        )
    elif provider == "openai":
        api_key = resolve_config_or_env(
            daemon_config.api_keys.openai_api_key,
            "OPENAI_API_KEY",
        )
    else:
        api_key = None

    llm_client = LLMClient(
        provider=daemon_config.llm.provider,
        model=daemon_config.llm.model,
        api_key=api_key,
        openai_base_url=resolve_config_or_env(
            daemon_config.api_keys.openai_api_base,
            "OPENAI_API_BASE",
        ),
    )

    if metrics_collector is not None:
        llm_client.set_metrics_collector(metrics_collector)

    return llm_client


def create_finbert_sentiment(device: str | None = None) -> "FinBERTSentiment":
    """Create FinBERT sentiment analyzer with lazy import.

    Thin wrapper over existing get_finbert_sentiment() factory.
    Maintains singleton behavior via existing implementation.
    Uses lazy import to avoid loading 440MB model on container creation.

    Args:
        device: Device for inference (cuda/cpu). Auto-detect if None.

    Returns:
        FinBERTSentiment singleton instance
    """
    from src.models.sentiment import get_finbert_sentiment

    return get_finbert_sentiment(device=device)
