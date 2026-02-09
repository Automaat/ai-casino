"""Tests for model providers."""

import os
from unittest.mock import MagicMock, patch

from src.daemon.config import DaemonConfig
from src.di.providers.models import create_finbert_sentiment, create_llm_client


def test_create_llm_client_default_config():
    """Test LLM client creation with default config."""
    config = DaemonConfig()

    with patch.dict(os.environ, {"LLM_PROVIDER": "ollama", "LLM_MODEL": "qwen3:14b"}):
        client = create_llm_client(config)

    assert client is not None
    assert client.provider == "ollama"
    assert client.model == "qwen3:14b"


def test_create_llm_client_config_priority_over_env():
    """Test config takes priority over env vars."""
    config = DaemonConfig()
    config.llm.provider = "anthropic"
    config.llm.model = "claude-sonnet-4"
    config.api_keys.anthropic_api_key = "test-key"

    with patch.dict(os.environ, {"LLM_PROVIDER": "ollama", "ANTHROPIC_API_KEY": "env-key"}):
        client = create_llm_client(config)

    assert client.provider == "anthropic"
    assert client.model == "claude-sonnet-4"


def test_create_llm_client_anthropic():
    """Test Anthropic provider API key resolution."""
    config = DaemonConfig()
    config.llm.provider = "anthropic"
    config.api_keys.anthropic_api_key = "config-key"

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-key"}):
        client = create_llm_client(config)

    # Config key takes priority
    assert client._api_key == "config-key"


def test_create_llm_client_openai():
    """Test OpenAI provider API key resolution."""
    config = DaemonConfig()
    config.llm.provider = "openai"
    config.api_keys.openai_api_key = None

    with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
        client = create_llm_client(config)

    # Falls back to env
    assert client._api_key == "env-key"


def test_create_llm_client_openai_base_url():
    """Test OpenAI base URL resolution."""
    config = DaemonConfig()
    config.llm.provider = "openai"
    config.api_keys.openai_api_base = "http://localhost:8080"
    config.api_keys.openai_api_key = "test-key"

    client = create_llm_client(config)

    assert client._openai_base_url == "http://localhost:8080"


def test_create_llm_client_with_metrics_collector():
    """Test metrics collector injection."""
    config = DaemonConfig()
    metrics_collector = MagicMock()

    client = create_llm_client(config, metrics_collector=metrics_collector)

    assert client._metrics_collector == metrics_collector


def test_create_finbert_sentiment_singleton():
    """Test FinBERT singleton behavior."""
    with patch("src.di.providers.models.get_finbert_sentiment") as mock_factory:
        mock_instance = MagicMock()
        mock_factory.return_value = mock_instance

        finbert1 = create_finbert_sentiment()
        finbert2 = create_finbert_sentiment()

        # Should call factory for both (factory handles singleton internally)
        assert finbert1 == mock_instance
        assert finbert2 == mock_instance
        assert mock_factory.call_count == 2


def test_create_finbert_sentiment_device():
    """Test device parameter passthrough."""
    with patch("src.di.providers.models.get_finbert_sentiment") as mock_factory:
        mock_instance = MagicMock()
        mock_factory.return_value = mock_instance

        create_finbert_sentiment(device="cuda")

        mock_factory.assert_called_once_with(device="cuda")
