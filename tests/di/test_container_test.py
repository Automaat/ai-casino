"""Tests for test container utilities."""

from tests.di.container_test import (
    create_mock_llm_client,
    create_test_config,
    create_test_container,
    reset_test_container,
)


def test_create_test_config():
    config = create_test_config()

    assert config.api_keys.alpha_vantage_api_key == "test_av_key"
    assert config.api_keys.marketaux_api_key == "test_marketaux_key"


def test_create_test_container_minimal(tmp_path):
    container = create_test_container(
        temp_cache_path=tmp_path / "test.db",
        override_llm=True,
        override_finbert=True,
        override_fetchers=False,
        override_broker=False,
    )

    assert container is not None

    # Verify LLM override works
    llm_client = container.llm_client()
    assert llm_client.provider == "ollama"

    reset_test_container(container)


def test_create_test_container_full(tmp_path):
    container = create_test_container(
        temp_cache_path=tmp_path / "test.db",
        override_llm=True,
        override_finbert=True,
        override_fetchers=True,
        override_broker=True,
    )

    assert container is not None

    # Verify all overrides work
    llm_client = container.llm_client()
    assert llm_client.provider == "ollama"

    finbert = container.finbert_sentiment()
    assert finbert.device == "cpu"

    market_fetcher = container.market_fetcher()
    assert market_fetcher is not None

    reset_test_container(container)


def test_test_container_fixture(test_container):
    """Test that test_container fixture works correctly."""
    assert test_container is not None

    # Verify worker can be created
    news_worker = test_container.news_worker()
    assert news_worker is not None


def test_test_container_full_fixture(test_container_full):
    """Test that test_container_full fixture works correctly."""
    assert test_container_full is not None

    # Verify workflow can be created
    workflow = test_container_full.workflow_momentum(container=test_container_full)
    assert workflow is not None


def test_mock_llm_client_creation():
    mock = create_mock_llm_client()

    assert mock.provider == "ollama"
    assert mock.model == "qwen3:14b"
    assert mock.supports_structured_output is True


async def test_mock_llm_client_acomplete():
    mock = create_mock_llm_client()

    result = await mock.acomplete("test prompt")
    assert "Mock LLM response" in result
    mock.acomplete.assert_called_once()


async def test_custom_override(test_container):
    """Test custom override pattern."""
    from unittest.mock import AsyncMock

    from dependency_injector import providers

    # Create custom LLM mock
    custom_llm = create_mock_llm_client()
    custom_llm.acomplete = AsyncMock(return_value="Custom response")

    # Override with Factory
    test_container.llm_client.override(providers.Factory(lambda: custom_llm))

    # Verify override works
    worker = test_container.news_worker()
    result = await worker.llm.acomplete("test")
    assert result == "Custom response"
