"""Test DI container infrastructure."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.di import create_container, load_daemon_config


def test_create_container_without_config():
    """Create container without config path."""
    container = create_container()
    assert container is not None
    assert hasattr(container, "config")
    assert hasattr(container, "daemon_config")


def test_daemon_config_returns_defaults_without_yaml():
    """Verify daemon_config() returns defaults when no config_path set."""
    container = create_container()

    # Should return DaemonConfig() defaults, not raise
    config = container.daemon_config()

    assert config is not None
    assert config.watchlist == ["AAPL", "TSLA", "GOOGL", "MSFT"]  # Default watchlist


def test_create_container_with_config(tmp_path):
    """Create container with valid config."""
    config_path = tmp_path / "daemon.yaml"
    config_path.write_text("""
daemon:
  watchlist: [AAPL, TSLA]
  interval_minutes: 5
  schedule:
    start_time: "09:30"
""")

    container = create_container(config_path)
    config = container.daemon_config()

    assert config.watchlist == ["AAPL", "TSLA"]
    assert config.interval_minutes == 5


def test_daemon_config_singleton(tmp_path):
    """DaemonConfig singleton - same instance."""
    config_path = tmp_path / "daemon.yaml"
    config_path.write_text("daemon:\n  watchlist: [AAPL]")

    container = create_container(config_path)
    config1 = container.daemon_config()
    config2 = container.daemon_config()

    assert config1 is config2


def test_load_daemon_config_nonexistent():
    """Error on nonexistent file."""
    with pytest.raises(FileNotFoundError):
        load_daemon_config(Path("missing.yaml"))


def test_load_daemon_config_none():
    """Returns defaults when no path."""
    config = load_daemon_config(None)
    assert config is not None
    assert config.watchlist == ["AAPL", "TSLA", "GOOGL", "MSFT"]  # Default watchlist


def test_load_daemon_config_valid(tmp_path):
    """Load valid config."""
    config_path = tmp_path / "daemon.yaml"
    config_path.write_text("daemon:\n  watchlist: [MSFT]")

    config = load_daemon_config(config_path)
    assert config is not None
    assert config.watchlist == ["MSFT"]


def test_llm_client_provider():
    """Test LLM client provider is accessible."""
    container = create_container()
    assert hasattr(container, "llm_client")

    # Should be callable
    client = container.llm_client()
    assert client is not None


def test_finbert_sentiment_provider():
    """Test FinBERT sentiment provider is accessible."""
    container = create_container()
    assert hasattr(container, "finbert_sentiment")

    # Should be callable (may take time on first call to download model)
    # We skip actual call to avoid downloading model in tests


def test_llm_client_factory():
    """Test LLM client is factory (new instance per request)."""
    container = create_container()

    client1 = container.llm_client()
    client2 = container.llm_client()

    # Factory: each call creates new instance
    assert client1 is not client2


def test_finbert_singleton():
    """Test FinBERT singleton behavior."""
    container = create_container()

    # Mock the underlying factory to avoid loading real 440MB model in CI
    with patch("src.models.sentiment.get_finbert_sentiment") as mock_factory:
        mock_instance = MagicMock()
        mock_factory.return_value = mock_instance

        finbert1 = container.finbert_sentiment()
        finbert2 = container.finbert_sentiment()

        # Singleton: same instance returned (factory called once, cached)
        assert finbert1 is finbert2
        assert finbert1 is mock_instance
        # Singleton provider calls factory once, caches result
        assert mock_factory.call_count == 1


def test_workflow_meta_provider():
    """Test workflow_meta provider instantiation."""
    container = create_container()

    # Mock FinBERT to avoid 440MB model download in CI
    with patch("src.models.sentiment.get_finbert_sentiment") as mock_factory:
        mock_factory.return_value = MagicMock()
        workflow = container.workflow_meta()

        assert workflow.use_meta_agent is True
        assert workflow.trump_mode is False
        # meta_agent is constructed in __init__ when use_meta_agent=True


def test_workflow_trump_provider():
    """Test workflow_trump provider instantiation."""
    container = create_container()

    # Mock FinBERT to avoid 440MB model download in CI
    with patch("src.models.sentiment.get_finbert_sentiment") as mock_factory:
        mock_factory.return_value = MagicMock()
        workflow = container.workflow_trump()

        assert workflow.use_meta_agent is True
        assert workflow.trump_mode is True
        # Trump analyst instantiated in __init__
        assert workflow.trump_analyst is not None


def test_workflow_momentum_provider():
    """Test workflow_momentum provider instantiation."""
    from src.strategies.momentum import MomentumStrategy

    container = create_container()

    # Mock FinBERT to avoid 440MB model download in CI
    with patch("src.models.sentiment.get_finbert_sentiment") as mock_factory:
        mock_factory.return_value = MagicMock()
        workflow = container.workflow_momentum()

        assert workflow.use_meta_agent is False
        assert workflow.trump_mode is False
        assert isinstance(workflow._default_strategy, MomentumStrategy)


def test_workflow_with_overrides():
    """Test workflow provider with runner-managed overrides."""
    from unittest.mock import MagicMock

    container = create_container()

    mock_broker = MagicMock()
    mock_metrics_tracker = MagicMock()

    # Mock FinBERT to avoid 440MB model download in CI
    with patch("src.models.sentiment.get_finbert_sentiment") as mock_factory:
        mock_factory.return_value = MagicMock()

        workflow = container.workflow_meta(
            broker=mock_broker,
            metrics_tracker=mock_metrics_tracker,
        )

        assert workflow.broker is mock_broker
        assert workflow.metrics_tracker is mock_metrics_tracker
