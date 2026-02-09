"""Test DI container infrastructure."""

from pathlib import Path

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


def test_llm_client_singleton():
    """Test LLM client is singleton."""
    container = create_container()

    client1 = container.llm_client()
    client2 = container.llm_client()

    assert client1 is client2


def test_finbert_singleton():
    """Test FinBERT is singleton."""
    container = create_container()

    # Note: finbert uses internal singleton, container provides factory wrapper
    # Each call to container.finbert_sentiment() calls factory which returns same instance
    finbert1 = container.finbert_sentiment()
    finbert2 = container.finbert_sentiment()

    assert finbert1 is finbert2
