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
    """None when no path."""
    assert load_daemon_config(None) is None


def test_load_daemon_config_valid(tmp_path):
    """Load valid config."""
    config_path = tmp_path / "daemon.yaml"
    config_path.write_text("daemon:\n  watchlist: [MSFT]")

    config = load_daemon_config(config_path)
    assert config is not None
    assert config.watchlist == ["MSFT"]
