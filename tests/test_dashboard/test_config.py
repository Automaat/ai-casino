"""Tests for dashboard configuration."""

import pytest
from pydantic import ValidationError

from src.dashboard.config import DashboardConfig


def test_dashboard_config_defaults() -> None:
    """Test default configuration values."""
    config = DashboardConfig()

    assert config.api_url == "http://localhost:8001"
    assert config.refresh_interval == 5000
    assert config.port == 8050
    assert config.host == "127.0.0.1"


def test_dashboard_config_custom_values() -> None:
    """Test custom configuration values."""
    config = DashboardConfig(
        api_url="http://example.com:9000",
        refresh_interval=10000,
        port=9050,
        host="0.0.0.0",  # noqa: S104
    )

    assert config.api_url == "http://example.com:9000"
    assert config.refresh_interval == 10000
    assert config.port == 9050
    assert config.host == "0.0.0.0"  # noqa: S104


def test_dashboard_config_refresh_interval_validation() -> None:
    """Test refresh_interval must be 1000-60000."""
    with pytest.raises(ValidationError):
        DashboardConfig(refresh_interval=500)

    with pytest.raises(ValidationError):
        DashboardConfig(refresh_interval=70000)

    # Valid boundaries
    config_min = DashboardConfig(refresh_interval=1000)
    assert config_min.refresh_interval == 1000

    config_max = DashboardConfig(refresh_interval=60000)
    assert config_max.refresh_interval == 60000


def test_dashboard_config_port_validation() -> None:
    """Test port must be 1-65535."""
    with pytest.raises(ValidationError):
        DashboardConfig(port=0)

    with pytest.raises(ValidationError):
        DashboardConfig(port=70000)

    # Valid boundaries
    config_min = DashboardConfig(port=1)
    assert config_min.port == 1

    config_max = DashboardConfig(port=65535)
    assert config_max.port == 65535


def test_dashboard_config_repr() -> None:
    """Test __repr__ output."""
    config = DashboardConfig()
    repr_str = repr(config)

    assert "DashboardConfig" in repr_str
    assert "http://localhost:8001" in repr_str
    assert "8050" in repr_str
    assert "5000ms" in repr_str
