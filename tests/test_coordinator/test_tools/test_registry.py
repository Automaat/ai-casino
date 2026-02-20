"""Tests for build_coordinator_registry."""

from unittest.mock import MagicMock

import pytest

from src.coordinator.tools import build_coordinator_registry


@pytest.fixture
def mock_container():
    """Create mock DI container with minimal setup to avoid side effects."""
    container = MagicMock()
    daemon_config = MagicMock()
    daemon_config.coordinator.confirmation_mode = "auto"
    daemon_config.database.enable_persistence = False
    container.daemon_config.return_value = daemon_config
    return container


class TestBuildCoordinatorRegistry:
    """Tests for build_coordinator_registry."""

    @pytest.mark.unit
    def test_registers_shared_tools(self, mock_container: MagicMock) -> None:
        """Shared src/tools tools are registered in coordinator registry."""
        registry = build_coordinator_registry(mock_container)

        tool_names = registry.tool_names
        assert "web_search" in tool_names
        assert "get_news" in tool_names
        assert "get_risk_metrics" in tool_names
        assert "get_social_sentiment" in tool_names
        assert "analyze_trump_posts" in tool_names

    @pytest.mark.unit
    def test_registers_coordinator_tools(self, mock_container: MagicMock) -> None:
        """Coordinator-specific tools are registered."""
        registry = build_coordinator_registry(mock_container)

        tool_names = registry.tool_names
        assert "market_overview" in tool_names
        assert "analyze_symbol" in tool_names
        assert "portfolio_status" in tool_names
        assert "execute_trade" in tool_names

    @pytest.mark.unit
    def test_reflect_tool_absent_without_coordinator(self, mock_container: MagicMock) -> None:
        """Reflect tool not registered when coordinator is not provided."""
        registry = build_coordinator_registry(mock_container)

        assert "reflect_on_decision" not in registry.tool_names

    @pytest.mark.unit
    def test_reflect_tool_present_with_coordinator(self, mock_container: MagicMock) -> None:
        """Reflect tool registered when coordinator is provided."""
        coordinator = MagicMock()

        registry = build_coordinator_registry(mock_container, coordinator=coordinator)

        assert "reflect_on_decision" in registry.tool_names
