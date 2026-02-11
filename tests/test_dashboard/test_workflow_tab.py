"""Tests for workflow tab."""

from unittest.mock import MagicMock

import pytest

from src.dashboard.tabs import workflow


@pytest.fixture
def mock_client():
    """Mock API client."""
    client = MagicMock()
    client.get_events.return_value = MagicMock(events=[])
    client.get_execution_metrics.return_value = MagicMock(metrics=[], count=0)
    return client


def test_render(mock_client):
    """Test render returns valid layout."""
    result = workflow.render(mock_client)

    assert isinstance(result, list)
    assert len(result) > 0


def test_register_callbacks():
    """Test register_callbacks runs without error."""
    app = MagicMock()
    app.api_client = MagicMock()
    app.api_client.get_events.return_value = MagicMock(events=[])
    app.api_client.get_execution_metrics.return_value = MagicMock(metrics=[], count=0)

    workflow.register_callbacks(app)

    assert app.callback.call_count > 0
