"""Tests for notification tool."""

from unittest.mock import AsyncMock, Mock

import pytest

from src.tools.notification import NotificationTool
from src.v1.notifications.models import NotificationMessage, NotificationSeverity
from src.v1.notifications.service import NotificationService


@pytest.fixture
def mock_notification_service() -> NotificationService:
    """Create mock notification service."""
    service = Mock(spec=NotificationService)
    service.notify = AsyncMock(return_value=None)
    return service


@pytest.fixture
def notification_tool(mock_notification_service: NotificationService) -> NotificationTool:
    """Create notification tool with mock service."""
    return NotificationTool(mock_notification_service)


def test_notification_tool_name(notification_tool: NotificationTool) -> None:
    """Verify tool name is send_notification."""
    assert notification_tool.name == "send_notification"


def test_notification_tool_definition(notification_tool: NotificationTool) -> None:
    """Verify ToolDefinition schema correct."""
    definition = notification_tool.get_tool_definition()

    assert definition.type == "function"
    assert definition.function.name == "send_notification"
    assert "title" in definition.function.parameters.properties
    assert "message" in definition.function.parameters.properties
    assert "priority" in definition.function.parameters.properties
    assert "context" in definition.function.parameters.properties
    assert definition.function.parameters.required == ["title", "message"]

    # Verify priority enum
    priority_param = definition.function.parameters.properties["priority"]
    assert priority_param.enum == ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert priority_param.default == "MEDIUM"


def test_notification_tool_requires_no_confirmation(notification_tool: NotificationTool) -> None:
    """Verify requires_confirmation=False."""
    assert notification_tool.requires_confirmation is False


@pytest.mark.asyncio
async def test_notification_tool_execute_success(
    notification_tool: NotificationTool, mock_notification_service: NotificationService
) -> None:
    """Test successful notification execution."""
    result = await notification_tool.aexecute(
        title="Test Alert",
        message="Critical market event detected",
        priority="HIGH",
        context={"symbol": "AAPL", "stage": "analysis"},
    )

    assert "sent successfully" in result.lower()
    mock_notification_service.notify.assert_called_once()

    # Verify notify was called with message as single positional arg
    call_args = mock_notification_service.notify.call_args
    notification_msg = call_args[0][0]
    assert isinstance(notification_msg, NotificationMessage)
    assert notification_msg.title == "Test Alert"
    assert notification_msg.body == "Critical market event detected"
    assert notification_msg.severity == NotificationSeverity.ERROR
    assert notification_msg.metadata["symbol"] == "AAPL"
    assert notification_msg.metadata["priority"] == "HIGH"
    assert notification_msg.metadata["workflow_stage"] == "analysis"


@pytest.mark.asyncio
async def test_notification_tool_execute_failure(
    notification_tool: NotificationTool, mock_notification_service: NotificationService
) -> None:
    """Test graceful error handling on service failure."""
    mock_notification_service.notify.side_effect = Exception("Network error")

    result = await notification_tool.aexecute(
        title="Test Alert",
        message="Test message",
    )

    assert "queued" in result.lower() or "unavailable" in result.lower()
    mock_notification_service.notify.assert_called_once()


@pytest.mark.asyncio
async def test_notification_tool_validates_title_length(notification_tool: NotificationTool) -> None:
    """Test title length validation (max 100 chars)."""
    long_title = "x" * 101

    result = await notification_tool.aexecute(
        title=long_title,
        message="Test message",
    )

    assert "error" in result.lower()
    assert "title" in result.lower()
    assert "100" in result


@pytest.mark.asyncio
async def test_notification_tool_validates_message_length(notification_tool: NotificationTool) -> None:
    """Test message length validation (max 1000 chars)."""
    long_message = "x" * 1001

    result = await notification_tool.aexecute(
        title="Test",
        message=long_message,
    )

    assert "error" in result.lower()
    assert "message" in result.lower()
    assert "1000" in result


@pytest.mark.asyncio
async def test_notification_tool_priority_levels(
    notification_tool: NotificationTool, mock_notification_service: NotificationService
) -> None:
    """Test all priority levels (LOW/MEDIUM/HIGH/CRITICAL)."""
    priority_to_severity = {
        "LOW": NotificationSeverity.INFO,
        "MEDIUM": NotificationSeverity.WARNING,
        "HIGH": NotificationSeverity.ERROR,
        "CRITICAL": NotificationSeverity.CRITICAL,
    }

    for priority, expected_severity in priority_to_severity.items():
        mock_notification_service.notify.reset_mock()

        result = await notification_tool.aexecute(
            title="Test",
            message="Test message",
            priority=priority,
        )

        assert "sent successfully" in result.lower()
        call_args = mock_notification_service.notify.call_args
        notification_msg = call_args[0][0]
        assert notification_msg.severity == expected_severity
        assert notification_msg.metadata["priority"] == priority


@pytest.mark.asyncio
async def test_notification_tool_invalid_priority(notification_tool: NotificationTool) -> None:
    """Test invalid priority level rejected."""
    result = await notification_tool.aexecute(
        title="Test",
        message="Test message",
        priority="INVALID",
    )

    assert "error" in result.lower()
    assert "priority" in result.lower()


@pytest.mark.asyncio
async def test_notification_tool_metadata_capture(
    notification_tool: NotificationTool, mock_notification_service: NotificationService
) -> None:
    """Verify metadata structure captured correctly."""
    result = await notification_tool.aexecute(
        title="Test",
        message="Test message",
        priority="MEDIUM",
        context={"symbol": "TSLA", "stage": "decision"},
    )

    assert "sent successfully" in result.lower()

    call_args = mock_notification_service.notify.call_args
    notification_msg = call_args[0][0]

    assert notification_msg.metadata["symbol"] == "TSLA"
    assert notification_msg.metadata["priority"] == "MEDIUM"
    assert notification_msg.metadata["agent_type"] == "coordinator"
    assert notification_msg.metadata["workflow_stage"] == "decision"


@pytest.mark.asyncio
async def test_notification_tool_default_priority(
    notification_tool: NotificationTool, mock_notification_service: NotificationService
) -> None:
    """Test default priority is MEDIUM."""
    result = await notification_tool.aexecute(
        title="Test",
        message="Test message",
    )

    assert "sent successfully" in result.lower()

    call_args = mock_notification_service.notify.call_args
    notification_msg = call_args[0][0]
    assert notification_msg.severity == NotificationSeverity.WARNING
    assert notification_msg.metadata["priority"] == "MEDIUM"


@pytest.mark.asyncio
async def test_notification_tool_missing_context(
    notification_tool: NotificationTool, mock_notification_service: NotificationService
) -> None:
    """Test handling when context not provided."""
    result = await notification_tool.aexecute(
        title="Test",
        message="Test message",
    )

    assert "sent successfully" in result.lower()

    call_args = mock_notification_service.notify.call_args
    notification_msg = call_args[0][0]

    assert notification_msg.metadata["symbol"] == "N/A"
    assert notification_msg.metadata["workflow_stage"] == "unknown"


def test_notification_tool_sync_execute_raises(notification_tool: NotificationTool) -> None:
    """Test that sync execute raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        notification_tool.execute(title="Test", message="Test message")


def test_notification_tool_repr(notification_tool: NotificationTool) -> None:
    """Test string representation."""
    assert repr(notification_tool) == "NotificationTool()"


@pytest.mark.asyncio
async def test_notification_tool_validates_empty_title(notification_tool: NotificationTool) -> None:
    """Test empty title validation."""
    result = await notification_tool.aexecute(
        title="   ",
        message="Test message",
    )

    assert "error" in result.lower()
    assert "title" in result.lower()
    assert "empty" in result.lower()


@pytest.mark.asyncio
async def test_notification_tool_validates_empty_message(notification_tool: NotificationTool) -> None:
    """Test empty message validation."""
    result = await notification_tool.aexecute(
        title="Test Title",
        message="   ",
    )

    assert "error" in result.lower()
    assert "message" in result.lower()
    assert "empty" in result.lower()


@pytest.mark.asyncio
async def test_notification_tool_workflow_stage_fallback(
    notification_tool: NotificationTool, mock_notification_service: NotificationService
) -> None:
    """Test workflow_stage preferred over legacy stage key."""
    result = await notification_tool.aexecute(
        title="Test",
        message="Test message",
        context={"workflow_stage": "execution", "stage": "analysis"},
    )

    assert "sent successfully" in result.lower()
    call_args = mock_notification_service.notify.call_args
    notification_msg = call_args[0][0]
    assert notification_msg.metadata["workflow_stage"] == "execution"

    # Test with only stage key (legacy)
    mock_notification_service.notify.reset_mock()
    result = await notification_tool.aexecute(
        title="Test",
        message="Test message",
        context={"stage": "analysis"},
    )

    assert "sent successfully" in result.lower()
    call_args = mock_notification_service.notify.call_args
    notification_msg = call_args[0][0]
    assert notification_msg.metadata["workflow_stage"] == "analysis"
