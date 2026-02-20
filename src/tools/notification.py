"""Notification tool for agent alerts."""

from datetime import UTC, datetime

from loguru import logger
from pydantic import ValidationError

from src.v1.notifications.models import NotificationMessage, NotificationSeverity
from src.v1.notifications.service import NotificationService
from src.tools.base import BaseTool
from src.tools.models import ToolDefinition, ToolFunction, ToolParameter, ToolParametersSchema

# Validation constants
MAX_TITLE_LENGTH = 100
MAX_MESSAGE_LENGTH = 1000


class NotificationTool(BaseTool):
    """Tool for agents to send notifications to user."""

    def __init__(self, notification_service: NotificationService) -> None:
        """Initialize tool with notification service.

        Args:
            notification_service: Notification service for sending alerts
        """
        self._service = notification_service

    @property
    def name(self) -> str:
        """Tool name."""
        return "send_notification"

    @property
    def requires_confirmation(self) -> bool:
        """Agents trusted to notify without confirmation."""
        return False

    def get_tool_definition(self) -> ToolDefinition:
        """Get tool definition in LiteLLM/OpenAI format.

        Returns:
            Tool definition for LLM function calling
        """
        return ToolDefinition(
            function=ToolFunction(
                name=self.name,
                description=(
                    "Send notification to user via configured channels (Telegram). Use when you "
                    "think something is important or interesting for the user - market insights, "
                    "anomalies, opportunities, or actionable information."
                ),
                parameters=ToolParametersSchema(
                    properties={
                        "title": ToolParameter(
                            type="string",
                            description="Brief notification title (max 100 chars)",
                        ),
                        "message": ToolParameter(
                            type="string",
                            description="Detailed notification message (max 1000 chars)",
                        ),
                        "priority": ToolParameter(
                            type="string",
                            description="Priority level",
                            enum=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                            default="MEDIUM",
                        ),
                        "context": ToolParameter(
                            type="object",
                            description="Additional context (symbol, workflow_stage, etc.)",
                        ),
                    },
                    required=["title", "message"],
                ),
            ),
        )

    def execute(self, **kwargs: str | int | float | bool | dict | object) -> str:
        """Execute notification synchronously (not supported).

        Args:
            **kwargs: Tool arguments

        Returns:
            Error message

        Raises:
            NotImplementedError: This tool requires async execution
        """
        msg = "NotificationTool requires async execution via aexecute()"
        raise NotImplementedError(msg)

    async def aexecute(self, **kwargs: str | int | float | bool | dict | object) -> str:
        """Execute notification tool asynchronously.

        Args:
            **kwargs: Tool arguments (title, message, priority, context)

        Returns:
            Success or error message
        """
        # Extract and validate inputs
        title = str(kwargs.get("title", ""))
        message = str(kwargs.get("message", ""))
        priority = str(kwargs.get("priority", "MEDIUM")).upper()
        context = kwargs.get("context", {})

        # Validate inputs (combined to reduce return statements)
        validation_error = self._validate_inputs(title, message, priority)
        if validation_error:
            return validation_error

        # Extract metadata from context
        if isinstance(context, dict):
            symbol = context.get("symbol", "N/A")
            # Prefer explicit workflow_stage, fall back to legacy stage key, then unknown
            workflow_stage = context.get("workflow_stage", context.get("stage", "unknown"))
        else:
            symbol = "N/A"
            workflow_stage = "unknown"

        metadata: dict[str, str | int | float | bool] = {
            "symbol": str(symbol),
            "priority": priority,
            "agent_type": "coordinator",
            "workflow_stage": str(workflow_stage),
        }

        # Build notification message
        try:
            notification = NotificationMessage(
                title=title,
                body=message,
                severity=NotificationSeverity.from_priority(priority),
                metadata=metadata,
                timestamp=datetime.now(UTC),
            )
        except ValidationError as e:
            logger.opt(exception=True).error(f"Failed to build notification: {e}")
            return f"Error: Failed to build notification: {e}"

        # Send notification (gracefully handle failures)
        try:
            await self._service.notify(notification)
            logger.info(f"Agent notification sent: {title}")
            return f"Notification sent successfully: {title}"
        except Exception as e:
            # Log warning but return success to LLM (don't crash agent workflow)
            logger.opt(exception=True).warning(f"Notification failed but continuing: {e}")
            return f"Notification queued: {title} (channel may be unavailable)"

    def _validate_inputs(self, title: str, message: str, priority: str) -> str | None:
        """Validate tool inputs.

        Args:
            title: Notification title
            message: Notification message
            priority: Priority level

        Returns:
            Error message if validation fails, None otherwise
        """
        if not title.strip():
            return "Error: Title cannot be empty"
        if not message.strip():
            return "Error: Message cannot be empty"
        if len(title) > MAX_TITLE_LENGTH:
            return f"Error: Title must be max {MAX_TITLE_LENGTH} characters"
        if len(message) > MAX_MESSAGE_LENGTH:
            return f"Error: Message must be max {MAX_MESSAGE_LENGTH} characters"
        if priority not in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
            return f"Error: Invalid priority '{priority}'. Must be LOW, MEDIUM, HIGH, or CRITICAL"
        return None

    def __repr__(self) -> str:
        """String representation."""
        return "NotificationTool()"
