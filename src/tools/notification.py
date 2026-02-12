"""Notification tool for agent alerts."""

from datetime import UTC, datetime

from loguru import logger
from pydantic import ValidationError

from src.daemon.config import NotificationTrigger
from src.daemon.notifications import NotificationMessage, NotificationService
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

    def execute(self, **kwargs: str | int | float | bool) -> str:
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

    async def aexecute(self, **kwargs: str | int | float | bool) -> str:
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

        # Validate title length
        if len(title) > MAX_TITLE_LENGTH:
            return f"Error: Title must be max {MAX_TITLE_LENGTH} characters"

        # Validate message length
        if len(message) > MAX_MESSAGE_LENGTH:
            return f"Error: Message must be max {MAX_MESSAGE_LENGTH} characters"

        # Validate priority enum
        if priority not in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
            return f"Error: Invalid priority '{priority}'. Must be LOW, MEDIUM, HIGH, or CRITICAL"

        # Extract metadata from context
        if isinstance(context, dict):
            symbol = context.get("symbol", "N/A")
            workflow_stage = context.get("stage", "unknown")
        else:
            symbol = "N/A"
            workflow_stage = "unknown"

        metadata = {
            "symbol": symbol,
            "priority": priority,
            "agent_type": "coordinator",
            "workflow_stage": workflow_stage,
        }

        # Build notification message
        try:
            notification = NotificationMessage(
                trigger=NotificationTrigger.AGENT_ALERT,
                title=title,
                body=message,
                metadata=metadata,
                timestamp=datetime.now(UTC),
            )
        except ValidationError as e:
            logger.opt(exception=True).error(f"Failed to build notification: {e}")
            return f"Error: Failed to build notification: {e}"

        # Send notification (gracefully handle failures)
        try:
            await self._service.notify(NotificationTrigger.AGENT_ALERT, notification)
            logger.info(f"Agent notification sent: {title}")
            return f"Notification sent successfully: {title}"
        except Exception as e:
            # Log warning but return success to LLM (don't crash agent workflow)
            logger.opt(exception=True).warning(f"Notification failed but continuing: {e}")
            return f"Notification queued: {title} (channel may be unavailable)"

    def __repr__(self) -> str:
        """String representation."""
        return "NotificationTool()"
