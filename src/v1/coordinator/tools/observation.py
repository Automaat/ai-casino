"""Save observation tool for coordinator."""

from typing import TYPE_CHECKING, ClassVar, Final

from loguru import logger

from src.tools.base import BaseTool
from src.tools.models import ToolDefinition, ToolFunction, ToolParameter, ToolParametersSchema

if TYPE_CHECKING:
    from src.v1.coordinator.memory import CoordinatorMemory

MIN_OBSERVATION_LENGTH: Final[int] = 10


class SaveObservationTool(BaseTool):
    """Tool to save coordinator learning observations."""

    VALID_CATEGORIES: ClassVar[set[str]] = {"market", "pattern", "error", "success", "general"}

    def __init__(self, memory: CoordinatorMemory) -> None:
        """Initialize tool with coordinator memory.

        Args:
            memory: Coordinator memory instance
        """
        self._memory = memory

    @property
    def name(self) -> str:
        """Tool name."""
        return "save_observation"

    def get_tool_definition(self) -> ToolDefinition:
        """Get tool definition in LiteLLM/OpenAI format.

        Returns:
            Tool definition for LLM function calling
        """
        return ToolDefinition(
            function=ToolFunction(
                name=self.name,
                description=(
                    "Save an observation for coordinator learning and future reference. "
                    "Use this to record patterns, insights, errors, or successes that should "
                    "be remembered. Minimum 10 characters required."
                ),
                parameters=ToolParametersSchema(
                    properties={
                        "observation": ToolParameter(
                            type="string",
                            description="Observation text (minimum 10 characters)",
                        ),
                        "category": ToolParameter(
                            type="string",
                            description=(
                                "Category: market, pattern, error, success, or general (default: general)"
                            ),
                        ),
                    },
                    required=["observation"],
                ),
            ),
        )

    async def aexecute(self, **kwargs: str | int | float | bool) -> str:
        """Execute observation save asynchronously.

        Args:
            **kwargs: Tool arguments (observation: str, category: str = "general")

        Returns:
            Confirmation message
        """
        observation = str(kwargs["observation"])
        category = str(kwargs.get("category", "general"))

        # Validate observation length
        if len(observation) < MIN_OBSERVATION_LENGTH:
            return f"Error: Observation must be at least {MIN_OBSERVATION_LENGTH} characters"

        # Validate category
        if category not in self.VALID_CATEGORIES:
            valid_categories = ", ".join(sorted(self.VALID_CATEGORIES))
            return f"Error: Invalid category '{category}'. Must be one of: {valid_categories}"

        logger.info(f"Saving observation (category={category})")

        try:
            await self._memory.save(observation, category)

            from datetime import UTC, datetime

            timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
            return f"Observation saved successfully at {timestamp} (category: {category})"

        except Exception as e:
            logger.opt(exception=True).error(f"Failed to save observation: {e}")
            return f"Failed to save observation: {e}"

    def execute(self, **kwargs: str | int | float | bool) -> str:
        """Sync wrapper (not used - coordinator calls aexecute).

        Raises:
            RuntimeError: If called from within a running event loop
        """
        import asyncio

        # Guard against being called from within an existing event loop
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop; safe to use asyncio.run
            return asyncio.run(self.aexecute(**kwargs))
        else:
            # There is a running loop; callers should use the async API directly
            msg = (
                "SaveObservationTool.execute() cannot be called from a running "
                "event loop. Use 'aexecute' instead."
            )
            raise RuntimeError(msg)

    def __repr__(self) -> str:
        """String representation."""
        return "SaveObservationTool()"
