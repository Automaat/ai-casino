"""Generate game plan tool for coordinator."""

import asyncio
import concurrent.futures
from typing import TYPE_CHECKING

from loguru import logger

from src.tools.base import BaseTool
from src.tools.models import ToolDefinition, ToolFunction, ToolParameter, ToolParametersSchema

if TYPE_CHECKING:
    from src.agents.game_plan import GamePlan, GamePlanAgent


class GenerateGamePlanTool(BaseTool):
    """Tool to generate daily trading game plan."""

    def __init__(self, game_plan_agent: GamePlanAgent) -> None:
        """Initialize tool with game plan agent.

        Args:
            game_plan_agent: Game plan agent instance
        """
        self._agent = game_plan_agent

    @property
    def name(self) -> str:
        """Tool name."""
        return "generate_game_plan"

    @property
    def requires_confirmation(self) -> bool:
        """Requires confirmation due to LLM calls."""
        return True

    def get_tool_definition(self) -> ToolDefinition:
        """Get tool definition in LiteLLM/OpenAI format.

        Returns:
            Tool definition for LLM function calling
        """
        return ToolDefinition(
            function=ToolFunction(
                name=self.name,
                description=(
                    "Generate daily trading game plan with risk stance, priority symbols, "
                    "sector focus, overnight summary, and rationale. Uses futures data and "
                    "watchlist to create strategic guidance for the trading day."
                ),
                parameters=ToolParametersSchema(
                    properties={
                        "watchlist": ToolParameter(
                            type="string",
                            description="Comma-separated watchlist symbols (e.g., AAPL,TSLA,NVDA)",
                        ),
                        "include_sector_context": ToolParameter(
                            type="boolean",
                            description="Include sector rotation context (default: false)",
                        ),
                        "include_earnings_context": ToolParameter(
                            type="boolean",
                            description="Include earnings calendar context (default: false)",
                        ),
                    },
                    required=["watchlist"],
                ),
            ),
        )

    async def aexecute(self, **kwargs: str | int | float | bool) -> str:
        """Execute game plan generation asynchronously.

        Args:
            **kwargs: Tool arguments (watchlist: str, include_sector_context: bool,
                     include_earnings_context: bool)

        Returns:
            Formatted game plan
        """
        watchlist_str = str(kwargs["watchlist"])
        watchlist = [s.strip().upper() for s in watchlist_str.split(",")]
        include_sector = bool(kwargs.get("include_sector_context", False))
        include_earnings = bool(kwargs.get("include_earnings_context", False))

        logger.info(f"Generating game plan for watchlist: {watchlist}")

        try:
            # Generate game plan (async)
            plan = await self._agent.generate(
                watchlist=watchlist,
                sector_context="Sector rotation analysis pending" if include_sector else None,
                earnings_context="Earnings calendar pending" if include_earnings else None,
            )

            return self._format_result(plan)

        except Exception as e:
            logger.error(f"Game plan generation failed: {e}")
            return f"Failed to generate game plan: {e}"

    def execute(self, **kwargs: str | int | float | bool) -> str:
        """Execute game plan generation synchronously.

        Args:
            **kwargs: Tool arguments

        Returns:
            Formatted game plan
        """

        def run_in_thread() -> str:
            return asyncio.run(self.aexecute(**kwargs))

        # Run in thread to avoid nested event loop issues
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_in_thread)
            return future.result()

    def _format_result(self, plan: GamePlan) -> str:
        """Format game plan as markdown.

        Args:
            plan: Generated game plan

        Returns:
            Formatted markdown string
        """
        lines = [
            f"# Game Plan - {plan.date.isoformat()}",
            "",
            f"**Risk Stance:** {plan.risk_stance}",
            f"**Confidence:** {plan.confidence:.0%}",
            "",
            "## Priority Symbols",
        ]

        for symbol in plan.priority_symbols:
            key_level = plan.key_levels.get(symbol)
            level_text = f" @ ${key_level:.2f}" if key_level else ""
            lines.append(f"- **{symbol}**{level_text}")

        lines.extend(
            [
                "",
                "## Sector Focus",
                *[f"- {sector}" for sector in plan.sector_focus],
                "",
                "## Overnight Summary",
                plan.overnight_summary,
                "",
                "## Strategic Rationale",
                plan.reasoning,
            ]
        )

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return "GenerateGamePlanTool()"
