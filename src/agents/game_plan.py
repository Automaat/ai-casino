"""Game plan agent for next-day trading strategy."""

import json
from datetime import UTC, date, datetime, tzinfo
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field

from src.data.market import MarketDataFetcher
from src.data.news import NewsFetcher
from src.models.llm import LLMClient, ToolCallingParams
from src.models.providers.base import StructuredOutputError
from src.prompts import PromptLoader
from src.tools.game_plan import (
    FetchMarketContextTool,
    FetchNewsHeadlinesTool,
    FetchPremarketMoversTool,
    FetchSectorPerformanceTool,
)
from src.tools.registry import ToolRegistry


class KeyLevel(BaseModel):
    """Price level for a symbol."""

    symbol: str
    price: float = Field(gt=0.0)


class GamePlanLLMResponse(BaseModel):
    """LLM response structure for game plan."""

    priority_symbols: list[str] = Field(description="3-5 priority symbols")
    risk_stance: Literal["AGGRESSIVE", "DEFENSIVE", "NEUTRAL"]
    sector_focus: list[str] = Field(description="1-3 sectors to focus on")
    key_levels: list[KeyLevel] = Field(default_factory=list, description="Key price levels (optional)")
    reasoning: str = Field(description="Strategic rationale (2-3 sentences)")
    confidence: float = Field(ge=0.0, le=1.0)


class GamePlan(BaseModel):
    """Generated trading game plan for the day."""

    date: date
    priority_symbols: list[str]
    risk_stance: Literal["AGGRESSIVE", "DEFENSIVE", "NEUTRAL"]
    sector_focus: list[str]
    key_levels: dict[str, float]
    overnight_summary: str
    reasoning: str
    confidence: float
    generated_at: datetime


class GamePlanAgent:
    """Agentic game plan generator with tool-calling research phase."""

    def __init__(
        self,
        llm_client: LLMClient,
        market_fetcher: MarketDataFetcher,
        news_fetcher: NewsFetcher,
    ) -> None:
        """Initialize game plan agent.

        Args:
            llm_client: LLM client for analysis
            market_fetcher: Market data fetcher for futures
            news_fetcher: News fetcher for headlines
        """
        self.llm = llm_client
        self.market_fetcher = market_fetcher
        self._news_fetcher = news_fetcher
        self._prompts = PromptLoader("game_plan")
        self._tool_registry = self._build_tool_registry()
        logger.info("Initialized GamePlanAgent (agentic)")

    def _build_tool_registry(self) -> ToolRegistry:
        """Build tool registry for research phase.

        Returns:
            ToolRegistry with game plan tools
        """
        registry = ToolRegistry()
        registry.register(FetchMarketContextTool(self.market_fetcher))
        registry.register(FetchPremarketMoversTool())
        registry.register(FetchSectorPerformanceTool())
        registry.register(FetchNewsHeadlinesTool(self._news_fetcher))
        return registry

    async def generate(
        self,
        watchlist: list[str],
        timezone: tzinfo = UTC,
    ) -> GamePlan:
        """Generate daily game plan via two-phase agentic approach.

        Phase 1: LLM calls tools to gather market context (futures, movers, sectors, news).
        Phase 2: Structured extraction from gathered context into GamePlan.

        Args:
            watchlist: Stock watchlist
            timezone: Timezone for date calculation (defaults to UTC)

        Returns:
            GamePlan with priorities, risk stance, sector focus
        """
        if not watchlist:
            watchlist = ["SPY", "QQQ", "AAPL"]
            logger.warning("Empty watchlist, using defaults")

        current_date = datetime.now(timezone).date().isoformat()

        # Phase 1: Agentic research — LLM decides which tools to call
        system = self._prompts.load("system")
        user_prompt = self._prompts.load("user", date=current_date, watchlist_symbols=", ".join(watchlist))
        tool_defs = self._tool_registry.get_definitions()

        research_context = await self.llm.acomplete_with_tools(
            ToolCallingParams(
                prompt=user_prompt,
                tools=tool_defs,
                tool_executor=self._tool_executor,
                system=system,
                temperature=0.5,
                max_tool_calls=8,
                max_tokens=2048,
            )
        )

        logger.info(f"Phase 1 complete: gathered research context ({len(research_context)} chars)")

        # Phase 2: Structured extraction
        extract_prompt = self._prompts.load(
            "extract",
            date=current_date,
            watchlist_symbols=", ".join(watchlist),
            research_context=research_context,
        )

        try:
            llm_response = await self.llm.astructured(
                extract_prompt, GamePlanLLMResponse, system=system, temperature=0.3, max_tokens=512
            )
        except StructuredOutputError as e:
            logger.opt(exception=True).warning(f"Structured output failed, falling back: {e}")
            text_response = await self.llm.acomplete(extract_prompt, system=system, temperature=0.3)
            return GamePlan(
                date=datetime.now(timezone).date(),
                priority_symbols=[],
                risk_stance="NEUTRAL",
                sector_focus=[],
                key_levels={},
                overnight_summary=research_context[:200],
                reasoning=text_response,
                confidence=0.5,
                generated_at=datetime.now(UTC),
            )

        key_levels: dict[str, float] = {}
        seen_symbols: set[str] = set()
        for kl in llm_response.key_levels:
            if kl.symbol in seen_symbols:
                logger.warning(
                    "Duplicate key level symbol: {symbol}. Keeping last price {price}.",
                    symbol=kl.symbol,
                    price=kl.price,
                )
            seen_symbols.add(kl.symbol)
            key_levels[kl.symbol] = kl.price

        return GamePlan(
            date=datetime.now(timezone).date(),
            priority_symbols=llm_response.priority_symbols,
            risk_stance=llm_response.risk_stance,
            sector_focus=llm_response.sector_focus,
            key_levels=key_levels,
            overnight_summary=research_context[:300],
            reasoning=llm_response.reasoning,
            confidence=llm_response.confidence,
            generated_at=datetime.now(UTC),
        )

    async def _tool_executor(self, name: str, args: dict) -> str:
        """Execute tool by name.

        Args:
            name: Tool name
            args: Tool arguments

        Returns:
            Tool result string
        """
        try:
            return await self._tool_registry.aexecute(name, args)
        except Exception as e:
            logger.opt(exception=True).error(f"Game plan tool failed: {name} - {e}")
            return f"Error: {e!s}"

    def persist(self, plan: GamePlan, plan_dir: str) -> Path:
        """Persist game plan to JSON.

        Args:
            plan: GamePlan to save
            plan_dir: Directory for plans

        Returns:
            Path to saved file
        """
        path = Path(plan_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)

        file_path = path / f"{plan.date}.json"

        with file_path.open("w") as f:
            json.dump(plan.model_dump(mode="json"), f, indent=2, default=str)

        logger.info(f"Persisted game plan to {file_path}")
        return file_path

    def __repr__(self) -> str:
        """String representation."""
        return f"GamePlanAgent(llm={self.llm.model})"
