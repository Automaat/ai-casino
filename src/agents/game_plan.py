"""Game plan agent for next-day trading strategy."""

import json
from datetime import UTC, date, datetime, tzinfo
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field

from src.data.market import MarketDataFetcher
from src.models.llm import LLMClient
from src.models.providers.base import StructuredOutputError
from src.prompts import PromptLoader


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
    """Agent for generating daily trading game plans."""

    def __init__(self, llm_client: LLMClient, market_fetcher: MarketDataFetcher) -> None:
        """Initialize game plan agent.

        Args:
            llm_client: LLM client for analysis
            market_fetcher: Market data fetcher for futures
        """
        self.llm = llm_client
        self.market_fetcher = market_fetcher
        self._prompts = PromptLoader("game_plan")
        logger.info("Initialized GamePlanAgent")

    async def generate(
        self,
        watchlist: list[str],
        futures_symbols: list[str] | None = None,
        sector_context: str | None = None,
        earnings_context: str | None = None,
        timezone: tzinfo = UTC,
    ) -> GamePlan:
        """Generate daily game plan.

        Args:
            watchlist: Stock watchlist
            futures_symbols: Futures to track (defaults to ES=F, NQ=F)
            sector_context: Optional sector rotation context
            earnings_context: Optional earnings calendar context
            timezone: Timezone for date calculation (defaults to UTC)

        Returns:
            GamePlan with priorities, risk stance, sector focus
        """
        if not watchlist:
            watchlist = ["SPY", "QQQ", "AAPL"]
            logger.warning("Empty watchlist, using defaults")

        futures_symbols = futures_symbols or ["ES=F", "NQ=F"]

        futures_context = self._fetch_futures_context(futures_symbols)
        premarket_movers = self._fetch_premarket_movers(watchlist)
        overnight_summary = self._format_overnight_summary(futures_context, premarket_movers)

        sector_section = f"\n## Sector Context\n{sector_context}\n" if sector_context else ""
        earnings_section = f"\n## Earnings Context\n{earnings_context}\n" if earnings_context else ""

        prompt = self._prompts.load(
            "user",
            date=datetime.now(timezone).date().isoformat(),
            futures_context=self._format_futures(futures_context),
            premarket_movers=premarket_movers,
            watchlist_symbols=", ".join(watchlist),
            sector_context_section=sector_section,
            earnings_context_section=earnings_section,
        )
        system = self._prompts.load("system")

        try:
            llm_response = await self.llm.astructured(
                prompt, GamePlanLLMResponse, system=system, temperature=0.7, max_tokens=512
            )
            priority_symbols = llm_response.priority_symbols
            risk_stance = llm_response.risk_stance
            sector_focus = llm_response.sector_focus
            key_levels: dict[str, float] = {}
            seen_symbols: set[str] = set()
            for kl in llm_response.key_levels:
                if kl.symbol in seen_symbols:
                    logger.warning(
                        "Duplicate key level symbol from LLM response: {symbol}. "
                        "Keeping last provided price {price}.",
                        symbol=kl.symbol,
                        price=kl.price,
                    )
                seen_symbols.add(kl.symbol)
                key_levels[kl.symbol] = kl.price
            reasoning = llm_response.reasoning
            confidence = llm_response.confidence
        except StructuredOutputError as e:
            logger.opt(exception=True).warning(f"Structured output failed, falling back: {e}")
            text_response = await self.llm.acomplete(prompt, system=system, temperature=0.7)
            priority_symbols = []
            risk_stance = "NEUTRAL"
            sector_focus = []
            key_levels = {}
            reasoning = text_response
            confidence = 0.5

        return GamePlan(
            date=datetime.now(timezone).date(),
            priority_symbols=priority_symbols,
            risk_stance=risk_stance,
            sector_focus=sector_focus,
            key_levels=key_levels,
            overnight_summary=overnight_summary,
            reasoning=reasoning,
            confidence=confidence,
            generated_at=datetime.now(UTC),
        )

    def _fetch_futures_context(self, symbols: list[str]) -> dict[str, float]:
        """Fetch overnight futures % change.

        Args:
            symbols: Futures symbols

        Returns:
            Dict mapping symbol to % change (empty if unavailable)
        """
        try:
            return self.market_fetcher.fetch_overnight_futures(symbols)
        except Exception as e:
            logger.opt(exception=True).warning(f"Unexpected error fetching futures: {e}")
            return {}

    def _fetch_premarket_movers(self, watchlist: list[str]) -> str:
        """Get top 3 pre-market gainers/losers.

        Args:
            watchlist: Symbols to check

        Returns:
            Formatted string of movers
        """
        try:
            import yfinance as yf

            # Limit to first 15 symbols to prevent slow/flaky yfinance loops (#270)
            limited_watchlist = watchlist[:15]
            movers = []
            for symbol in limited_watchlist:
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period="2d")
                    if data.empty or len(data) < 2:
                        continue

                    prev_close = data["Close"].iloc[-2]
                    current = data["Close"].iloc[-1]
                    pct_change = ((current - prev_close) / prev_close) * 100
                    movers.append((symbol, pct_change))
                except Exception as e:
                    logger.debug(f"Failed to fetch pre-market for {symbol}: {e}")
                    continue

            if not movers:
                return "No pre-market data available"

            movers.sort(key=lambda x: abs(x[1]), reverse=True)
            top_movers = movers[:3]

            gainers = [f"{s} +{p:.1f}%" for s, p in top_movers if p > 0]
            losers = [f"{s} {p:.1f}%" for s, p in top_movers if p < 0]

            result = []
            if gainers:
                result.append(f"Gainers: {', '.join(gainers)}")
            if losers:
                result.append(f"Losers: {', '.join(losers)}")

            return " | ".join(result) if result else "Flat pre-market"

        except Exception as e:
            logger.opt(exception=True).warning(f"Pre-market movers failed: {e}")
            return "Pre-market data unavailable"

    def _format_futures(self, futures: dict[str, float]) -> str:
        """Format futures data for prompt.

        Args:
            futures: Symbol to % change mapping

        Returns:
            Formatted string
        """
        if not futures:
            return "Futures data unavailable"

        lines = []
        for symbol, change in futures.items():
            direction = "up" if change > 0 else "down"
            lines.append(f"- {symbol}: {change:+.2f}% ({direction})")

        return "\n".join(lines)

    def _format_overnight_summary(self, futures: dict[str, float], movers: str) -> str:
        """Format overnight summary.

        Args:
            futures: Futures data
            movers: Pre-market movers string

        Returns:
            Summary string
        """
        futures_str = self._format_futures(futures)
        return f"Futures: {futures_str} | Pre-market: {movers}"

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
