"""Fundamental analysis worker for stock valuation with earnings calendar integration."""

import asyncio
from datetime import date
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import BaseModel, Field

from src.agents.fundamental import FundamentalAnalysis
from src.agents.models import FundamentalMetrics
from src.data.earnings import EarningsCalendarFetcher
from src.data.fundamental import FundamentalDataFetcher
from src.models.llm import LLMClient
from src.models.providers.base import StructuredOutputError
from src.prompts import PromptLoader
from src.tools.models import ToolDefinition, ToolFunction, ToolParameter

if TYPE_CHECKING:
    from src.data.earnings import EarningsCalendar

# Constants for valuation thresholds
EARNINGS_WARNING_DAYS = 5  # Flag trades within ±5 days of earnings
PE_RATIO_UNDERVALUED = 15  # P/E < 15 is undervalued
PE_RATIO_OVERVALUED = 30  # P/E > 30 is overvalued


class EarningsFlags(BaseModel):
    """Earnings calendar flags."""

    upcoming_earnings: bool
    days_until_earnings: int | None = None
    earnings_date: date | None = None
    estimate_eps: float | None = None


class FundamentalLLMResponse(BaseModel):
    """LLM response for fundamental worker."""

    interpretation: str = Field(description="Fundamental health interpretation (2-3 sentences)")
    confidence_keywords: list[str] = Field(
        description="Confidence keywords: 'strong', 'high confidence', 'weak', 'limited data', etc."
    )


class FundamentalWorker:
    """Worker for fundamental analysis with earnings calendar integration."""

    def __init__(
        self,
        llm_client: LLMClient,
        fundamental_fetcher: FundamentalDataFetcher,
        earnings_fetcher: EarningsCalendarFetcher,
    ) -> None:
        """Initialize fundamental worker.

        Args:
            llm_client: LLM client for generating interpretations
            fundamental_fetcher: Fundamental data fetcher
            earnings_fetcher: Earnings calendar fetcher
        """
        self.llm = llm_client
        self.fundamental_fetcher = fundamental_fetcher
        self.earnings_fetcher = earnings_fetcher
        self._prompts = PromptLoader("fundamental_worker")
        logger.info("Initialized FundamentalWorker")

    async def analyze(self, symbol: str, current_price: float | None = None) -> FundamentalAnalysis:
        """Fetch fundamentals + earnings, generate analysis.

        Args:
            symbol: Stock ticker symbol
            current_price: Current stock price (optional, for context)

        Returns:
            FundamentalAnalysis with earnings_flags
        """
        logger.info(f"FundamentalWorker analyzing {symbol}")

        # Parallel fetch: metrics + earnings
        metrics_task = asyncio.to_thread(self.fundamental_fetcher.fetch_overview, symbol)
        earnings_task = asyncio.to_thread(self.earnings_fetcher.fetch_earnings_dates, [symbol])

        try:
            overview, earnings_calendar = await asyncio.gather(metrics_task, earnings_task)
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to fetch data for {symbol}: {e}")
            raise

        # Extract metrics
        metrics = self._extract_metrics(overview)

        # Calculate earnings flags (5-day window)
        earnings_flags = self._calculate_earnings_flags(earnings_calendar, symbol)

        # Assess valuation
        valuation = self._assess_valuation(metrics)

        # Build LLM prompt
        metrics_section = self._build_metrics_section(metrics, valuation, current_price)
        earnings_section = self._build_earnings_section(earnings_flags)
        prompt = self._prompts.load(
            "user",
            symbol=symbol,
            valuation=valuation,
            metrics_section=metrics_section,
            earnings_section=earnings_section,
        )
        system = self._prompts.load("system")

        # Get LLM interpretation with structured output and fallback
        try:
            llm_response = await self.llm.astructured(
                prompt, FundamentalLLMResponse, system=system, temperature=0.5
            )
            interpretation = llm_response.interpretation
            confidence = self._calculate_confidence_from_keywords(metrics, llm_response.confidence_keywords)
        except StructuredOutputError as e:
            logger.opt(exception=True).warning(f"Structured output failed, falling back to text parsing: {e}")
            interpretation = await self.llm.acomplete(prompt, system=system, temperature=0.5)
            confidence = self._calculate_confidence(metrics, interpretation)

        return FundamentalAnalysis(
            valuation=valuation,
            earnings_flags=earnings_flags,
            pe_ratio=metrics.pe_ratio,
            eps=metrics.eps,
            revenue_growth_yoy=metrics.revenue_growth_yoy,
            earnings_growth_yoy=metrics.earnings_growth_yoy,
            debt_to_equity=metrics.debt_to_equity,
            current_ratio=metrics.current_ratio,
            interpretation=interpretation,
            confidence=confidence,
        )

    def _calculate_earnings_flags(
        self,
        earnings_calendar: EarningsCalendar | None,
        symbol: str,
    ) -> EarningsFlags:
        """Calculate earnings flags - 5-day window (±5 days).

        Args:
            earnings_calendar: Earnings calendar data
            symbol: Stock ticker symbol

        Returns:
            EarningsFlags with upcoming_earnings status
        """
        from datetime import UTC, datetime

        # Safe default if fetch failed
        if earnings_calendar is None or not hasattr(earnings_calendar, "events"):
            logger.warning(f"No earnings calendar data for {symbol}, defaulting to no upcoming earnings")
            return EarningsFlags(upcoming_earnings=False)

        # Find event for symbol
        today = datetime.now(UTC).date()
        for event in earnings_calendar.events:
            if event.symbol.upper() == symbol.upper() and event.earnings_date:
                days_until = (event.earnings_date - today).days
                # Flag if within threshold
                if abs(days_until) <= EARNINGS_WARNING_DAYS:
                    return EarningsFlags(
                        upcoming_earnings=True,
                        days_until_earnings=days_until,
                        earnings_date=event.earnings_date,
                        estimate_eps=event.estimate_eps,
                    )

        return EarningsFlags(upcoming_earnings=False)

    def _extract_metrics(self, overview: dict[str, Any]) -> FundamentalMetrics:
        """Extract key fundamental metrics from overview data.

        Args:
            overview: Raw overview data from Alpha Vantage

        Returns:
            FundamentalMetrics with parsed values (None for missing/invalid)
        """
        return FundamentalMetrics(
            pe_ratio=self._parse_float(overview.get("PERatio")),
            eps=self._parse_float(overview.get("EPS")),
            revenue_growth_yoy=self._parse_float(overview.get("QuarterlyRevenueGrowthYOY")),
            earnings_growth_yoy=self._parse_float(overview.get("QuarterlyEarningsGrowthYOY")),
            debt_to_equity=self._parse_float(overview.get("DebtToEquity")),
            current_ratio=self._parse_float(overview.get("CurrentRatio")),
        )

    def _assess_valuation(self, metrics: FundamentalMetrics) -> str:
        """Assess company valuation based on P/E ratio.

        Args:
            metrics: Extracted fundamental metrics

        Returns:
            Valuation string (UNDERVALUED | FAIRLY_VALUED | OVERVALUED)
        """
        pe_ratio = metrics.pe_ratio

        if pe_ratio is None:
            return "FAIRLY_VALUED"

        if pe_ratio < PE_RATIO_UNDERVALUED:
            return "UNDERVALUED"
        if pe_ratio > PE_RATIO_OVERVALUED:
            return "OVERVALUED"
        return "FAIRLY_VALUED"

    def _build_metrics_section(
        self,
        metrics: FundamentalMetrics,
        valuation: str,
        current_price: float | None,
    ) -> str:
        """Build metrics section for prompt.

        Args:
            metrics: Extracted metrics
            valuation: Assessed valuation
            current_price: Current price (optional)

        Returns:
            Formatted metrics string
        """
        prompt_parts = []

        if current_price:
            prompt_parts.append(f"Current Price: ${current_price:.2f}")

        prompt_parts.append(f"Valuation: {valuation}")

        # Add available metrics
        if metrics.pe_ratio is not None:
            prompt_parts.append(f"P/E Ratio: {metrics.pe_ratio:.2f}")
        if metrics.eps is not None:
            prompt_parts.append(f"EPS: ${metrics.eps:.2f}")
        if metrics.revenue_growth_yoy is not None:
            prompt_parts.append(f"Revenue Growth YoY: {metrics.revenue_growth_yoy * 100:.1f}%")
        if metrics.earnings_growth_yoy is not None:
            prompt_parts.append(f"Earnings Growth YoY: {metrics.earnings_growth_yoy * 100:.1f}%")
        if metrics.debt_to_equity is not None:
            prompt_parts.append(f"Debt-to-Equity: {metrics.debt_to_equity:.2f}")
        if metrics.current_ratio is not None:
            prompt_parts.append(f"Current Ratio: {metrics.current_ratio:.2f}")

        return "\n".join(prompt_parts)

    def _build_earnings_section(self, earnings_flags: EarningsFlags) -> str:
        """Build earnings section for prompt.

        Args:
            earnings_flags: Earnings flags

        Returns:
            Formatted earnings string
        """
        if not earnings_flags.upcoming_earnings:
            return "No upcoming earnings within 5 days"

        parts = []
        if earnings_flags.days_until_earnings is not None:
            if earnings_flags.days_until_earnings < 0:
                parts.append(f"Earnings reported {abs(earnings_flags.days_until_earnings)} days ago")
            elif earnings_flags.days_until_earnings == 0:
                parts.append("Earnings report TODAY")
            else:
                parts.append(f"Earnings in {earnings_flags.days_until_earnings} days")

        if earnings_flags.earnings_date:
            parts.append(f"Date: {earnings_flags.earnings_date.isoformat()}")

        if earnings_flags.estimate_eps is not None:
            parts.append(f"Estimated EPS: ${earnings_flags.estimate_eps:.2f}")

        return ", ".join(parts)

    def _calculate_confidence(self, metrics: FundamentalMetrics, interpretation: str) -> float:
        """Calculate confidence score based on data completeness and LLM signals.

        Args:
            metrics: Extracted metrics
            interpretation: LLM interpretation text

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence
        confidence = 0.5

        # Boost based on data completeness
        confidence += 0.3 * metrics.completeness_ratio

        # Adjust based on LLM signals
        interpretation_lower = interpretation.lower()
        if any(word in interpretation_lower for word in ["strong", "high confidence", "clear"]):
            confidence += 0.1
        if any(word in interpretation_lower for word in ["uncertain", "limited data", "unclear"]):
            confidence -= 0.2

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, confidence))

    def _calculate_confidence_from_keywords(self, metrics: FundamentalMetrics, keywords: list[str]) -> float:
        """Calculate confidence score based on data completeness and extracted keywords.

        Args:
            metrics: Extracted metrics
            keywords: Confidence keywords from LLM

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence
        confidence = 0.5

        # Boost based on data completeness
        confidence += 0.3 * metrics.completeness_ratio

        # Adjust based on keywords
        keywords_lower = [k.lower() for k in keywords]
        if any(word in keywords_lower for word in ["strong", "high confidence", "clear"]):
            confidence += 0.1
        if any(word in keywords_lower for word in ["uncertain", "limited data", "unclear"]):
            confidence -= 0.2

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, confidence))

    def _parse_float(self, value: str | float | None) -> float | None:
        """Parse float value from API response.

        Args:
            value: Value to parse (can be str, float, None, or "-")

        Returns:
            Parsed float or None if invalid/missing
        """
        if value is None or value in {"-", "None"}:
            return None

        try:
            return float(value)
        except ValueError, TypeError:
            return None

    def get_tool_definition(self) -> ToolDefinition:
        """Get tool definition for supervisor integration.

        Returns:
            ToolDefinition for fundamental analysis
        """
        return ToolDefinition(
            type="function",
            function=ToolFunction(
                name="analyze_fundamental",
                description="Analyze fundamental metrics and earnings calendar for a stock",
                parameters=ToolParameter(
                    type="object",
                    properties={
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol (e.g., AAPL, TSLA)",
                        },
                        "current_price": {
                            "type": "number",
                            "description": "Current stock price (optional, for context)",
                        },
                    },
                    required=["symbol"],
                ),
            ),
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"FundamentalWorker(llm={self.llm})"
