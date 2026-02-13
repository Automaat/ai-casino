"""Fundamental analysis agent for stock valuation."""

from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from src.agents.models import EarningsFlags, FundamentalMetrics
from src.data.fundamental import FundamentalDataFetcher
from src.execution_tracking import track_agent
from src.models.llm import LLMClient
from src.models.providers.base import StructuredOutputError
from src.prompts import PromptLoader


class FundamentalLLMResponse(BaseModel):
    """LLM response structure for fundamental analysis."""

    interpretation: str = Field(description="Analysis interpretation of the fundamental metrics")
    confidence_keywords: list[str] = Field(
        description="Keywords indicating confidence level: 'strong', 'high confidence', 'clear', 'uncertain', etc."
    )


class FundamentalAnalysis(BaseModel):
    """Fundamental analysis result."""

    valuation: str  # UNDERVALUED | FAIRLY_VALUED | OVERVALUED
    earnings_flags: EarningsFlags | None = None
    pe_ratio: float | None
    eps: float | None
    revenue_growth_yoy: float | None
    earnings_growth_yoy: float | None
    debt_to_equity: float | None
    current_ratio: float | None
    interpretation: str
    confidence: float


class FundamentalAnalyst:
    """Analyzes fundamental company metrics for valuation."""

    def __init__(self, llm_client: LLMClient, fetcher: FundamentalDataFetcher) -> None:
        """Initialize the fundamental analyst.

        Args:
            llm_client: LLM client for generating interpretations
            fetcher: Fundamental data fetcher
        """
        self.llm = llm_client
        self.fetcher = fetcher
        self._prompts = PromptLoader("fundamental")
        logger.info("Initialized FundamentalAnalyst")

    @track_agent
    async def analyze(self, symbol: str, current_price: float | None = None) -> FundamentalAnalysis:
        """Perform fundamental analysis on a company.

        Args:
            symbol: Stock ticker symbol
            current_price: Current stock price (optional, for context)

        Returns:
            FundamentalAnalysis with valuation, metrics, interpretation, and confidence
        """
        logger.info(f"Analyzing {symbol} fundamentals")

        try:
            overview = self.fetcher.fetch_overview(symbol)
            metrics = self._extract_metrics(overview)
            valuation = self._assess_valuation(metrics)

            # Build LLM prompt with available metrics
            metrics_section = self._build_metrics_section(metrics, valuation, current_price)
            prompt = self._prompts.load("user", symbol=symbol, metrics_section=metrics_section)
            system = self._prompts.load("system")

            try:
                llm_response = await self.llm.astructured(
                    prompt, FundamentalLLMResponse, system=system, temperature=0.5
                )
                interpretation = llm_response.interpretation
                confidence = self._calculate_confidence_from_keywords(
                    metrics, llm_response.confidence_keywords
                )
            except StructuredOutputError as e:
                logger.opt(exception=True).warning(
                    f"Structured output failed, falling back to text parsing: {e}"
                )
                interpretation = await self.llm.acomplete(prompt, system=system, temperature=0.5)
                confidence = self._calculate_confidence(metrics, interpretation)

            return FundamentalAnalysis(
                valuation=valuation,
                pe_ratio=metrics.pe_ratio,
                eps=metrics.eps,
                revenue_growth_yoy=metrics.revenue_growth_yoy,
                earnings_growth_yoy=metrics.earnings_growth_yoy,
                debt_to_equity=metrics.debt_to_equity,
                current_ratio=metrics.current_ratio,
                interpretation=interpretation,
                confidence=confidence,
            )

        except Exception as e:
            logger.opt(exception=True).error(f"Fundamental analysis failed for {symbol}: {e}")
            raise

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

        if pe_ratio < 15:
            return "UNDERVALUED"
        if pe_ratio > 30:
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

    def __repr__(self) -> str:
        """Return string representation."""
        return f"FundamentalAnalyst(llm={self.llm}, fetcher={self.fetcher})"
