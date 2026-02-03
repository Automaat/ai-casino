"""Fundamental analysis agent for stock valuation."""

from typing import Any

from loguru import logger
from pydantic import BaseModel

from src.data.fundamental import FundamentalDataFetcher
from src.models.llm import LLMClient


class FundamentalAnalysis(BaseModel):
    """Fundamental analysis result."""

    valuation: str  # UNDERVALUED | FAIRLY_VALUED | OVERVALUED
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
        logger.info("Initialized FundamentalAnalyst")

    def analyze(self, symbol: str, current_price: float | None = None) -> FundamentalAnalysis:
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
            prompt = self._build_analysis_prompt(symbol, metrics, valuation, current_price)
            system = (
                "You are a fundamental analyst. Provide a concise interpretation "
                "of the company's financial health and valuation in 2-3 sentences."
            )

            interpretation = self.llm.complete(prompt, system=system, temperature=0.5)
            confidence = self._calculate_confidence(metrics, interpretation)

            return FundamentalAnalysis(
                valuation=valuation,
                pe_ratio=metrics.get("pe_ratio"),
                eps=metrics.get("eps"),
                revenue_growth_yoy=metrics.get("revenue_growth_yoy"),
                earnings_growth_yoy=metrics.get("earnings_growth_yoy"),
                debt_to_equity=metrics.get("debt_to_equity"),
                current_ratio=metrics.get("current_ratio"),
                interpretation=interpretation,
                confidence=confidence,
            )

        except Exception as e:
            logger.error(f"Fundamental analysis failed for {symbol}: {e}")
            raise

    def _extract_metrics(self, overview: dict[str, Any]) -> dict[str, float | None]:
        """Extract key fundamental metrics from overview data.

        Args:
            overview: Raw overview data from Alpha Vantage

        Returns:
            Dictionary of parsed metrics (None for missing/invalid values)
        """
        return {
            "pe_ratio": self._parse_float(overview.get("PERatio")),
            "eps": self._parse_float(overview.get("EPS")),
            "revenue_growth_yoy": self._parse_float(overview.get("QuarterlyRevenueGrowthYOY")),
            "earnings_growth_yoy": self._parse_float(overview.get("QuarterlyEarningsGrowthYOY")),
            "debt_to_equity": self._parse_float(overview.get("DebtToEquity")),
            "current_ratio": self._parse_float(overview.get("CurrentRatio")),
        }

    def _assess_valuation(self, metrics: dict[str, float | None]) -> str:
        """Assess company valuation based on P/E ratio.

        Args:
            metrics: Extracted fundamental metrics

        Returns:
            Valuation string (UNDERVALUED | FAIRLY_VALUED | OVERVALUED)
        """
        pe_ratio = metrics.get("pe_ratio")

        if pe_ratio is None:
            return "FAIRLY_VALUED"

        if pe_ratio < 15:
            return "UNDERVALUED"
        if pe_ratio > 30:
            return "OVERVALUED"
        return "FAIRLY_VALUED"

    def _build_analysis_prompt(
        self,
        symbol: str,
        metrics: dict[str, float | None],
        valuation: str,
        current_price: float | None,
    ) -> str:
        """Build prompt for LLM interpretation.

        Args:
            symbol: Stock ticker
            metrics: Extracted metrics
            valuation: Assessed valuation
            current_price: Current price (optional)

        Returns:
            Formatted prompt string
        """
        prompt_parts = [f"Analyze fundamental data for {symbol}:"]

        if current_price:
            prompt_parts.append(f"Current Price: ${current_price:.2f}")

        prompt_parts.append(f"Valuation: {valuation}")

        # Add available metrics
        if metrics.get("pe_ratio") is not None:
            prompt_parts.append(f"P/E Ratio: {metrics['pe_ratio']:.2f}")
        if metrics.get("eps") is not None:
            prompt_parts.append(f"EPS: ${metrics['eps']:.2f}")
        if metrics.get("revenue_growth_yoy") is not None:
            prompt_parts.append(f"Revenue Growth YoY: {metrics['revenue_growth_yoy'] * 100:.1f}%")
        if metrics.get("earnings_growth_yoy") is not None:
            prompt_parts.append(f"Earnings Growth YoY: {metrics['earnings_growth_yoy'] * 100:.1f}%")
        if metrics.get("debt_to_equity") is not None:
            prompt_parts.append(f"Debt-to-Equity: {metrics['debt_to_equity']:.2f}")
        if metrics.get("current_ratio") is not None:
            prompt_parts.append(f"Current Ratio: {metrics['current_ratio']:.2f}")

        return "\n".join(prompt_parts)

    def _calculate_confidence(self, metrics: dict[str, float | None], interpretation: str) -> float:
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
        total_metrics = len(metrics)
        non_none_metrics = sum(1 for v in metrics.values() if v is not None)
        completeness_ratio = non_none_metrics / total_metrics if total_metrics > 0 else 0
        confidence += 0.3 * completeness_ratio

        # Adjust based on LLM signals
        interpretation_lower = interpretation.lower()
        if any(word in interpretation_lower for word in ["strong", "high confidence", "clear"]):
            confidence += 0.1
        if any(word in interpretation_lower for word in ["uncertain", "limited data", "unclear"]):
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
        except (ValueError, TypeError):
            return None

    def __repr__(self) -> str:
        """Return string representation."""
        return f"FundamentalAnalyst(llm={self.llm}, fetcher={self.fetcher})"
