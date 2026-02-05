"""Bullish researcher agent for constructing optimistic investment thesis."""

import re
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel, Field

from src.agents.comparative import ComparativeAnalysis
from src.agents.fundamental import FundamentalAnalysis
from src.agents.news import NewsAnalysis
from src.agents.sentiment import SentimentAnalysis
from src.agents.technical import TechnicalAnalysis
from src.models.llm import LLMClient
from src.prompts import PromptLoader
from src.strategies.momentum import Signal

if TYPE_CHECKING:
    from src.agents.trump import TrumpAnalysis


class BullishResearchAnalysis(BaseModel):
    """Bullish research analysis result."""

    thesis: str = Field(description="Bull thesis (3-4 sentences)")
    key_strengths: list[str] = Field(description="Top 3-5 bullish signals")
    target_upside: float | None = Field(description="Expected upside % or None", default=None)
    confidence: float = Field(description="Confidence in bull case (0.0-1.0)", ge=0.0, le=1.0)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BullishResearchAnalysis(thesis='{self.thesis[:50]}...', "
            f"strengths={len(self.key_strengths)}, upside={self.target_upside}, "
            f"confidence={self.confidence:.2f})"
        )


class BullishResearcher:
    """Bullish researcher agent - synthesizes optimistic case from all analyses."""

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize bullish researcher.

        Args:
            llm_client: LLM client for generating bull thesis
        """
        self.llm = llm_client
        self._prompts = PromptLoader("bullish_researcher")
        logger.info("Initialized BullishResearcher")

    async def analyze(
        self,
        symbol: str,
        technical: TechnicalAnalysis,
        sentiment: SentimentAnalysis,
        news: NewsAnalysis,
        fundamental: FundamentalAnalysis | None,
        comparative: ComparativeAnalysis | None = None,
        trump_analysis: "TrumpAnalysis | None" = None,
    ) -> BullishResearchAnalysis:
        """Construct bullish thesis from all analyses.

        Args:
            symbol: Stock ticker symbol
            technical: Technical analysis result
            sentiment: Sentiment analysis result
            news: News analysis result
            fundamental: Fundamental analysis result (None if unavailable due to API rate limit)
            comparative: Comparative analysis result (optional)
            trump_analysis: Trump social media analysis (optional)

        Returns:
            BullishResearchAnalysis with thesis, strengths, upside, and confidence
        """
        logger.info(f"Constructing bull thesis for {symbol}")

        prompt_vars = self._build_prompt_vars(
            symbol, technical, sentiment, news, fundamental, comparative, trump_analysis
        )

        prompt = self._prompts.load("user", **prompt_vars)
        system_prompt = self._prompts.load("system")

        response = await self.llm.acomplete(prompt, system=system_prompt, temperature=0.5)

        thesis = self._extract_thesis(response)
        key_strengths = self._extract_key_strengths(response)
        target_upside = self._extract_target_upside(response)
        confidence = self._calculate_confidence(technical, sentiment, news, fundamental)

        logger.info(
            f"Bull thesis for {symbol}: {len(key_strengths)} strengths, "
            f"upside={target_upside}, confidence={confidence:.2f}"
        )

        return BullishResearchAnalysis(
            thesis=thesis,
            key_strengths=key_strengths,
            target_upside=target_upside,
            confidence=confidence,
        )

    def _build_prompt_vars(
        self,
        symbol: str,
        technical: TechnicalAnalysis,
        sentiment: SentimentAnalysis,
        news: NewsAnalysis,
        fundamental: FundamentalAnalysis | None,
        comparative: ComparativeAnalysis | None = None,
        trump_analysis: "TrumpAnalysis | None" = None,
    ) -> dict[str, str]:
        """Build LLM prompt variables from all analyses.

        Args:
            symbol: Stock ticker symbol
            technical: Technical analysis result
            sentiment: Sentiment analysis result
            news: News analysis result
            fundamental: Fundamental analysis result
            comparative: Comparative analysis result (optional)
            trump_analysis: Trump social media analysis (optional)

        Returns:
            Dictionary of prompt variables
        """
        # Technical section
        rsi_str = f"{technical.rsi:.1f}" if technical.rsi is not None else "N/A"
        macd_str = f"{technical.macd_hist:.3f}" if technical.macd_hist is not None else "N/A"
        tech_str = (
            f"{technical.signal.value} (RSI {rsi_str}, "
            f"MACD {macd_str}, confidence {technical.confidence:.2f})"
        )

        # Sentiment section
        sent_str = (
            f"{sentiment.overall_sentiment} "
            f"(score {sentiment.sentiment_score:.2f}, {sentiment.article_count} articles)"
        )

        # News section
        news_themes = ", ".join(news.key_themes) if news.key_themes else "None"
        news_str = f"{news_themes}, impact: {news.impact_assessment}"

        # Fundamental section
        if fundamental:
            pe_str = f"{fundamental.pe_ratio:.1f}" if fundamental.pe_ratio is not None else "N/A"
            eps_str = f"{fundamental.eps:.2f}" if fundamental.eps is not None else "N/A"
            growth_str = (
                f"{fundamental.revenue_growth_yoy:.1%}"
                if fundamental.revenue_growth_yoy is not None
                else "N/A"
            )
            fund_str = f"{fundamental.valuation} (P/E {pe_str}, EPS {eps_str}, growth {growth_str})"
        else:
            fund_str = "N/A (API rate limited)"

        # Comparative section
        comp_str = "N/A"
        if comparative:
            pe_vs_sector = f"{comparative.pe_vs_sector:.2f}x" if comparative.pe_vs_sector else "N/A"
            perf_vs_sector = (
                f"{comparative.perf_vs_sector_3m:+.1f}%" if comparative.perf_vs_sector_3m else "N/A"
            )
            comp_str = (
                f"{comparative.relative_valuation.value} "
                f"(P/E vs sector: {pe_vs_sector}, 3M perf vs sector: {perf_vs_sector})"
            )

        # Trump analysis section
        trump_str = "N/A"
        if trump_analysis:
            trump_str = (
                f"{trump_analysis.signal.value} "
                f"(sentiment: {trump_analysis.sentiment}, "
                f"confidence: {trump_analysis.confidence:.2f}, "
                f"market_relevant: {trump_analysis.market_relevant})"
            )

        return {
            "symbol": symbol,
            "tech_str": tech_str,
            "sent_str": sent_str,
            "news_str": news_str,
            "fund_str": fund_str,
            "comp_str": comp_str,
            "trump_str": trump_str,
        }

    def _extract_thesis(self, response: str) -> str:
        """Extract bull thesis from LLM response.

        Args:
            response: LLM response text

        Returns:
            Extracted thesis text
        """
        # Look for THESIS: section
        match = re.search(r"THESIS:\s*(.+?)(?=STRENGTHS:|$)", response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Fallback: use first 3-4 sentences
        sentences = response.split(".")[:4]
        return ".".join(sentences).strip() + "."

    def _extract_key_strengths(self, response: str) -> list[str]:
        """Extract key strengths from LLM response.

        Args:
            response: LLM response text

        Returns:
            List of strength bullet points
        """
        # Look for STRENGTHS: section
        match = re.search(r"STRENGTHS:\s*(.+?)(?=UPSIDE:|$)", response, re.DOTALL | re.IGNORECASE)
        if not match:
            # Fallback: look for bullet points anywhere
            bullets = re.findall(r"[-•]\s*(.+)", response)
            return [b.strip() for b in bullets[:5]] if bullets else []

        strengths_text = match.group(1).strip()
        bullets = re.findall(r"[-•]\s*(.+)", strengths_text)

        return [b.strip() for b in bullets[:5]] if bullets else []

    def _extract_target_upside(self, response: str) -> float | None:
        """Extract target upside percentage from LLM response.

        Args:
            response: LLM response text

        Returns:
            Upside percentage as float or None if not available
        """
        # Look for UPSIDE: section
        match = re.search(r"UPSIDE:\s*(.+)", response, re.IGNORECASE)
        if not match:
            return None

        upside_text = match.group(1).strip()

        # Check for N/A or similar
        if re.search(r"n/?a|not\s+available|uncertain|unknown", upside_text, re.IGNORECASE):
            return None

        # Extract percentage number
        num_match = re.search(r"(\d+(?:\.\d+)?)\s*%?", upside_text)
        if num_match:
            return float(num_match.group(1))

        return None

    def _calculate_confidence(
        self,
        technical: TechnicalAnalysis,
        sentiment: SentimentAnalysis,
        _news: NewsAnalysis,
        fundamental: FundamentalAnalysis | None,
    ) -> float:
        """Calculate confidence in bull case.

        Args:
            technical: Technical analysis result
            sentiment: Sentiment analysis result
            _news: News analysis result (unused, for API consistency)
            fundamental: Fundamental analysis result

        Returns:
            Confidence score (0.0-1.0)
        """
        confidence = 0.5  # Base confidence

        # Technical boost/penalty
        if technical.signal == Signal.BUY:
            confidence += 0.15
        elif technical.signal == Signal.SELL:
            confidence -= 0.2

        # Sentiment boost/penalty
        if sentiment.sentiment_score > 0.3:
            confidence += 0.1
        elif sentiment.sentiment_score < -0.3:
            confidence -= 0.15

        # Fundamental boost/penalty (skip if unavailable)
        if fundamental:
            if fundamental.valuation in ["UNDERVALUED", "FAIRLY_VALUED"]:
                confidence += 0.1
            elif fundamental.valuation == "OVERVALUED":
                confidence -= 0.1

            # Growth boost
            if fundamental.revenue_growth_yoy and fundamental.revenue_growth_yoy > 0.1:  # >10% growth
                confidence += 0.05

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, confidence))

    def __repr__(self) -> str:
        """String representation."""
        return f"BullishResearcher(llm={self.llm.provider}/{self.llm.model})"
