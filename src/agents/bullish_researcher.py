"""Bullish researcher agent for constructing optimistic investment thesis."""

import re

from loguru import logger
from pydantic import BaseModel, Field

from src.agents.fundamental import FundamentalAnalysis
from src.agents.news import NewsAnalysis
from src.agents.sentiment import SentimentAnalysis
from src.agents.technical import TechnicalAnalysis
from src.models.llm import LLMClient
from src.strategies.momentum import Signal


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
        logger.info("Initialized BullishResearcher")

    def analyze(
        self,
        symbol: str,
        technical: TechnicalAnalysis,
        sentiment: SentimentAnalysis,
        news: NewsAnalysis,
        fundamental: FundamentalAnalysis,
    ) -> BullishResearchAnalysis:
        """Construct bullish thesis from all analyses.

        Args:
            symbol: Stock ticker symbol
            technical: Technical analysis result
            sentiment: Sentiment analysis result
            news: News analysis result
            fundamental: Fundamental analysis result

        Returns:
            BullishResearchAnalysis with thesis, strengths, upside, and confidence
        """
        logger.info(f"Constructing bull thesis for {symbol}")

        prompt = self._build_prompt(symbol, technical, sentiment, news, fundamental)

        system_prompt = (
            "You are an optimistic investment researcher who identifies upside opportunities "
            "and constructs bull theses. Focus on strengths, catalysts, and positive scenarios."
        )

        response = self.llm.complete(prompt, system=system_prompt, temperature=0.5)

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

    def _build_prompt(
        self,
        symbol: str,
        technical: TechnicalAnalysis,
        sentiment: SentimentAnalysis,
        news: NewsAnalysis,
        fundamental: FundamentalAnalysis,
    ) -> str:
        """Build LLM prompt from all analyses.

        Args:
            symbol: Stock ticker symbol
            technical: Technical analysis result
            sentiment: Sentiment analysis result
            news: News analysis result
            fundamental: Fundamental analysis result

        Returns:
            Formatted prompt string
        """
        # Technical section
        tech_str = (
            f"{technical.signal.value} (RSI {technical.rsi:.1f}, "
            f"MACD {technical.macd_hist:.3f}, confidence {technical.confidence:.2f})"
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
        pe_str = f"{fundamental.pe_ratio:.1f}" if fundamental.pe_ratio is not None else "N/A"
        eps_str = f"{fundamental.eps:.2f}" if fundamental.eps is not None else "N/A"
        growth_str = (
            f"{fundamental.revenue_growth_yoy:.1%}" if fundamental.revenue_growth_yoy is not None else "N/A"
        )
        fund_str = f"{fundamental.valuation} (P/E {pe_str}, EPS {eps_str}, growth {growth_str})"

        return f"""Construct a bull thesis for {symbol} based on:

TECHNICAL: {tech_str}
SENTIMENT: {sent_str}
NEWS: {news_str}
FUNDAMENTAL: {fund_str}

Provide:
1. Bull thesis (3-4 sentences explaining why this stock has upside potential)
2. Key strengths (3-5 bullet points, each starting with '- ')
3. Target upside % (reasonable estimate or "N/A" if uncertain)

Format your response as:
THESIS: [your thesis here]
STRENGTHS:
- [strength 1]
- [strength 2]
- [strength 3]
UPSIDE: [percentage or N/A]"""

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
        fundamental: FundamentalAnalysis,
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

        # Fundamental boost/penalty
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
