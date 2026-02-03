"""Bearish researcher agent for constructing pessimistic investment thesis."""

import re

from loguru import logger
from pydantic import BaseModel, Field

from src.agents.fundamental import FundamentalAnalysis
from src.agents.news import NewsAnalysis
from src.agents.sentiment import SentimentAnalysis
from src.agents.technical import TechnicalAnalysis
from src.models.llm import LLMClient
from src.strategies.momentum import Signal


class BearishResearchAnalysis(BaseModel):
    """Bearish research analysis result."""

    thesis: str = Field(description="Bear thesis (3-4 sentences)")
    key_weaknesses: list[str] = Field(description="Top 3-5 bearish signals")
    target_downside: float | None = Field(description="Expected downside % or None", default=None)
    confidence: float = Field(description="Confidence in bear case (0.0-1.0)", ge=0.0, le=1.0)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BearishResearchAnalysis(thesis='{self.thesis[:50]}...', "
            f"weaknesses={len(self.key_weaknesses)}, downside={self.target_downside}, "
            f"confidence={self.confidence:.2f})"
        )


class BearishResearcher:
    """Bearish researcher agent - synthesizes pessimistic case from all analyses."""

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize bearish researcher.

        Args:
            llm_client: LLM client for generating bear thesis
        """
        self.llm = llm_client
        logger.info("Initialized BearishResearcher")

    async def analyze(
        self,
        symbol: str,
        technical: TechnicalAnalysis,
        sentiment: SentimentAnalysis,
        news: NewsAnalysis,
        fundamental: FundamentalAnalysis,
    ) -> BearishResearchAnalysis:
        """Construct bearish thesis from all analyses.

        Args:
            symbol: Stock ticker symbol
            technical: Technical analysis result
            sentiment: Sentiment analysis result
            news: News analysis result
            fundamental: Fundamental analysis result

        Returns:
            BearishResearchAnalysis with thesis, weaknesses, downside, and confidence
        """
        logger.info(f"Constructing bear thesis for {symbol}")

        prompt = self._build_prompt(symbol, technical, sentiment, news, fundamental)

        system_prompt = (
            "You are a skeptical investment researcher who identifies risks, weaknesses, "
            "and downside scenarios. Focus on vulnerabilities, threats, and negative catalysts."
        )

        response = await self.llm.acomplete(prompt, system=system_prompt, temperature=0.5)

        thesis = self._extract_thesis(response)
        key_weaknesses = self._extract_key_weaknesses(response)
        target_downside = self._extract_target_downside(response)
        confidence = self._calculate_confidence(technical, sentiment, news, fundamental)

        logger.info(
            f"Bear thesis for {symbol}: {len(key_weaknesses)} weaknesses, "
            f"downside={target_downside}, confidence={confidence:.2f}"
        )

        return BearishResearchAnalysis(
            thesis=thesis,
            key_weaknesses=key_weaknesses,
            target_downside=target_downside,
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
        debt_str = f"{fundamental.debt_to_equity:.2f}" if fundamental.debt_to_equity is not None else "N/A"
        fund_str = (
            f"{fundamental.valuation} (P/E {pe_str}, EPS {eps_str}, growth {growth_str}, D/E {debt_str})"
        )

        return f"""Construct a bear thesis for {symbol} based on:

TECHNICAL: {tech_str}
SENTIMENT: {sent_str}
NEWS: {news_str}
FUNDAMENTAL: {fund_str}

Provide:
1. Bear thesis (3-4 sentences explaining why this stock has downside risk)
2. Key weaknesses (3-5 bullet points, each starting with '- ')
3. Target downside % (reasonable estimate or "N/A" if uncertain)

Format your response as:
THESIS: [your thesis here]
WEAKNESSES:
- [weakness 1]
- [weakness 2]
- [weakness 3]
DOWNSIDE: [percentage or N/A]"""

    def _extract_thesis(self, response: str) -> str:
        """Extract bear thesis from LLM response.

        Args:
            response: LLM response text

        Returns:
            Extracted thesis text
        """
        # Look for THESIS: section
        match = re.search(r"THESIS:\s*(.+?)(?=WEAKNESSES:|$)", response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Fallback: use first 3-4 sentences
        sentences = response.split(".")[:4]
        return ".".join(sentences).strip() + "."

    def _extract_key_weaknesses(self, response: str) -> list[str]:
        """Extract key weaknesses from LLM response.

        Args:
            response: LLM response text

        Returns:
            List of weakness bullet points
        """
        # Look for WEAKNESSES: section
        match = re.search(r"WEAKNESSES:\s*(.+?)(?=DOWNSIDE:|$)", response, re.DOTALL | re.IGNORECASE)
        if not match:
            # Fallback: look for bullet points anywhere
            bullets = re.findall(r"[-•]\s*(.+)", response)
            return [b.strip() for b in bullets[:5]] if bullets else []

        weaknesses_text = match.group(1).strip()
        bullets = re.findall(r"[-•]\s*(.+)", weaknesses_text)

        return [b.strip() for b in bullets[:5]] if bullets else []

    def _extract_target_downside(self, response: str) -> float | None:
        """Extract target downside percentage from LLM response.

        Args:
            response: LLM response text

        Returns:
            Downside percentage as float or None if not available
        """
        # Look for DOWNSIDE: section
        match = re.search(r"DOWNSIDE:\s*(.+)", response, re.IGNORECASE)
        if not match:
            return None

        downside_text = match.group(1).strip()

        # Check for N/A or similar
        if re.search(r"n/?a|not\s+available|uncertain|unknown", downside_text, re.IGNORECASE):
            return None

        # Extract percentage number
        num_match = re.search(r"(\d+(?:\.\d+)?)\s*%?", downside_text)
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
        """Calculate confidence in bear case.

        Args:
            technical: Technical analysis result
            sentiment: Sentiment analysis result
            _news: News analysis result (unused, for API consistency)
            fundamental: Fundamental analysis result

        Returns:
            Confidence score (0.0-1.0)
        """
        confidence = 0.5  # Base confidence

        # Technical boost/penalty (INVERTED from bullish)
        if technical.signal == Signal.SELL:
            confidence += 0.15
        elif technical.signal == Signal.BUY:
            confidence -= 0.2

        # Sentiment boost/penalty (INVERTED from bullish)
        if sentiment.sentiment_score < -0.3:
            confidence += 0.1
        elif sentiment.sentiment_score > 0.3:
            confidence -= 0.15

        # Fundamental boost/penalty (INVERTED from bullish)
        if fundamental.valuation == "OVERVALUED":
            confidence += 0.1
        elif fundamental.valuation == "UNDERVALUED":
            confidence -= 0.1

        # High debt boost (bearish signal)
        if fundamental.debt_to_equity and fundamental.debt_to_equity > 2.0:
            confidence += 0.05

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, confidence))

    def __repr__(self) -> str:
        """String representation."""
        return f"BearishResearcher(llm={self.llm.provider}/{self.llm.model})"
