"""Base researcher agent with shared logic for bullish/bearish analysis."""

import re
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel

from src.agents.comparative import ComparativeAnalysis
from src.agents.fundamental import FundamentalAnalysis
from src.agents.news import NewsAnalysis
from src.agents.sentiment import SentimentAnalysis
from src.agents.technical import TechnicalAnalysis
from src.models.llm import LLMClient
from src.models.providers.base import StructuredOutputError
from src.prompts import PromptLoader

if TYPE_CHECKING:
    from src.agents.trump import TrumpAnalysis


class ResearchDirection(StrEnum):
    """Research direction for thesis construction."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


# Section name mapping for direction-based regex patterns
SECTION_NAMES = {
    ResearchDirection.BULLISH: {
        "thesis_anchor": "STRENGTHS:",
        "key_points": "STRENGTHS:",
        "target": "UPSIDE:",
    },
    ResearchDirection.BEARISH: {
        "thesis_anchor": "WEAKNESSES:",
        "key_points": "WEAKNESSES:",
        "target": "DOWNSIDE:",
    },
}


class BaseResearcher(ABC):
    """Base researcher agent with shared logic for thesis construction."""

    def __init__(self, llm_client: LLMClient, direction: ResearchDirection, prompt_dir: str) -> None:
        """Initialize base researcher.

        Args:
            llm_client: LLM client for generating thesis
            direction: Research direction (BULLISH or BEARISH)
            prompt_dir: Prompt directory name
        """
        self.llm = llm_client
        self.direction = direction
        self._prompts = PromptLoader(prompt_dir)
        logger.info(f"Initialized {self.__class__.__name__}")

    @property
    @abstractmethod
    def llm_response_model(self) -> type[BaseModel]:
        """LLM response model type - must be implemented by subclass."""
        ...

    async def analyze(
        self,
        symbol: str,
        technical: TechnicalAnalysis,
        sentiment: SentimentAnalysis,
        news: NewsAnalysis,
        fundamental: FundamentalAnalysis | None,
        comparative: ComparativeAnalysis | None = None,
        trump_analysis: "TrumpAnalysis | None" = None,
    ) -> BaseModel:
        """Construct thesis from all analyses.

        Args:
            symbol: Stock ticker symbol
            technical: Technical analysis result
            sentiment: Sentiment analysis result
            news: News analysis result
            fundamental: Fundamental analysis result (None if unavailable due to API rate limit)
            comparative: Comparative analysis result (optional)
            trump_analysis: Trump social media analysis (optional)

        Returns:
            Analysis result with thesis, key points, target, and confidence
        """
        direction_str = "bull" if self.direction == ResearchDirection.BULLISH else "bear"
        logger.info(f"Constructing {direction_str} thesis for {symbol}")

        prompt_vars = self._build_prompt_vars(
            symbol, technical, sentiment, news, fundamental, comparative, trump_analysis
        )

        prompt = self._prompts.load("user", **prompt_vars)
        system_prompt = self._prompts.load("system")

        try:
            llm_response = await self.llm.astructured(
                prompt, self.llm_response_model, system=system_prompt, temperature=0.5
            )
            # Access dynamic attributes - type checker sees BaseModel but runtime has specific fields
            thesis = llm_response.thesis
            key_points = getattr(llm_response, self._get_key_points_field())
            target = getattr(llm_response, self._get_target_field())
        except StructuredOutputError as e:
            logger.warning(f"Structured output failed, falling back to text parsing: {e}")
            response = await self.llm.acomplete(prompt, system=system_prompt, temperature=0.5)
            thesis = self._extract_thesis(response)
            key_points = self._extract_key_points(response)
            target = self._extract_target(response)

        confidence = self._calculate_confidence(technical, sentiment, news, fundamental)

        points_label = "strengths" if self.direction == ResearchDirection.BULLISH else "weaknesses"
        target_label = "upside" if self.direction == ResearchDirection.BULLISH else "downside"

        logger.info(
            f"{direction_str.capitalize()} thesis for {symbol}: {len(key_points)} {points_label}, "
            f"{target_label}={target}, confidence={confidence:.2f}"
        )

        return self._build_analysis(thesis, key_points, target, confidence)

    @abstractmethod
    def _build_analysis(
        self, thesis: str, key_points: list[str], target: float | None, confidence: float
    ) -> BaseModel:
        """Build analysis result model - must be implemented by subclass.

        Args:
            thesis: Thesis text
            key_points: Key strengths or weaknesses
            target: Target upside or downside percentage
            confidence: Confidence score

        Returns:
            Analysis result model instance
        """
        ...

    @abstractmethod
    def _calculate_confidence(
        self,
        technical: TechnicalAnalysis,
        sentiment: SentimentAnalysis,
        news: NewsAnalysis,
        fundamental: FundamentalAnalysis | None,
    ) -> float:
        """Calculate confidence score - must be implemented by subclass with direction-specific logic.

        Args:
            technical: Technical analysis result
            sentiment: Sentiment analysis result
            news: News analysis result
            fundamental: Fundamental analysis result

        Returns:
            Confidence score (0.0-1.0)
        """
        ...

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
            debt_str = (
                f"{fundamental.debt_to_equity:.2f}" if fundamental.debt_to_equity is not None else "N/A"
            )
            fund_str = (
                f"{fundamental.valuation} (P/E {pe_str}, EPS {eps_str}, growth {growth_str}, D/E {debt_str})"
            )
        else:
            fund_str = "N/A (API rate limited)"

        # Comparative section
        comp_str = "N/A"
        if comparative:
            pe_vs_sector = (
                f"{comparative.pe_vs_sector:.2f}x" if comparative.pe_vs_sector is not None else "N/A"
            )
            perf_vs_sector = (
                f"{comparative.perf_vs_sector_3m:+.1f}%"
                if comparative.perf_vs_sector_3m is not None
                else "N/A"
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
        """Extract thesis from LLM response.

        Args:
            response: LLM response text

        Returns:
            Extracted thesis text
        """
        # Look for THESIS: section using direction-based anchor
        anchor = SECTION_NAMES[self.direction]["thesis_anchor"]
        match = re.search(rf"THESIS:\s*(.+?)(?={anchor}|$)", response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Fallback: use first 3-4 sentences
        sentences = response.split(".")[:4]
        return ".".join(sentences).strip() + "."

    def _extract_key_points(self, response: str) -> list[str]:
        """Extract key points from LLM response.

        Args:
            response: LLM response text

        Returns:
            List of key point bullet points
        """
        # Look for key points section using direction-based name
        section_name = SECTION_NAMES[self.direction]["key_points"]
        target_section = SECTION_NAMES[self.direction]["target"]
        match = re.search(
            rf"{section_name}\s*(.+?)(?={target_section}|$)", response, re.DOTALL | re.IGNORECASE
        )
        if not match:
            # Fallback: look for bullet points anywhere
            bullets = re.findall(r"[-•]\s*(.+)", response)
            return [b.strip() for b in bullets[:5]] if bullets else []

        points_text = match.group(1).strip()
        bullets = re.findall(r"[-•]\s*(.+)", points_text)

        return [b.strip() for b in bullets[:5]] if bullets else []

    def _extract_target(self, response: str) -> float | None:
        """Extract target percentage from LLM response.

        Args:
            response: LLM response text

        Returns:
            Target percentage as float or None if not available
        """
        # Look for target section using direction-based name
        target_section = SECTION_NAMES[self.direction]["target"]
        match = re.search(rf"{target_section}\s*(.+)", response, re.IGNORECASE)
        if not match:
            return None

        target_text = match.group(1).strip()

        # Check for N/A or similar
        if re.search(r"n/?a|not\s+available|uncertain|unknown", target_text, re.IGNORECASE):
            return None

        # Extract percentage number
        num_match = re.search(r"(\d+(?:\.\d+)?)\s*%?", target_text)
        if num_match:
            return float(num_match.group(1))

        return None

    def _get_key_points_field(self) -> str:
        """Get key points field name based on direction.

        Returns:
            Field name (key_strengths or key_weaknesses)
        """
        return "key_strengths" if self.direction == ResearchDirection.BULLISH else "key_weaknesses"

    def _get_target_field(self) -> str:
        """Get target field name based on direction.

        Returns:
            Field name (target_upside or target_downside)
        """
        return "target_upside" if self.direction == ResearchDirection.BULLISH else "target_downside"

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(llm={self.llm.provider}/{self.llm.model})"
