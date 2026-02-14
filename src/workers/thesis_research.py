"""Unified thesis research worker for bullish and bearish analysis."""

from typing import Literal

from loguru import logger
from pydantic import BaseModel

from src.agents.base_researcher import ResearchDirection
from src.agents.comparative import ComparativeAnalysis
from src.agents.fundamental import FundamentalAnalysis
from src.agents.news import NewsAnalysis
from src.agents.sentiment import SentimentAnalysis
from src.agents.technical import TechnicalAnalysis
from src.agents.thesis_researcher import (
    BearishConfidenceCalculator,
    BullishConfidenceCalculator,
    ConfidenceCalculator,
    ResearchAnalysis,
    ResearchLLMResponse,
)
from src.agents.trump import TrumpAnalysis
from src.models.llm import LLMClient
from src.models.providers.base import StructuredOutputError
from src.prompts import PromptLoader
from src.tools.models import ToolDefinition, ToolFunction, ToolParameter, ToolParametersSchema


class AnalysisInputs(BaseModel):
    """Container for all analysis inputs needed for thesis research."""

    technical: TechnicalAnalysis
    sentiment: SentimentAnalysis
    news: NewsAnalysis
    fundamental: FundamentalAnalysis | None
    comparative: ComparativeAnalysis | None = None
    trump_analysis: TrumpAnalysis | None = None

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


class ThesisResearchWorker:
    """Worker for thesis research - supports both bullish and bearish direction."""

    def __init__(self, llm_client: LLMClient, direction: Literal["bullish", "bearish"]) -> None:
        """Initialize thesis research worker.

        Args:
            llm_client: LLM client for generating thesis
            direction: Research direction (bullish or bearish)
        """
        self.llm = llm_client
        self.direction = ResearchDirection.BULLISH if direction == "bullish" else ResearchDirection.BEARISH
        prompt_dir = "bullish_researcher" if direction == "bullish" else "bearish_researcher"
        self._prompts = PromptLoader(prompt_dir)
        self._confidence_calculator = self._create_confidence_calculator()
        logger.info(f"Initialized ThesisResearchWorker (direction={direction})")

    def _create_confidence_calculator(self) -> ConfidenceCalculator:
        """Create direction-specific confidence calculator."""
        if self.direction == ResearchDirection.BULLISH:
            return BullishConfidenceCalculator()
        return BearishConfidenceCalculator()

    async def analyze(self, symbol: str, inputs: AnalysisInputs) -> ResearchAnalysis:
        """Construct thesis from all analyses.

        Args:
            symbol: Stock ticker symbol
            inputs: Container with all required analysis results

        Returns:
            ResearchAnalysis with thesis, key points, target, confidence
        """
        direction_str = "Bullish" if self.direction == ResearchDirection.BULLISH else "Bearish"
        logger.info(f"ThesisResearchWorker generating {direction_str} thesis for {symbol}")

        # Build prompt
        prompt = self._build_prompt(symbol, inputs)
        system = self._prompts.load("system")

        # Try structured output with fallback
        try:
            llm_response = await self.llm.astructured(
                prompt, ResearchLLMResponse, system=system, temperature=0.5
            )
            thesis = llm_response.thesis
            key_points = (
                llm_response.key_strengths
                if self.direction == ResearchDirection.BULLISH
                else llm_response.key_weaknesses
            )
            target = (
                llm_response.target_upside
                if self.direction == ResearchDirection.BULLISH
                else llm_response.target_downside
            )
        except StructuredOutputError as e:
            logger.opt(exception=True).warning(f"Structured output failed, falling back: {e}")
            response = await self.llm.acomplete(prompt, system=system, temperature=0.5)
            # Parse response (fallback)
            thesis = response[:500]  # First 500 chars as thesis
            key_points = []
            target = None

        # Calculate confidence
        confidence = self._calculate_confidence(
            inputs.technical, inputs.sentiment, inputs.news, inputs.fundamental
        )

        logger.info(
            f"{direction_str} thesis complete for {symbol}: "
            f"points={len(key_points)}, confidence={confidence:.2f}"
        )

        return ResearchAnalysis(
            direction=self.direction,
            thesis=thesis,
            key_points=key_points,
            target=target,
            confidence=confidence,
        )

    def _build_prompt(self, symbol: str, inputs: AnalysisInputs) -> str:
        """Build prompt for LLM.

        Args:
            symbol: Stock ticker
            inputs: Container with all analysis results

        Returns:
            Formatted prompt string
        """
        # Build analysis summaries
        technical_summary = (
            f"Signal: {inputs.technical.signal.value}, "
            f"RSI: {inputs.technical.rsi:.1f if inputs.technical.rsi else 'N/A'}, "
            f"Confidence: {inputs.technical.confidence:.2f}"
        )

        sentiment_summary = (
            f"Score: {inputs.sentiment.sentiment_score:.2f}, Confidence: {inputs.sentiment.confidence:.2f}"
        )

        news_summary = (
            f"{inputs.news.impact_assessment[:200]}..."
            if inputs.news.impact_assessment
            else "No news analysis"
        )

        fundamental_summary = (
            f"Valuation: {inputs.fundamental.valuation}, "
            f"P/E: {inputs.fundamental.pe_ratio:.2f if inputs.fundamental.pe_ratio else 'N/A'}, "
            f"Confidence: {inputs.fundamental.confidence:.2f}"
            if inputs.fundamental
            else "No fundamental data available"
        )

        comparative_summary = (
            f"Relative Valuation: {inputs.comparative.relative_valuation.value}, "
            f"Confidence: {inputs.comparative.confidence:.2f}"
            if inputs.comparative
            else "No comparative data available"
        )

        trump_summary = (
            f"Sentiment: {inputs.trump_analysis.sentiment}, "
            f"Confidence: {inputs.trump_analysis.confidence:.2f}"
            if inputs.trump_analysis
            else "No Trump analysis available"
        )

        return self._prompts.load(
            "user",
            symbol=symbol,
            technical_summary=technical_summary,
            sentiment_summary=sentiment_summary,
            news_summary=news_summary,
            fundamental_summary=fundamental_summary,
            comparative_summary=comparative_summary,
            trump_summary=trump_summary,
        )

    def _calculate_confidence(
        self,
        technical: TechnicalAnalysis,
        sentiment: SentimentAnalysis,
        news: NewsAnalysis,
        fundamental: FundamentalAnalysis | None,
    ) -> float:
        """Calculate confidence using direction-specific strategy."""
        return self._confidence_calculator.calculate(technical, sentiment, news, fundamental)

    def get_tool_definition(self) -> ToolDefinition:
        """Get tool definition for supervisor integration.

        Returns:
            Tool definition
        """
        direction_str = "bullish" if self.direction == ResearchDirection.BULLISH else "bearish"
        return ToolDefinition(
            type="function",
            function=ToolFunction(
                name=f"research_{direction_str}_thesis",
                description=f"Generate {direction_str} investment thesis from all analyses",
                parameters=ToolParametersSchema(
                    type="object",
                    properties={
                        "symbol": ToolParameter(type="string", description="Stock ticker symbol"),
                    },
                    required=["symbol"],
                ),
            ),
        )

    def __repr__(self) -> str:
        """String representation."""
        direction_str = "bullish" if self.direction == ResearchDirection.BULLISH else "bearish"
        return f"ThesisResearchWorker(direction={direction_str})"
