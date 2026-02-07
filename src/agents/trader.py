"""Trader Agent for final decision making."""

from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field

from src.agents.bearish_researcher import BearishResearchAnalysis
from src.agents.bullish_researcher import BullishResearchAnalysis
from src.agents.comparative import ComparativeAnalysis
from src.agents.fundamental import FundamentalAnalysis
from src.agents.news import NewsAnalysis
from src.agents.sentiment import SentimentAnalysis
from src.agents.technical import TechnicalAnalysis
from src.models.llm import LLMClient
from src.models.providers.base import StructuredOutputError
from src.prompts import PromptLoader
from src.strategies.signal import Signal


class TraderLLMResponse(BaseModel):
    """LLM response structure for trading decision."""

    action: Literal["BUY", "SELL", "HOLD"] = Field(description="Trading action")
    confidence: float = Field(description="Confidence in the decision (0.0-1.0)", ge=0.0, le=1.0)
    risk_level: Literal["LOW", "MEDIUM", "HIGH"] = Field(description="Risk level of the trade")
    reasoning: list[str] = Field(description="1-2 punchy sentences explaining the decision")


class TradingDecision(BaseModel):
    """Final trading decision."""

    action: Signal
    confidence: float
    reasoning: list[str]
    risk_level: str
    owns_position: bool = False
    position_qty: float | None = None

    @property
    def display_action(self) -> str:
        """User-friendly action label based on portfolio context."""
        if self.action == Signal.HOLD and not self.owns_position:
            return "WAIT"
        return self.action.value


class TraderAgent:
    """Agent that synthesizes all analyses to make trading decisions."""

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize trader agent.

        Args:
            llm_client: LLM client for decision synthesis
        """
        self.llm = llm_client
        self._prompts = PromptLoader("trader")
        logger.info("Initialized TraderAgent")

    async def decide(
        self,
        symbol: str,
        technical: TechnicalAnalysis,
        sentiment: SentimentAnalysis,
        news: NewsAnalysis,
        fundamental: FundamentalAnalysis | None,
        bullish: BullishResearchAnalysis,
        bearish: BearishResearchAnalysis,
        comparative: ComparativeAnalysis | None = None,
        owns_position: bool = False,
        position_qty: float | None = None,
        sector_context: str | None = None,
    ) -> TradingDecision:
        """Make final trading decision based on all analyses.

        Args:
            symbol: Stock ticker symbol
            technical: Technical analysis results
            sentiment: Sentiment analysis results
            news: News analysis results
            fundamental: Fundamental analysis results (None if unavailable due to API rate limit)
            bullish: Bullish research analysis
            bearish: Bearish research analysis
            comparative: Comparative analysis results (optional)
            owns_position: Whether user owns this stock
            position_qty: Number of shares owned (if any)
            sector_context: Formatted sector rotation context (optional)

        Returns:
            TradingDecision with action and reasoning
        """
        logger.info(f"Making trading decision for {symbol} (owns={owns_position}, qty={position_qty})")

        if owns_position:
            portfolio_section = self._prompts.load(
                "section_portfolio_holding",
                symbol=symbol,
                position_qty=position_qty,
            )
        else:
            portfolio_section = self._prompts.load("section_portfolio_no_holding", symbol=symbol)

        fundamental_section = self._build_fundamental_section(fundamental)
        comparative_section = self._build_comparative_section(comparative)
        sector_rotation_section = self._build_sector_rotation_section(sector_context)

        prompt = self._prompts.load(
            "user_base",
            symbol=symbol,
            portfolio_section=portfolio_section,
            technical_signal=technical.signal.value,
            rsi=f"{technical.rsi:.2f}" if technical.rsi is not None else "N/A",
            macd_hist=f"{technical.macd_hist:.4f}" if technical.macd_hist is not None else "N/A",
            technical_confidence=f"{technical.confidence:.2f}",
            technical_interpretation=technical.interpretation,
            sentiment_overall=sentiment.overall_sentiment,
            sentiment_score=f"{sentiment.sentiment_score:.2f}",
            sentiment_article_count=sentiment.article_count,
            sentiment_summary=sentiment.summary,
            news_themes=", ".join(news.key_themes),
            news_impact=news.impact_assessment,
            news_recommendation=news.recommendation,
            fundamental_section=fundamental_section,
            bullish_thesis=bullish.thesis,
            bullish_strengths=", ".join(bullish.key_strengths),
            bullish_upside=f"{bullish.target_upside:.1f}%" if bullish.target_upside is not None else "N/A",
            bullish_confidence=f"{bullish.confidence:.2f}",
            bearish_thesis=bearish.thesis,
            bearish_weaknesses=", ".join(bearish.key_weaknesses),
            bearish_downside=f"{bearish.target_downside:.1f}%"
            if bearish.target_downside is not None
            else "N/A",
            bearish_confidence=f"{bearish.confidence:.2f}",
            comparative_section=comparative_section,
            sector_rotation_section=sector_rotation_section,
        )

        system_prompt = self._prompts.load("system")

        try:
            llm_response = await self.llm.astructured(
                prompt, TraderLLMResponse, system=system_prompt, temperature=0.5
            )
            action = Signal(llm_response.action)
            confidence = llm_response.confidence
            risk_level = llm_response.risk_level
            reasoning = llm_response.reasoning
        except StructuredOutputError as e:
            logger.warning(f"Structured output failed, falling back to text parsing: {e}")
            response = await self.llm.acomplete(prompt, system=system_prompt, temperature=0.5)
            action = self._extract_action(response, technical.signal)
            confidence = self._extract_confidence(response, technical, sentiment, bullish, bearish, action)
            risk_level = self._extract_risk_level(response, confidence)
            reasoning = [response]  # Wrap fallback text in list

        logger.info(f"Decision: {action.value} (confidence={confidence:.2f}, risk={risk_level})")

        return TradingDecision(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            risk_level=risk_level,
            owns_position=owns_position,
            position_qty=position_qty,
        )

    def _build_fundamental_section(self, fundamental: FundamentalAnalysis | None) -> str:
        """Build fundamental analysis section for prompt.

        Args:
            fundamental: Fundamental analysis results (optional)

        Returns:
            Formatted section string
        """
        if not fundamental:
            return self._prompts.load("section_fundamental_unavailable")

        return self._prompts.load(
            "section_fundamental",
            valuation=fundamental.valuation,
            pe_ratio=fundamental.pe_ratio if fundamental.pe_ratio is not None else "N/A",
            eps=f"${fundamental.eps}" if fundamental.eps is not None else "N/A",
            revenue_growth=(
                f"{fundamental.revenue_growth_yoy * 100:.1f}%"
                if fundamental.revenue_growth_yoy is not None
                else "N/A"
            ),
            earnings_growth=(
                f"{fundamental.earnings_growth_yoy * 100:.1f}%"
                if fundamental.earnings_growth_yoy is not None
                else "N/A"
            ),
            debt_to_equity=fundamental.debt_to_equity if fundamental.debt_to_equity is not None else "N/A",
            current_ratio=fundamental.current_ratio if fundamental.current_ratio is not None else "N/A",
            confidence=f"{fundamental.confidence:.2f}",
            interpretation=fundamental.interpretation,
        )

    def _build_comparative_section(self, comparative: ComparativeAnalysis | None) -> str:
        """Build comparative analysis section for prompt.

        Args:
            comparative: Comparative analysis results (optional)

        Returns:
            Formatted section string
        """
        if not comparative:
            return ""

        return self._prompts.load(
            "section_comparative",
            relative_valuation=comparative.relative_valuation.value,
            sector_etf=comparative.sector_etf,
            pe_vs_sector=f"{comparative.pe_vs_sector:.2f}x" if comparative.pe_vs_sector else "N/A",
            pe_vs_market=f"{comparative.pe_vs_market:.2f}x" if comparative.pe_vs_market else "N/A",
            perf_vs_sector_ytd=(
                f"{comparative.perf_vs_sector_ytd:+.1f}%"
                if comparative.perf_vs_sector_ytd is not None
                else "N/A"
            ),
            perf_vs_sector_3m=(
                f"{comparative.perf_vs_sector_3m:+.1f}%"
                if comparative.perf_vs_sector_3m is not None
                else "N/A"
            ),
            perf_vs_market_ytd=(
                f"{comparative.perf_vs_market_ytd:+.1f}%"
                if comparative.perf_vs_market_ytd is not None
                else "N/A"
            ),
            perf_vs_market_3m=(
                f"{comparative.perf_vs_market_3m:+.1f}%"
                if comparative.perf_vs_market_3m is not None
                else "N/A"
            ),
            confidence=f"{comparative.confidence:.2f}",
            interpretation=comparative.interpretation,
        )

    def _build_sector_rotation_section(self, sector_context: str | None) -> str:
        """Build sector rotation section for prompt.

        Args:
            sector_context: Formatted sector rotation context (optional)

        Returns:
            Formatted section string (empty if None)
        """
        if not sector_context:
            return ""

        # Parse leading/lagging from context (first two lines)
        lines = sector_context.strip().split("\n")
        leading = ""
        lagging = ""
        details_lines = []
        for line in lines:
            if line.startswith("Leading Sectors:"):
                leading = line.replace("Leading Sectors: ", "")
            elif line.startswith("Lagging Sectors:"):
                lagging = line.replace("Lagging Sectors: ", "")
            elif line.strip():
                details_lines.append(line)

        return self._prompts.load(
            "section_sector_rotation",
            leading_sectors=leading,
            lagging_sectors=lagging,
            sector_details="\n".join(details_lines),
        )

    def _extract_action(self, response: str, technical_signal: Signal) -> Signal:
        """Extract trading action from response.

        Args:
            response: LLM response text
            technical_signal: Fallback technical signal

        Returns:
            Trading signal
        """
        import re

        # Normalize: remove markdown bold/italic, collapse whitespace
        normalized = re.sub(r"\*+", "", response.lower())
        normalized = re.sub(r"\s+", " ", normalized)

        # Pattern 1: "action: buy/sell/hold" or "action : buy" (with optional space)
        action_patterns = [
            r"action\s*:\s*(buy|sell|hold)",
            r"decision\s*:\s*(buy|sell|hold)",
            r"1\.\s*action\s*:\s*(buy|sell|hold)",
        ]

        for pattern in action_patterns:
            match = re.search(pattern, normalized)
            if match:
                action_str = match.group(1).upper()
                return Signal(action_str)

        # Pattern 2: Look for explicit decision statements
        decision_patterns = [
            r"my decision is (buy|sell|hold)",
            r"i recommend (buy|sell|hold)",
            r"recommendation:\s*(buy|sell|hold)",
        ]

        for pattern in decision_patterns:
            match = re.search(pattern, normalized)
            if match:
                action_str = match.group(1).upper()
                return Signal(action_str)

        logger.warning(f"Could not extract action, using technical signal: {technical_signal}")
        return technical_signal

    def _extract_confidence(
        self,
        response: str,
        technical: TechnicalAnalysis,
        sentiment: SentimentAnalysis,
        bullish: BullishResearchAnalysis,
        bearish: BearishResearchAnalysis,
        action: Signal,
    ) -> float:
        """Extract or calculate confidence score.

        Args:
            response: LLM response text
            technical: Technical analysis
            sentiment: Sentiment analysis
            bullish: Bullish research analysis
            bearish: Bearish research analysis
            action: Trading action (affects bull/bear weighting)

        Returns:
            Confidence score (0.0-1.0)
        """
        parsed = self._parse_confidence_from_response(response)
        if parsed is not None:
            return parsed

        bull_weight, bear_weight = self._calculate_bull_bear_weights(action, bullish, bearish)
        base_confidence = (technical.confidence + bull_weight + bear_weight) / 3

        if abs(sentiment.sentiment_score) > 0.3:
            sentiment_boost = abs(sentiment.sentiment_score) * 0.2
            base_confidence = min(base_confidence + sentiment_boost, 1.0)

        response_lower = response.lower()
        if "high confidence" in response_lower or "strongly" in response_lower:
            return min(base_confidence + 0.1, 1.0)
        if "low confidence" in response_lower or "uncertain" in response_lower:
            return max(base_confidence - 0.1, 0.0)

        return base_confidence

    def _parse_confidence_from_response(self, response: str) -> float | None:
        """Parse confidence value from LLM response text."""
        for line in response.split("\n"):
            if "confidence" not in line.lower():
                continue
            try:
                parts = line.split(":")
                if len(parts) > 1:
                    value = float(parts[1].strip().split()[0])
                    if 0.0 <= value <= 1.0:
                        return value
            except (ValueError, IndexError):
                continue
        return None

    def _calculate_bull_bear_weights(
        self,
        action: Signal,
        bullish: BullishResearchAnalysis,
        bearish: BearishResearchAnalysis,
    ) -> tuple[float, float]:
        """Calculate bull/bear weights based on trading action."""
        if action == Signal.BUY:
            # BUY: high bullish = good, high bearish = bad
            return bullish.confidence, 1 - bearish.confidence
        if action == Signal.SELL:
            # SELL: high bearish = good, high bullish = bad
            return 1 - bullish.confidence, bearish.confidence
        # HOLD: neutral weights
        return 0.5, 0.5

    def _extract_risk_level(self, response: str, confidence: float) -> str:
        """Determine risk level from response or confidence.

        Args:
            response: LLM response text
            confidence: Confidence score

        Returns:
            Risk level (LOW/MEDIUM/HIGH)
        """
        response_lower = response.lower()

        if "risk: high" in response_lower or "high risk" in response_lower:
            return "HIGH"
        if "risk: low" in response_lower or "low risk" in response_lower:
            return "LOW"
        if "risk: medium" in response_lower or "medium risk" in response_lower:
            return "MEDIUM"

        if confidence >= 0.75:
            return "LOW"
        if confidence >= 0.5:
            return "MEDIUM"
        return "HIGH"

    def __repr__(self) -> str:
        """String representation."""
        return f"TraderAgent(llm={self.llm.provider})"
