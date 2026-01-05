"""Trader Agent for final decision making."""

from loguru import logger
from pydantic import BaseModel

from src.agents.news import NewsAnalysis
from src.agents.sentiment import SentimentAnalysis
from src.agents.technical import TechnicalAnalysis
from src.models.llm import LLMClient
from src.strategies.momentum import Signal


class TradingDecision(BaseModel):
    """Final trading decision."""

    action: Signal
    confidence: float
    reasoning: str
    risk_level: str


class TraderAgent:
    """Agent that synthesizes all analyses to make trading decisions."""

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize trader agent.

        Args:
            llm_client: LLM client for decision synthesis
        """
        self.llm = llm_client
        logger.info("Initialized TraderAgent")

    def decide(
        self,
        symbol: str,
        technical: TechnicalAnalysis,
        sentiment: SentimentAnalysis,
        news: NewsAnalysis,
    ) -> TradingDecision:
        """Make final trading decision based on all analyses.

        Args:
            symbol: Stock ticker symbol
            technical: Technical analysis results
            sentiment: Sentiment analysis results
            news: News analysis results

        Returns:
            TradingDecision with action and reasoning
        """
        logger.info(f"Making trading decision for {symbol}")

        prompt = f"""You are a professional trader making a decision for {symbol}.

TECHNICAL ANALYSIS:
Signal: {technical.signal.value}
RSI: {technical.rsi:.2f}
MACD Histogram: {technical.macd_hist:.4f}
Confidence: {technical.confidence:.2f}
Analysis: {technical.interpretation}

SENTIMENT ANALYSIS:
Overall: {sentiment.overall_sentiment}
Score: {sentiment.sentiment_score:.2f}
Articles: {sentiment.article_count}
Summary: {sentiment.summary}

NEWS ANALYSIS:
Key Themes: {", ".join(news.key_themes)}
Impact: {news.impact_assessment}
Recommendation: {news.recommendation}

Based on these three independent analyses, make your trading decision:
1. Action: BUY, SELL, or HOLD
2. Confidence: 0.0-1.0 (how confident in this decision)
3. Risk Level: LOW, MEDIUM, or HIGH
4. Reasoning: 2-3 sentences explaining your decision

Consider agreement/disagreement between signals. Higher agreement = higher confidence.
"""

        system_prompt = (
            "You are an experienced trader who synthesizes technical, sentiment, "
            "and news analysis to make informed trading decisions. Be decisive but cautious."
        )

        response = self.llm.complete(prompt, system=system_prompt, temperature=0.5)

        action = self._extract_action(response, technical.signal)
        confidence = self._extract_confidence(response, technical, sentiment)
        risk_level = self._extract_risk_level(response, confidence)

        logger.info(f"Decision: {action.value} (confidence={confidence:.2f}, risk={risk_level})")

        return TradingDecision(
            action=action,
            confidence=confidence,
            reasoning=response,
            risk_level=risk_level,
        )

    def _extract_action(self, response: str, technical_signal: Signal) -> Signal:
        """Extract trading action from response.

        Args:
            response: LLM response text
            technical_signal: Fallback technical signal

        Returns:
            Trading signal
        """
        response_lower = response.lower()

        if "action: buy" in response_lower or "decision: buy" in response_lower:
            return Signal.BUY
        if "action: sell" in response_lower or "decision: sell" in response_lower:
            return Signal.SELL
        if "action: hold" in response_lower or "decision: hold" in response_lower:
            return Signal.HOLD

        for line in response.split("\n"):
            if "buy" in line.lower() and len(line) < 50:
                return Signal.BUY
            if "sell" in line.lower() and len(line) < 50:
                return Signal.SELL

        logger.warning(f"Could not extract action, using technical signal: {technical_signal}")
        return technical_signal

    def _extract_confidence(
        self,
        response: str,
        technical: TechnicalAnalysis,
        sentiment: SentimentAnalysis,
    ) -> float:
        """Extract or calculate confidence score.

        Args:
            response: LLM response text
            technical: Technical analysis
            sentiment: Sentiment analysis

        Returns:
            Confidence score (0.0-1.0)
        """
        response_lower = response.lower()

        for line in response.split("\n"):
            if "confidence" in line.lower():
                try:
                    parts = line.split(":")
                    if len(parts) > 1:
                        value = float(parts[1].strip().split()[0])
                        if 0.0 <= value <= 1.0:
                            return value
                except (ValueError, IndexError):
                    continue

        base_confidence = technical.confidence

        if abs(sentiment.sentiment_score) > 0.3:
            sentiment_boost = abs(sentiment.sentiment_score) * 0.2
            base_confidence = min(base_confidence + sentiment_boost, 1.0)

        if "high confidence" in response_lower or "strongly" in response_lower:
            return min(base_confidence + 0.1, 1.0)
        if "low confidence" in response_lower or "uncertain" in response_lower:
            return max(base_confidence - 0.1, 0.0)

        return base_confidence

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
