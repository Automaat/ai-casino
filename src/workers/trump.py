"""Trump social media analysis worker."""

import re
from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field

from src.agents.trump import COMPANY_TICKERS, MARKET_KEYWORDS, TrumpAnalysis
from src.data.truth_social import TruthPost
from src.models.llm import LLMClient
from src.models.providers.base import StructuredOutputError
from src.prompts import PromptLoader
from src.strategies.signal import Signal
from src.tools.models import ToolDefinition, ToolFunction, ToolParameter, ToolParametersSchema

# Text processing thresholds
MIN_PHRASE_LENGTH = 10  # Minimum character length for extracted phrases
MAX_PHRASE_LENGTH = 200  # Maximum character length for extracted phrases
POST_CONTENT_PREVIEW_LENGTH = 500  # Character limit for post content preview
MIN_CLEANED_LINE_LENGTH = 20  # Minimum length for cleaned summary lines


class TrumpLLMResponse(BaseModel):
    """LLM response structure for Trump analysis."""

    signal: Literal["BUY", "SELL", "HOLD"] = Field(description="Trading signal based on posts")
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="Overall sentiment of posts")
    confidence: float = Field(description="Confidence in the signal (0.0-1.0)", ge=0.0, le=1.0)
    interpretation: str = Field(description="Summary interpretation of the posts")


class TrumpWorker:
    """Worker for analyzing Trump's social media posts - stateless implementation."""

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize Trump worker.

        Args:
            llm_client: LLM client for analysis
        """
        self.llm = llm_client
        self._prompts = PromptLoader("trump")
        logger.info("Initialized TrumpWorker")

    async def analyze(self, posts: list[TruthPost]) -> TrumpAnalysis:
        """Analyze Trump posts for trading implications.

        Args:
            posts: List of Truth Social posts

        Returns:
            TrumpAnalysis with signal and interpretation
        """
        logger.info(f"TrumpWorker analyzing {len(posts)} posts")

        if not posts:
            logger.warning("No posts provided for Trump analysis")
            return TrumpAnalysis(
                market_relevant=False,
                signal=Signal.HOLD,
                mentioned_tickers=[],
                sentiment="neutral",
                confidence=0.0,
                key_phrases=[],
                interpretation="No recent posts to analyze",
                post_count=0,
            )

        # Extract tickers from all posts
        all_tickers = set()
        all_key_phrases = []
        market_relevant_posts = []

        for post in posts:
            tickers = self._extract_tickers(post.content)
            all_tickers.update(tickers)

            if self._is_market_relevant(post.content):
                market_relevant_posts.append(post)
                phrases = self._extract_key_phrases(post.content)
                all_key_phrases.extend(phrases)

        market_relevant = len(market_relevant_posts) > 0
        posts_text = self._format_posts(posts[:10])  # Limit to 10 most recent

        system_prompt = self._prompts.load("system")
        user_prompt = self._prompts.load("user", posts_text=posts_text)

        try:
            llm_response = await self.llm.astructured(
                user_prompt, TrumpLLMResponse, system=system_prompt, temperature=0.4, max_tokens=256
            )
            sentiment = llm_response.sentiment
            signal = Signal(llm_response.signal)
            confidence = llm_response.confidence
            interpretation = llm_response.interpretation
        except StructuredOutputError as e:
            logger.opt(exception=True).warning(f"Structured output failed, falling back to text parsing: {e}")
            response = await self.llm.acomplete(user_prompt, system=system_prompt, temperature=0.4)
            sentiment = self._extract_sentiment(response)
            signal = self._extract_signal(response)
            confidence = self._extract_confidence(response)
            interpretation = self._extract_interpretation(response)

        logger.info(f"Trump analysis complete: signal={signal}, confidence={confidence:.2f}")

        return TrumpAnalysis(
            market_relevant=market_relevant,
            signal=signal,
            mentioned_tickers=sorted(all_tickers),
            sentiment=sentiment,
            confidence=confidence,
            key_phrases=all_key_phrases[:10],
            interpretation=interpretation,
            post_count=len(posts),
        )

    def _is_market_relevant(self, text: str) -> bool:
        """Check if text contains market-relevant keywords."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in MARKET_KEYWORDS)

    def _extract_tickers(self, text: str) -> set[str]:
        """Extract stock tickers from text."""
        tickers = set()

        # Direct $TICKER mentions
        ticker_pattern = r"\$([A-Z]{1,5})\b"
        for match in re.finditer(ticker_pattern, text):
            tickers.add(match.group(1))

        # Company name mentions with word boundaries
        text_lower = text.lower()
        for company, ticker in COMPANY_TICKERS.items():
            pattern = r"\b" + re.escape(company) + r"\b"
            if re.search(pattern, text_lower):
                tickers.add(ticker)

        return tickers

    def _extract_key_phrases(self, text: str) -> list[str]:
        """Extract market-relevant phrases from text."""
        phrases = []
        text_lower = text.lower()

        for keyword in MARKET_KEYWORDS:
            if keyword in text_lower:
                # Find sentence containing keyword
                sentences = text.split(".")
                for sentence in sentences:
                    if keyword in sentence.lower():
                        cleaned = sentence.strip()
                        if MIN_PHRASE_LENGTH < len(cleaned) < MAX_PHRASE_LENGTH:
                            phrases.append(cleaned)
                        break

        return phrases[:5]

    def _format_posts(self, posts: list[TruthPost]) -> str:
        """Format posts for LLM prompt."""
        lines = []
        for i, post in enumerate(posts, 1):
            date_str = post.created_at.strftime("%Y-%m-%d %H:%M")
            content = (
                post.content[:POST_CONTENT_PREVIEW_LENGTH]
                if len(post.content) > POST_CONTENT_PREVIEW_LENGTH
                else post.content
            )
            lines.append(f"{i}. [{date_str}] {content}")
            lines.append(f"   (Likes: {post.likes}, Reposts: {post.reposts})")
            lines.append("")

        return "\n".join(lines)

    def _extract_sentiment(self, response: str) -> str:
        """Extract sentiment from LLM response."""
        response_lower = response.lower()
        if "positive" in response_lower:
            return "positive"
        if "negative" in response_lower:
            return "negative"
        return "neutral"

    def _extract_signal(self, response: str) -> Signal:
        """Extract trading signal from LLM response."""
        response_upper = response.upper()
        if "BUY" in response_upper and "SELL" not in response_upper:
            return Signal.BUY
        if "SELL" in response_upper and "BUY" not in response_upper:
            return Signal.SELL
        return Signal.HOLD

    def _extract_confidence(self, response: str) -> float:
        """Extract confidence score from LLM response."""
        # Look for patterns like "0.7", "confidence: 0.8", "70%"
        patterns = [
            r"confidence[:\s]+(\d+\.?\d*)",
            r"(\d+\.?\d*)\s*confidence",
            r"(\d+)%",
            r"\b(0\.\d+)\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                value = float(match.group(1))
                if value > 1:
                    value = value / 100  # Convert percentage
                return min(max(value, 0.0), 1.0)

        # Default based on sentiment strength
        if "strong" in response.lower() or "clear" in response.lower():
            return 0.7
        return 0.5

    def _extract_interpretation(self, response: str) -> str:
        """Extract interpretation from LLM response."""
        lines = response.split("\n")
        for line in lines:
            line_lower = line.lower()
            is_interpretation = (
                "interpret" in line_lower or "analysis" in line_lower or "summary" in line_lower
            )
            if is_interpretation and ":" in line:
                return line.split(":", 1)[1].strip()

        # Return last substantial line
        for line in reversed(lines):
            cleaned = line.strip()
            if len(cleaned) > MIN_CLEANED_LINE_LENGTH and not cleaned[0].isdigit():
                return cleaned[:MAX_PHRASE_LENGTH]

        return response[:200]

    def get_tool_definition(self) -> ToolDefinition:
        """Get tool definition for supervisor integration.

        Returns:
            Tool definition
        """
        return ToolDefinition(
            type="function",
            function=ToolFunction(
                name="analyze_trump",
                description="Analyze Trump social media posts for trading implications",
                parameters=ToolParametersSchema(
                    type="object",
                    properties={
                        "posts": ToolParameter(
                            type="array",
                            items={"type": "object"},
                            description="List of Truth Social posts",
                        ),
                    },
                    required=["posts"],
                ),
            ),
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"TrumpWorker(llm={self.llm.provider})"
