"""LLM-based ticker extraction from Reddit content."""

from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel, Field

from src.daemon.config.reddit import RedditScraperConfig
from src.daemon.events import Sentiment
from src.data.reddit import RedditComment, RedditPost, TickerMention
from src.prompts import PromptLoader

if TYPE_CHECKING:
    from src.models.llm import LLMClient


class TickerExtractionResponse(BaseModel):
    """LLM response for ticker extraction."""

    mentions: list[TickerMention] = Field(default_factory=list, description="List of ticker mentions found")


class RedditTickerExtractor:
    """Extract stock tickers from Reddit posts and comments using LLM."""

    def __init__(
        self,
        llm_client: "LLMClient",
        config: RedditScraperConfig,
    ) -> None:
        """Initialize Reddit ticker extractor.

        Args:
            llm_client: LLM client for structured output
            config: Reddit scraper configuration
        """
        self.llm_client = llm_client
        self.config = config
        self.prompt_loader = PromptLoader(agent_name="reddit")

    async def extract_tickers(
        self,
        post: RedditPost,
        comments: list[RedditComment] | None = None,
    ) -> list[TickerMention]:
        """Extract ticker mentions from post and comments.

        Args:
            post: RedditPost to extract from
            comments: Optional list of top comments (uses top 3 if provided)

        Returns:
            List of TickerMention objects with high confidence (>0.7)
        """
        if not self.config.use_llm_extraction:
            return []

        # Build input text (limit to max_tokens)
        input_parts = [
            f"Title: {post.title}",
        ]

        if post.body:
            input_parts.append(f"Body: {post.body}")

        if comments:
            top_comments = sorted(comments, key=lambda c: c.score, reverse=True)[:3]
            for idx, comment in enumerate(top_comments, 1):
                input_parts.append(f"Comment {idx}: {comment.body}")

        input_text = "\n\n".join(input_parts)

        # Truncate if too long (rough token estimate: 1 token ~= 4 chars)
        max_chars = self.config.extraction_max_tokens * 4
        if len(input_text) > max_chars:
            input_text = input_text[:max_chars] + "..."
            logger.debug(
                f"Truncated input to {max_chars} chars (~{self.config.extraction_max_tokens} tokens)"
            )

        # Load prompt template
        try:
            prompt_template = self.prompt_loader.load("ticker_extraction.txt")
            prompt = prompt_template.format(
                title=post.title,
                body=post.body or "(no body)",
                comments="\n".join(f"{idx}. {c.body[:200]}" for idx, c in enumerate(comments[:3], 1))
                if comments
                else "(no comments)",
            )
        except Exception:
            logger.opt(exception=True).warning("Failed to load prompt template, using fallback")
            prompt = f"""Extract stock tickers mentioned in this Reddit post.

For each ticker:
- Symbol (e.g., AAPL, TSLA) - uppercase, 1-5 chars
- Sentiment: BULLISH (positive/buying), BEARISH (negative/selling), NEUTRAL (mention only)
- Context: brief snippet showing why ticker was mentioned (max 50 chars)
- Confidence: 0.0-1.0 (how certain this is a stock ticker)

Ignore:
- Common abbreviations (CEO, IPO, DD, YOLO, ATH)
- Non-stock mentions (company names without ticker context)
- Sarcasm or jokes (e.g., "$ROPE", "$MOON")

{input_text}

Extract tickers:"""

        try:
            # Use LLM structured output
            response = await self.llm_client.astructured(
                prompt=prompt,
                response_model=TickerExtractionResponse,
                temperature=self.config.extraction_temperature,
            )

            # Filter by confidence threshold
            high_confidence_mentions = [
                mention
                for mention in response.mentions
                if mention.confidence >= self.config.extraction_min_confidence
            ]

            # Validate symbols (uppercase, 1-5 chars, alphanumeric)
            valid_mentions = [
                mention for mention in high_confidence_mentions if self._is_valid_symbol(mention.symbol)
            ]

            if len(valid_mentions) < len(response.mentions):
                filtered_count = len(response.mentions) - len(valid_mentions)
                logger.debug(f"Filtered out {filtered_count} mentions (low confidence or invalid symbols)")

            logger.info(f"Extracted {len(valid_mentions)} tickers from post {post.id} (r/{post.subreddit})")
            return valid_mentions

        except TimeoutError:
            logger.warning(f"LLM extraction timeout for post {post.id}")
            return []
        except Exception:
            logger.opt(exception=True).warning(f"Failed to extract tickers from post {post.id}")
            return []

    def _is_valid_symbol(self, symbol: str) -> bool:
        """Validate stock ticker symbol.

        Args:
            symbol: Ticker symbol to validate

        Returns:
            True if valid, False otherwise
        """
        # Uppercase, 1-5 alphanumeric chars (allow . for BRK.B)
        if not symbol or not (1 <= len(symbol) <= 6):
            return False

        # Must be mostly uppercase letters (allow one . for class shares)
        if not symbol.replace(".", "").replace("-", "").isalnum():
            return False

        if not symbol[0].isupper():
            return False

        # Exclude common false positives
        false_positives = {
            "CEO",
            "CFO",
            "IPO",
            "ETF",
            "GDP",
            "SEC",
            "FBI",
            "USA",
            "NYSE",
            "NASDAQ",
            "WSB",
            "DD",
            "YOLO",
            "FOMO",
            "FUD",
            "HODL",
            "EOD",
            "ATH",
            "ATL",
            "OTM",
            "ITM",
            "IV",
            "DTE",
            "EPS",
            "PE",
            "PM",
            "AM",
            "OP",
            "IMO",
            "IMHO",
            "TL",
            "DR",
            "TLDR",
            "EDIT",
            "PSA",
            "FYI",
            "LMAO",
            "LOL",
            "WTF",
            "BTW",
            "AMA",
            "RIP",
            "ASAP",
            "MON",
            "TUE",
            "WED",
            "THU",
            "FRI",
            "SAT",
            "SUN",
            "JAN",
            "FEB",
            "MAR",
            "APR",
            "MAY",
            "JUN",
            "JUL",
            "AUG",
            "SEP",
            "OCT",
            "NOV",
            "DEC",
            "BUY",
            "SELL",
            "HOLD",
            "LONG",
            "SHORT",
            "CALL",
            "PUT",
            "ALL",
            "NEW",
            "OLD",
            "BIG",
            "LOW",
            "HIGH",
            "UP",
            "DOWN",
            "OUT",
            "RH",
            "MOON",
            "ROPE",
        }

        if symbol.upper() in false_positives:
            logger.debug(f"Filtered false positive: {symbol}")
            return False

        return True

    def _normalize_sentiment(self, sentiment: str) -> str:
        """Normalize sentiment to enum value.

        Args:
            sentiment: Raw sentiment string

        Returns:
            Normalized sentiment (BULLISH, BEARISH, NEUTRAL)
        """
        sentiment_upper = sentiment.upper()

        if sentiment_upper in {Sentiment.BULLISH, Sentiment.BEARISH, Sentiment.NEUTRAL}:
            return sentiment_upper

        # Fallback mappings
        if sentiment_upper in {"POSITIVE", "BUY", "LONG"}:
            return Sentiment.BULLISH
        if sentiment_upper in {"NEGATIVE", "SELL", "SHORT"}:
            return Sentiment.BEARISH

        return Sentiment.NEUTRAL

    def __repr__(self) -> str:
        """Return string representation."""
        return f"RedditTickerExtractor(model={self.config.extraction_model})"
