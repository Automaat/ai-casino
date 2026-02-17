"""LLM-based ticker extraction from Reddit content."""

import asyncio
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
        llm_client: LLMClient,
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

    def _truncate_content(
        self, post: RedditPost, comments: list[RedditComment] | None, input_text: str
    ) -> tuple[str | None, list[RedditComment] | None]:
        """Truncate post body and comments if needed.

        Args:
            post: Original post
            comments: Original comments
            input_text: Combined input text for length check

        Returns:
            Tuple of (truncated_body, truncated_comments)
        """
        max_chars = self.config.extraction_max_tokens * 4
        if len(input_text) <= max_chars:
            return post.body, comments

        # Truncate body proportionally
        body_ratio = len(post.body or "") / len(input_text) if input_text else 0
        max_body_chars = int(max_chars * body_ratio)
        truncated_body = post.body
        if post.body and len(post.body) > max_body_chars:
            truncated_body = post.body[:max_body_chars] + "..."

        # Limit comments to top 3
        truncated_comments = comments[:3] if comments else None

        logger.debug(f"Truncated input to ~{max_chars} chars (~{self.config.extraction_max_tokens} tokens)")
        return truncated_body, truncated_comments

    def _build_fallback_prompt(
        self, post: RedditPost, truncated_body: str | None, truncated_comments: list[RedditComment] | None
    ) -> str:
        """Build fallback prompt when template fails to load.

        Args:
            post: Reddit post
            truncated_body: Truncated post body
            truncated_comments: Truncated comments list

        Returns:
            Fallback prompt string
        """
        fallback_parts = [f"Title: {post.title}"]
        if truncated_body:
            fallback_parts.append(f"Body: {truncated_body}")
        if truncated_comments:
            for idx, comment in enumerate(truncated_comments[:3], 1):
                fallback_parts.append(f"Comment {idx}: {comment.body[:200]}")
        truncated_input = "\n\n".join(fallback_parts)

        return f"""Extract stock tickers mentioned in this Reddit post.

For each ticker:
- Symbol (e.g., AAPL, TSLA) - uppercase, 1-5 chars
- Sentiment: BULLISH (positive/buying), BEARISH (negative/selling), NEUTRAL (mention only)
- Context: brief snippet showing why ticker was mentioned (max 50 chars)
- Confidence: 0.0-1.0 (how certain this is a stock ticker)

Ignore:
- Common abbreviations (CEO, IPO, DD, YOLO, ATH)
- Non-stock mentions (company names without ticker context)
- Sarcasm or jokes (e.g., "$ROPE", "$MOON")

{truncated_input}

Extract tickers:"""

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

        # Build input text for length check
        input_parts = [f"Title: {post.title}"]
        if post.body:
            input_parts.append(f"Body: {post.body}")
        if comments:
            top_comments = sorted(comments, key=lambda c: c.score, reverse=True)[:3]
            for idx, comment in enumerate(top_comments, 1):
                input_parts.append(f"Comment {idx}: {comment.body}")

        input_text = "\n\n".join(input_parts)
        truncated_body, truncated_comments = self._truncate_content(post, comments, input_text)

        # Load and render prompt
        try:
            prompt = self.prompt_loader.load(
                "ticker_extraction",
                title=post.title,
                body=truncated_body or "(no body)",
                comments="\n".join(
                    f"{idx}. {c.body[:200]}" for idx, c in enumerate(truncated_comments[:3], 1)
                )
                if truncated_comments
                else "(no comments)",
            )
        except Exception:
            logger.opt(exception=True).warning("Failed to load prompt template, using fallback")
            prompt = self._build_fallback_prompt(post, truncated_body, truncated_comments)

        try:
            # Use LLM structured output with timeout
            response = await asyncio.wait_for(
                self.llm_client.astructured(
                    prompt=prompt,
                    response_model=TickerExtractionResponse,
                    temperature=self.config.extraction_temperature,
                ),
                timeout=self.config.extraction_timeout_s,
            )

            # Filter by confidence threshold
            high_confidence_mentions = [
                mention
                for mention in response.mentions
                if mention.confidence >= self.config.extraction_min_confidence
            ]

            # Normalize sentiment and validate symbols
            valid_mentions = []
            for mention in high_confidence_mentions:
                if self._is_valid_symbol(mention.symbol):
                    # Normalize sentiment before adding
                    normalized_mention = TickerMention(
                        symbol=mention.symbol,
                        sentiment=self._normalize_sentiment(mention.sentiment),
                        context=mention.context,
                        confidence=mention.confidence,
                    )
                    valid_mentions.append(normalized_mention)

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
