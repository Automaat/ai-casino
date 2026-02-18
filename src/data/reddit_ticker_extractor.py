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


class BatchTickerMention(BaseModel):
    """Single ticker mention with source post attribution."""

    post_id: str = Field(description="Reddit post ID this mention came from")
    symbol: str = Field(description="Stock ticker symbol (e.g., AAPL)")
    sentiment: str = Field(description="BULLISH, BEARISH, or NEUTRAL")
    context: str = Field(description="Brief context snippet (max 50 chars)")
    confidence: float = Field(description="0.0-1.0 certainty this is a valid ticker")


class BatchTickerExtractionResponse(BaseModel):
    """LLM response for batch ticker extraction."""

    mentions: list[BatchTickerMention] = Field(
        default_factory=list, description="Ticker mentions with post attribution"
    )


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

    def _build_extraction_input(
        self, post: RedditPost, comments: list[RedditComment] | None
    ) -> tuple[str, str | None, list[RedditComment] | None]:
        """Build input text and truncate if needed.

        Args:
            post: RedditPost to extract from
            comments: Optional list of comments

        Returns:
            Tuple of (input_text, truncated_body, truncated_comments)
        """
        input_parts = [f"Title: {post.title}"]
        if post.body:
            input_parts.append(f"Body: {post.body}")
        if comments:
            top_comments = sorted(comments, key=lambda c: c.score, reverse=True)[:3]
            for idx, comment in enumerate(top_comments, 1):
                input_parts.append(f"Comment {idx}: {comment.body}")

        input_text = "\n\n".join(input_parts)
        truncated_body, truncated_comments = self._truncate_content(post, comments, input_text)
        return input_text, truncated_body, truncated_comments

    def _prepare_prompt(
        self, post: RedditPost, truncated_body: str | None, truncated_comments: list[RedditComment] | None
    ) -> str:
        """Prepare extraction prompt with fallback.

        Args:
            post: Reddit post
            truncated_body: Truncated post body
            truncated_comments: Truncated comments list

        Returns:
            Formatted prompt string
        """
        try:
            return self.prompt_loader.load(
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
            return self._build_fallback_prompt(post, truncated_body, truncated_comments)

    def _filter_and_validate_mentions(
        self, mentions: list[TickerMention], total_mentions: int
    ) -> list[TickerMention]:
        """Filter mentions by confidence and validate symbols.

        Args:
            mentions: Raw mentions from LLM
            total_mentions: Total count before filtering

        Returns:
            List of valid, normalized mentions
        """
        high_confidence = [m for m in mentions if m.confidence >= self.config.extraction_min_confidence]

        valid_mentions = []
        for mention in high_confidence:
            if self._is_valid_symbol(mention.symbol):
                normalized_mention = TickerMention(
                    symbol=mention.symbol,
                    sentiment=self._normalize_sentiment(mention.sentiment),
                    context=mention.context,
                    confidence=mention.confidence,
                )
                valid_mentions.append(normalized_mention)

        if len(valid_mentions) < total_mentions:
            filtered_count = total_mentions - len(valid_mentions)
            logger.debug(f"Filtered out {filtered_count} mentions (low confidence or invalid symbols)")

        return valid_mentions

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

        _, truncated_body, truncated_comments = self._build_extraction_input(post, comments)
        prompt = self._prepare_prompt(post, truncated_body, truncated_comments)

        try:
            response = await asyncio.wait_for(
                self.llm_client.astructured(
                    prompt=prompt,
                    response_model=TickerExtractionResponse,
                    temperature=self.config.extraction_temperature,
                    max_tokens=256,
                ),
                timeout=self.config.extraction_timeout_s,
            )

            valid_mentions = self._filter_and_validate_mentions(response.mentions, len(response.mentions))
            logger.info(f"Extracted {len(valid_mentions)} tickers from post {post.id} (r/{post.subreddit})")
            return valid_mentions

        except TimeoutError:
            logger.warning(f"LLM extraction timeout for post {post.id}")
            return []
        except Exception:
            logger.opt(exception=True).warning(f"Failed to extract tickers from post {post.id}")
            return []

    async def extract_tickers_batch(
        self,
        posts_with_comments: list[tuple[RedditPost, list[RedditComment]]],
        batch_size: int = 5,
    ) -> dict[str, list[TickerMention]]:
        """Extract tickers from multiple posts in batched LLM calls.

        Args:
            posts_with_comments: List of (post, comments) tuples
            batch_size: Posts per LLM call

        Returns:
            Dict mapping post_id to list of TickerMention
        """
        if not self.config.use_llm_extraction or not posts_with_comments:
            return {}

        result: dict[str, list[TickerMention]] = {}
        batches = [
            posts_with_comments[i : i + batch_size] for i in range(0, len(posts_with_comments), batch_size)
        ]

        tasks = [self._extract_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for batch_result in batch_results:
            if isinstance(batch_result, BaseException):
                logger.warning(f"Batch extraction failed: {batch_result}")
                continue
            for post_id, mentions in batch_result.items():
                result[post_id] = mentions

        total = sum(len(m) for m in result.values())
        logger.info(f"Batch extraction: {total} tickers from {len(result)} posts ({len(batches)} batches)")
        return result

    def _build_batch_prompt(
        self,
        batch: list[tuple[RedditPost, list[RedditComment]]],
    ) -> str:
        """Build combined prompt for a batch of posts.

        Args:
            batch: List of (post, comments) tuples

        Returns:
            Formatted prompt string
        """
        post_sections = []
        for post, comments in batch:
            section_parts = [f"[post_id={post.id}]", f"Title: {post.title}"]
            if post.body:
                body = post.body[:500] if len(post.body) > 500 else post.body
                section_parts.append(f"Body: {body}")
            if comments:
                top_comments = sorted(comments, key=lambda c: c.score, reverse=True)[:2]
                for idx, c in enumerate(top_comments, 1):
                    section_parts.append(f"Comment {idx}: {c.body[:150]}")
            post_sections.append("\n".join(section_parts))

        posts_text = "\n\n---\n\n".join(post_sections)

        try:
            return self.prompt_loader.load("ticker_extraction_batch", posts=posts_text)
        except Exception:
            logger.opt(exception=True).warning("Failed to load batch prompt template, using inline")
            return (
                "Extract stock tickers from these Reddit posts. "
                "Include post_id for each mention.\n\n" + posts_text
            )

    def _validate_batch_mentions(
        self,
        mentions: list[BatchTickerMention],
        valid_post_ids: set[str],
    ) -> dict[str, list[TickerMention]]:
        """Validate and group batch mentions by post_id.

        Args:
            mentions: Raw mentions from LLM
            valid_post_ids: Set of valid post IDs from the batch

        Returns:
            Dict mapping post_id to list of validated TickerMention
        """
        result: dict[str, list[TickerMention]] = {}
        for mention in mentions:
            if mention.post_id not in valid_post_ids:
                continue
            if mention.confidence < self.config.extraction_min_confidence:
                continue
            if not self._is_valid_symbol(mention.symbol):
                continue

            ticker_mention = TickerMention(
                symbol=mention.symbol,
                sentiment=self._normalize_sentiment(mention.sentiment),
                context=mention.context,
                confidence=mention.confidence,
            )
            result.setdefault(mention.post_id, []).append(ticker_mention)
        return result

    async def _extract_batch(
        self,
        batch: list[tuple[RedditPost, list[RedditComment]]],
    ) -> dict[str, list[TickerMention]]:
        """Extract tickers from a single batch of posts.

        Args:
            batch: List of (post, comments) tuples

        Returns:
            Dict mapping post_id to list of TickerMention
        """
        prompt = self._build_batch_prompt(batch)

        try:
            response = await asyncio.wait_for(
                self.llm_client.astructured(
                    prompt=prompt,
                    response_model=BatchTickerExtractionResponse,
                    temperature=self.config.extraction_temperature,
                ),
                timeout=self.config.extraction_timeout_s * len(batch),
            )
        except TimeoutError:
            post_ids = [p.id for p, _ in batch]
            logger.warning(f"Batch extraction timeout for posts: {post_ids}")
            return {}
        except Exception:
            logger.opt(exception=True).warning("Batch extraction LLM call failed")
            return {}

        valid_post_ids = {post.id for post, _ in batch}
        return self._validate_batch_mentions(response.mentions, valid_post_ids)

    def _is_valid_symbol(self, symbol: str) -> bool:
        """Validate stock ticker symbol.

        Args:
            symbol: Ticker symbol to validate

        Returns:
            True if valid, False otherwise
        """
        # Uppercase, 1-5 chars (allow . for BRK.B style class shares)
        if not symbol or not (1 <= len(symbol) <= 5):
            return False

        # Must be uppercase letters only — no digits (SP500, etc. are indices not tickers)
        if not symbol.replace(".", "").replace("-", "").isalpha():
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
