"""Reddit sentiment data fetcher using PRAW."""

import hashlib
import os
import re
from datetime import UTC, datetime
from pathlib import Path

import praw
import prawcore
from diskcache import Cache
from loguru import logger
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Cache TTL in seconds
REDDIT_CACHE_TTL = 900  # 15 minutes

DEFAULT_SUBREDDITS = ["wallstreetbets", "stocks", "investing"]

# Common words to exclude from ticker detection
EXCLUDED_WORDS = frozenset(
    {
        # Single letters
        "I",
        "A",
        # Common words
        "THE",
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
        # Reddit/WSB slang
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
        # Time-related
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
        # Common verbs/adjectives
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
        "RH",  # Robinhood abbreviation
    }
)

_NO_RETRY_EXCEPTIONS = prawcore.exceptions.ResponseException

HTTP_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception) & retry_if_not_exception_type(_NO_RETRY_EXCEPTIONS),
    reraise=True,
    before_sleep=lambda retry_state: logger.warning(
        f"Retry {retry_state.attempt_number} after {retry_state.outcome.exception()}"
    ),
)


class RedditPost(BaseModel):
    """Single Reddit post."""

    id: str
    title: str
    body: str
    subreddit: str
    score: int
    upvote_ratio: float
    url: str
    created_utc: datetime
    num_comments: int


class RedditSentimentData(BaseModel):
    """Reddit sentiment data for a symbol."""

    symbol: str
    posts: list[RedditPost]
    mention_count: int
    avg_score: float
    avg_upvote_ratio: float
    fetched_at: datetime


class TrendingTicker(BaseModel):
    """Trending ticker from Reddit."""

    symbol: str
    mention_count: int
    total_score: int
    avg_upvote_ratio: float
    sample_posts: list[RedditPost]


class RedditFetcher:
    """Fetch Reddit sentiment data using PRAW."""

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        user_agent: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        """Initialize Reddit fetcher.

        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            user_agent: Reddit API user agent
            cache_dir: Cache directory path
        """
        self._client_id = client_id or os.getenv("REDDIT_CLIENT_ID")
        self._client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET")
        self._user_agent = user_agent or os.getenv("REDDIT_USER_AGENT", "ai-casino/1.0")

        self._cache_dir = Path(cache_dir or "data/cache/reddit")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = Cache(str(self._cache_dir))

        self._reddit: praw.Reddit | None = None

        if not self._client_id or not self._client_secret:
            logger.warning("Reddit credentials not set - API calls will fail")
        else:
            logger.info(f"Initialized RedditFetcher (cache_dir={self._cache_dir})")

    def _get_reddit(self) -> praw.Reddit:
        """Lazy init PRAW client.

        Returns:
            Initialized PRAW Reddit instance
        """
        if self._reddit is None:
            if not self._client_id or not self._client_secret:
                msg = "Reddit credentials not configured"
                raise ValueError(msg)
            self._reddit = praw.Reddit(
                client_id=self._client_id,
                client_secret=self._client_secret,
                user_agent=self._user_agent,
            )
        return self._reddit

    def _cache_key(self, prefix: str, *args: str) -> str:
        """Generate cache key.

        Args:
            prefix: Cache key prefix
            args: Additional key components

        Returns:
            Cache key string
        """
        raw = f"{prefix}:{':'.join(str(a) for a in args)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def _submission_to_post(self, submission: praw.models.Submission) -> RedditPost:
        """Convert PRAW submission to RedditPost.

        Args:
            submission: PRAW Submission object

        Returns:
            RedditPost model
        """
        body = submission.selftext or ""
        if len(body) > 2000:
            body = body[:2000]

        return RedditPost(
            id=submission.id,
            title=submission.title,
            body=body,
            subreddit=submission.subreddit.display_name,
            score=submission.score,
            upvote_ratio=submission.upvote_ratio,
            url=f"https://reddit.com{submission.permalink}",
            created_utc=datetime.fromtimestamp(submission.created_utc, tz=UTC),
            num_comments=submission.num_comments,
        )

    def _contains_symbol(self, text: str, symbol: str) -> bool:
        """Check if text contains the symbol.

        Args:
            text: Text to search
            symbol: Stock symbol to find

        Returns:
            True if symbol found
        """
        pattern = rf"\${symbol}\b|\b{symbol}\b"
        return bool(re.search(pattern, text, re.IGNORECASE))

    @HTTP_RETRY
    def fetch_mentions(
        self,
        symbol: str,
        subreddits: list[str] | None = None,
        limit: int = 25,
        time_filter: str = "day",
    ) -> RedditSentimentData:
        """Fetch Reddit mentions for a symbol.

        Args:
            symbol: Stock ticker symbol
            subreddits: List of subreddits to search
            limit: Max posts per subreddit
            time_filter: Time filter (hour, day, week, month, year, all)

        Returns:
            RedditSentimentData with posts and aggregates
        """
        logger.info(f"Fetching Reddit mentions for {symbol}")
        subreddits = subreddits or DEFAULT_SUBREDDITS

        cache_key = self._cache_key("mentions", symbol, ",".join(subreddits), str(limit), time_filter)
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for {symbol} mentions")
            return RedditSentimentData.model_validate(cached)

        try:
            reddit = self._get_reddit()
            posts: list[RedditPost] = []

            for subreddit_name in subreddits:
                subreddit = reddit.subreddit(subreddit_name)
                query = f"${symbol} OR {symbol}"

                for submission in subreddit.search(query, limit=limit, time_filter=time_filter):
                    text = f"{submission.title} {submission.selftext or ''}"
                    if self._contains_symbol(text, symbol):
                        posts.append(self._submission_to_post(submission))

            avg_score = sum(p.score for p in posts) / len(posts) if posts else 0.0
            avg_upvote_ratio = sum(p.upvote_ratio for p in posts) / len(posts) if posts else 0.0

            result = RedditSentimentData(
                symbol=symbol,
                posts=posts,
                mention_count=len(posts),
                avg_score=avg_score,
                avg_upvote_ratio=avg_upvote_ratio,
                fetched_at=datetime.now(),
            )

            self._cache.set(cache_key, result.model_dump(), expire=REDDIT_CACHE_TTL)
            logger.info(f"Fetched {len(posts)} Reddit mentions for {symbol}")
            return result

        except Exception as e:
            logger.error(f"Reddit fetch failed: {e}")
            raise

    def _extract_tickers(self, text: str) -> set[str]:
        """Extract stock tickers from text.

        Args:
            text: Text to extract tickers from

        Returns:
            Set of ticker symbols
        """
        tickers = set()
        # Match $SYMBOL or standalone 2-5 letter uppercase words
        pattern = r"\$([A-Z]{1,5})\b|\b([A-Z]{2,5})\b"

        for match in re.finditer(pattern, text):
            ticker = match.group(1) or match.group(2)
            if ticker and ticker not in EXCLUDED_WORDS:
                tickers.add(ticker)

        return tickers

    @HTTP_RETRY
    def fetch_trending_tickers(
        self,
        subreddits: list[str] | None = None,
        limit: int = 100,
        time_filter: str = "day",
        min_mentions: int = 3,
    ) -> list[TrendingTicker]:
        """Fetch trending tickers from Reddit.

        Args:
            subreddits: List of subreddits to scan
            limit: Max posts per subreddit
            time_filter: Time filter (hour, day, week, month, year, all)
            min_mentions: Minimum mentions to include ticker

        Returns:
            List of TrendingTicker sorted by mention_count
        """
        logger.info("Fetching trending tickers from Reddit")
        subreddits = subreddits or DEFAULT_SUBREDDITS

        cache_key = self._cache_key(
            "trending", ",".join(subreddits), str(limit), time_filter, str(min_mentions)
        )
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug("Cache hit for trending tickers")
            return [TrendingTicker.model_validate(t) for t in cached]

        try:
            reddit = self._get_reddit()

            # Aggregate by ticker: {symbol: {"posts": [], "total_score": 0, "ratios": []}}
            ticker_data: dict[str, dict] = {}

            for subreddit_name in subreddits:
                subreddit = reddit.subreddit(subreddit_name)

                for submission in subreddit.hot(limit=limit):
                    text = f"{submission.title} {submission.selftext or ''}"
                    tickers = self._extract_tickers(text)

                    for ticker in tickers:
                        if ticker not in ticker_data:
                            ticker_data[ticker] = {"posts": [], "total_score": 0, "ratios": []}

                        post = self._submission_to_post(submission)
                        ticker_data[ticker]["posts"].append(post)
                        ticker_data[ticker]["total_score"] += submission.score
                        ticker_data[ticker]["ratios"].append(submission.upvote_ratio)

            # Build results
            results: list[TrendingTicker] = []
            for symbol, data in ticker_data.items():
                if len(data["posts"]) < min_mentions:
                    continue

                # Sort posts by score and take top 5
                sorted_posts = sorted(data["posts"], key=lambda p: p.score, reverse=True)[:5]
                avg_ratio = sum(data["ratios"]) / len(data["ratios"])

                results.append(
                    TrendingTicker(
                        symbol=symbol,
                        mention_count=len(data["posts"]),
                        total_score=data["total_score"],
                        avg_upvote_ratio=avg_ratio,
                        sample_posts=sorted_posts,
                    )
                )

            # Sort by mention_count descending
            results.sort(key=lambda t: t.mention_count, reverse=True)

            self._cache.set(cache_key, [r.model_dump() for r in results], expire=REDDIT_CACHE_TTL)
            logger.info(f"Found {len(results)} trending tickers")
            return results

        except Exception as e:
            logger.error(f"Reddit trending fetch failed: {e}")
            raise

    def clear_cache(self) -> None:
        """Clear all cached Reddit data."""
        self._cache.clear()
        logger.info("Cleared Reddit cache")

    def __repr__(self) -> str:
        """String representation."""
        authenticated = bool(self._client_id and self._client_secret)
        return f"RedditFetcher(authenticated={authenticated}, cache_dir={self._cache_dir})"
