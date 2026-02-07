"""Truth Social data fetcher for Trump posts."""

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests
from dateutil import parser
from diskcache import Cache
from loguru import logger
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.cache.historical import HistoricalCache

CACHE_TTL = 300  # 5 minutes (matches archive update frequency)

HTTP_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(
        (
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        )
    ),
    reraise=True,
    before_sleep=lambda retry_state: logger.warning(
        f"Retry {retry_state.attempt_number} after {retry_state.outcome.exception()}"
    ),
)


class TruthPost(BaseModel):
    """Single Truth Social post."""

    id: str
    content: str
    created_at: datetime
    likes: int
    reposts: int
    replies: int
    url: str


class TrumpPostData(BaseModel):
    """Collection of Trump posts with metadata."""

    posts: list[TruthPost]
    total_count: int
    latest_post_at: datetime | None
    fetched_at: datetime


class TruthSocialFetcher:
    """Fetch Trump's Truth Social posts from CNN archive."""

    CNN_ARCHIVE_URL = "https://ix.cnn.io/data/truth-social/truth_archive.json"

    def __init__(
        self,
        cache_dir: str | None = None,
        historical_cache: HistoricalCache | None = None,
    ) -> None:
        """Initialize Truth Social fetcher.

        Args:
            cache_dir: Cache directory path
            historical_cache: Optional permanent cache for posts
        """
        self._cache_dir = Path(cache_dir or "data/cache/truth_social")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = Cache(str(self._cache_dir))
        self._historical_cache = historical_cache
        logger.info(f"Initialized TruthSocialFetcher (cache_dir={self._cache_dir})")

    def _cache_key(self, prefix: str, *args: str) -> str:
        """Generate cache key."""
        raw = f"{prefix}:{':'.join(str(a) for a in args)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def _parse_datetime(self, dt_str: str) -> datetime:
        """Parse datetime string from archive using python-dateutil.

        Args:
            dt_str: ISO 8601 datetime string

        Returns:
            Parsed datetime with UTC timezone

        Raises:
            ValueError: If dt_str is empty or invalid
        """
        if not dt_str or not dt_str.strip():
            msg = "Empty datetime string - missing created_at field"
            raise ValueError(msg)

        parsed = parser.parse(dt_str)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed

    def _raw_to_post(self, raw: dict) -> TruthPost:
        """Convert raw archive entry to TruthPost."""
        post_id = str(raw.get("id", ""))
        return TruthPost(
            id=post_id,
            content=raw.get("content", "") or raw.get("text", ""),
            created_at=self._parse_datetime(raw.get("created_at", raw.get("createdAt", ""))),
            likes=int(raw.get("favourites_count", raw.get("likes", 0)) or 0),
            reposts=int(raw.get("reblogs_count", raw.get("reposts", 0)) or 0),
            replies=int(raw.get("replies_count", raw.get("replies", 0)) or 0),
            url=raw.get("url", f"https://truthsocial.com/@realDonaldTrump/posts/{post_id}"),
        )

    @HTTP_RETRY
    def _fetch_archive(self) -> list[dict]:
        """Fetch raw archive data."""
        cache_key = self._cache_key("archive")
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug("Cache hit for Truth Social archive")
            return cached

        logger.info("Fetching Truth Social archive from CNN")
        response = requests.get(self.CNN_ARCHIVE_URL, timeout=30)
        response.raise_for_status()

        data = response.json()
        self._cache.set(cache_key, data, expire=CACHE_TTL)
        logger.info(f"Fetched {len(data)} posts from archive")
        return data

    def fetch_recent(self, hours: int = 24) -> TrumpPostData:
        """Fetch recent Trump posts.

        Args:
            hours: Number of hours to look back

        Returns:
            TrumpPostData with filtered posts
        """
        logger.info(f"Fetching Trump posts from last {hours} hours")
        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        return self.fetch_since(cutoff)

    def fetch_since(self, since: datetime) -> TrumpPostData:
        """Fetch Trump posts since a specific time.

        Args:
            since: Datetime cutoff (posts after this time)

        Returns:
            TrumpPostData with filtered posts
        """
        if since.tzinfo is None:
            since = since.replace(tzinfo=UTC)

        raw_data = self._fetch_archive()
        posts: list[TruthPost] = []

        for raw in raw_data:
            post = self._raw_to_post(raw)
            if post.created_at >= since:
                posts.append(post)

        # Sort by created_at descending (newest first)
        posts.sort(key=lambda p: p.created_at, reverse=True)

        self._store_posts_to_cache(posts)

        latest = posts[0].created_at if posts else None
        logger.info(f"Found {len(posts)} posts since {since}")

        return TrumpPostData(
            posts=posts,
            total_count=len(posts),
            latest_post_at=latest,
            fetched_at=datetime.now(UTC),
        )

    def _store_posts_to_cache(self, posts: list[TruthPost]) -> None:
        """Store posts to permanent cache if available."""
        if self._historical_cache and posts:
            self._historical_cache.store_truth_social_posts(posts)

    def fetch_all(self) -> TrumpPostData:
        """Fetch all available Trump posts."""
        logger.info("Fetching all Trump posts from archive")
        raw_data = self._fetch_archive()

        posts = [self._raw_to_post(raw) for raw in raw_data]
        posts.sort(key=lambda p: p.created_at, reverse=True)

        self._store_posts_to_cache(posts)

        latest = posts[0].created_at if posts else None

        return TrumpPostData(
            posts=posts,
            total_count=len(posts),
            latest_post_at=latest,
            fetched_at=datetime.now(UTC),
        )

    def get_latest_post_id(self) -> str | None:
        """Get the ID of the most recent post."""
        raw_data = self._fetch_archive()
        if not raw_data:
            return None

        # Find most recent by created_at
        latest = max(raw_data, key=lambda x: x.get("created_at", x.get("createdAt", "")))
        return str(latest.get("id", ""))

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        logger.info("Cleared Truth Social cache")

    def __repr__(self) -> str:
        """String representation."""
        return f"TruthSocialFetcher(cache_dir={self._cache_dir})"
