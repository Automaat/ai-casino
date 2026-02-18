"""Reddit web scraper using Playwright with anti-bot measures."""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from playwright.async_api import Browser, Page, async_playwright
from playwright_stealth import Stealth

from src.cache.historical import HistoricalCache
from src.daemon.config.reddit import RedditScraperConfig
from src.data.reddit import RedditComment, RedditPost

# User agent pool (realistic Chrome/Firefox 2024-2026)
USER_AGENTS = [
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.2 Safari/605.1.15"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.3 Safari/605.1.15"
    ),
    ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    "Mozilla/5.0 (X11; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:122.0) Gecko/20100101 Firefox/122.0",
]

_JS_EXTRACT_POSTS = """
() => Array.from(document.querySelectorAll('div.thing.link[data-fullname]:not(.promoted)')).map(el => ({
    id: (el.dataset.fullname || '').replace('t3_', ''),
    title: el.querySelector('a.title')?.textContent?.trim() || '',
    url: (() => {
        const h = el.querySelector('a.title')?.getAttribute('href');
        return h && !h.startsWith('http') ? 'https://old.reddit.com' + h : (h || '');
    })(),
    score: parseInt(el.querySelector('div.score.unvoted')?.textContent?.replace(/[^0-9]/g, '') || '0') || 0,
    num_comments: parseInt(
        (el.querySelector('a.comments')?.textContent || '0 comments').split(' ')[0]
    ) || 0,
    created_utc: el.querySelector('time')?.getAttribute('datetime') || null,
    body: el.querySelector('div.usertext-body div.md')?.textContent?.trim() || ''
}))
"""

_JS_EXTRACT_COMMENTS = """
(limit) => Array.from(document.querySelectorAll('div.comment[data-fullname]'))
    .slice(0, limit)
    .map(el => ({
        id: (el.dataset.fullname || '').replace('t1_', ''),
        body: el.querySelector('div.md')?.textContent?.trim() || '',
        score: (() => {
            const t = el.querySelector('div.score.unvoted')?.textContent?.trim().split(' ')[0];
            return parseInt(t) || 1;
        })()
    }))
"""


class RedditPlaywrightScraper:
    """Web scraper for old.reddit.com using Playwright."""

    def __init__(
        self,
        config: RedditScraperConfig,
        cache: HistoricalCache | None = None,
    ) -> None:
        """Initialize Reddit Playwright scraper.

        Args:
            config: Reddit scraper configuration
            cache: Optional cache for tracking scraped posts
        """
        self.config = config
        self.cache = cache
        self._browser: Browser | None = None
        self._playwright_context = None
        self._viewport_width = 0
        self._viewport_height = 0

        # Cookie persistence
        self._cookie_file = Path.home() / ".ai-casino" / "cache" / "reddit_cookies.json"
        self._cookie_file.parent.mkdir(parents=True, exist_ok=True)

    async def __aenter__(self) -> RedditPlaywrightScraper:
        """Enter async context manager."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit async context manager."""
        await self.close()

    async def start(self) -> None:
        """Start Playwright browser."""
        if self._browser:
            logger.debug("Browser already running, skipping start")
            return

        logger.info("Starting Playwright browser for Reddit scraping")
        self._playwright_context = async_playwright()
        playwright = await self._playwright_context.__aenter__()

        # Random viewport
        self._viewport_width = random.randint(self.config.viewport_width_min, self.config.viewport_width_max)
        self._viewport_height = random.randint(
            self.config.viewport_height_min, self.config.viewport_height_max
        )

        # Launch browser with anti-detection
        self._browser = await playwright.chromium.launch(
            headless=self.config.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ],
        )

        logger.info(
            f"Playwright browser started (headless={self.config.headless}, "
            f"viewport={self._viewport_width}x{self._viewport_height})"
        )

    async def close(self) -> None:
        """Close Playwright browser."""
        if self._browser:
            await self._browser.close()
            self._browser = None

        if self._playwright_context:
            await self._playwright_context.__aexit__(None, None, None)
            self._playwright_context = None

        logger.info("Playwright browser closed")

    async def _apply_stealth(self, page: Page) -> None:
        """Apply stealth mode to page.

        Args:
            page: Playwright page
        """
        if self.config.use_stealth_mode:
            await Stealth().apply_stealth_async(page)

    async def _random_delay(self, min_s: float, max_s: float) -> None:
        """Apply random delay.

        Args:
            min_s: Minimum delay in seconds
            max_s: Maximum delay in seconds
        """
        delay = random.uniform(min_s, max_s)
        await asyncio.sleep(delay)

    async def _load_cookies(self, page: Page) -> None:
        """Load cookies from file if exists.

        Args:
            page: Playwright page
        """
        if self._cookie_file.exists():
            try:
                cookies = json.loads(self._cookie_file.read_text())
                await page.context.add_cookies(cookies)
                logger.debug(f"Loaded {len(cookies)} cookies from {self._cookie_file}")
            except Exception:
                logger.opt(exception=True).warning("Failed to load cookies")

    async def _save_cookies(self, page: Page) -> None:
        """Save cookies to file.

        Args:
            page: Playwright page
        """
        try:
            cookies = await page.context.cookies()
            self._cookie_file.write_text(json.dumps(cookies))
            logger.debug(f"Saved {len(cookies)} cookies to {self._cookie_file}")
        except Exception:
            logger.opt(exception=True).warning("Failed to save cookies")

    def _parse_post(self, raw: dict, subreddit: str) -> RedditPost | None:
        """Parse raw JS-evaluated dict into RedditPost.

        Args:
            raw: Dict from page.evaluate JS extraction
            subreddit: Subreddit name

        Returns:
            RedditPost or None if id missing
        """
        if not raw.get("id"):
            return None

        created_utc = datetime.now(UTC)
        with contextlib.suppress(ValueError, AttributeError):
            if raw.get("created_utc"):
                created_utc = datetime.fromisoformat(raw["created_utc"])

        return RedditPost(
            id=raw["id"],
            title=raw.get("title", ""),
            body=raw.get("body", ""),
            subreddit=subreddit,
            score=raw.get("score", 0),
            upvote_ratio=0.0,  # Sentinel value (not available on listing)
            url=raw.get("url", ""),
            created_utc=created_utc,
            num_comments=raw.get("num_comments", 0),
        )

    def _parse_comment(self, raw: dict, parent_post_id: str) -> RedditComment | None:
        """Parse raw JS-evaluated dict into RedditComment.

        Args:
            raw: Dict from page.evaluate JS extraction
            parent_post_id: Parent post Reddit ID

        Returns:
            RedditComment or None if id or body missing
        """
        if not raw.get("id") or not raw.get("body"):
            return None

        return RedditComment(
            id=raw["id"],
            parent_post_id=parent_post_id,
            body=raw["body"],
            score=raw.get("score", 1),
            created_utc=datetime.now(UTC),
        )

    async def scrape_subreddit_posts(
        self,
        subreddit: str,
        limit: int = 50,
    ) -> list[RedditPost]:
        """Scrape hot posts from subreddit.

        Args:
            subreddit: Subreddit name (without r/)
            limit: Maximum number of posts to scrape

        Returns:
            List of RedditPost objects
        """
        if not self._browser:
            await self.start()

        if not self._browser:
            msg = "Browser failed to start"
            raise RuntimeError(msg)

        user_agent = random.choice(USER_AGENTS)
        context = await self._browser.new_context(
            user_agent=user_agent,
            viewport={"width": self._viewport_width, "height": self._viewport_height},
        )
        page = await context.new_page()

        await self._apply_stealth(page)
        await self._load_cookies(page)

        try:
            url = f"https://old.reddit.com/r/{subreddit}/"
            logger.info(f"Scraping r/{subreddit} (limit={limit})")

            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await self._random_delay(self.config.delay_page_load_min, self.config.delay_page_load_max)

            for scroll_num in range(3):
                await page.evaluate("window.scrollBy(0, window.innerHeight * 2)")
                await self._random_delay(self.config.delay_action_min, self.config.delay_action_max)
                logger.debug(f"Scroll {scroll_num + 1}/3")

            raw_posts: list[dict] = await page.evaluate(_JS_EXTRACT_POSTS)
            logger.info(f"Found {len(raw_posts)} post containers on r/{subreddit}")

            posts = []
            for raw in raw_posts[:limit]:
                try:
                    post = self._parse_post(raw, subreddit)
                    if post:
                        posts.append(post)
                except Exception:
                    logger.opt(exception=True).warning("Failed to parse post")

            await self._save_cookies(page)
            logger.info(f"Scraped {len(posts)} posts from r/{subreddit}")
            return posts

        except Exception:
            logger.opt(exception=True).error(f"Failed to scrape r/{subreddit}")
            return []

        finally:
            await page.close()
            await context.close()

    async def scrape_post_comments(
        self,
        post: RedditPost,
        limit: int = 10,
    ) -> list[RedditComment]:
        """Scrape top comments from post.

        Args:
            post: RedditPost to scrape comments from
            limit: Maximum number of comments to scrape

        Returns:
            List of RedditComment objects
        """
        if not self._browser:
            await self.start()

        if not self._browser:
            msg = "Browser failed to start"
            raise RuntimeError(msg)

        user_agent = random.choice(USER_AGENTS)
        context = await self._browser.new_context(
            user_agent=user_agent,
            viewport={"width": self._viewport_width, "height": self._viewport_height},
        )
        page = await context.new_page()

        await self._apply_stealth(page)
        await self._load_cookies(page)

        try:
            logger.debug(f"Scraping comments for post {post.id} (limit={limit})")

            comment_url = f"https://old.reddit.com/r/{post.subreddit}/comments/{post.id}/"
            await page.goto(comment_url, wait_until="domcontentloaded", timeout=30000)
            await self._random_delay(self.config.delay_page_load_min, self.config.delay_page_load_max)

            raw_comments: list[dict] = await page.evaluate(_JS_EXTRACT_COMMENTS, limit)

            comments = []
            for raw in raw_comments:
                try:
                    comment = self._parse_comment(raw, post.id)
                    if comment:
                        comments.append(comment)
                except Exception:
                    logger.opt(exception=True).debug("Failed to parse comment")

            logger.debug(f"Scraped {len(comments)} comments for post {post.id}")
            return comments

        except Exception:
            logger.opt(exception=True).warning(f"Failed to scrape comments for post {post.id}")
            return []

        finally:
            await page.close()
            await context.close()

    def __repr__(self) -> str:
        """Return string representation."""
        return f"RedditPlaywrightScraper(headless={self.config.headless})"
