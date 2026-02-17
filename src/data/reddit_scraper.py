"""Reddit web scraper using Playwright with anti-bot measures."""

import asyncio
import json
import random
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from playwright.async_api import Browser, ElementHandle, Page, async_playwright
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
        self._viewport_width = random.randint(
            self.config.viewport_width_min, self.config.viewport_width_max
        )
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

        # Create new page with random user agent and viewport
        user_agent = random.choice(USER_AGENTS)
        context = await self._browser.new_context(
            user_agent=user_agent,
            viewport={"width": self._viewport_width, "height": self._viewport_height},
        )
        page = await context.new_page()

        # Apply stealth
        await self._apply_stealth(page)

        # Load cookies
        await self._load_cookies(page)

        try:
            # Navigate to subreddit
            url = f"https://old.reddit.com/r/{subreddit}/"
            logger.info(f"Scraping r/{subreddit} (limit={limit})")

            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await self._random_delay(self.config.delay_page_load_min, self.config.delay_page_load_max)

            # Scroll to trigger lazy load
            for scroll_num in range(3):
                await page.evaluate("window.scrollBy(0, window.innerHeight * 2)")
                await self._random_delay(self.config.delay_action_min, self.config.delay_action_max)
                logger.debug(f"Scroll {scroll_num + 1}/3")

            # Extract posts
            posts = []
            post_containers = await page.query_selector_all("div.thing.link[data-fullname]")

            logger.info(f"Found {len(post_containers)} post containers on r/{subreddit}")

            for container in post_containers[:limit]:
                try:
                    post = await self._extract_post(container, subreddit)
                    if post:
                        posts.append(post)
                except Exception:
                    logger.opt(exception=True).warning("Failed to extract post")
                    continue

            # Save cookies
            await self._save_cookies(page)

            logger.info(f"Scraped {len(posts)} posts from r/{subreddit}")
            return posts

        except Exception:
            logger.opt(exception=True).error(f"Failed to scrape r/{subreddit}")
            return []

        finally:
            await page.close()
            await context.close()

    async def _extract_post(self, container: ElementHandle, subreddit: str) -> RedditPost | None:
        """Extract post data from container element.

        Args:
            container: Post container element
            subreddit: Subreddit name

        Returns:
            RedditPost or None if extraction fails
        """
        try:
            # Reddit ID
            reddit_id_attr = await container.get_attribute("data-fullname")
            if not reddit_id_attr:
                return None
            reddit_id = reddit_id_attr.replace("t3_", "")

            # Title
            title_elem = await container.query_selector("a.title")
            title = await title_elem.text_content() if title_elem else ""
            title = title.strip() if title else ""

            # URL
            url_elem = await container.query_selector("a.title")
            url = await url_elem.get_attribute("href") if url_elem else None
            if url and not url.startswith("http"):
                url = f"https://old.reddit.com{url}"
            if not url:
                url = ""

            # Score
            score_elem = await container.query_selector("div.score.unvoted")
            score_text = await score_elem.text_content() if score_elem else "0"
            score_text = score_text.strip().replace("•", "").strip() if score_text else "0"
            try:
                score = int(score_text) if score_text and score_text.isdigit() else 0
            except ValueError:
                score = 0

            # Comments count
            comments_elem = await container.query_selector("a.comments")
            comments_text = await comments_elem.text_content() if comments_elem else "0 comments"
            num_comments = 0
            if comments_text and "comment" in comments_text:
                num_text = comments_text.split()[0]
                try:
                    num_comments = int(num_text) if num_text.isdigit() else 0
                except ValueError:
                    num_comments = 0

            # Body (self-posts only)
            body_elem = await container.query_selector("div.usertext-body div.md")
            body = await body_elem.text_content() if body_elem else ""
            body = body.strip() if body else ""

            # Created UTC (extract from time element)
            created_utc = datetime.now(UTC)
            time_elem = await container.query_selector("time")
            if time_elem:
                timestamp_attr = await time_elem.get_attribute("datetime")
                if timestamp_attr:
                    try:
                        created_utc = datetime.fromisoformat(timestamp_attr.replace("Z", "+00:00"))
                    except (ValueError, AttributeError):
                        logger.debug(f"Failed to parse timestamp: {timestamp_attr}")

            # Upvote ratio (not available on listing, use sentinel)
            upvote_ratio = 0.0  # Sentinel value (real data requires post detail page)

            return RedditPost(
                id=reddit_id,
                title=title,
                body=body,
                subreddit=subreddit,
                score=score,
                upvote_ratio=upvote_ratio,
                url=url,
                created_utc=created_utc,
                num_comments=num_comments,
            )

        except Exception:
            logger.opt(exception=True).debug("Failed to extract post data")
            return None

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

        # Create new page with random user agent and viewport
        user_agent = random.choice(USER_AGENTS)
        context = await self._browser.new_context(
            user_agent=user_agent,
            viewport={"width": self._viewport_width, "height": self._viewport_height},
        )
        page = await context.new_page()

        # Apply stealth
        await self._apply_stealth(page)

        # Load cookies
        await self._load_cookies(page)

        try:
            logger.debug(f"Scraping comments for post {post.id} (limit={limit})")

            await page.goto(post.url, wait_until="domcontentloaded", timeout=30000)
            await self._random_delay(self.config.delay_page_load_min, self.config.delay_page_load_max)

            # Extract comments
            comments = []
            comment_containers = await page.query_selector_all("div.comment[data-fullname]")

            for container in comment_containers[:limit]:
                try:
                    comment = await self._extract_comment(container, post.id)
                    if comment:
                        comments.append(comment)
                except Exception:
                    logger.opt(exception=True).debug("Failed to extract comment")
                    continue

            logger.debug(f"Scraped {len(comments)} comments for post {post.id}")
            return comments

        except Exception:
            logger.opt(exception=True).warning(f"Failed to scrape comments for post {post.id}")
            return []

        finally:
            await page.close()
            await context.close()

    async def _extract_comment(self, container: ElementHandle, parent_post_id: str) -> RedditComment | None:
        """Extract comment data from container element.

        Args:
            container: Comment container element
            parent_post_id: Parent post Reddit ID

        Returns:
            RedditComment or None if extraction fails
        """
        try:
            # Comment ID
            comment_id_attr = await container.get_attribute("data-fullname")
            if not comment_id_attr:
                return None
            comment_id = comment_id_attr.replace("t1_", "")

            # Body
            body_elem = await container.query_selector("div.md")
            body = await body_elem.text_content() if body_elem else ""
            body = body.strip() if body else ""

            if not body:
                return None

            # Score
            score_elem = await container.query_selector("div.score.unvoted")
            score_text = await score_elem.text_content() if score_elem else "1"
            score_text = score_text.strip().split()[0] if score_text else "1"  # "123 points" -> "123"
            try:
                score = int(score_text) if score_text.lstrip("-").isdigit() else 1
            except ValueError:
                score = 1

            # Created UTC (not easily available, use current time)
            created_utc = datetime.now(UTC)

            return RedditComment(
                id=comment_id,
                parent_post_id=parent_post_id,
                body=body,
                score=score,
                created_utc=created_utc,
            )

        except Exception:
            logger.opt(exception=True).debug("Failed to extract comment data")
            return None

    def __repr__(self) -> str:
        """Return string representation."""
        return f"RedditPlaywrightScraper(headless={self.config.headless})"
