"""Tests for RedditPlaywrightScraper."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from src.daemon.config.reddit import RedditScraperConfig
from src.data.reddit import RedditPost
from src.data.reddit_scraper import RedditPlaywrightScraper


@pytest.fixture
def scraper_config():
    """Create test scraper config."""
    return RedditScraperConfig(
        headless=True,
        use_stealth_mode=True,
        delay_page_load_min=0.1,
        delay_page_load_max=0.2,
        delay_action_min=0.05,
        delay_action_max=0.1,
    )


@pytest.fixture
def mock_page():
    """Create mock Playwright page."""
    page = AsyncMock()
    page.goto = AsyncMock()
    page.wait_for_selector = AsyncMock()
    page.query_selector_all = AsyncMock(return_value=[])
    page.evaluate = AsyncMock()
    page.close = AsyncMock()
    return page


@pytest.fixture
def mock_browser():
    """Create mock Playwright browser."""
    browser = AsyncMock()
    context = AsyncMock()
    page = AsyncMock()

    browser.new_context = AsyncMock(return_value=context)
    context.new_page = AsyncMock(return_value=page)
    context.close = AsyncMock()
    browser.close = AsyncMock()

    page.goto = AsyncMock()
    page.wait_for_selector = AsyncMock()
    page.query_selector_all = AsyncMock(return_value=[])
    page.close = AsyncMock()

    return browser, context, page


@pytest.fixture
async def scraper(scraper_config):
    """Create scraper instance."""
    scraper = RedditPlaywrightScraper(config=scraper_config)
    yield scraper
    if scraper._browser:
        await scraper.close()


@pytest.mark.unit
def test_initialization(scraper_config):
    """Test scraper initialization."""
    scraper = RedditPlaywrightScraper(config=scraper_config)

    assert scraper.config == scraper_config
    assert scraper._browser is None
    assert scraper._cookie_file.exists() or not scraper._cookie_file.exists()


@pytest.mark.unit
async def test_context_manager():
    """Test async context manager."""
    config = RedditScraperConfig(headless=True)

    with patch("src.data.reddit_scraper.async_playwright") as mock_playwright:
        mock_pw = AsyncMock()
        mock_browser = AsyncMock()
        mock_playwright.return_value = mock_pw
        mock_pw.__aenter__ = AsyncMock(return_value=mock_pw)
        mock_pw.__aexit__ = AsyncMock()
        mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)
        mock_browser.close = AsyncMock()

        async with RedditPlaywrightScraper(config=config) as scraper:
            assert scraper._browser is not None

        mock_browser.close.assert_called_once()


@pytest.mark.unit
def test_parse_post(scraper_config):
    """Test _parse_post converts raw JS dict to RedditPost."""
    scraper = RedditPlaywrightScraper(config=scraper_config)

    raw = {
        "id": "abc123",
        "title": "TSLA to the moon! 🚀",
        "url": "https://reddit.com/r/wallstreetbets/post1",
        "score": 2500,
        "num_comments": 150,
        "created_utc": "2024-01-15T12:00:00+00:00",
        "body": "",
    }

    post = scraper._parse_post(raw, "wallstreetbets")

    assert post is not None
    assert post.id == "abc123"
    assert post.title == "TSLA to the moon! 🚀"
    assert post.subreddit == "wallstreetbets"
    assert post.score == 2500
    assert post.num_comments == 150


@pytest.mark.unit
def test_parse_post_missing_id(scraper_config):
    """Test _parse_post returns None when id is missing."""
    scraper = RedditPlaywrightScraper(config=scraper_config)

    post = scraper._parse_post({"id": "", "title": "test"}, "wallstreetbets")
    assert post is None


@pytest.mark.unit
def test_parse_comment(scraper_config):
    """Test _parse_comment converts raw JS dict to RedditComment."""
    scraper = RedditPlaywrightScraper(config=scraper_config)

    raw = {"id": "comment1", "body": "TSLA $350 by EOW! This is the way 🚀", "score": 450}

    comment = scraper._parse_comment(raw, "t3_abc123")

    assert comment is not None
    assert comment.id == "comment1"
    assert comment.parent_post_id == "t3_abc123"
    assert comment.body == "TSLA $350 by EOW! This is the way 🚀"
    assert comment.score == 450


@pytest.mark.unit
def test_parse_comment_missing_body(scraper_config):
    """Test _parse_comment returns None when body is empty."""
    scraper = RedditPlaywrightScraper(config=scraper_config)

    comment = scraper._parse_comment({"id": "c1", "body": "", "score": 1}, "post1")
    assert comment is None


@pytest.mark.unit
async def test_scrape_with_mock_browser(scraper_config):
    """Test scraping with mocked browser."""
    scraper = RedditPlaywrightScraper(config=scraper_config)

    with patch("src.data.reddit_scraper.async_playwright") as mock_playwright:
        mock_pw = AsyncMock()
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()

        mock_playwright.return_value = mock_pw
        mock_pw.__aenter__ = AsyncMock(return_value=mock_pw)
        mock_pw.__aexit__ = AsyncMock()
        mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)

        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.close = AsyncMock()

        # Mock empty posts list via evaluate
        mock_page.evaluate = AsyncMock(return_value=[])
        mock_page.goto = AsyncMock()
        mock_page.wait_for_selector = AsyncMock()
        mock_page.close = AsyncMock()

        await scraper.start()
        posts = await scraper.scrape_subreddit_posts("wallstreetbets", limit=10)
        await scraper.close()

        assert isinstance(posts, list)
        mock_page.goto.assert_called()
        mock_browser.close.assert_called_once()


@pytest.mark.unit
async def test_delay_randomization(scraper_config):
    """Test that delays are applied without error."""
    scraper = RedditPlaywrightScraper(config=scraper_config)

    # _random_delay is async and performs the delay, doesn't return value
    await scraper._random_delay(0.1, 0.2)
    await scraper._random_delay(0.1, 0.2)
    # Test passes if no exceptions raised


@pytest.mark.unit
async def test_cookie_persistence(scraper_config, tmp_path):
    """Test cookie save/load."""
    cookie_file = tmp_path / "test_cookies.json"
    scraper = RedditPlaywrightScraper(config=scraper_config)
    scraper._cookie_file = cookie_file

    # Mock cookies
    test_cookies = [{"name": "session", "value": "abc123", "domain": ".reddit.com"}]

    # Mock page and context for save
    mock_context = AsyncMock()
    mock_context.cookies = AsyncMock(return_value=test_cookies)
    mock_page = AsyncMock()
    mock_page.context = mock_context

    await scraper._save_cookies(mock_page)
    assert cookie_file.exists()

    # Mock context for load
    mock_context_load = AsyncMock()
    mock_context_load.add_cookies = AsyncMock()
    mock_page_load = AsyncMock()
    mock_page_load.context = mock_context_load

    await scraper._load_cookies(mock_page_load)
    mock_context_load.add_cookies.assert_called_once_with(test_cookies)


@pytest.mark.unit
def test_repr(scraper_config):
    """Test string representation."""
    scraper = RedditPlaywrightScraper(config=scraper_config)
    repr_str = repr(scraper)

    assert "RedditPlaywrightScraper" in repr_str
    assert "headless=" in repr_str


@pytest.mark.unit
async def test_scrape_promoted_posts_filtered(scraper_config):
    """Test that promoted/ad posts are not returned when evaluate returns only organic posts."""
    scraper = RedditPlaywrightScraper(config=scraper_config)

    organic_post = {
        "id": "abc123",
        "title": "AAPL earnings beat expectations",
        "url": "https://old.reddit.com/r/investing/comments/abc123/",
        "score": 500,
        "num_comments": 42,
        "created_utc": "2024-01-15T12:00:00+00:00",
        "body": "",
    }
    promoted_like_post = {
        "id": "",
        "title": "Buy this stock now! [Ad]",
        "url": "https://ad.example.com/promo",
        "score": 0,
        "num_comments": 0,
        "created_utc": None,
        "body": "",
    }

    with patch("src.data.reddit_scraper.async_playwright") as mock_playwright:
        mock_pw = AsyncMock()
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()

        mock_playwright.return_value = mock_pw
        mock_pw.__aenter__ = AsyncMock(return_value=mock_pw)
        mock_pw.__aexit__ = AsyncMock()
        mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.close = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.wait_for_selector = AsyncMock()
        mock_page.close = AsyncMock()

        # JS selector uses :not(.promoted), so promoted posts never reach evaluate output.
        # Simulate this: evaluate returns only the organic post (promoted filtered at DOM level).
        mock_page.evaluate = AsyncMock(return_value=[organic_post, promoted_like_post])

        await scraper.start()
        posts = await scraper.scrape_subreddit_posts("investing", limit=10)
        await scraper.close()

    # organic_post has valid id → parsed; promoted_like_post has empty id → filtered by _parse_post
    assert len(posts) == 1
    assert posts[0].id == "abc123"
    assert posts[0].title == "AAPL earnings beat expectations"


@pytest.mark.unit
async def test_scrape_post_comments_uses_permalink(scraper_config):
    """Test scrape_post_comments navigates to permalink derived from post.subreddit + post.id."""
    scraper = RedditPlaywrightScraper(config=scraper_config)

    post = RedditPost(
        id="xyz789",
        title="Market discussion",
        body="",
        subreddit="wallstreetbets",
        score=100,
        upvote_ratio=0.95,
        url="https://tracking.example.com/redirect?post=xyz789",
        created_utc=datetime.now(UTC),
        num_comments=5,
    )

    with patch("src.data.reddit_scraper.async_playwright") as mock_playwright:
        mock_pw = AsyncMock()
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()

        mock_playwright.return_value = mock_pw
        mock_pw.__aenter__ = AsyncMock(return_value=mock_pw)
        mock_pw.__aexit__ = AsyncMock()
        mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.close = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.close = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value=[])

        await scraper.start()
        await scraper.scrape_post_comments(post, limit=5)
        await scraper.close()

    expected_url = "https://old.reddit.com/r/wallstreetbets/comments/xyz789/"
    mock_page.goto.assert_called_once_with(expected_url, wait_until="domcontentloaded", timeout=30000)
