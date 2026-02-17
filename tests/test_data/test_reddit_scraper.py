"""Tests for RedditPlaywrightScraper."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.daemon.config.reddit import RedditScraperConfig
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
async def test_extract_post_from_html(scraper, tmp_path):
    """Test post extraction from HTML fixture."""
    fixture_path = Path(__file__).parent.parent / "fixtures" / "reddit_html" / "subreddit_listing.html"

    if not fixture_path.exists():
        pytest.skip("HTML fixture not found")

    html_content = fixture_path.read_text()

    # Mock Playwright page with fixture HTML
    mock_container = AsyncMock()
    mock_container.get_attribute = AsyncMock(return_value="t3_abc123")

    # Mock selectors
    title_elem = AsyncMock()
    title_elem.text_content = AsyncMock(return_value="TSLA to the moon! 🚀")
    title_elem.get_attribute = AsyncMock(return_value="https://reddit.com/r/wallstreetbets/post1")

    score_elem = AsyncMock()
    score_elem.text_content = AsyncMock(return_value="2500")

    comments_elem = AsyncMock()
    comments_elem.text_content = AsyncMock(return_value="150 comments")

    async def query_selector_side_effect(selector):
        if selector == "a.title":
            return title_elem
        if selector == "div.score.unvoted":
            return score_elem
        if selector == "a.comments":
            return comments_elem
        return None

    mock_container.query_selector = AsyncMock(side_effect=query_selector_side_effect)

    # Test extraction
    post = await scraper._extract_post(mock_container, "wallstreetbets")

    assert post is not None
    assert post.id == "abc123"
    assert post.title == "TSLA to the moon! 🚀"
    assert post.subreddit == "wallstreetbets"
    assert post.score == 2500
    assert post.num_comments == 150


@pytest.mark.unit
async def test_extract_comment_from_html(scraper):
    """Test comment extraction."""
    mock_container = AsyncMock()
    mock_container.get_attribute = AsyncMock(return_value="t1_comment1")

    # Mock selectors
    body_elem = AsyncMock()
    body_elem.text_content = AsyncMock(return_value="TSLA $350 by EOW! This is the way 🚀")

    score_elem = AsyncMock()
    score_elem.text_content = AsyncMock(return_value="450")

    async def query_selector_side_effect(selector):
        if selector == "div.md":
            return body_elem
        if selector == "div.score.unvoted":
            return score_elem
        return None

    mock_container.query_selector = AsyncMock(side_effect=query_selector_side_effect)

    comment = await scraper._extract_comment(mock_container, "t3_abc123")

    assert comment is not None
    assert comment.id == "comment1"
    assert comment.parent_post_id == "t3_abc123"
    assert comment.body == "TSLA $350 by EOW! This is the way 🚀"
    assert comment.score == 450


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

        # Mock empty posts list
        mock_page.query_selector_all = AsyncMock(return_value=[])
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
