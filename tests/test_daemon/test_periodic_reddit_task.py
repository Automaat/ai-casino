"""Tests for PeriodicRedditScrapingTask."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.daemon.config.reddit import RedditScraperConfig
from src.daemon.tasks.data_tasks import PeriodicRedditScrapingTask
from src.data.reddit import RedditComment, RedditPost, TickerMention


@pytest.fixture
def mock_components():
    """Create mock daemon components."""
    components = Mock()
    components.config = Mock()
    components.config.reddit_scraper = RedditScraperConfig(
        enabled=True,
        use_llm_extraction=True,
        high_priority_subreddits=["wallstreetbets", "stocks"],
        posts_per_subreddit=10,
        comments_per_post=5,
    )
    components.container = Mock()
    components.container.llm_client = Mock(return_value=Mock())
    components.state = AsyncMock()
    components.state.get_last_reddit_scraping = AsyncMock(return_value=None)
    components.state.set_last_reddit_scraping = AsyncMock()
    return components


@pytest.fixture
def mock_container():
    """Create mock DI container."""
    container = Mock()
    container.llm_client = Mock(return_value=Mock())
    return container


@pytest.fixture
def reddit_task(mock_components, mock_container):
    """Create PeriodicRedditScrapingTask instance."""
    return PeriodicRedditScrapingTask(components=mock_components, container=mock_container)


@pytest.fixture
def sample_posts():
    """Create sample posts."""
    return [
        RedditPost(
            id="post1",
            title="TSLA to the moon",
            body="Bullish on Tesla",
            subreddit="wallstreetbets",
            score=1500,
            upvote_ratio=0.95,
            url="https://reddit.com/r/wallstreetbets/comments/post1",
            created_utc=datetime.now(UTC),
            num_comments=100,
        ),
        RedditPost(
            id="post2",
            title="NVDA earnings beat",
            body="AI boom continues",
            subreddit="wallstreetbets",
            score=800,
            upvote_ratio=0.90,
            url="https://reddit.com/r/wallstreetbets/comments/post2",
            created_utc=datetime.now(UTC),
            num_comments=50,
        ),
    ]


@pytest.fixture
def sample_comments():
    """Create sample comments."""
    return [
        RedditComment(
            id="c1",
            parent_post_id="post1",
            body="TSLA $350 EOW!",
            score=200,
            created_utc=datetime.now(UTC),
        ),
        RedditComment(
            id="c2",
            parent_post_id="post1",
            body="Loading more calls",
            score=150,
            created_utc=datetime.now(UTC),
        ),
    ]


@pytest.fixture
def sample_mentions():
    """Create sample ticker mentions."""
    return [
        TickerMention(symbol="TSLA", sentiment="BULLISH", context="moon", confidence=0.95),
        TickerMention(symbol="NVDA", sentiment="BULLISH", context="AI boom", confidence=0.90),
    ]


@pytest.mark.unit
def test_initialization(reddit_task):
    """Test task initialization."""
    assert reddit_task.task_name == "Reddit Scraping"
    assert reddit_task._posts_scraped == 0
    assert reddit_task._comments_scraped == 0
    assert reddit_task._mentions_extracted == 0
    assert reddit_task._skipped is False


@pytest.mark.unit
async def test_execute_skipped_when_disabled(mock_components, mock_container):
    """Test task is skipped when disabled in config."""
    mock_components.config.reddit_scraper.enabled = False
    task = PeriodicRedditScrapingTask(components=mock_components, container=mock_container)

    await task.execute()

    assert task._skipped is True
    assert task._posts_scraped == 0


@pytest.mark.unit
async def test_execute_success(reddit_task, sample_posts, sample_comments, sample_mentions):
    """Test successful execution."""
    mock_scraper = AsyncMock()
    mock_scraper.start = AsyncMock()
    mock_scraper.close = AsyncMock()
    mock_scraper.scrape_subreddit_posts = AsyncMock(return_value=sample_posts)
    mock_scraper.scrape_post_comments = AsyncMock(return_value=sample_comments)

    mock_extractor = Mock()
    mock_extractor.extract_tickers_batch = AsyncMock(
        return_value={"post1": sample_mentions, "post2": sample_mentions}
    )

    mock_session = AsyncMock()
    mock_post_repo = Mock()
    mock_post_repo.bulk_insert = AsyncMock(return_value=2)
    mock_comment_repo = Mock()
    mock_comment_repo.bulk_insert = AsyncMock(return_value=2)
    mock_mention_repo = Mock()
    mock_mention_repo.bulk_insert_all = AsyncMock(return_value=2)
    mock_sentiment_repo = Mock()
    mock_sentiment_repo.compute_hourly_aggregates = AsyncMock(return_value=1)

    with (
        patch("src.data.reddit_scraper.RedditPlaywrightScraper", return_value=mock_scraper),
        patch("src.data.reddit_ticker_extractor.RedditTickerExtractor", return_value=mock_extractor),
        patch("src.database.connection.get_session") as mock_get_session,
        patch("src.database.repositories.reddit.RedditPostRepository", return_value=mock_post_repo),
        patch("src.database.repositories.reddit.RedditCommentRepository", return_value=mock_comment_repo),
        patch(
            "src.database.repositories.reddit.RedditTickerMentionRepository",
            return_value=mock_mention_repo,
        ),
        patch(
            "src.database.repositories.reddit.RedditTickerSentimentRepository",
            return_value=mock_sentiment_repo,
        ),
    ):
        mock_get_session.return_value.__aenter__.return_value = mock_session

        await reddit_task.execute()

        # Verify scraper methods called (parallel subreddit listing)
        assert mock_scraper.start.called
        assert mock_scraper.scrape_subreddit_posts.call_count == 2  # 2 subreddits
        assert mock_scraper.close.called

        # Verify batch extraction called
        assert mock_extractor.extract_tickers_batch.called

        # Verify DB inserts
        assert mock_post_repo.bulk_insert.called
        assert mock_comment_repo.bulk_insert.called
        assert mock_mention_repo.bulk_insert_all.called
        assert mock_sentiment_repo.compute_hourly_aggregates.called

        # Verify stats recorded
        assert reddit_task._posts_scraped == 2
        assert reddit_task._comments_scraped == 2


@pytest.mark.unit
async def test_execute_handles_subreddit_failure(reddit_task, sample_posts):
    """Test graceful handling of subreddit scraping failure."""
    mock_scraper = AsyncMock()
    mock_scraper.start = AsyncMock()
    mock_scraper.close = AsyncMock()

    # First subreddit succeeds, second fails (caught by _scrape_listing)
    mock_scraper.scrape_subreddit_posts = AsyncMock(side_effect=[sample_posts, Exception("Subreddit banned")])
    mock_scraper.scrape_post_comments = AsyncMock(return_value=[])

    mock_extractor = Mock()
    mock_extractor.extract_tickers_batch = AsyncMock(return_value={})

    mock_session = AsyncMock()
    mock_post_repo = Mock()
    mock_post_repo.bulk_insert = AsyncMock(return_value=2)
    mock_comment_repo = Mock()
    mock_comment_repo.bulk_insert = AsyncMock(return_value=0)
    mock_mention_repo = Mock()
    mock_mention_repo.bulk_insert_all = AsyncMock(return_value=0)
    mock_sentiment_repo = Mock()
    mock_sentiment_repo.compute_hourly_aggregates = AsyncMock(return_value=0)

    with (
        patch("src.data.reddit_scraper.RedditPlaywrightScraper", return_value=mock_scraper),
        patch("src.data.reddit_ticker_extractor.RedditTickerExtractor", return_value=mock_extractor),
        patch("src.database.connection.get_session") as mock_get_session,
        patch("src.database.repositories.reddit.RedditPostRepository", return_value=mock_post_repo),
        patch("src.database.repositories.reddit.RedditCommentRepository", return_value=mock_comment_repo),
        patch(
            "src.database.repositories.reddit.RedditTickerMentionRepository",
            return_value=mock_mention_repo,
        ),
        patch(
            "src.database.repositories.reddit.RedditTickerSentimentRepository",
            return_value=mock_sentiment_repo,
        ),
    ):
        mock_get_session.return_value.__aenter__.return_value = mock_session

        # Should not raise, just log warning
        await reddit_task.execute()

        # First subreddit posts should still be inserted
        assert reddit_task._posts_scraped == 2


@pytest.mark.unit
async def test_execute_closes_scraper_on_exception(reddit_task):
    """Test scraper is closed even if exception occurs."""
    mock_scraper = AsyncMock()
    mock_scraper.scrape_subreddit_posts = AsyncMock(side_effect=Exception("Fatal error"))

    mock_extractor = AsyncMock()

    # Mock database to avoid table errors
    mock_session = AsyncMock()
    mock_post_repo = AsyncMock()
    mock_post_repo.bulk_insert = AsyncMock(return_value=0)
    mock_comment_repo = AsyncMock()
    mock_comment_repo.bulk_insert = AsyncMock(return_value=0)
    mock_mention_repo = AsyncMock()
    mock_sentiment_repo = AsyncMock()
    mock_sentiment_repo.compute_hourly_aggregates = AsyncMock(return_value=0)

    with (
        patch("src.data.reddit_scraper.RedditPlaywrightScraper", return_value=mock_scraper),
        patch("src.data.reddit_ticker_extractor.RedditTickerExtractor", return_value=mock_extractor),
        patch("src.database.connection.get_session") as mock_get_session,
        patch("src.database.repositories.reddit.RedditPostRepository", return_value=mock_post_repo),
        patch("src.database.repositories.reddit.RedditCommentRepository", return_value=mock_comment_repo),
        patch(
            "src.database.repositories.reddit.RedditTickerMentionRepository", return_value=mock_mention_repo
        ),
        patch(
            "src.database.repositories.reddit.RedditTickerSentimentRepository",
            return_value=mock_sentiment_repo,
        ),
    ):
        mock_get_session.return_value.__aenter__.return_value = mock_session

        # Should not raise - exceptions in scraping are caught and logged
        await reddit_task.execute()

    # Scraper should still be closed even after exception
    assert mock_scraper.close.called
    # No posts/comments scraped due to exception
    assert reddit_task._posts_scraped == 0


@pytest.mark.unit
async def test_get_last_run(reddit_task):
    """Test get_last_run delegates to state."""
    last_run = await reddit_task.get_last_run()
    assert last_run is None
    reddit_task.components.state.get_last_reddit_scraping.assert_called_once()


@pytest.mark.unit
async def test_record_success(reddit_task):
    """Test record_success persists timestamp via state."""
    reddit_task._posts_scraped = 10
    reddit_task._comments_scraped = 50
    reddit_task._mentions_extracted = 15

    await reddit_task.record_success(duration=5.0)

    reddit_task.components.state.set_last_reddit_scraping.assert_called_once()


@pytest.mark.unit
async def test_record_success_skipped(reddit_task):
    """Test record_success does nothing when skipped."""
    reddit_task._skipped = True

    # Should return early
    await reddit_task.record_success(duration=5.0)
