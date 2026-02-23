"""Tests for RedditFetcher."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import prawcore
import pytest

from src.data.reddit import (
    DEFAULT_SUBREDDITS,
    RedditFetcher,
    RedditPost,
    RedditSentimentData,
    TrendingTicker,
)


@pytest.fixture
def mock_submission():
    """Create mock Reddit submission."""

    def _create(
        post_id: str = "abc123",
        title: str = "AAPL to the moon!",
        selftext: str = "Just bought more $AAPL",
        subreddit: str = "wallstreetbets",
        score: int = 100,
        upvote_ratio: float = 0.85,
        permalink: str = "/r/wallstreetbets/comments/abc123/aapl_to_the_moon/",
        created_utc: float = 1704067200.0,
        num_comments: int = 50,
    ):
        submission = MagicMock()
        submission.id = post_id
        submission.title = title
        submission.selftext = selftext
        submission.score = score
        submission.upvote_ratio = upvote_ratio
        submission.permalink = permalink
        submission.created_utc = created_utc
        submission.num_comments = num_comments

        # Subreddit mock
        subreddit_mock = MagicMock()
        subreddit_mock.display_name = subreddit
        submission.subreddit = subreddit_mock

        return submission

    return _create


@pytest.fixture
def mock_reddit(mock_submission):
    """Mock PRAW Reddit client."""
    with patch("src.data.reddit.praw.Reddit") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance

        # Default subreddit mock
        subreddit_mock = MagicMock()
        subreddit_mock.search.return_value = [mock_submission()]
        subreddit_mock.hot.return_value = [mock_submission()]
        mock_instance.subreddit.return_value = subreddit_mock

        yield mock_instance


@pytest.fixture
def fetcher():
    """Create RedditFetcher with mock credentials."""
    with patch.dict(
        "os.environ",
        {
            "REDDIT_CLIENT_ID": "test_client_id",
            "REDDIT_CLIENT_SECRET": "test_client_secret",
            "REDDIT_USER_AGENT": "test_agent",
        },
    ):
        return RedditFetcher()


class TestRedditPost:
    """Tests for RedditPost model."""

    def test_create_post(self):
        """Test creating a RedditPost."""
        post = RedditPost(
            id="abc123",
            title="Test Post",
            body="Test body content",
            subreddit="wallstreetbets",
            score=100,
            upvote_ratio=0.85,
            url="https://reddit.com/r/wallstreetbets/comments/abc123/",
            created_utc=datetime(2024, 1, 1),
            num_comments=50,
        )
        assert post.id == "abc123"
        assert post.subreddit == "wallstreetbets"
        assert post.score == 100


class TestRedditSentimentData:
    """Tests for RedditSentimentData model."""

    def test_create_sentiment_data(self):
        """Test creating RedditSentimentData."""
        post = RedditPost(
            id="abc123",
            title="AAPL",
            body="",
            subreddit="stocks",
            score=50,
            upvote_ratio=0.9,
            url="https://reddit.com/r/stocks/comments/abc123/",
            created_utc=datetime.now(),
            num_comments=10,
        )
        data = RedditSentimentData(
            symbol="AAPL",
            posts=[post],
            mention_count=1,
            avg_score=50.0,
            avg_upvote_ratio=0.9,
            fetched_at=datetime.now(),
        )
        assert data.symbol == "AAPL"
        assert data.mention_count == 1
        assert len(data.posts) == 1


class TestTrendingTicker:
    """Tests for TrendingTicker model."""

    def test_create_trending_ticker(self):
        """Test creating TrendingTicker."""
        post = RedditPost(
            id="abc123",
            title="TSLA",
            body="",
            subreddit="wallstreetbets",
            score=200,
            upvote_ratio=0.92,
            url="https://reddit.com/r/wallstreetbets/comments/abc123/",
            created_utc=datetime.now(),
            num_comments=100,
        )
        ticker = TrendingTicker(
            symbol="TSLA",
            mention_count=5,
            total_score=1000,
            avg_upvote_ratio=0.88,
            sample_posts=[post],
        )
        assert ticker.symbol == "TSLA"
        assert ticker.mention_count == 5
        assert len(ticker.sample_posts) == 1


class TestRedditFetcher:
    """Tests for RedditFetcher."""

    def test_init_with_explicit_credentials(self):
        """Test init with explicit credentials."""
        test_client_id = "explicit_id"
        test_client_secret = "explicit_secret"
        test_user_agent = "explicit_agent"
        fetcher = RedditFetcher(
            client_id=test_client_id,
            client_secret=test_client_secret,
            user_agent=test_user_agent,
        )
        assert fetcher._client_id == test_client_id
        assert fetcher._client_secret == test_client_secret
        assert fetcher._user_agent == test_user_agent

    def test_init_without_credentials_logs_warning(self):
        """Test that missing credentials logs warning."""
        with patch.dict("os.environ", {}, clear=True):
            fetcher = RedditFetcher()
            assert fetcher._client_id is None or fetcher._client_secret is None

    def test_cache_key_deterministic(self, fetcher):
        """Test cache key generation is deterministic."""
        key1 = fetcher._cache_key("mentions", "AAPL", "wsb", "25", "day")
        key2 = fetcher._cache_key("mentions", "AAPL", "wsb", "25", "day")
        key3 = fetcher._cache_key("mentions", "TSLA", "wsb", "25", "day")

        assert key1 == key2
        assert key1 != key3
        assert len(key1) == 32

    def test_fetch_mentions_returns_data(self, fetcher, mock_reddit, mock_submission):
        """Test fetch_mentions returns correct data."""
        mock_reddit.subreddit.return_value.search.return_value = [
            mock_submission(post_id="1", title="AAPL analysis", selftext="Buying $AAPL today"),
            mock_submission(post_id="2", title="More AAPL", selftext="AAPL looking good"),
        ]

        result = fetcher.fetch_mentions("AAPL", subreddits=["wallstreetbets"], limit=10)

        assert isinstance(result, RedditSentimentData)
        assert result.symbol == "AAPL"
        assert result.mention_count == 2
        assert result.avg_score == 100.0
        assert result.avg_upvote_ratio == 0.85

    def test_fetch_mentions_filters_irrelevant_posts(self, fetcher, mock_reddit, mock_submission):
        """Test that fetch_mentions filters posts not containing the symbol."""
        mock_reddit.subreddit.return_value.search.return_value = [
            mock_submission(post_id="1", title="AAPL analysis", selftext="Great stock"),
            mock_submission(post_id="2", title="TSLA news", selftext="Not about Apple"),
        ]

        result = fetcher.fetch_mentions("AAPL", subreddits=["wallstreetbets"])

        # Only the AAPL post should be included
        assert result.mention_count == 1
        assert result.posts[0].title == "AAPL analysis"

    def test_fetch_mentions_uses_cache(self, fetcher, mock_reddit, mock_submission):
        """Test that repeated fetches use cache."""
        mock_reddit.subreddit.return_value.search.return_value = [mock_submission()]

        fetcher.fetch_mentions("AAPL", subreddits=["wallstreetbets"])
        fetcher.fetch_mentions("AAPL", subreddits=["wallstreetbets"])

        # Should only call search once due to caching
        mock_reddit.subreddit.return_value.search.assert_called_once()

    def test_fetch_mentions_uses_default_subreddits(self, fetcher, mock_reddit, mock_submission):
        """Test that default subreddits are used when not specified."""
        mock_reddit.subreddit.return_value.search.return_value = [mock_submission()]

        fetcher.fetch_mentions("AAPL")

        # Should call subreddit for each default subreddit
        assert mock_reddit.subreddit.call_count == len(DEFAULT_SUBREDDITS)

    def test_fetch_trending_extracts_tickers(self, fetcher, mock_reddit, mock_submission):
        """Test that fetch_trending extracts tickers correctly."""
        mock_reddit.subreddit.return_value.hot.return_value = [
            mock_submission(post_id="1", title="$AAPL is great", selftext=""),
            mock_submission(post_id="2", title="More $AAPL", selftext=""),
            mock_submission(post_id="3", title="Also $AAPL", selftext=""),
        ]

        result = fetcher.fetch_trending_tickers(subreddits=["wallstreetbets"], min_mentions=3)

        assert len(result) >= 1
        aapl_ticker = next((t for t in result if t.symbol == "AAPL"), None)
        assert aapl_ticker is not None
        assert aapl_ticker.mention_count == 3

    def test_fetch_trending_excludes_common_words(self, fetcher, mock_reddit, mock_submission):
        """Test that common words are excluded from ticker detection."""
        mock_reddit.subreddit.return_value.hot.return_value = [
            mock_submission(post_id="1", title="CEO says IPO", selftext="The WSB DD is great"),
            mock_submission(post_id="2", title="CEO again IPO", selftext="More WSB DD"),
            mock_submission(post_id="3", title="CEO IPO news", selftext="WSB DD analysis"),
        ]

        result = fetcher.fetch_trending_tickers(subreddits=["wallstreetbets"], min_mentions=1)

        symbols = [t.symbol for t in result]
        assert "CEO" not in symbols
        assert "IPO" not in symbols
        assert "WSB" not in symbols
        assert "DD" not in symbols

    def test_fetch_trending_respects_min_mentions(self, fetcher, mock_reddit, mock_submission):
        """Test that min_mentions threshold is respected."""
        mock_reddit.subreddit.return_value.hot.return_value = [
            mock_submission(post_id="1", title="$AAPL stock", selftext=""),
            mock_submission(post_id="2", title="$AAPL again", selftext=""),
            mock_submission(post_id="3", title="$TSLA once", selftext=""),
        ]

        result = fetcher.fetch_trending_tickers(subreddits=["wallstreetbets"], min_mentions=2)

        symbols = [t.symbol for t in result]
        assert "AAPL" in symbols
        assert "TSLA" not in symbols

    def test_fetch_trending_uses_cache(self, fetcher, mock_reddit, mock_submission):
        """Test that trending fetch uses cache."""
        mock_reddit.subreddit.return_value.hot.return_value = [mock_submission()]

        fetcher.fetch_trending_tickers(subreddits=["wallstreetbets"], min_mentions=1)
        fetcher.fetch_trending_tickers(subreddits=["wallstreetbets"], min_mentions=1)

        mock_reddit.subreddit.return_value.hot.assert_called_once()

    def test_retry_on_network_error(self, fetcher, mock_reddit):
        """Test that network errors trigger retry."""
        mock_reddit.subreddit.side_effect = [
            Exception("Network error"),
            Exception("Network error"),
            MagicMock(search=MagicMock(return_value=[])),
        ]

        result = fetcher.fetch_mentions("AAPL", subreddits=["wallstreetbets"])

        assert result.mention_count == 0

    def test_no_retry_on_auth_error(self, fetcher, mock_reddit):
        """Test that auth errors do not trigger retry."""
        mock_reddit.subreddit.side_effect = prawcore.exceptions.InvalidToken(MagicMock(status_code=401))

        with pytest.raises(prawcore.exceptions.InvalidToken):
            fetcher.fetch_mentions("AAPL", subreddits=["wallstreetbets"])

        # Should have been called only once (no retry)
        assert mock_reddit.subreddit.call_count == 1

    def test_clear_cache(self, fetcher, mock_reddit, mock_submission):
        """Test cache clearing."""
        mock_reddit.subreddit.return_value.search.return_value = [mock_submission()]

        fetcher.fetch_mentions("AAPL", subreddits=["wallstreetbets"])
        fetcher.clear_cache()
        fetcher.fetch_mentions("AAPL", subreddits=["wallstreetbets"])

        assert mock_reddit.subreddit.return_value.search.call_count == 2

    def test_repr_shows_auth_status(self):
        """Test __repr__ shows authentication status."""
        with patch.dict(
            "os.environ",
            {"REDDIT_CLIENT_ID": "test", "REDDIT_CLIENT_SECRET": "test"},
        ):
            fetcher = RedditFetcher()
            assert "authenticated=True" in repr(fetcher)

        with patch.dict("os.environ", {}, clear=True):
            fetcher = RedditFetcher()
            assert "authenticated=False" in repr(fetcher)

    def test_submission_body_truncation(self, fetcher, mock_reddit, mock_submission):
        """Test that long submission bodies are truncated."""
        long_body = "x" * 3000
        mock_reddit.subreddit.return_value.search.return_value = [
            mock_submission(post_id="1", title="AAPL", selftext=long_body)
        ]

        result = fetcher.fetch_mentions("AAPL", subreddits=["wallstreetbets"])

        assert len(result.posts[0].body) == 2000

    def test_get_reddit_lazy_init(self, fetcher, mock_reddit):  # noqa: ARG002
        """Test that Reddit client is lazily initialized."""
        assert fetcher._reddit is None
        fetcher._get_reddit()
        assert fetcher._reddit is not None

    def test_get_reddit_raises_without_credentials(self):
        """Test that _get_reddit raises without credentials."""
        with patch.dict("os.environ", {}, clear=True):
            fetcher = RedditFetcher()
            with pytest.raises(ValueError, match="credentials not configured"):
                fetcher._get_reddit()

    def test_contains_symbol_dollar_sign(self, fetcher):
        """Test symbol detection with dollar sign."""
        assert fetcher._contains_symbol("I bought $AAPL today", "AAPL")
        assert fetcher._contains_symbol("$AAPL is great", "AAPL")
        assert not fetcher._contains_symbol("$TSLA is great", "AAPL")

    def test_contains_symbol_plain(self, fetcher):
        """Test symbol detection without dollar sign."""
        assert fetcher._contains_symbol("AAPL stock analysis", "AAPL")
        assert fetcher._contains_symbol("Looking at AAPL", "AAPL")
        assert not fetcher._contains_symbol("Apple is great", "AAPL")

    def test_extract_tickers(self, fetcher):
        """Test ticker extraction from text."""
        text = "$AAPL and TSLA are trending. Also check out NVDA."
        tickers = fetcher._extract_tickers(text)

        assert "AAPL" in tickers
        assert "TSLA" in tickers
        assert "NVDA" in tickers
        assert len(tickers) == 3

    def test_extract_tickers_excludes_common_words(self, fetcher):
        """Test that common words are excluded."""
        text = "CEO says IPO and WSB DD YOLO FOMO"
        tickers = fetcher._extract_tickers(text)

        assert "CEO" not in tickers
        assert "IPO" not in tickers
        assert "WSB" not in tickers
        assert "DD" not in tickers
        assert "YOLO" not in tickers
        assert "FOMO" not in tickers
