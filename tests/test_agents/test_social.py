"""Tests for social sentiment analyst."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.social import SocialSentimentAnalysis, SocialSentimentAnalyst, SocialSentimentLLMResponse
from src.data.finnhub import (
    BuzzData,
    NewsSentimentData,
    SentimentBreakdown,
    SocialSentimentData,
    SocialSentimentEntry,
)
from src.data.reddit import RedditPost, RedditSentimentData
from src.models.sentiment import SentimentScore


@pytest.fixture
def mock_finnhub_fetcher():
    """Mock Finnhub fetcher."""
    mock = MagicMock()

    # Social sentiment data
    now = datetime.now(tz=UTC)
    reddit_entries = [
        SocialSentimentEntry(at_time=now - timedelta(days=i), mention=10 + i, score=0.5 + (i * 0.05))
        for i in range(7)
    ]
    twitter_entries = [
        SocialSentimentEntry(at_time=now - timedelta(days=i), mention=20 + i, score=0.6 + (i * 0.03))
        for i in range(7)
    ]
    mock.fetch_social_sentiment.return_value = SocialSentimentData(
        symbol="AAPL", reddit=reddit_entries, twitter=twitter_entries, fetched_at=now
    )

    # News sentiment data
    mock.fetch_sentiment_indicator.return_value = NewsSentimentData(
        symbol="AAPL",
        buzz=BuzzData(articles_in_last_week=50, buzz=1.5, weekly_average=1.2),
        company_news_score=0.7,
        sector_avg_bullish_percent=55.0,
        sector_avg_news_score=0.6,
        sentiment=SentimentBreakdown(bearish_percent=30.0, bullish_percent=70.0),
        fetched_at=now,
    )

    return mock


@pytest.fixture
def mock_reddit_fetcher():
    """Mock Reddit fetcher."""
    mock = MagicMock()

    posts = [
        RedditPost(
            id=f"post{i}",
            title=f"AAPL bullish analysis {i}",
            body=f"Body {i}",
            subreddit="wallstreetbets",
            score=100 + i * 10,
            upvote_ratio=0.85 + (i * 0.01),
            url=f"https://reddit.com/post{i}",
            created_utc=datetime.now(tz=UTC),
            num_comments=50,
        )
        for i in range(5)
    ]

    mock.fetch_mentions.return_value = RedditSentimentData(
        symbol="AAPL",
        posts=posts,
        mention_count=len(posts),
        avg_score=110.0,
        avg_upvote_ratio=0.87,
        fetched_at=datetime.now(tz=UTC),
    )

    return mock


@pytest.fixture
def mock_finbert():
    """Mock FinBERT sentiment analyzer."""
    mock = MagicMock()
    mock.analyze_batch.return_value = [
        SentimentScore(positive=0.7, neutral=0.2, negative=0.1, label="positive", score=0.6) for _ in range(5)
    ]
    return mock


async def test_analyze_full_data(test_container, mock_finnhub_fetcher, mock_reddit_fetcher, mock_finbert):
    """Test analysis with all sources available."""

    async def astructured_response(*args, **kwargs):
        return SocialSentimentLLMResponse(
            interpretation="Strong bullish sentiment across social platforms",
            sentiment_label="BULLISH",
            confidence_keywords=["strong", "clear"],
        )

    analyst = test_container.social_sentiment_analyst()
    analyst.finnhub_fetcher = mock_finnhub_fetcher
    analyst.reddit_fetcher = mock_reddit_fetcher
    analyst.finbert = mock_finbert
    analyst.llm.astructured = AsyncMock(side_effect=astructured_response)

    # Mock _fetch_all_sources to return the expected data
    async def fetch_all_mock(symbol):
        return (
            mock_finnhub_fetcher.fetch_social_sentiment(symbol),
            mock_finnhub_fetcher.fetch_sentiment_indicator(symbol),
            mock_reddit_fetcher.fetch_mentions(symbol),
        )

    analyst._fetch_all_sources = fetch_all_mock

    result = await analyst.analyze("AAPL")

    assert isinstance(result, SocialSentimentAnalysis)
    assert result.finnhub_sentiment is not None
    assert result.reddit_sentiment is not None
    assert -1.0 <= result.overall_social_score <= 1.0
    assert result.social_momentum in ["rising", "falling", "stable"]
    assert result.wsb_mentions_24h == 5
    assert result.sentiment_label == "BULLISH"
    assert result.interpretation == "Strong bullish sentiment across social platforms"
    assert 0.0 <= result.confidence <= 1.0


async def test_analyze_missing_finnhub(test_container, mock_reddit_fetcher, mock_finbert):
    """Test analysis when Finnhub fails."""
    analyst = test_container.social_sentiment_analyst()
    analyst.reddit_fetcher = mock_reddit_fetcher
    analyst.finbert = mock_finbert

    # Make Finnhub fail
    async def fetch_all_mock(symbol):
        return None, None, mock_reddit_fetcher.fetch_mentions(symbol)

    analyst._fetch_all_sources = fetch_all_mock

    result = await analyst.analyze("AAPL")

    assert isinstance(result, SocialSentimentAnalysis)
    assert result.finnhub_sentiment is None
    assert result.reddit_sentiment is not None
    assert result.wsb_mentions_24h == 5
    assert 0.0 <= result.confidence <= 1.0


async def test_analyze_no_reddit_mentions(test_container, mock_finnhub_fetcher, mock_finbert):
    """Test analysis with zero Reddit posts."""
    mock_reddit = MagicMock()
    mock_reddit.fetch_mentions.return_value = RedditSentimentData(
        symbol="AAPL",
        posts=[],
        mention_count=0,
        avg_score=0.0,
        avg_upvote_ratio=0.0,
        fetched_at=datetime.now(tz=UTC),
    )

    analyst = test_container.social_sentiment_analyst()
    analyst.finnhub_fetcher = mock_finnhub_fetcher
    analyst.reddit_fetcher = mock_reddit
    analyst.finbert = mock_finbert

    result = await analyst.analyze("AAPL")

    assert result.wsb_mentions_24h == 0
    assert result.reddit_sentiment is None


async def test_analyze_all_sources_failed(test_container, mock_finbert):
    """Test analysis when all APIs fail."""
    analyst = test_container.social_sentiment_analyst()
    analyst.finbert = mock_finbert

    # Make all sources fail
    async def fetch_all_fail(symbol):
        return None, None, None

    analyst._fetch_all_sources = fetch_all_fail

    result = await analyst.analyze("AAPL")

    assert result.overall_social_score == 0.0
    assert result.confidence < 0.5  # Low confidence due to no data
    assert result.sentiment_label in ["BULLISH", "BEARISH", "NEUTRAL"]
    assert result.wsb_mentions_24h == 0


def test_compute_overall_score_weighted():
    """Test weighted score computation."""
    analyst = SocialSentimentAnalyst(MagicMock(), MagicMock(), MagicMock(), MagicMock())

    # Create mock data
    now = datetime.now(tz=UTC)
    finnhub_social = SocialSentimentData(
        symbol="AAPL",
        reddit=[SocialSentimentEntry(at_time=now, mention=10, score=0.8)],
        twitter=[SocialSentimentEntry(at_time=now, mention=20, score=0.6)],
        fetched_at=now,
    )

    finnhub_news = NewsSentimentData(
        symbol="AAPL",
        buzz=BuzzData(articles_in_last_week=50, buzz=1.5, weekly_average=1.2),
        company_news_score=0.7,
        sector_avg_bullish_percent=55.0,
        sector_avg_news_score=0.6,
        sentiment=SentimentBreakdown(bearish_percent=30.0, bullish_percent=70.0),
        fetched_at=now,
    )

    reddit_sentiment = 0.5

    score = analyst._compute_overall_social_score(finnhub_social, finnhub_news, reddit_sentiment)

    assert -1.0 <= score <= 1.0


def test_compute_overall_score_missing_sources():
    """Test score computation with missing sources."""
    analyst = SocialSentimentAnalyst(MagicMock(), MagicMock(), MagicMock(), MagicMock())

    # Only reddit sentiment
    score = analyst._compute_overall_social_score(None, None, 0.5)
    assert score == 0.5

    # No sources
    score = analyst._compute_overall_social_score(None, None, None)
    assert score == 0.0


def test_compute_momentum_thresholds():
    """Test momentum threshold detection."""
    analyst = SocialSentimentAnalyst(MagicMock(), MagicMock(), MagicMock(), MagicMock())

    now = datetime.now(tz=UTC)

    # Rising momentum: recent > older by 0.15
    recent_entries = [
        SocialSentimentEntry(at_time=now - timedelta(days=i), mention=10, score=0.8) for i in range(3)
    ]
    older_entries = [
        SocialSentimentEntry(at_time=now - timedelta(days=i + 3), mention=10, score=0.5) for i in range(4)
    ]

    finnhub_social = SocialSentimentData(
        symbol="AAPL", reddit=recent_entries + older_entries, twitter=[], fetched_at=now
    )

    momentum = analyst._compute_social_momentum(finnhub_social)
    assert momentum == "rising"

    # Falling momentum: recent < older by 0.15
    recent_entries = [
        SocialSentimentEntry(at_time=now - timedelta(days=i), mention=10, score=0.4) for i in range(3)
    ]
    older_entries = [
        SocialSentimentEntry(at_time=now - timedelta(days=i + 3), mention=10, score=0.7) for i in range(4)
    ]

    finnhub_social = SocialSentimentData(
        symbol="AAPL", reddit=recent_entries + older_entries, twitter=[], fetched_at=now
    )

    momentum = analyst._compute_social_momentum(finnhub_social)
    assert momentum == "falling"

    # Stable momentum: diff < 0.1
    recent_entries = [
        SocialSentimentEntry(at_time=now - timedelta(days=i), mention=10, score=0.6) for i in range(3)
    ]
    older_entries = [
        SocialSentimentEntry(at_time=now - timedelta(days=i + 3), mention=10, score=0.62) for i in range(4)
    ]

    finnhub_social = SocialSentimentData(
        symbol="AAPL", reddit=recent_entries + older_entries, twitter=[], fetched_at=now
    )

    momentum = analyst._compute_social_momentum(finnhub_social)
    assert momentum == "stable"


def test_compute_confidence_high():
    """Test high confidence computation."""
    analyst = SocialSentimentAnalyst(MagicMock(), MagicMock(), MagicMock(), MagicMock())

    now = datetime.now(tz=UTC)
    finnhub_social = SocialSentimentData(
        symbol="AAPL",
        reddit=[SocialSentimentEntry(at_time=now, mention=10, score=0.7)],
        twitter=[SocialSentimentEntry(at_time=now, mention=20, score=0.7)],
        fetched_at=now,
    )

    finnhub_news = NewsSentimentData(
        symbol="AAPL",
        buzz=BuzzData(articles_in_last_week=50, buzz=1.5, weekly_average=1.2),
        company_news_score=0.7,
        sector_avg_bullish_percent=55.0,
        sector_avg_news_score=0.6,
        sentiment=SentimentBreakdown(bearish_percent=30.0, bullish_percent=70.0),
        fetched_at=now,
    )

    # Create real RedditPost objects
    posts = [
        RedditPost(
            id=f"{i}",
            title=f"Post {i}",
            body="",
            subreddit="wallstreetbets",
            score=100,
            upvote_ratio=0.85,
            url=f"https://reddit.com/{i}",
            created_utc=now,
            num_comments=50,
        )
        for i in range(60)
    ]

    reddit_data = RedditSentimentData(
        symbol="AAPL", posts=posts, mention_count=60, avg_score=100.0, avg_upvote_ratio=0.85, fetched_at=now
    )

    confidence = analyst._compute_confidence(
        finnhub_social, finnhub_news, reddit_data, 0.7, ["strong", "clear"]
    )

    assert confidence >= 0.8


def test_compute_confidence_low():
    """Test low confidence computation."""
    analyst = SocialSentimentAnalyst(MagicMock(), MagicMock(), MagicMock(), MagicMock())

    # Missing sources, low mentions
    reddit_data = RedditSentimentData(
        symbol="AAPL",
        posts=[],
        mention_count=5,
        avg_score=0.0,
        avg_upvote_ratio=0.0,
        fetched_at=datetime.now(tz=UTC),
    )

    confidence = analyst._compute_confidence(None, None, reddit_data, None, [])

    assert confidence <= 0.5


async def test_compute_reddit_sentiment_weighted():
    """Test Reddit sentiment weighting."""
    analyst = SocialSentimentAnalyst(MagicMock(), MagicMock(), MagicMock(), MagicMock())

    # Mock FinBERT to return positive sentiment
    analyst.finbert.analyze_batch.return_value = [
        SentimentScore(positive=0.8, neutral=0.1, negative=0.1, label="positive", score=0.7) for _ in range(2)
    ]

    posts = [
        RedditPost(
            id="1",
            title="AAPL bullish",
            body="",
            subreddit="wsb",
            score=100,
            upvote_ratio=0.9,
            url="https://reddit.com/1",
            created_utc=datetime.now(tz=UTC),
            num_comments=50,
        ),
        RedditPost(
            id="2",
            title="AAPL analysis",
            body="",
            subreddit="wsb",
            score=50,
            upvote_ratio=0.7,
            url="https://reddit.com/2",
            created_utc=datetime.now(tz=UTC),
            num_comments=20,
        ),
    ]

    reddit_data = RedditSentimentData(
        symbol="AAPL",
        posts=posts,
        mention_count=2,
        avg_score=75.0,
        avg_upvote_ratio=0.8,
        fetched_at=datetime.now(tz=UTC),
    )

    sentiment = await analyst._compute_reddit_sentiment(reddit_data)

    assert sentiment is not None
    assert -1.0 <= sentiment <= 1.0


async def test_structured_output_fallback(
    test_container, mock_finnhub_fetcher, mock_reddit_fetcher, mock_finbert
):
    """Test fallback when structured output fails."""
    from src.models.providers.base import StructuredOutputError

    analyst = test_container.social_sentiment_analyst()
    analyst.finnhub_fetcher = mock_finnhub_fetcher
    analyst.reddit_fetcher = mock_reddit_fetcher
    analyst.finbert = mock_finbert

    # Fail structured output
    async def astructured_fail(*args, **kwargs):
        msg = "Structured output failed"
        raise StructuredOutputError(msg, raw_response=None)

    analyst.llm.astructured = AsyncMock(side_effect=astructured_fail)
    analyst.llm.acomplete = AsyncMock(return_value="The stock shows bullish sentiment with strong signals")

    result = await analyst.analyze("AAPL")

    assert result.sentiment_label == "BULLISH"
    assert len(result.interpretation) > 0


def test_format_finnhub_summary():
    """Test Finnhub summary formatting."""
    analyst = SocialSentimentAnalyst(MagicMock(), MagicMock(), MagicMock(), MagicMock())

    now = datetime.now(tz=UTC)
    data = SocialSentimentData(
        symbol="AAPL",
        reddit=[SocialSentimentEntry(at_time=now, mention=10, score=0.7)],
        twitter=[SocialSentimentEntry(at_time=now, mention=20, score=0.6)],
        fetched_at=now,
    )

    summary = analyst._format_finnhub_summary(data)

    assert "Reddit" in summary
    assert "Twitter" in summary
    assert "0.7" in summary or "0.70" in summary


def test_format_reddit_posts():
    """Test Reddit posts formatting."""
    analyst = SocialSentimentAnalyst(MagicMock(), MagicMock(), MagicMock(), MagicMock())

    posts = [
        RedditPost(
            id=f"{i}",
            title=f"Post {i}",
            body="",
            subreddit="wallstreetbets",
            score=100 - i * 10,
            upvote_ratio=0.85,
            url=f"https://reddit.com/{i}",
            created_utc=datetime.now(tz=UTC),
            num_comments=50,
        )
        for i in range(10)
    ]

    reddit_data = RedditSentimentData(
        symbol="AAPL",
        posts=posts,
        mention_count=10,
        avg_score=50.0,
        avg_upvote_ratio=0.85,
        fetched_at=datetime.now(tz=UTC),
    )

    formatted = analyst._format_reddit_posts(reddit_data, limit=5)

    lines = formatted.split("\n")
    assert len(lines) == 5  # Top 5 posts
    assert "Post 0" in formatted  # Highest score
