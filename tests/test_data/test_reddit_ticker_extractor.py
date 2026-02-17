"""Tests for RedditTickerExtractor."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

import pytest

from src.daemon.config.reddit import RedditScraperConfig
from src.data.reddit import RedditComment, RedditPost, TickerMention
from src.data.reddit_ticker_extractor import RedditTickerExtractor, TickerExtractionResponse


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client."""
    client = Mock()
    client.astructured = AsyncMock()
    return client


@pytest.fixture
def extractor_config():
    """Create test extractor config."""
    return RedditScraperConfig(
        use_llm_extraction=True,
        extraction_model="claude-3-5-sonnet-20241022",
        extraction_temperature=0.3,
        extraction_min_confidence=0.7,
    )


@pytest.fixture
def extractor(mock_llm_client, extractor_config):
    """Create extractor instance."""
    return RedditTickerExtractor(llm_client=mock_llm_client, config=extractor_config)


@pytest.fixture
def sample_post():
    """Create sample Reddit post."""
    return RedditPost(
        id="abc123",
        title="TSLA to the moon! 🚀",
        body="Tesla is going parabolic! Just loaded calls. Price target $300.",
        subreddit="wallstreetbets",
        score=2500,
        upvote_ratio=0.95,
        url="https://reddit.com/r/wallstreetbets/comments/abc123",
        created_utc=datetime.now(UTC),
        num_comments=150,
    )


@pytest.fixture
def sample_comments():
    """Create sample comments."""
    return [
        RedditComment(
            id="c1",
            parent_post_id="abc123",
            body="TSLA $350 by EOW! This is the way 🚀",
            score=450,
            created_utc=datetime.now(UTC),
        ),
        RedditComment(
            id="c2",
            parent_post_id="abc123",
            body="Also watching NVDA and AMD for tech exposure.",
            score=120,
            created_utc=datetime.now(UTC),
        ),
        RedditComment(
            id="c3",
            parent_post_id="abc123",
            body="TSLA puts printing. This is overbought.",
            score=85,
            created_utc=datetime.now(UTC),
        ),
    ]


@pytest.mark.unit
async def test_extract_tickers_success(extractor, mock_llm_client, sample_post, sample_comments):
    """Test successful ticker extraction."""
    # Mock LLM response
    mock_response = TickerExtractionResponse(
        mentions=[
            TickerMention(
                symbol="TSLA",
                sentiment="BULLISH",
                context="Tesla is going parabolic",
                confidence=0.95,
            ),
            TickerMention(
                symbol="NVDA",
                sentiment="NEUTRAL",
                context="watching NVDA for tech exposure",
                confidence=0.80,
            ),
        ]
    )
    mock_llm_client.astructured.return_value = mock_response

    mentions = await extractor.extract_tickers(sample_post, sample_comments)

    assert len(mentions) == 2
    assert mentions[0].symbol == "TSLA"
    assert mentions[0].sentiment == "BULLISH"
    assert mentions[0].confidence == 0.95
    assert mentions[1].symbol == "NVDA"
    assert mentions[1].sentiment == "NEUTRAL"
    mock_llm_client.astructured.assert_called_once()


@pytest.mark.unit
async def test_extract_tickers_filters_low_confidence(extractor, mock_llm_client, sample_post):
    """Test low confidence tickers are filtered."""
    mock_response = TickerExtractionResponse(
        mentions=[
            TickerMention(
                symbol="TSLA",
                sentiment="BULLISH",
                context="Tesla mentioned",
                confidence=0.95,
            ),
            TickerMention(
                symbol="FAKE",
                sentiment="NEUTRAL",
                context="maybe fake ticker",
                confidence=0.50,  # Below 0.7 threshold
            ),
        ]
    )
    mock_llm_client.astructured.return_value = mock_response

    mentions = await extractor.extract_tickers(sample_post)

    assert len(mentions) == 1
    assert mentions[0].symbol == "TSLA"


@pytest.mark.unit
async def test_extract_tickers_filters_false_positives(extractor, mock_llm_client, sample_post):
    """Test false positive tickers are filtered."""
    mock_response = TickerExtractionResponse(
        mentions=[
            TickerMention(symbol="TSLA", sentiment="BULLISH", context="Tesla", confidence=0.95),
            TickerMention(symbol="CEO", sentiment="NEUTRAL", context="CEO announced", confidence=0.85),
            TickerMention(symbol="YOLO", sentiment="BULLISH", context="YOLO into calls", confidence=0.90),
            TickerMention(symbol="DD", sentiment="NEUTRAL", context="DD posted", confidence=0.88),
        ]
    )
    mock_llm_client.astructured.return_value = mock_response

    mentions = await extractor.extract_tickers(sample_post)

    # Only TSLA should pass (CEO, YOLO, DD are in FALSE_POSITIVES)
    assert len(mentions) == 1
    assert mentions[0].symbol == "TSLA"


@pytest.mark.unit
async def test_extract_tickers_validates_symbol_format(extractor, mock_llm_client, sample_post):
    """Test symbol format validation."""
    mock_response = TickerExtractionResponse(
        mentions=[
            TickerMention(symbol="TSLA", sentiment="BULLISH", context="Tesla", confidence=0.95),
            TickerMention(symbol="BRK.B", sentiment="NEUTRAL", context="Berkshire", confidence=0.90),
            TickerMention(symbol="INVALID_LONG", sentiment="NEUTRAL", context="too long", confidence=0.85),
            TickerMention(symbol="12", sentiment="NEUTRAL", context="numbers", confidence=0.85),
        ]
    )
    mock_llm_client.astructured.return_value = mock_response

    mentions = await extractor.extract_tickers(sample_post)

    # Only TSLA and BRK.B should pass (valid format)
    symbols = [m.symbol for m in mentions]
    assert "TSLA" in symbols
    assert "BRK.B" in symbols
    assert "INVALID_LONG" not in symbols
    assert "12" not in symbols


@pytest.mark.unit
async def test_extract_tickers_handles_empty_response(extractor, mock_llm_client, sample_post):
    """Test handling of empty LLM response."""
    mock_response = TickerExtractionResponse(mentions=[])
    mock_llm_client.astructured.return_value = mock_response

    mentions = await extractor.extract_tickers(sample_post)

    assert mentions == []


@pytest.mark.unit
async def test_extract_tickers_truncates_long_content(extractor, mock_llm_client):
    """Test content truncation for long posts."""
    long_body = "A" * 3000  # Exceeds max tokens
    long_post = RedditPost(
        id="long123",
        title="Long post",
        body=long_body,
        subreddit="wallstreetbets",
        score=100,
        upvote_ratio=0.8,
        url="https://reddit.com/r/wallstreetbets/comments/long123",
        created_utc=datetime.now(UTC),
        num_comments=10,
    )

    mock_response = TickerExtractionResponse(mentions=[])
    mock_llm_client.astructured.return_value = mock_response

    await extractor.extract_tickers(long_post)

    # Verify LLM was called (content was truncated, not rejected)
    mock_llm_client.astructured.assert_called_once()
    call_args = mock_llm_client.astructured.call_args
    prompt = call_args[1]["prompt"]

    # Prompt should be truncated (rough check)
    assert len(prompt) < len(long_body) + 500


@pytest.mark.unit
async def test_extract_tickers_includes_top_comments(extractor, mock_llm_client, sample_post, sample_comments):
    """Test that top comments are included in extraction."""
    mock_response = TickerExtractionResponse(mentions=[])
    mock_llm_client.astructured.return_value = mock_response

    await extractor.extract_tickers(sample_post, sample_comments)

    call_args = mock_llm_client.astructured.call_args
    prompt = call_args[1]["prompt"]

    # Verify comments are in prompt
    assert "TSLA $350 by EOW" in prompt
    assert "watching NVDA" in prompt


@pytest.mark.unit
async def test_extract_tickers_limits_comments(extractor, mock_llm_client, sample_post):
    """Test comment limit (max 3)."""
    many_comments = [
        RedditComment(
            id=f"c{i}",
            parent_post_id="abc123",
            body=f"Comment {i}",
            score=100 - i,
            created_utc=datetime.now(UTC),
        )
        for i in range(10)
    ]

    mock_response = TickerExtractionResponse(mentions=[])
    mock_llm_client.astructured.return_value = mock_response

    await extractor.extract_tickers(sample_post, many_comments)

    call_args = mock_llm_client.astructured.call_args
    prompt = call_args[1]["prompt"]

    # Top 3 comments should be included
    assert "Comment 0" in prompt
    assert "Comment 1" in prompt
    assert "Comment 2" in prompt
    # Later comments should not be included
    assert "Comment 9" not in prompt


@pytest.mark.unit
async def test_sentiment_detection(extractor, mock_llm_client, sample_post):
    """Test sentiment is correctly extracted."""
    mock_response = TickerExtractionResponse(
        mentions=[
            TickerMention(symbol="TSLA", sentiment="BULLISH", context="moon", confidence=0.95),
            TickerMention(symbol="GME", sentiment="BEARISH", context="crash", confidence=0.90),
            TickerMention(symbol="AAPL", sentiment="NEUTRAL", context="holding", confidence=0.85),
        ]
    )
    mock_llm_client.astructured.return_value = mock_response

    mentions = await extractor.extract_tickers(sample_post)

    assert mentions[0].sentiment == "BULLISH"
    assert mentions[1].sentiment == "BEARISH"
    assert mentions[2].sentiment == "NEUTRAL"


@pytest.mark.unit
def test_repr(extractor):
    """Test string representation."""
    repr_str = repr(extractor)

    assert "RedditTickerExtractor" in repr_str
    assert "model=" in repr_str
