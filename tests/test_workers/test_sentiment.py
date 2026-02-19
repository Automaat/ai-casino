"""Tests for sentiment worker."""

from src.tools.models import ToolDefinition
from src.workers.sentiment import SentimentAnalysis


def test_sentiment_worker_init(test_container):
    """Test worker initialization."""
    worker = test_container.sentiment_worker()

    assert worker.finbert is not None
    tool_def = worker.get_tool_definition()
    assert tool_def is not None


async def test_sentiment_worker_analyze_with_articles(test_container, sample_news_articles):
    """Test analysis with news articles."""
    worker = test_container.sentiment_worker()

    result = await worker.analyze("AAPL", sample_news_articles)

    assert isinstance(result, SentimentAnalysis)
    assert result.overall_sentiment in ["positive", "negative", "neutral"]
    assert -1.0 <= result.sentiment_score <= 1.0
    assert 0.0 <= result.positive_ratio <= 1.0
    assert 0.0 <= result.negative_ratio <= 1.0
    assert 0.0 <= result.neutral_ratio <= 1.0
    assert result.article_count == len(sample_news_articles)
    assert result.summary


async def test_sentiment_worker_analyze_empty_articles(test_container):
    """Test analysis with no articles returns neutral sentiment."""
    worker = test_container.sentiment_worker()

    result = await worker.analyze("AAPL", [])

    assert isinstance(result, SentimentAnalysis)
    assert result.overall_sentiment == "neutral"
    assert result.sentiment_score == 0.0
    assert result.positive_ratio == 0.0
    assert result.negative_ratio == 0.0
    assert result.neutral_ratio == 1.0
    assert result.article_count == 0
    assert "No news articles" in result.summary


async def test_sentiment_worker_analyze_positive_sentiment(test_container):
    """Test positive sentiment classification."""
    from datetime import datetime

    from src.data.news import NewsArticle

    positive_articles = [
        NewsArticle(
            title="Company Reports Record Profits",
            description="Strong growth and excellent performance.",
            url="http://example.com/1",
            published_at=datetime.now(),
            source="Test",
        ),
        NewsArticle(
            title="Stock Surges on Great News",
            description="Investors celebrate outstanding results.",
            url="http://example.com/2",
            published_at=datetime.now(),
            source="Test",
        ),
    ]

    worker = test_container.sentiment_worker()
    result = await worker.analyze("AAPL", positive_articles)

    assert isinstance(result, SentimentAnalysis)
    assert result.overall_sentiment in ["positive", "neutral"]  # FinBERT might vary
    assert result.article_count == 2


async def test_sentiment_worker_analyze_negative_sentiment(test_container):
    """Test negative sentiment classification."""
    from datetime import datetime

    from src.data.news import NewsArticle

    negative_articles = [
        NewsArticle(
            title="Company Faces Major Losses",
            description="Terrible performance and declining revenue.",
            url="http://example.com/1",
            published_at=datetime.now(),
            source="Test",
        ),
        NewsArticle(
            title="Stock Plummets on Bad News",
            description="Investors flee amid concerning developments.",
            url="http://example.com/2",
            published_at=datetime.now(),
            source="Test",
        ),
    ]

    worker = test_container.sentiment_worker()
    result = await worker.analyze("AAPL", negative_articles)

    assert isinstance(result, SentimentAnalysis)
    assert result.overall_sentiment in ["negative", "neutral"]  # FinBERT might vary
    assert result.article_count == 2


async def test_sentiment_worker_aggregation(test_container, sample_news_articles):
    """Test sentiment score aggregation logic."""
    worker = test_container.sentiment_worker()

    result = await worker.analyze("AAPL", sample_news_articles)

    # Verify ratios sum to 1.0
    total_ratio = result.positive_ratio + result.negative_ratio + result.neutral_ratio
    assert abs(total_ratio - 1.0) < 0.01  # Allow small floating-point error

    # Verify counts match total
    total = result.article_count
    assert total == len(sample_news_articles)


def test_sentiment_worker_tool_definition(test_container):
    """Test tool definition for supervisor integration."""
    worker = test_container.sentiment_worker()

    tool_def = worker.get_tool_definition()

    assert isinstance(tool_def, ToolDefinition)
    assert tool_def.type == "function"
    assert tool_def.function.name == "analyze_sentiment"
    assert "symbol" in tool_def.function.parameters.properties
    assert "symbol" in tool_def.function.parameters.required


def test_sentiment_worker_repr(test_container):
    """Test string representation."""
    worker = test_container.sentiment_worker()

    repr_str = repr(worker)

    assert "SentimentWorker" in repr_str
    assert "FinBERT" in repr_str


async def test_sentiment_worker_remote_device_path(sample_news_articles):
    """Test remote device routes via asyncio.to_thread, not ProcessPoolExecutor."""
    from unittest.mock import MagicMock, patch

    from src.models.sentiment import SentimentScore
    from src.workers.sentiment import SentimentWorker

    mock_finbert = MagicMock()
    mock_finbert.device = "remote"
    mock_finbert.analyze_batch.return_value = [
        SentimentScore(positive=0.7, negative=0.1, neutral=0.2),
        SentimentScore(positive=0.6, negative=0.2, neutral=0.2),
        SentimentScore(positive=0.8, negative=0.05, neutral=0.15),
    ]

    worker = SentimentWorker(finbert=mock_finbert)

    with patch("src.workers.sentiment.get_finbert_executor") as mock_executor:
        result = await worker.analyze("AAPL", sample_news_articles)
        mock_executor.assert_not_called()

    assert isinstance(result, SentimentAnalysis)
    assert result.article_count == len(sample_news_articles)
    assert result.overall_sentiment in ["positive", "negative", "neutral"]
    mock_finbert.analyze_batch.assert_called_once()
