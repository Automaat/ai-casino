"""Tests for sentiment analyst agent."""

from src.agents.sentiment import SentimentAnalysis, SentimentAnalyst


def test_sentiment_analyst_init(mock_finbert):
    analyst = SentimentAnalyst(mock_finbert)

    assert analyst.finbert == mock_finbert


def test_sentiment_analyst_analyze(mock_finbert, sample_news_articles):
    analyst = SentimentAnalyst(mock_finbert)

    result = analyst.analyze("AAPL", sample_news_articles)

    assert isinstance(result, SentimentAnalysis)
    assert result.overall_sentiment in ["positive", "negative", "neutral"]
    assert -1.0 <= result.sentiment_score <= 1.0
    assert result.article_count == 3
    assert result.summary
    mock_finbert.analyze_batch.assert_called_once()


def test_sentiment_analyst_analyze_empty_articles(mock_finbert):
    analyst = SentimentAnalyst(mock_finbert)

    result = analyst.analyze("AAPL", [])

    assert result.overall_sentiment == "neutral"
    assert result.sentiment_score == 0.0
    assert result.article_count == 0
    mock_finbert.analyze_batch.assert_not_called()


def test_aggregate_sentiment(mock_finbert):
    from src.models.sentiment import SentimentScore

    analyst = SentimentAnalyst(mock_finbert)

    scores = [
        SentimentScore(positive=0.8, negative=0.1, neutral=0.1),
        SentimentScore(positive=0.6, negative=0.2, neutral=0.2),
        SentimentScore(positive=0.7, negative=0.15, neutral=0.15),
    ]

    result = analyst._aggregate_sentiment(scores)

    assert -1.0 <= result <= 1.0
    assert result > 0


def test_get_sentiment_label(mock_finbert):
    analyst = SentimentAnalyst(mock_finbert)

    assert analyst._get_sentiment_label(0.5) == "positive"
    assert analyst._get_sentiment_label(-0.5) == "negative"
    assert analyst._get_sentiment_label(0.1) == "neutral"


def test_generate_summary(mock_finbert):
    analyst = SentimentAnalyst(mock_finbert)

    summary = analyst._generate_summary("AAPL", "positive", 0.6, 8, 2, 10)

    assert "AAPL" in summary
    assert "positive" in summary
    assert "10" in summary


def test_repr(mock_finbert):
    analyst = SentimentAnalyst(mock_finbert)

    assert repr(analyst) == "SentimentAnalyst(model=FinBERT)"
