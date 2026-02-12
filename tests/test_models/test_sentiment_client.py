"""Tests for FinBERT HTTP client."""

from unittest.mock import Mock, patch

import httpx
import pytest

from src.models.sentiment_client import FinBERTClient, SentimentScore


@pytest.fixture
def mock_httpx_response():
    """Mock httpx response for successful request."""
    mock_resp = Mock()
    mock_resp.json.return_value = {
        "scores": [
            {"positive": 0.8, "negative": 0.1, "neutral": 0.1},
            {"positive": 0.2, "negative": 0.7, "neutral": 0.1},
        ],
        "batch_size": 2,
        "inference_time_ms": 123.4,
    }
    mock_resp.raise_for_status = Mock()
    return mock_resp


def test_analyze_batch_success(mock_httpx_response):
    """Test successful batch analysis."""
    with patch("httpx.Client") as mock_client_class:
        mock_client = Mock()
        mock_client.post.return_value = mock_httpx_response
        mock_client_class.return_value = mock_client

        client = FinBERTClient("http://localhost:8485")
        scores = client.analyze_batch(["bullish news", "bearish news"])

        assert len(scores) == 2
        assert isinstance(scores[0], SentimentScore)
        assert scores[0].dominant == "positive"
        assert scores[1].dominant == "negative"
        assert scores[0].score > 0
        assert scores[1].score < 0


def test_analyze_batch_empty_returns_empty_list():
    """Test empty batch returns empty list."""
    client = FinBERTClient("http://localhost:8485")
    scores = client.analyze_batch([])
    assert scores == []


def test_sentiment_score_dominant():
    """Test dominant sentiment calculation."""
    score_positive = SentimentScore(positive=0.8, negative=0.1, neutral=0.1)
    assert score_positive.dominant == "positive"

    score_negative = SentimentScore(positive=0.1, negative=0.8, neutral=0.1)
    assert score_negative.dominant == "negative"

    score_neutral = SentimentScore(positive=0.2, negative=0.2, neutral=0.6)
    assert score_neutral.dominant == "neutral"


def test_sentiment_score_score():
    """Test overall score calculation."""
    score = SentimentScore(positive=0.8, negative=0.1, neutral=0.1)
    assert score.score == pytest.approx(0.7, abs=0.01)

    score_negative = SentimentScore(positive=0.1, negative=0.8, neutral=0.1)
    assert score_negative.score == pytest.approx(-0.7, abs=0.01)


def test_analyze_batch_http_error():
    """Test error handling when HTTP request fails."""
    with patch("httpx.Client") as mock_client_class:
        mock_client = Mock()
        mock_client.post.side_effect = httpx.HTTPError("Connection failed")
        mock_client_class.return_value = mock_client

        client = FinBERTClient("http://localhost:8485")

        with pytest.raises(httpx.HTTPError, match="Connection failed"):
            client.analyze_batch(["test"])


@pytest.mark.asyncio
async def test_analyze_batch_async(mock_httpx_response):
    """Test async batch analysis."""
    with patch("httpx.Client") as mock_client_class:
        mock_client = Mock()
        mock_client.post.return_value = mock_httpx_response
        mock_client_class.return_value = mock_client

        client = FinBERTClient("http://localhost:8485")
        scores = await client.analyze_batch_async(["bullish news", "bearish news"])

        assert len(scores) == 2
        assert isinstance(scores[0], SentimentScore)


def test_client_repr():
    """Test client string representation."""
    client = FinBERTClient("http://localhost:8485")
    assert "FinBERTClient" in repr(client)
    assert "http://localhost:8485" in repr(client)
