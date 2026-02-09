"""Tests for agent providers."""

from unittest.mock import MagicMock, patch

from src.di import create_container


def test_news_analyst_provider():
    """Test NewsAnalyst provider is accessible."""
    container = create_container()
    assert hasattr(container, "news_analyst")

    with patch("src.agents.news.NewsAnalyst") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance

        analyst = container.news_analyst()
        assert analyst is not None


def test_sentiment_analyst_provider():
    """Test SentimentAnalyst provider is accessible."""
    container = create_container()
    assert hasattr(container, "sentiment_analyst")

    with patch("src.agents.sentiment.SentimentAnalyst") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance

        analyst = container.sentiment_analyst()
        assert analyst is not None


def test_trump_analyst_provider():
    """Test TrumpAnalyst provider is accessible."""
    container = create_container()
    assert hasattr(container, "trump_analyst")

    with patch("src.agents.trump.TrumpAnalyst") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance

        analyst = container.trump_analyst()
        assert analyst is not None


def test_fundamental_analyst_provider(monkeypatch):
    """Test FundamentalAnalyst provider is accessible."""
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test_key")
    container = create_container()
    assert hasattr(container, "fundamental_analyst")

    with patch("src.agents.fundamental.FundamentalAnalyst") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance

        analyst = container.fundamental_analyst()
        assert analyst is not None


def test_social_sentiment_analyst_provider():
    """Test SocialSentimentAnalyst provider is accessible."""
    container = create_container()
    assert hasattr(container, "social_sentiment_analyst")

    with patch("src.agents.social.SocialSentimentAnalyst") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance

        analyst = container.social_sentiment_analyst()
        assert analyst is not None


def test_news_analyst_factory():
    """Test NewsAnalyst is factory (new instance per call)."""
    container = create_container()

    with patch("src.agents.news.NewsAnalyst") as mock_class:
        mock_instance1 = MagicMock()
        mock_instance2 = MagicMock()
        mock_class.side_effect = [mock_instance1, mock_instance2]

        analyst1 = container.news_analyst()
        analyst2 = container.news_analyst()

        assert analyst1 is not analyst2
        assert mock_class.call_count == 2


def test_sentiment_analyst_factory():
    """Test SentimentAnalyst is factory (new instance per call)."""
    container = create_container()

    with patch("src.agents.sentiment.SentimentAnalyst") as mock_class:
        mock_instance1 = MagicMock()
        mock_instance2 = MagicMock()
        mock_class.side_effect = [mock_instance1, mock_instance2]

        analyst1 = container.sentiment_analyst()
        analyst2 = container.sentiment_analyst()

        assert analyst1 is not analyst2
        assert mock_class.call_count == 2


def test_trump_analyst_factory():
    """Test TrumpAnalyst is factory (new instance per call)."""
    container = create_container()

    with patch("src.agents.trump.TrumpAnalyst") as mock_class:
        mock_instance1 = MagicMock()
        mock_instance2 = MagicMock()
        mock_class.side_effect = [mock_instance1, mock_instance2]

        analyst1 = container.trump_analyst()
        analyst2 = container.trump_analyst()

        assert analyst1 is not analyst2
        assert mock_class.call_count == 2


def test_fundamental_analyst_factory(monkeypatch):
    """Test FundamentalAnalyst is factory (new instance per call)."""
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test_key")
    container = create_container()

    with patch("src.agents.fundamental.FundamentalAnalyst") as mock_class:
        mock_instance1 = MagicMock()
        mock_instance2 = MagicMock()
        mock_class.side_effect = [mock_instance1, mock_instance2]

        analyst1 = container.fundamental_analyst()
        analyst2 = container.fundamental_analyst()

        assert analyst1 is not analyst2
        assert mock_class.call_count == 2


def test_social_sentiment_analyst_factory():
    """Test SocialSentimentAnalyst is factory (new instance per call)."""
    container = create_container()

    with patch("src.agents.social.SocialSentimentAnalyst") as mock_class:
        mock_instance1 = MagicMock()
        mock_instance2 = MagicMock()
        mock_class.side_effect = [mock_instance1, mock_instance2]

        analyst1 = container.social_sentiment_analyst()
        analyst2 = container.social_sentiment_analyst()

        assert analyst1 is not analyst2
        assert mock_class.call_count == 2


def test_shared_finbert_singleton():
    """Test FinBERT singleton shared between sentiment_analyst and social_sentiment_analyst."""
    container = create_container()

    with (
        patch("src.models.sentiment.get_finbert_sentiment") as mock_finbert_factory,
        patch("src.agents.sentiment.SentimentAnalyst") as mock_sentiment_class,
        patch("src.agents.social.SocialSentimentAnalyst") as mock_social_class,
    ):
        mock_finbert = MagicMock()
        mock_finbert_factory.return_value = mock_finbert

        mock_sentiment_instance = MagicMock()
        mock_social_instance = MagicMock()
        mock_sentiment_class.return_value = mock_sentiment_instance
        mock_social_class.return_value = mock_social_instance

        # Create both analysts
        container.sentiment_analyst()
        container.social_sentiment_analyst()

        # FinBERT factory called once (singleton)
        assert mock_finbert_factory.call_count == 1

        # Both analysts initialized with same FinBERT instance
        mock_sentiment_class.assert_called_once()
        sentiment_finbert_arg = mock_sentiment_class.call_args[0][0]

        mock_social_class.assert_called_once()
        social_finbert_arg = mock_social_class.call_args[0][3]  # 4th arg

        assert sentiment_finbert_arg is mock_finbert
        assert social_finbert_arg is mock_finbert
        assert sentiment_finbert_arg is social_finbert_arg
