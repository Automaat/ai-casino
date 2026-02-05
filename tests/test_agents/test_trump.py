"""Tests for Trump analyst agent."""

import pytest

from src.agents.trump import COMPANY_TICKERS, MARKET_KEYWORDS, TrumpAnalysis, TrumpAnalyst
from src.strategies.momentum import Signal


def test_trump_analyst_init(mock_llm_client):
    analyst = TrumpAnalyst(mock_llm_client)
    assert analyst.llm == mock_llm_client


@pytest.mark.asyncio
async def test_trump_analyst_analyze(mock_llm_client, sample_trump_posts):
    mock_llm_client.acomplete.return_value = """
    Sentiment: positive
    Signal: BUY
    Confidence: 0.75
    Interpretation: Trump's posts show bullish sentiment toward markets and specific stocks.
    """

    analyst = TrumpAnalyst(mock_llm_client)
    result = await analyst.analyze(sample_trump_posts)

    assert isinstance(result, TrumpAnalysis)
    assert result.post_count == 3
    assert result.market_relevant is True
    assert "TSLA" in result.mentioned_tickers
    mock_llm_client.acomplete.assert_called_once()


@pytest.mark.asyncio
async def test_trump_analyst_analyze_empty(mock_llm_client):
    analyst = TrumpAnalyst(mock_llm_client)
    result = await analyst.analyze([])

    assert isinstance(result, TrumpAnalysis)
    assert result.market_relevant is False
    assert result.signal == Signal.HOLD
    assert result.confidence == 0.0
    assert result.post_count == 0
    mock_llm_client.acomplete.assert_not_called()


def test_is_market_relevant(mock_llm_client):
    analyst = TrumpAnalyst(mock_llm_client)

    assert analyst._is_market_relevant("Great time to BUY stocks!") is True
    assert analyst._is_market_relevant("Tariffs on China paused") is True
    assert analyst._is_market_relevant("Bitcoin is the future!") is True
    assert analyst._is_market_relevant("Had a great golf game today") is False


def test_extract_tickers(mock_llm_client):
    analyst = TrumpAnalyst(mock_llm_client)

    # Direct ticker mentions
    tickers = analyst._extract_tickers("I love $TSLA and $AAPL!")
    assert "TSLA" in tickers
    assert "AAPL" in tickers

    # Company name mentions
    tickers = analyst._extract_tickers("Tesla and Apple are doing great!")
    assert "TSLA" in tickers
    assert "AAPL" in tickers

    # No tickers
    tickers = analyst._extract_tickers("Had a great day!")
    assert len(tickers) == 0


def test_extract_key_phrases(mock_llm_client):
    analyst = TrumpAnalyst(mock_llm_client)

    phrases = analyst._extract_key_phrases("Great time to buy stocks. Tariffs are working!")

    assert len(phrases) > 0
    assert any("buy" in p.lower() or "tariff" in p.lower() for p in phrases)


def test_format_posts(mock_llm_client, sample_trump_posts):
    analyst = TrumpAnalyst(mock_llm_client)
    formatted = analyst._format_posts(sample_trump_posts)

    assert "TSLA" in formatted
    assert "Likes:" in formatted
    assert len(formatted) > 0


def test_extract_sentiment(mock_llm_client):
    analyst = TrumpAnalyst(mock_llm_client)

    assert analyst._extract_sentiment("The sentiment is positive") == "positive"
    assert analyst._extract_sentiment("The sentiment is negative") == "negative"
    assert analyst._extract_sentiment("The sentiment is neutral") == "neutral"
    assert analyst._extract_sentiment("No clear sentiment") == "neutral"


def test_extract_signal(mock_llm_client):
    analyst = TrumpAnalyst(mock_llm_client)

    assert analyst._extract_signal("Signal: BUY") == Signal.BUY
    assert analyst._extract_signal("Signal: SELL") == Signal.SELL
    assert analyst._extract_signal("Signal: HOLD") == Signal.HOLD
    assert analyst._extract_signal("Mixed signals") == Signal.HOLD


def test_extract_confidence(mock_llm_client):
    analyst = TrumpAnalyst(mock_llm_client)

    assert analyst._extract_confidence("confidence: 0.8") == 0.8
    assert analyst._extract_confidence("80% confidence") == 0.8
    assert analyst._extract_confidence("strong signal") == 0.7
    assert analyst._extract_confidence("some text") == 0.5


def test_repr(mock_llm_client):
    analyst = TrumpAnalyst(mock_llm_client)
    repr_str = repr(analyst)

    assert "TrumpAnalyst" in repr_str
    assert "ollama" in repr_str


def test_market_keywords():
    """Verify market keywords are properly defined."""
    assert "buy" in MARKET_KEYWORDS
    assert "tariff" in MARKET_KEYWORDS
    assert "bitcoin" in MARKET_KEYWORDS
    assert "tesla" in MARKET_KEYWORDS


def test_company_tickers():
    """Verify company ticker mappings."""
    assert COMPANY_TICKERS["tesla"] == "TSLA"
    assert COMPANY_TICKERS["apple"] == "AAPL"
    assert COMPANY_TICKERS["amazon"] == "AMZN"
