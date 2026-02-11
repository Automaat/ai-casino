"""Tests for Trump analyst agent."""

from src.agents.trump import COMPANY_TICKERS, MARKET_KEYWORDS, TrumpAnalysis
from src.strategies.signal import Signal


def test_trump_analyst_init(test_container):
    analyst = test_container.trump_analyst()
    assert analyst.llm is not None


async def test_trump_analyst_analyze(test_container, sample_trump_posts):
    # Mock updated to Trump's authentic voice (2024-2026 style)
    analyst = test_container.trump_analyst()
    analyst.llm.acomplete.return_value = """
    Sentiment: positive
    Signal: BUY
    Confidence: 0.95
    Interpretation: The Trump Stock Market is ROARING back! My tariff policies are WORKING -
        China is BEGGING for a deal. Time to BUY AMERICAN! 🇺🇸
    """

    result = await analyst.analyze(sample_trump_posts)

    assert isinstance(result, TrumpAnalysis)
    assert result.post_count == 3
    assert result.market_relevant is True
    assert "TSLA" in result.mentioned_tickers


async def test_trump_analyst_analyze_empty(test_container):
    analyst = test_container.trump_analyst()
    result = await analyst.analyze([])

    assert isinstance(result, TrumpAnalysis)
    assert result.market_relevant is False
    assert result.signal == Signal.HOLD
    assert result.confidence == 0.0
    assert result.post_count == 0


def test_is_market_relevant(test_container):
    analyst = test_container.trump_analyst()

    assert analyst._is_market_relevant("Great time to BUY stocks!") is True
    assert analyst._is_market_relevant("Tariffs on China paused") is True
    assert analyst._is_market_relevant("Bitcoin is the future!") is True
    assert analyst._is_market_relevant("Had a great golf game today") is False


def test_extract_tickers(test_container):
    analyst = test_container.trump_analyst()

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


def test_extract_key_phrases(test_container):
    analyst = test_container.trump_analyst()

    phrases = analyst._extract_key_phrases("Great time to buy stocks. Tariffs are working!")

    assert len(phrases) > 0
    assert any("buy" in p.lower() or "tariff" in p.lower() for p in phrases)


def test_format_posts(test_container, sample_trump_posts):
    analyst = test_container.trump_analyst()
    formatted = analyst._format_posts(sample_trump_posts)

    assert "TSLA" in formatted
    assert "Likes:" in formatted
    assert len(formatted) > 0


def test_extract_sentiment(test_container):
    analyst = test_container.trump_analyst()

    assert analyst._extract_sentiment("The sentiment is positive") == "positive"
    assert analyst._extract_sentiment("The sentiment is negative") == "negative"
    assert analyst._extract_sentiment("The sentiment is neutral") == "neutral"
    assert analyst._extract_sentiment("No clear sentiment") == "neutral"


def test_extract_signal(test_container):
    analyst = test_container.trump_analyst()

    assert analyst._extract_signal("Signal: BUY") == Signal.BUY
    assert analyst._extract_signal("Signal: SELL") == Signal.SELL
    assert analyst._extract_signal("Signal: HOLD") == Signal.HOLD
    assert analyst._extract_signal("Mixed signals") == Signal.HOLD


def test_extract_confidence(test_container):
    analyst = test_container.trump_analyst()

    assert analyst._extract_confidence("confidence: 0.8") == 0.8
    assert analyst._extract_confidence("80% confidence") == 0.8
    assert analyst._extract_confidence("strong signal") == 0.7
    assert analyst._extract_confidence("some text") == 0.5


def test_repr(test_container):
    analyst = test_container.trump_analyst()
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
