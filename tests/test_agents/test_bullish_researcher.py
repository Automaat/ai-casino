"""Tests for bullish researcher agent."""

from unittest.mock import AsyncMock

import pytest

from src.agents.bullish_researcher import BullishResearchAnalysis, BullishResearcher
from src.agents.fundamental import FundamentalAnalysis
from src.agents.news import NewsAnalysis
from src.agents.sentiment import SentimentAnalysis
from src.agents.technical import TechnicalAnalysis
from src.strategies.momentum import Signal


@pytest.fixture
def bullish_researcher(mock_llm_client):
    """Create bullish researcher instance."""
    return BullishResearcher(mock_llm_client)


@pytest.fixture
def sample_technical_analysis():
    """Sample technical analysis with BUY signal."""
    return TechnicalAnalysis(
        signal=Signal.BUY,
        rsi=65.0,
        macd_hist=0.5,
        interpretation="Strong upward momentum",
        confidence=0.8,
    )


@pytest.fixture
def sample_sentiment_analysis():
    """Sample sentiment analysis with positive sentiment."""
    return SentimentAnalysis(
        overall_sentiment="POSITIVE",
        sentiment_score=0.6,
        positive_ratio=0.7,
        negative_ratio=0.1,
        neutral_ratio=0.2,
        article_count=10,
        summary="Positive sentiment across articles",
    )


@pytest.fixture
def sample_news_analysis():
    """Sample news analysis."""
    return NewsAnalysis(
        key_themes=["earnings", "growth", "innovation"],
        impact_assessment="Very positive - strong fundamentals",
        recommendation="Consider buying on positive momentum",
    )


@pytest.fixture
def sample_fundamental_analysis():
    """Sample fundamental analysis with undervalued assessment."""
    return FundamentalAnalysis(
        valuation="UNDERVALUED",
        pe_ratio=18.5,
        eps=5.25,
        revenue_growth_yoy=0.15,
        earnings_growth_yoy=0.12,
        debt_to_equity=1.5,
        current_ratio=1.2,
        interpretation="Strong growth at reasonable valuation",
        confidence=0.75,
    )


@pytest.fixture
def sample_llm_response():
    """Sample LLM response for bull thesis."""
    return (
        "THESIS: This stock shows strong momentum with improving fundamentals and positive "
        "sentiment. Technical indicators suggest continued upward trajectory, supported by solid "
        "earnings growth and market enthusiasm. The combination of undervaluation and strong "
        "growth creates compelling upside opportunity.\n\n"
        "STRENGTHS:\n"
        "- Strong technical momentum with RSI at 65 and positive MACD\n"
        "- Positive sentiment across 10 recent articles\n"
        "- 15% revenue growth demonstrating strong business momentum\n"
        "- Undervalued at 18.5x P/E relative to growth rate\n"
        "- Key themes of earnings, growth, and innovation driving narrative\n\n"
        "UPSIDE: 25%"
    )


class TestBullishResearcher:
    """Test suite for BullishResearcher."""

    def test_initialization(self, mock_llm_client):
        """Test researcher initialization."""
        researcher = BullishResearcher(mock_llm_client)

        assert researcher.llm == mock_llm_client
        assert repr(researcher) == "BullishResearcher(llm=ollama/qwen3:14b)"

    @pytest.mark.asyncio
    async def test_analyze_returns_bullish_research_analysis(
        self,
        bullish_researcher,
        sample_technical_analysis,
        sample_sentiment_analysis,
        sample_news_analysis,
        sample_fundamental_analysis,
        sample_llm_response,
    ):
        """Test analyze returns BullishResearchAnalysis."""
        bullish_researcher.llm.acomplete = AsyncMock(return_value=sample_llm_response)

        result = await bullish_researcher.analyze(
            "AAPL",
            sample_technical_analysis,
            sample_sentiment_analysis,
            sample_news_analysis,
            sample_fundamental_analysis,
        )

        assert isinstance(result, BullishResearchAnalysis)
        assert result.thesis
        assert len(result.key_strengths) >= 3
        assert result.target_upside == 25.0
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_analyze_calls_llm(
        self,
        bullish_researcher,
        sample_technical_analysis,
        sample_sentiment_analysis,
        sample_news_analysis,
        sample_fundamental_analysis,
        sample_llm_response,
    ):
        """Test analyze calls LLM with correct parameters."""
        bullish_researcher.llm.acomplete = AsyncMock(return_value=sample_llm_response)

        await bullish_researcher.analyze(
            "TSLA",
            sample_technical_analysis,
            sample_sentiment_analysis,
            sample_news_analysis,
            sample_fundamental_analysis,
        )

        bullish_researcher.llm.acomplete.assert_called_once()
        call_args = bullish_researcher.llm.acomplete.call_args
        assert "TSLA" in call_args[0][0]
        assert "optimistic investment researcher" in call_args[1]["system"]
        assert call_args[1]["temperature"] == 0.5

    def test_build_prompt_contains_all_analyses(
        self,
        bullish_researcher,
        sample_technical_analysis,
        sample_sentiment_analysis,
        sample_news_analysis,
        sample_fundamental_analysis,
    ):
        """Test prompt includes all analysis components."""
        prompt = bullish_researcher._build_prompt(
            "AAPL",
            sample_technical_analysis,
            sample_sentiment_analysis,
            sample_news_analysis,
            sample_fundamental_analysis,
        )

        assert "AAPL" in prompt
        assert "TECHNICAL:" in prompt
        assert "BUY" in prompt
        assert "65.0" in prompt  # RSI
        assert "SENTIMENT:" in prompt
        assert "POSITIVE" in prompt
        assert "NEWS:" in prompt
        assert "earnings" in prompt
        assert "FUNDAMENTAL:" in prompt
        assert "UNDERVALUED" in prompt

    def test_calculate_confidence_strong_bull_signals(
        self, bullish_researcher, sample_sentiment_analysis, sample_news_analysis
    ):
        """Test confidence calculation with strong bullish signals."""
        technical = TechnicalAnalysis(
            signal=Signal.BUY, rsi=70.0, macd_hist=1.0, interpretation="Strong buy", confidence=0.9
        )
        fundamental = FundamentalAnalysis(
            valuation="UNDERVALUED",
            pe_ratio=15.0,
            eps=6.0,
            revenue_growth_yoy=0.20,
            earnings_growth_yoy=0.15,
            debt_to_equity=1.0,
            current_ratio=1.5,
            interpretation="Undervalued with strong growth",
            confidence=0.85,
        )

        confidence = bullish_researcher._calculate_confidence(
            technical, sample_sentiment_analysis, sample_news_analysis, fundamental
        )

        # Base 0.5 + BUY 0.15 + positive sentiment 0.1 + undervalued 0.1 + growth 0.05 = 0.9
        assert confidence == 0.9

    def test_calculate_confidence_weak_signals(self, bullish_researcher, sample_news_analysis):
        """Test confidence calculation with weak signals."""
        technical = TechnicalAnalysis(
            signal=Signal.HOLD, rsi=50.0, macd_hist=0.0, interpretation="Neutral", confidence=0.5
        )
        sentiment = SentimentAnalysis(
            overall_sentiment="NEUTRAL",
            sentiment_score=0.0,
            positive_ratio=0.33,
            negative_ratio=0.33,
            neutral_ratio=0.34,
            article_count=5,
            summary="Neutral sentiment",
        )
        fundamental = FundamentalAnalysis(
            valuation="FAIRLY_VALUED",
            pe_ratio=25.0,
            eps=3.0,
            revenue_growth_yoy=0.05,
            earnings_growth_yoy=0.03,
            debt_to_equity=2.0,
            current_ratio=1.0,
            interpretation="Fair value",
            confidence=0.6,
        )

        confidence = bullish_researcher._calculate_confidence(
            technical, sentiment, sample_news_analysis, fundamental
        )

        # Base 0.5 + fairly valued 0.1 = 0.6
        assert confidence == 0.6

    def test_calculate_confidence_bearish_signals(self, bullish_researcher, sample_news_analysis):
        """Test confidence calculation with bearish signals."""
        technical = TechnicalAnalysis(
            signal=Signal.SELL, rsi=25.0, macd_hist=-1.0, interpretation="Strong sell", confidence=0.8
        )
        sentiment = SentimentAnalysis(
            overall_sentiment="NEGATIVE",
            sentiment_score=-0.5,
            positive_ratio=0.1,
            negative_ratio=0.7,
            neutral_ratio=0.2,
            article_count=15,
            summary="Negative sentiment",
        )
        fundamental = FundamentalAnalysis(
            valuation="OVERVALUED",
            pe_ratio=50.0,
            eps=1.0,
            revenue_growth_yoy=-0.05,
            earnings_growth_yoy=-0.10,
            debt_to_equity=4.0,
            current_ratio=0.5,
            interpretation="Overvalued",
            confidence=0.7,
        )

        confidence = bullish_researcher._calculate_confidence(
            technical, sentiment, sample_news_analysis, fundamental
        )

        # Base 0.5 - SELL 0.2 - negative sentiment 0.15 - overvalued 0.1 = 0.05
        assert confidence == pytest.approx(0.05)

    def test_calculate_confidence_clamped_to_range(
        self, bullish_researcher, sample_sentiment_analysis, sample_news_analysis
    ):
        """Test confidence is clamped to [0.0, 1.0]."""
        # Extreme bullish
        technical_bull = TechnicalAnalysis(
            signal=Signal.BUY, rsi=80.0, macd_hist=2.0, interpretation="Very strong", confidence=1.0
        )
        fundamental_bull = FundamentalAnalysis(
            valuation="UNDERVALUED",
            pe_ratio=10.0,
            eps=10.0,
            revenue_growth_yoy=0.50,
            earnings_growth_yoy=0.40,
            debt_to_equity=0.5,
            current_ratio=2.0,
            interpretation="Very undervalued",
            confidence=1.0,
        )

        confidence_bull = bullish_researcher._calculate_confidence(
            technical_bull, sample_sentiment_analysis, sample_news_analysis, fundamental_bull
        )
        # Base 0.5 + BUY 0.15 + positive sentiment 0.1 + undervalued 0.1 + growth 0.05 = 0.9
        assert confidence_bull == 0.9

        # Extreme bearish
        technical_bear = TechnicalAnalysis(
            signal=Signal.SELL, rsi=20.0, macd_hist=-2.0, interpretation="Very weak", confidence=1.0
        )
        sentiment_bear = SentimentAnalysis(
            overall_sentiment="NEGATIVE",
            sentiment_score=-0.8,
            positive_ratio=0.05,
            negative_ratio=0.85,
            neutral_ratio=0.1,
            article_count=20,
            summary="Very negative sentiment",
        )
        fundamental_bear = FundamentalAnalysis(
            valuation="OVERVALUED",
            pe_ratio=100.0,
            eps=0.1,
            revenue_growth_yoy=-0.30,
            earnings_growth_yoy=-0.40,
            debt_to_equity=5.0,
            current_ratio=0.3,
            interpretation="Very overvalued",
            confidence=0.9,
        )

        confidence_bear = bullish_researcher._calculate_confidence(
            technical_bear, sentiment_bear, sample_news_analysis, fundamental_bear
        )
        # Base 0.5 - SELL 0.2 - negative sentiment 0.15 - overvalued 0.1 = 0.05
        assert confidence_bear == pytest.approx(0.05)

    def test_extract_thesis(self, bullish_researcher, sample_llm_response):
        """Test thesis extraction from LLM response."""
        thesis = bullish_researcher._extract_thesis(sample_llm_response)

        assert "strong momentum" in thesis.lower()
        assert "growth" in thesis.lower()
        assert len(thesis) > 50

    def test_extract_thesis_fallback(self, bullish_researcher):
        """Test thesis extraction fallback when no THESIS: marker."""
        response = (
            "This is a good stock. It has strong fundamentals. Growth is impressive. Valuation is attractive."
        )

        thesis = bullish_researcher._extract_thesis(response)

        assert "This is a good stock" in thesis
        assert len(thesis) > 20

    def test_extract_key_strengths(self, bullish_researcher, sample_llm_response):
        """Test key strengths extraction from LLM response."""
        strengths = bullish_researcher._extract_key_strengths(sample_llm_response)

        assert len(strengths) == 5
        assert any("technical momentum" in s.lower() for s in strengths)
        assert any("positive sentiment" in s.lower() for s in strengths)
        assert any("revenue growth" in s.lower() for s in strengths)

    def test_extract_key_strengths_fallback(self, bullish_researcher):
        """Test key strengths extraction fallback."""
        response = """
        Some text here.
        - Strong earnings
        - Good momentum
        - Positive sentiment
        More text.
        """

        strengths = bullish_researcher._extract_key_strengths(response)

        assert len(strengths) == 3
        assert "Strong earnings" in strengths
        assert "Good momentum" in strengths

    def test_extract_key_strengths_empty(self, bullish_researcher):
        """Test key strengths extraction with no bullets."""
        response = "No bullet points in this response at all."

        strengths = bullish_researcher._extract_key_strengths(response)

        assert strengths == []

    def test_extract_target_upside_valid(self, bullish_researcher, sample_llm_response):
        """Test target upside extraction with valid percentage."""
        upside = bullish_researcher._extract_target_upside(sample_llm_response)

        assert upside == 25.0

    def test_extract_target_upside_with_percent_sign(self, bullish_researcher):
        """Test target upside extraction with percent sign."""
        response = "UPSIDE: 30%"

        upside = bullish_researcher._extract_target_upside(response)

        assert upside == 30.0

    def test_extract_target_upside_decimal(self, bullish_researcher):
        """Test target upside extraction with decimal."""
        response = "UPSIDE: 15.5%"

        upside = bullish_researcher._extract_target_upside(response)

        assert upside == 15.5

    def test_extract_target_upside_missing(self, bullish_researcher):
        """Test target upside extraction with N/A."""
        response = "UPSIDE: N/A"

        upside = bullish_researcher._extract_target_upside(response)

        assert upside is None

    def test_extract_target_upside_uncertain(self, bullish_researcher):
        """Test target upside extraction with uncertain text."""
        response = "UPSIDE: Not available due to market volatility"

        upside = bullish_researcher._extract_target_upside(response)

        assert upside is None

    def test_extract_target_upside_no_section(self, bullish_researcher):
        """Test target upside extraction with no UPSIDE section."""
        response = "Some text without upside section."

        upside = bullish_researcher._extract_target_upside(response)

        assert upside is None

    @pytest.mark.asyncio
    async def test_analyze_with_missing_fundamental_data(
        self,
        bullish_researcher,
        sample_technical_analysis,
        sample_sentiment_analysis,
        sample_news_analysis,
        sample_llm_response,
    ):
        """Test analyze with missing fundamental data."""
        fundamental = FundamentalAnalysis(
            valuation="UNKNOWN",
            pe_ratio=None,
            eps=None,
            revenue_growth_yoy=None,
            earnings_growth_yoy=None,
            debt_to_equity=None,
            current_ratio=None,
            interpretation="Insufficient data",
            confidence=0.3,
        )

        bullish_researcher.llm.acomplete = AsyncMock(return_value=sample_llm_response)

        result = await bullish_researcher.analyze(
            "AAPL",
            sample_technical_analysis,
            sample_sentiment_analysis,
            sample_news_analysis,
            fundamental,
        )

        assert isinstance(result, BullishResearchAnalysis)
        # Should still work with missing data
        assert result.thesis
        assert len(result.key_strengths) > 0

    def test_repr(self, bullish_researcher):
        """Test string representation."""
        repr_str = repr(bullish_researcher)

        assert "BullishResearcher" in repr_str
        assert "ollama" in repr_str
        assert "qwen3:14b" in repr_str

    def test_bullish_research_analysis_repr(self):
        """Test BullishResearchAnalysis string representation."""
        analysis = BullishResearchAnalysis(
            thesis="This is a great investment opportunity with strong fundamentals",
            key_strengths=["Strong growth", "Positive sentiment", "Good valuation"],
            target_upside=25.0,
            confidence=0.8,
        )

        repr_str = repr(analysis)

        assert "BullishResearchAnalysis" in repr_str
        assert "strengths=3" in repr_str
        assert "upside=25.0" in repr_str
        assert "confidence=0.80" in repr_str

    def test_build_prompt_fundamental_none(
        self,
        bullish_researcher,
        sample_technical_analysis,
        sample_sentiment_analysis,
        sample_news_analysis,
    ):
        """Test prompt contains N/A message when fundamental is None."""
        prompt = bullish_researcher._build_prompt(
            "AAPL",
            sample_technical_analysis,
            sample_sentiment_analysis,
            sample_news_analysis,
            None,
        )

        assert "N/A (API rate limited)" in prompt

    def test_calculate_confidence_skips_fundamental_when_none(self, bullish_researcher, sample_news_analysis):
        """Test confidence calculation skips fundamental factors when None."""
        technical = TechnicalAnalysis(
            signal=Signal.BUY, rsi=65.0, macd_hist=0.5, interpretation="Buy", confidence=0.8
        )
        sentiment = SentimentAnalysis(
            overall_sentiment="POSITIVE",
            sentiment_score=0.5,
            positive_ratio=0.6,
            negative_ratio=0.1,
            neutral_ratio=0.3,
            article_count=5,
            summary="Positive",
        )

        confidence = bullish_researcher._calculate_confidence(
            technical, sentiment, sample_news_analysis, None
        )

        # Base 0.5 + BUY 0.15 + positive sentiment 0.1 = 0.75 (no fundamental adjustment)
        assert confidence == pytest.approx(0.75)
