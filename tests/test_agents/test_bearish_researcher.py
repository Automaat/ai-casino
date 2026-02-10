"""Tests for bearish researcher agent."""

from unittest.mock import AsyncMock

import pytest

from src.agents.bearish_researcher import BearishResearchAnalysis
from src.agents.fundamental import FundamentalAnalysis
from src.agents.news import NewsAnalysis
from src.agents.sentiment import SentimentAnalysis
from src.agents.technical import TechnicalAnalysis
from src.strategies.signal import Signal


@pytest.fixture
def bearish_researcher(test_container):
    """Create bearish researcher instance."""
    return test_container.bearish_researcher()


@pytest.fixture
def sample_technical_analysis():
    """Sample technical analysis with SELL signal."""
    return TechnicalAnalysis(
        signal=Signal.SELL,
        rsi=25.0,
        macd_hist=-0.5,
        interpretation="Weak downward momentum",
        confidence=0.8,
    )


@pytest.fixture
def sample_sentiment_analysis():
    """Sample sentiment analysis with negative sentiment."""
    return SentimentAnalysis(
        overall_sentiment="NEGATIVE",
        sentiment_score=-0.6,
        positive_ratio=0.1,
        negative_ratio=0.7,
        neutral_ratio=0.2,
        article_count=10,
        summary="Negative sentiment across articles",
    )


@pytest.fixture
def sample_news_analysis():
    """Sample news analysis."""
    return NewsAnalysis(
        key_themes=["layoffs", "revenue decline", "competition"],
        impact_assessment="Very negative - weakening fundamentals",
        recommendation="Consider selling on negative momentum",
    )


@pytest.fixture
def sample_fundamental_analysis():
    """Sample fundamental analysis with overvalued assessment."""
    return FundamentalAnalysis(
        valuation="OVERVALUED",
        pe_ratio=45.0,
        eps=2.25,
        revenue_growth_yoy=-0.05,
        earnings_growth_yoy=-0.08,
        debt_to_equity=3.5,
        current_ratio=0.8,
        interpretation="Overvalued with weak growth and high debt",
        confidence=0.75,
    )


@pytest.fixture
def sample_llm_response():
    """Sample LLM response for bear thesis."""
    return (
        "THESIS: This stock shows weak momentum with deteriorating fundamentals and negative "
        "sentiment. Technical indicators suggest continued downward trajectory, compounded by poor "
        "earnings outlook and excessive valuation. The combination of overvaluation and weak "
        "growth creates compelling downside risk.\n\n"
        "WEAKNESSES:\n"
        "- Weak technical momentum with RSI at 25 and negative MACD\n"
        "- Negative sentiment across 10 recent articles\n"
        "- Revenue declining 5% YoY demonstrating weak business momentum\n"
        "- Overvalued at 45x P/E relative to declining growth\n"
        "- High debt-to-equity ratio of 3.5 straining balance sheet\n\n"
        "DOWNSIDE: 25%"
    )


class TestBearishResearcher:
    """Test suite for BearishResearcher."""

    def test_initialization(self, test_container):
        """Test researcher initialization."""
        researcher = test_container.bearish_researcher()

        assert repr(researcher) == "BearishResearcher(llm=ollama/qwen3:14b)"

    async def test_analyze_returns_bearish_research_analysis(
        self,
        bearish_researcher,
        sample_technical_analysis,
        sample_sentiment_analysis,
        sample_news_analysis,
        sample_fundamental_analysis,
        sample_llm_response,
    ):
        """Test analyze returns BearishResearchAnalysis."""
        bearish_researcher.llm.acomplete = AsyncMock(return_value=sample_llm_response)

        result = await bearish_researcher.analyze(
            "AAPL",
            sample_technical_analysis,
            sample_sentiment_analysis,
            sample_news_analysis,
            sample_fundamental_analysis,
        )

        assert isinstance(result, BearishResearchAnalysis)
        assert result.thesis
        assert len(result.key_weaknesses) >= 3
        assert result.target_downside == 25.0
        assert 0.0 <= result.confidence <= 1.0

    async def test_analyze_calls_llm(
        self,
        bearish_researcher,
        sample_technical_analysis,
        sample_sentiment_analysis,
        sample_news_analysis,
        sample_fundamental_analysis,
        sample_llm_response,
    ):
        """Test analyze calls LLM with correct parameters."""
        bearish_researcher.llm.acomplete = AsyncMock(return_value=sample_llm_response)

        await bearish_researcher.analyze(
            "TSLA",
            sample_technical_analysis,
            sample_sentiment_analysis,
            sample_news_analysis,
            sample_fundamental_analysis,
        )

        bearish_researcher.llm.acomplete.assert_called_once()
        call_args = bearish_researcher.llm.acomplete.call_args
        assert "TSLA" in call_args[0][0]
        assert "skeptical investment researcher" in call_args[1]["system"]
        assert call_args[1]["temperature"] == 0.5

    def test_build_prompt_contains_all_analyses(
        self,
        bearish_researcher,
        sample_technical_analysis,
        sample_sentiment_analysis,
        sample_news_analysis,
        sample_fundamental_analysis,
    ):
        """Test prompt includes all analysis components."""
        prompt_vars = bearish_researcher._build_prompt_vars(
            "AAPL",
            sample_technical_analysis,
            sample_sentiment_analysis,
            sample_news_analysis,
            sample_fundamental_analysis,
        )
        prompt = bearish_researcher._prompts.load("user", **prompt_vars)

        assert "AAPL" in prompt
        assert "TECHNICAL:" in prompt
        assert "SELL" in prompt
        assert "25.0" in prompt  # RSI
        assert "SENTIMENT:" in prompt
        assert "NEGATIVE" in prompt
        assert "NEWS:" in prompt
        assert "layoffs" in prompt
        assert "FUNDAMENTAL:" in prompt
        assert "OVERVALUED" in prompt
        assert "D/E" in prompt  # Debt-to-equity included

    def test_calculate_confidence_strong_bear_signals(
        self, bearish_researcher, sample_sentiment_analysis, sample_news_analysis
    ):
        """Test confidence calculation with strong bearish signals."""
        technical = TechnicalAnalysis(
            signal=Signal.SELL, rsi=20.0, macd_hist=-1.0, interpretation="Strong sell", confidence=0.9
        )
        fundamental = FundamentalAnalysis(
            valuation="OVERVALUED",
            pe_ratio=50.0,
            eps=1.0,
            revenue_growth_yoy=-0.10,
            earnings_growth_yoy=-0.15,
            debt_to_equity=3.0,
            current_ratio=0.5,
            interpretation="Overvalued with weak growth",
            confidence=0.85,
        )

        confidence = bearish_researcher._calculate_confidence(
            technical, sample_sentiment_analysis, sample_news_analysis, fundamental
        )

        # Base 0.5 + SELL 0.15 + negative sentiment 0.1 + overvalued 0.1 + high debt 0.05 = 0.9
        assert confidence == 0.9

    def test_calculate_confidence_bullish_signals(self, bearish_researcher, sample_news_analysis):
        """Test confidence calculation with bullish signals (low bear confidence)."""
        technical = TechnicalAnalysis(
            signal=Signal.BUY, rsi=70.0, macd_hist=1.0, interpretation="Strong buy", confidence=0.9
        )
        sentiment = SentimentAnalysis(
            overall_sentiment="POSITIVE",
            sentiment_score=0.6,
            positive_ratio=0.7,
            negative_ratio=0.1,
            neutral_ratio=0.2,
            article_count=10,
            summary="Positive sentiment",
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

        confidence = bearish_researcher._calculate_confidence(
            technical, sentiment, sample_news_analysis, fundamental
        )

        # Base 0.5 - BUY 0.2 - positive sentiment 0.15 - undervalued 0.1 = 0.05
        assert confidence == pytest.approx(0.05)

    def test_calculate_confidence_neutral_signals(self, bearish_researcher, sample_news_analysis):
        """Test confidence calculation with neutral signals."""
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
            debt_to_equity=1.5,
            current_ratio=1.0,
            interpretation="Fair value",
            confidence=0.6,
        )

        confidence = bearish_researcher._calculate_confidence(
            technical, sentiment, sample_news_analysis, fundamental
        )

        # Base 0.5, no adjustments
        assert confidence == 0.5

    def test_calculate_confidence_clamped_to_range(
        self, bearish_researcher, sample_sentiment_analysis, sample_news_analysis
    ):
        """Test confidence is clamped to [0.0, 1.0]."""
        # Extreme bearish
        technical_bear = TechnicalAnalysis(
            signal=Signal.SELL, rsi=15.0, macd_hist=-2.0, interpretation="Very weak", confidence=1.0
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

        confidence_bear = bearish_researcher._calculate_confidence(
            technical_bear, sample_sentiment_analysis, sample_news_analysis, fundamental_bear
        )
        # Base 0.5 + SELL 0.15 + negative sentiment 0.1 + overvalued 0.1 + high debt 0.05 = 0.9
        assert confidence_bear == 0.9

        # Extreme bullish (low bear confidence)
        technical_bull = TechnicalAnalysis(
            signal=Signal.BUY, rsi=80.0, macd_hist=2.0, interpretation="Very strong", confidence=1.0
        )
        sentiment_bull = SentimentAnalysis(
            overall_sentiment="POSITIVE",
            sentiment_score=0.8,
            positive_ratio=0.85,
            negative_ratio=0.05,
            neutral_ratio=0.1,
            article_count=20,
            summary="Very positive sentiment",
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

        confidence_bull = bearish_researcher._calculate_confidence(
            technical_bull, sentiment_bull, sample_news_analysis, fundamental_bull
        )
        # Base 0.5 - BUY 0.2 - positive sentiment 0.15 - undervalued 0.1 = 0.05, clamped at 0.0
        assert confidence_bull == pytest.approx(0.05)

    def test_extract_thesis(self, bearish_researcher, sample_llm_response):
        """Test thesis extraction from LLM response."""
        thesis = bearish_researcher._extract_thesis(sample_llm_response)

        assert "weak momentum" in thesis.lower()
        assert "deteriorating" in thesis.lower()
        assert len(thesis) > 50

    def test_extract_thesis_fallback(self, bearish_researcher):
        """Test thesis extraction fallback when no THESIS: marker."""
        response = (
            "This is a risky stock. It has weak fundamentals. Growth is declining. Valuation is excessive."
        )

        thesis = bearish_researcher._extract_thesis(response)

        assert "This is a risky stock" in thesis
        assert len(thesis) > 20

    def test_extract_key_weaknesses(self, bearish_researcher, sample_llm_response):
        """Test key weaknesses extraction from LLM response."""
        weaknesses = bearish_researcher._extract_key_weaknesses(sample_llm_response)

        assert len(weaknesses) == 5
        assert any("technical momentum" in w.lower() for w in weaknesses)
        assert any("negative sentiment" in w.lower() for w in weaknesses)
        assert any("revenue" in w.lower() for w in weaknesses)

    def test_extract_key_weaknesses_fallback(self, bearish_researcher):
        """Test key weaknesses extraction fallback."""
        response = """
        Some text here.
        - Weak earnings
        - Poor momentum
        - Negative sentiment
        More text.
        """

        weaknesses = bearish_researcher._extract_key_weaknesses(response)

        assert len(weaknesses) == 3
        assert "Weak earnings" in weaknesses
        assert "Poor momentum" in weaknesses

    def test_extract_key_weaknesses_empty(self, bearish_researcher):
        """Test key weaknesses extraction with no bullets."""
        response = "No bullet points in this response at all."

        weaknesses = bearish_researcher._extract_key_weaknesses(response)

        assert weaknesses == []

    def test_extract_target_downside_valid(self, bearish_researcher, sample_llm_response):
        """Test target downside extraction with valid percentage."""
        downside = bearish_researcher._extract_target_downside(sample_llm_response)

        assert downside == 25.0

    def test_extract_target_downside_with_percent_sign(self, bearish_researcher):
        """Test target downside extraction with percent sign."""
        response = "DOWNSIDE: 30%"

        downside = bearish_researcher._extract_target_downside(response)

        assert downside == 30.0

    def test_extract_target_downside_decimal(self, bearish_researcher):
        """Test target downside extraction with decimal."""
        response = "DOWNSIDE: 15.5%"

        downside = bearish_researcher._extract_target_downside(response)

        assert downside == 15.5

    def test_extract_target_downside_missing(self, bearish_researcher):
        """Test target downside extraction with N/A."""
        response = "DOWNSIDE: N/A"

        downside = bearish_researcher._extract_target_downside(response)

        assert downside is None

    def test_extract_target_downside_uncertain(self, bearish_researcher):
        """Test target downside extraction with uncertain text."""
        response = "DOWNSIDE: Not available due to market volatility"

        downside = bearish_researcher._extract_target_downside(response)

        assert downside is None

    def test_extract_target_downside_no_section(self, bearish_researcher):
        """Test target downside extraction with no DOWNSIDE section."""
        response = "Some text without downside section."

        downside = bearish_researcher._extract_target_downside(response)

        assert downside is None

    async def test_analyze_with_missing_fundamental_data(
        self,
        bearish_researcher,
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

        bearish_researcher.llm.acomplete = AsyncMock(return_value=sample_llm_response)

        result = await bearish_researcher.analyze(
            "AAPL",
            sample_technical_analysis,
            sample_sentiment_analysis,
            sample_news_analysis,
            fundamental,
        )

        assert isinstance(result, BearishResearchAnalysis)
        # Should still work with missing data
        assert result.thesis
        assert len(result.key_weaknesses) > 0

    def test_repr(self, bearish_researcher):
        """Test string representation."""
        repr_str = repr(bearish_researcher)

        assert "BearishResearcher" in repr_str
        assert "ollama" in repr_str
        assert "qwen3:14b" in repr_str

    def test_bearish_research_analysis_repr(self):
        """Test BearishResearchAnalysis string representation."""
        analysis = BearishResearchAnalysis(
            thesis="This is a risky investment with weak fundamentals",
            key_weaknesses=["Weak growth", "Negative sentiment", "High valuation"],
            target_downside=25.0,
            confidence=0.8,
        )

        repr_str = repr(analysis)

        assert "BearishResearchAnalysis" in repr_str
        assert "weaknesses=3" in repr_str
        assert "downside=25.0" in repr_str
        assert "confidence=0.80" in repr_str

    def test_build_prompt_fundamental_none(
        self,
        bearish_researcher,
        sample_technical_analysis,
        sample_sentiment_analysis,
        sample_news_analysis,
    ):
        """Test prompt contains N/A message when fundamental is None."""
        prompt_vars = bearish_researcher._build_prompt_vars(
            "AAPL",
            sample_technical_analysis,
            sample_sentiment_analysis,
            sample_news_analysis,
            None,
        )
        prompt = bearish_researcher._prompts.load("user", **prompt_vars)

        assert "N/A (API rate limited)" in prompt

    def test_calculate_confidence_skips_fundamental_when_none(self, bearish_researcher, sample_news_analysis):
        """Test confidence calculation skips fundamental factors when None."""
        technical = TechnicalAnalysis(
            signal=Signal.SELL, rsi=25.0, macd_hist=-0.5, interpretation="Sell", confidence=0.8
        )
        sentiment = SentimentAnalysis(
            overall_sentiment="NEGATIVE",
            sentiment_score=-0.5,
            positive_ratio=0.1,
            negative_ratio=0.6,
            neutral_ratio=0.3,
            article_count=5,
            summary="Negative",
        )

        confidence = bearish_researcher._calculate_confidence(
            technical, sentiment, sample_news_analysis, None
        )

        # Base 0.5 + SELL 0.15 + negative sentiment 0.1 = 0.75 (no fundamental adjustment)
        assert confidence == pytest.approx(0.75)
