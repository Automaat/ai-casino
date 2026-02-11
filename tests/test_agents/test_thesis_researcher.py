"""Tests for unified thesis researcher agent (bullish and bearish)."""

from unittest.mock import AsyncMock

import pytest

from src.agents.base_researcher import ResearchDirection
from src.agents.fundamental import FundamentalAnalysis
from src.agents.news import NewsAnalysis
from src.agents.sentiment import SentimentAnalysis
from src.agents.technical import TechnicalAnalysis
from src.agents.thesis_researcher import ResearchAnalysis
from src.strategies.signal import Signal


@pytest.fixture(params=[ResearchDirection.BULLISH, ResearchDirection.BEARISH])
def direction(request):
    """Parametrize tests with both directions."""
    return request.param


@pytest.fixture
def researcher(test_container, direction):
    """Create thesis researcher instance for given direction."""
    if direction == ResearchDirection.BULLISH:
        return test_container.bullish_researcher()
    return test_container.bearish_researcher()


@pytest.fixture
def sample_technical_analysis(direction):
    """Sample technical analysis with direction-appropriate signal."""
    if direction == ResearchDirection.BULLISH:
        return TechnicalAnalysis(
            signal=Signal.BUY,
            rsi=65.0,
            macd_hist=0.5,
            interpretation="Strong upward momentum",
            confidence=0.8,
        )
    return TechnicalAnalysis(
        signal=Signal.SELL,
        rsi=35.0,
        macd_hist=-0.5,
        interpretation="Strong downward momentum",
        confidence=0.8,
    )


@pytest.fixture
def sample_sentiment_analysis(direction):
    """Sample sentiment analysis with direction-appropriate sentiment."""
    if direction == ResearchDirection.BULLISH:
        return SentimentAnalysis(
            overall_sentiment="POSITIVE",
            sentiment_score=0.6,
            positive_ratio=0.7,
            negative_ratio=0.1,
            neutral_ratio=0.2,
            article_count=10,
            summary="Positive sentiment across articles",
        )
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
def sample_news_analysis(direction):
    """Sample news analysis with direction-appropriate themes."""
    if direction == ResearchDirection.BULLISH:
        return NewsAnalysis(
            key_themes=["earnings", "growth", "innovation"],
            impact_assessment="Very positive - strong fundamentals",
            recommendation="Consider buying on positive momentum",
        )
    return NewsAnalysis(
        key_themes=["losses", "decline", "risks"],
        impact_assessment="Very negative - weak fundamentals",
        recommendation="Consider selling on negative momentum",
    )


@pytest.fixture
def sample_fundamental_analysis(direction):
    """Sample fundamental analysis with direction-appropriate valuation."""
    if direction == ResearchDirection.BULLISH:
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
    return FundamentalAnalysis(
        valuation="OVERVALUED",
        pe_ratio=35.0,
        eps=2.0,
        revenue_growth_yoy=-0.05,
        earnings_growth_yoy=-0.08,
        debt_to_equity=3.5,
        current_ratio=0.8,
        interpretation="Weak fundamentals at high valuation",
        confidence=0.75,
    )


@pytest.fixture
def sample_llm_response(direction):
    """Sample LLM response for thesis."""
    if direction == ResearchDirection.BULLISH:
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
    return (
        "THESIS: This stock faces significant headwinds with deteriorating fundamentals and negative "
        "sentiment. Technical indicators suggest continued downward pressure, compounded by weak "
        "earnings and market pessimism. The combination of overvaluation and declining growth "
        "creates substantial downside risk.\n\n"
        "WEAKNESSES:\n"
        "- Weak technical momentum with RSI at 35 and negative MACD\n"
        "- Negative sentiment across 10 recent articles\n"
        "- 5% revenue decline demonstrating weak business momentum\n"
        "- Overvalued at 35.0x P/E relative to declining growth\n"
        "- Key themes of losses, decline, and risks driving narrative\n\n"
        "DOWNSIDE: 30%"
    )


class TestThesisResearcher:
    """Test suite for unified ThesisResearcher."""

    def test_initialization(self, test_container, direction):
        """Test researcher initialization for both directions."""
        if direction == ResearchDirection.BULLISH:
            researcher = test_container.bullish_researcher()
            assert "BullishResearcher" in repr(researcher) or "ThesisResearcher" in repr(researcher)
        else:
            researcher = test_container.bearish_researcher()
            assert "BearishResearcher" in repr(researcher) or "ThesisResearcher" in repr(researcher)

        assert researcher.direction == direction

    async def test_analyze_returns_research_analysis(
        self,
        researcher,
        direction,
        sample_technical_analysis,
        sample_sentiment_analysis,
        sample_news_analysis,
        sample_fundamental_analysis,
        sample_llm_response,
    ):
        """Test analyze returns ResearchAnalysis with correct direction."""
        researcher.llm.acomplete = AsyncMock(return_value=sample_llm_response)

        result = await researcher.analyze(
            "AAPL",
            sample_technical_analysis,
            sample_sentiment_analysis,
            sample_news_analysis,
            sample_fundamental_analysis,
        )

        assert isinstance(result, ResearchAnalysis)
        assert result.direction == direction
        assert result.thesis
        assert len(result.key_points) >= 3
        assert 0.0 <= result.confidence <= 1.0

        # Check target and backward compatibility properties
        if direction == ResearchDirection.BULLISH:
            assert result.target == 25.0
            assert result.target_upside == 25.0
            assert result.key_strengths == result.key_points
            assert result.target_downside is None
            assert result.key_weaknesses is None
        else:
            assert result.target == 30.0
            assert result.target_downside == 30.0
            assert result.key_weaknesses == result.key_points
            assert result.target_upside is None
            assert result.key_strengths is None

    async def test_analyze_calls_llm(
        self,
        researcher,
        direction,
        sample_technical_analysis,
        sample_sentiment_analysis,
        sample_news_analysis,
        sample_fundamental_analysis,
        sample_llm_response,
    ):
        """Test analyze calls LLM with correct parameters."""
        researcher.llm.acomplete = AsyncMock(return_value=sample_llm_response)

        await researcher.analyze(
            "TSLA",
            sample_technical_analysis,
            sample_sentiment_analysis,
            sample_news_analysis,
            sample_fundamental_analysis,
        )

        researcher.llm.acomplete.assert_called_once()
        call_args = researcher.llm.acomplete.call_args
        assert "TSLA" in call_args[0][0]

        # Check system prompt contains direction-appropriate language
        system_prompt = call_args[1]["system"]
        if direction == ResearchDirection.BULLISH:
            assert "optimistic" in system_prompt.lower()
        else:
            assert "skeptical" in system_prompt.lower() or "pessimistic" in system_prompt.lower()

        assert call_args[1]["temperature"] == 0.5

    def test_build_prompt_contains_all_analyses(
        self,
        researcher,
        sample_technical_analysis,
        sample_sentiment_analysis,
        sample_news_analysis,
        sample_fundamental_analysis,
    ):
        """Test prompt includes all analysis components."""
        prompt_vars = researcher._build_prompt_vars(
            "AAPL",
            sample_technical_analysis,
            sample_sentiment_analysis,
            sample_news_analysis,
            sample_fundamental_analysis,
        )

        assert prompt_vars["symbol"] == "AAPL"
        # Check that sentiment score is included
        assert "score" in prompt_vars["sent_str"].lower()
        assert prompt_vars["fund_str"]  # Contains fundamental data

    def test_calculate_confidence_strong_signals(
        self, researcher, direction, sample_sentiment_analysis, sample_news_analysis
    ):
        """Test confidence calculation with strong direction-appropriate signals."""
        if direction == ResearchDirection.BULLISH:
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
            expected_confidence = 0.9  # Base 0.5 + BUY 0.15 + pos sent 0.1 + undervalued 0.1 + growth 0.05
        else:
            technical = TechnicalAnalysis(
                signal=Signal.SELL, rsi=30.0, macd_hist=-1.0, interpretation="Strong sell", confidence=0.9
            )
            fundamental = FundamentalAnalysis(
                valuation="OVERVALUED",
                pe_ratio=40.0,
                eps=2.0,
                revenue_growth_yoy=-0.05,
                earnings_growth_yoy=-0.08,
                debt_to_equity=3.0,
                current_ratio=0.8,
                interpretation="Overvalued with declining growth",
                confidence=0.85,
            )
            # Bearish uses inverted sentiment
            neg_sentiment = SentimentAnalysis(
                overall_sentiment="NEGATIVE",
                sentiment_score=-0.6,
                positive_ratio=0.1,
                negative_ratio=0.7,
                neutral_ratio=0.2,
                article_count=10,
                summary="Negative sentiment",
            )
            expected_confidence = 0.9  # Base 0.5 + SELL 0.15 + neg sent 0.1 + overvalued 0.1 + high debt 0.05

            confidence = researcher._calculate_confidence(
                technical, neg_sentiment, sample_news_analysis, fundamental
            )
            assert confidence == expected_confidence
            return

        confidence = researcher._calculate_confidence(
            technical, sample_sentiment_analysis, sample_news_analysis, fundamental
        )

        assert confidence == expected_confidence

    def test_calculate_confidence_weak_signals(self, researcher, direction, sample_news_analysis):
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

        confidence = researcher._calculate_confidence(technical, sentiment, sample_news_analysis, fundamental)

        # Both directions get same boost for fairly valued
        if direction == ResearchDirection.BULLISH:
            assert confidence == 0.6  # Base 0.5 + fairly valued 0.1
        else:
            assert confidence == 0.5  # Base 0.5, no adjustments for bearish with neutral signals

    def test_calculate_confidence_opposite_signals(self, researcher, direction, sample_news_analysis):
        """Test confidence calculation with signals opposite to direction."""
        if direction == ResearchDirection.BULLISH:
            # Bearish signals for bullish researcher
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
                pe_ratio=45.0,
                eps=2.0,
                revenue_growth_yoy=-0.05,
                earnings_growth_yoy=-0.08,
                debt_to_equity=4.0,
                current_ratio=0.7,
                interpretation="Overvalued with decline",
                confidence=0.7,
            )
            # Base 0.5 - SELL 0.2 - neg sent 0.15 - overvalued 0.1 = 0.05
            expected_min = 0.05
        else:
            # Bullish signals for bearish researcher
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
            # Base 0.5 - BUY 0.2 - pos sent 0.15 - undervalued 0.1 = 0.05
            expected_min = 0.05

        confidence = researcher._calculate_confidence(technical, sentiment, sample_news_analysis, fundamental)

        assert confidence == pytest.approx(expected_min, abs=1e-9)

    async def test_analyze_with_structured_output(
        self,
        researcher,
        direction,
        sample_technical_analysis,
        sample_sentiment_analysis,
        sample_news_analysis,
        sample_fundamental_analysis,
    ):
        """Test analyze uses structured output path when available."""
        from src.agents.thesis_researcher import ResearchLLMResponse

        # Create valid structured response with direction-appropriate fields
        if direction == ResearchDirection.BULLISH:
            structured_response = ResearchLLMResponse(
                thesis="Strong bullish thesis with compelling upside potential.",
                key_strengths=[
                    "Strong technical momentum",
                    "Positive sentiment",
                    "Undervalued fundamentals",
                ],
                key_weaknesses=[],
                target_upside=25.0,
                target_downside=None,
            )
        else:
            structured_response = ResearchLLMResponse(
                thesis="Significant bearish thesis with downside risk.",
                key_strengths=[],
                key_weaknesses=[
                    "Weak technical momentum",
                    "Negative sentiment",
                    "Overvalued fundamentals",
                ],
                target_upside=None,
                target_downside=30.0,
            )

        # Mock astructured to return valid response
        researcher.llm.astructured = AsyncMock(return_value=structured_response)
        researcher.llm.acomplete = AsyncMock()  # Should not be called

        result = await researcher.analyze(
            "AAPL",
            sample_technical_analysis,
            sample_sentiment_analysis,
            sample_news_analysis,
            sample_fundamental_analysis,
        )

        # Verify astructured was used, not acomplete fallback
        researcher.llm.astructured.assert_called_once()
        researcher.llm.acomplete.assert_not_called()

        # Verify result matches structured response
        assert isinstance(result, ResearchAnalysis)
        assert result.direction == direction
        assert result.thesis == structured_response.thesis
        assert len(result.key_points) == 3

        if direction == ResearchDirection.BULLISH:
            assert result.target == 25.0
            assert result.key_points == structured_response.key_strengths
        else:
            assert result.target == 30.0
            assert result.key_points == structured_response.key_weaknesses
