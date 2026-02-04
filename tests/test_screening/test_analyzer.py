"""Tests for screening analyzer."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.screening.analyzer import ScreeningAnalysis, ScreeningAnalyzer
from src.screening.screener import ScreeningCriteria, ScreeningOutput, ScreeningResult
from src.strategies.momentum import Signal


@pytest.fixture
def sample_screening_output():
    """Sample screening output for testing."""
    return ScreeningOutput(
        criteria=ScreeningCriteria.MOMENTUM,
        universe="SP500",
        results=[
            ScreeningResult(
                symbol="AAPL",
                name="Apple Inc.",
                sector="Technology",
                score=0.85,
                signal=Signal.BUY,
                metrics={"rsi": 28.5, "macd_hist": 0.15, "close": 150.0, "sma50": 145.0},
                reason="RSI 28.5 (oversold), MACD bullish (0.15), above 50-day MA",
            ),
            ScreeningResult(
                symbol="MSFT",
                name="Microsoft Corp",
                sector="Technology",
                score=0.78,
                signal=Signal.BUY,
                metrics={"rsi": 32.1, "macd_hist": 0.12, "close": 380.0, "sma50": 370.0},
                reason="RSI 32.1 (oversold), MACD bullish (0.12), above 50-day MA",
            ),
            ScreeningResult(
                symbol="JPM",
                name="JPMorgan Chase",
                sector="Financials",
                score=0.72,
                signal=Signal.BUY,
                metrics={"rsi": 35.0, "macd_hist": 0.08, "close": 180.0, "sma50": 175.0},
                reason="RSI 35.0 (oversold), MACD bullish (0.08), above 50-day MA",
            ),
        ],
        total_screened=500,
        errors=["FAILED1", "FAILED2"],
        screened_at=datetime.now(),
    )


@pytest.fixture
def mock_llm_response():
    """Mock LLM response in expected format."""
    return """SUMMARY: Found 3 tech and financial stocks with oversold conditions, bullish reversal signals.

TOP_PICKS:
1. AAPL - Strongest RSI oversold signal with highest MACD histogram
2. MSFT - Solid momentum setup with price above key moving average
3. JPM - Diversification play in financials with similar technical setup

SECTOR_INSIGHTS: Heavy concentration in Technology (2 of 3 picks), sector-wide oversold.

RISK_FACTORS: Similar technical patterns create correlation risk if sentiment shifts.

NEXT_STEPS: Research fundamental catalysts and set position sizing based on allocation."""


@pytest.fixture
def mock_llm_client(mock_llm_response):
    """Mock LLM client for testing."""
    mock = MagicMock()
    mock.provider = "ollama"
    mock.model = "qwen3:14b"
    mock.complete.return_value = mock_llm_response
    return mock


@pytest.fixture
def analyzer(mock_llm_client):
    """Create ScreeningAnalyzer with mock LLM."""
    return ScreeningAnalyzer(llm_client=mock_llm_client)


class TestScreeningAnalysis:
    """Tests for ScreeningAnalysis model."""

    def test_create(self):
        """Test ScreeningAnalysis creation."""
        analysis = ScreeningAnalysis(
            summary="Test summary",
            top_picks=["AAPL - reason", "MSFT - reason"],
            sector_insights="Sector insight",
            risk_factors="Risk factors",
            next_steps="Next steps",
        )

        assert analysis.summary == "Test summary"
        assert len(analysis.top_picks) == 2
        assert "AAPL" in analysis.top_picks[0]


class TestScreeningAnalyzer:
    """Tests for ScreeningAnalyzer."""

    def test_init(self, analyzer):
        """Test analyzer initialization."""
        assert "ScreeningAnalyzer" in repr(analyzer)

    def test_analyze(self, analyzer, sample_screening_output, mock_llm_client):
        """Test screening analysis."""
        analysis = analyzer.analyze(sample_screening_output)

        assert isinstance(analysis, ScreeningAnalysis)
        assert analysis.summary
        assert len(analysis.top_picks) <= 3
        mock_llm_client.complete.assert_called_once()

    def test_analyze_with_market_context(self, analyzer, sample_screening_output, mock_llm_client):
        """Test analysis with market context."""
        context = "Market is in a corrective phase after recent highs."

        analysis = analyzer.analyze(sample_screening_output, market_context=context)

        call_args = mock_llm_client.complete.call_args
        assert "corrective phase" in call_args[0][0]
        assert isinstance(analysis, ScreeningAnalysis)

    def test_build_prompt(self, analyzer, sample_screening_output):
        """Test prompt building."""
        prompt = analyzer._build_prompt(sample_screening_output, None)

        assert "AAPL" in prompt
        assert "momentum" in prompt
        assert "SP500" in prompt
        assert "rsi=28.5" in prompt

    def test_build_prompt_with_context(self, analyzer, sample_screening_output):
        """Test prompt building with context."""
        context = "Bear market conditions"
        prompt = analyzer._build_prompt(sample_screening_output, context)

        assert "Bear market conditions" in prompt

    def test_parse_response_standard(self, analyzer, mock_llm_response):
        """Test parsing standard LLM response."""
        analysis = analyzer._parse_response(mock_llm_response)

        assert "oversold conditions" in analysis.summary.lower() or "screening" in analysis.summary.lower()
        assert len(analysis.top_picks) == 3
        assert "Technology" in analysis.sector_insights
        assert analysis.risk_factors
        assert analysis.next_steps

    def test_parse_response_minimal(self, analyzer):
        """Test parsing minimal response."""
        minimal_response = """SUMMARY: Brief summary.

TOP_PICKS:
1. AAPL - reason

SECTOR_INSIGHTS: Sector note.

RISK_FACTORS: Risk note.

NEXT_STEPS: Action item."""

        analysis = analyzer._parse_response(minimal_response)

        assert analysis.summary == "Brief summary."
        assert len(analysis.top_picks) == 1
        assert "AAPL" in analysis.top_picks[0]

    def test_parse_response_empty(self, analyzer):
        """Test parsing empty response provides defaults."""
        analysis = analyzer._parse_response("")

        assert analysis.summary
        assert analysis.top_picks
        assert analysis.sector_insights
        assert analysis.risk_factors
        assert analysis.next_steps

    def test_parse_response_malformed(self, analyzer):
        """Test parsing malformed response."""
        malformed = """Some random text
        that doesn't follow the format
        at all."""

        analysis = analyzer._parse_response(malformed)

        # Should still return valid object with defaults
        assert isinstance(analysis, ScreeningAnalysis)
        assert analysis.summary  # Has default
