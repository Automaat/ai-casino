"""Tests for FundamentalAnalyst agent."""

import pytest

from src.agents.fundamental import FundamentalAnalysis
from src.agents.models import FundamentalMetrics


class TestFundamentalAnalyst:
    """Tests for FundamentalAnalyst."""

    def test_initialization(self, test_container):
        """Test analyst initialization."""
        analyst = test_container.fundamental_analyst()

        assert analyst.llm is not None
        assert analyst.fetcher is not None

    async def test_analyze_returns_fundamental_analysis(self, test_container_full):
        """Test analyze returns FundamentalAnalysis with correct types."""
        analyst = test_container_full.fundamental_analyst()

        result = await analyst.analyze("AAPL", current_price=150.0)

        assert isinstance(result, FundamentalAnalysis)
        assert result.valuation in ["UNDERVALUED", "FAIRLY_VALUED", "OVERVALUED"]
        assert isinstance(result.pe_ratio, float)
        assert isinstance(result.eps, float)
        assert isinstance(result.revenue_growth_yoy, float)
        assert isinstance(result.earnings_growth_yoy, float)
        assert isinstance(result.debt_to_equity, float)
        assert isinstance(result.current_ratio, float)
        assert isinstance(result.interpretation, str)
        assert 0.0 <= result.confidence <= 1.0

    async def test_analyze_calls_fetcher_and_llm(self, test_container_full):
        """Test analyze calls fetcher and LLM."""
        analyst = test_container_full.fundamental_analyst()
        mock_fundamental_fetcher = test_container_full.fundamental_fetcher()

        await analyst.analyze("AAPL")

        mock_fundamental_fetcher.fetch_overview.assert_called_once_with("AAPL")

    def test_extract_metrics_complete_data(self, test_container, sample_fundamental_overview):
        """Test metrics extraction with complete data."""
        analyst = test_container.fundamental_analyst()

        metrics = analyst._extract_metrics(sample_fundamental_overview)

        assert metrics.pe_ratio == 28.5
        assert metrics.eps == 6.15
        assert metrics.revenue_growth_yoy == 0.062
        assert metrics.earnings_growth_yoy == 0.102
        assert metrics.debt_to_equity == 2.05
        assert metrics.current_ratio == 0.94

    def test_extract_metrics_missing_data(self, test_container):
        """Test metrics extraction with missing data."""
        analyst = test_container.fundamental_analyst()
        overview = {"Symbol": "TEST"}

        metrics = analyst._extract_metrics(overview)

        assert metrics.pe_ratio is None
        assert metrics.eps is None
        assert metrics.revenue_growth_yoy is None
        assert metrics.earnings_growth_yoy is None
        assert metrics.debt_to_equity is None
        assert metrics.current_ratio is None

    def test_extract_metrics_invalid_data(self, test_container):
        """Test metrics extraction with invalid data."""
        analyst = test_container.fundamental_analyst()
        overview = {
            "PERatio": "-",
            "EPS": "N/A",
            "QuarterlyRevenueGrowthYOY": "invalid",
            "DebtToEquity": None,
        }

        metrics = analyst._extract_metrics(overview)

        assert metrics.pe_ratio is None
        assert metrics.eps is None
        assert metrics.revenue_growth_yoy is None
        assert metrics.debt_to_equity is None

    def test_assess_valuation_undervalued(self, test_container):
        """Test valuation assessment for undervalued stock (P/E < 15)."""
        analyst = test_container.fundamental_analyst()
        metrics = FundamentalMetrics(pe_ratio=12.0)

        valuation = analyst._assess_valuation(metrics)

        assert valuation == "UNDERVALUED"

    def test_assess_valuation_overvalued(self, test_container):
        """Test valuation assessment for overvalued stock (P/E > 30)."""
        analyst = test_container.fundamental_analyst()
        metrics = FundamentalMetrics(pe_ratio=35.0)

        valuation = analyst._assess_valuation(metrics)

        assert valuation == "OVERVALUED"

    def test_assess_valuation_fairly_valued(self, test_container):
        """Test valuation assessment for fairly valued stock (15 <= P/E <= 30)."""
        analyst = test_container.fundamental_analyst()
        metrics = FundamentalMetrics(pe_ratio=20.0)

        valuation = analyst._assess_valuation(metrics)

        assert valuation == "FAIRLY_VALUED"

    def test_assess_valuation_no_pe(self, test_container):
        """Test valuation assessment with no P/E ratio."""
        analyst = test_container.fundamental_analyst()
        metrics = FundamentalMetrics(pe_ratio=None)

        valuation = analyst._assess_valuation(metrics)

        assert valuation == "FAIRLY_VALUED"

    def test_calculate_confidence_high_completeness(self, test_container):
        """Test confidence calculation with high data completeness."""
        analyst = test_container.fundamental_analyst()
        metrics = FundamentalMetrics(
            pe_ratio=28.5,
            eps=6.15,
            revenue_growth_yoy=0.062,
            earnings_growth_yoy=0.102,
            debt_to_equity=2.05,
            current_ratio=0.94,
        )
        interpretation = "Strong financial performance with high confidence."

        confidence = analyst._calculate_confidence(metrics, interpretation)

        assert confidence >= 0.8  # 0.5 base + 0.3 completeness + 0.1 signal

    def test_calculate_confidence_low_completeness(self, test_container):
        """Test confidence calculation with low data completeness."""
        analyst = test_container.fundamental_analyst()
        metrics = FundamentalMetrics(
            pe_ratio=None,
            eps=None,
            revenue_growth_yoy=None,
            earnings_growth_yoy=None,
            debt_to_equity=None,
            current_ratio=None,
        )
        interpretation = "Limited data available."

        confidence = analyst._calculate_confidence(metrics, interpretation)

        assert confidence <= 0.5  # 0.5 base - 0.2 uncertainty signal

    def test_calculate_confidence_uncertain_signal(self, test_container):
        """Test confidence calculation with uncertain LLM signal."""
        analyst = test_container.fundamental_analyst()
        metrics = FundamentalMetrics(
            pe_ratio=28.5,
            eps=6.15,
            revenue_growth_yoy=None,
            earnings_growth_yoy=None,
            debt_to_equity=None,
            current_ratio=None,
        )
        interpretation = "Analysis uncertain due to limited data and unclear trends."

        confidence = analyst._calculate_confidence(metrics, interpretation)

        # 0.5 base + 0.3 * (2/6) = 0.6, then -0.2 for "uncertain" = 0.4
        assert confidence < 0.5  # Should be reduced due to "uncertain" signal

    def test_parse_float_valid_string(self, test_container):
        """Test float parsing with valid string."""
        analyst = test_container.fundamental_analyst()

        result = analyst._parse_float("28.5")

        assert result == 28.5

    def test_parse_float_valid_float(self, test_container):
        """Test float parsing with valid float."""
        analyst = test_container.fundamental_analyst()

        result = analyst._parse_float(28.5)

        assert result == 28.5

    def test_parse_float_none(self, test_container):
        """Test float parsing with None."""
        analyst = test_container.fundamental_analyst()

        result = analyst._parse_float(None)

        assert result is None

    def test_parse_float_dash(self, test_container):
        """Test float parsing with dash."""
        analyst = test_container.fundamental_analyst()

        result = analyst._parse_float("-")

        assert result is None

    def test_parse_float_invalid_string(self, test_container):
        """Test float parsing with invalid string."""
        analyst = test_container.fundamental_analyst()

        result = analyst._parse_float("N/A")

        assert result is None

    def test_build_analysis_prompt_complete_data(self, test_container):
        """Test prompt building with complete data."""
        analyst = test_container.fundamental_analyst()
        metrics = FundamentalMetrics(
            pe_ratio=28.5,
            eps=6.15,
            revenue_growth_yoy=0.062,
            earnings_growth_yoy=0.102,
            debt_to_equity=2.05,
            current_ratio=0.94,
        )

        prompt = analyst._build_metrics_section(metrics, "FAIRLY_VALUED", 150.0)

        assert "$150.00" in prompt
        assert "FAIRLY_VALUED" in prompt
        assert "28.5" in prompt
        assert "6.15" in prompt
        assert "6.2%" in prompt
        assert "10.2%" in prompt
        assert "2.05" in prompt
        assert "0.94" in prompt

    def test_build_analysis_prompt_partial_data(self, test_container):
        """Test prompt building with partial data."""
        analyst = test_container.fundamental_analyst()
        metrics = FundamentalMetrics(
            pe_ratio=28.5,
            eps=None,
            revenue_growth_yoy=None,
            earnings_growth_yoy=None,
            debt_to_equity=None,
            current_ratio=None,
        )

        prompt = analyst._build_metrics_section(metrics, "FAIRLY_VALUED", None)

        assert "FAIRLY_VALUED" in prompt
        assert "28.5" in prompt
        assert "$" not in prompt  # No price

    async def test_analyze_without_current_price(self, test_container_full):
        """Test analyze without providing current price."""
        analyst = test_container_full.fundamental_analyst()

        result = await analyst.analyze("AAPL")

        assert isinstance(result, FundamentalAnalysis)
        assert result.confidence > 0.0

    async def test_analyze_edge_case_negative_earnings(self, test_container_full):
        """Test analyze with negative earnings."""
        mock_fundamental_fetcher = test_container_full.fundamental_fetcher()
        mock_fundamental_fetcher.fetch_overview.return_value = {
            "Symbol": "TEST",
            "PERatio": "-10.0",
            "EPS": "-2.50",
            "QuarterlyEarningsGrowthYOY": "-0.15",
        }
        analyst = test_container_full.fundamental_analyst()

        result = await analyst.analyze("TEST")

        assert isinstance(result, FundamentalAnalysis)
        assert result.eps is not None
        assert result.eps < 0

    async def test_analyze_raises_on_fetcher_error(self, test_container_full):
        """Test analyze raises exception when fetcher fails."""
        mock_fundamental_fetcher = test_container_full.fundamental_fetcher()
        mock_fundamental_fetcher.fetch_overview.side_effect = ValueError("API error")
        analyst = test_container_full.fundamental_analyst()

        with pytest.raises(ValueError, match="API error"):
            await analyst.analyze("INVALID")

    def test_repr(self, test_container):
        """Test string representation."""
        analyst = test_container.fundamental_analyst()

        repr_str = repr(analyst)

        assert "FundamentalAnalyst" in repr_str
        assert "llm=" in repr_str
        assert "fetcher=" in repr_str
