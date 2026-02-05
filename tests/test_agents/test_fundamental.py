"""Tests for FundamentalAnalyst agent."""

import pytest

from src.agents.fundamental import FundamentalAnalysis, FundamentalAnalyst


class TestFundamentalAnalyst:
    """Tests for FundamentalAnalyst."""

    def test_initialization(self, mock_llm_client, mock_fundamental_fetcher):
        """Test analyst initialization."""
        analyst = FundamentalAnalyst(mock_llm_client, mock_fundamental_fetcher)

        assert analyst.llm == mock_llm_client
        assert analyst.fetcher == mock_fundamental_fetcher

    @pytest.mark.asyncio
    async def test_analyze_returns_fundamental_analysis(self, mock_llm_client, mock_fundamental_fetcher):
        """Test analyze returns FundamentalAnalysis with correct types."""
        analyst = FundamentalAnalyst(mock_llm_client, mock_fundamental_fetcher)

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

    @pytest.mark.asyncio
    async def test_analyze_calls_fetcher_and_llm(self, mock_llm_client, mock_fundamental_fetcher):
        """Test analyze calls fetcher and LLM."""
        analyst = FundamentalAnalyst(mock_llm_client, mock_fundamental_fetcher)

        await analyst.analyze("AAPL")

        mock_fundamental_fetcher.fetch_overview.assert_called_once_with("AAPL")
        mock_llm_client.acomplete.assert_called_once()

    def test_extract_metrics_complete_data(self, mock_fundamental_fetcher, sample_fundamental_overview):
        """Test metrics extraction with complete data."""
        analyst = FundamentalAnalyst(None, mock_fundamental_fetcher)

        metrics = analyst._extract_metrics(sample_fundamental_overview)

        assert metrics["pe_ratio"] == 28.5
        assert metrics["eps"] == 6.15
        assert metrics["revenue_growth_yoy"] == 0.062
        assert metrics["earnings_growth_yoy"] == 0.102
        assert metrics["debt_to_equity"] == 2.05
        assert metrics["current_ratio"] == 0.94

    def test_extract_metrics_missing_data(self, mock_fundamental_fetcher):
        """Test metrics extraction with missing data."""
        analyst = FundamentalAnalyst(None, mock_fundamental_fetcher)
        overview = {"Symbol": "TEST"}

        metrics = analyst._extract_metrics(overview)

        assert metrics["pe_ratio"] is None
        assert metrics["eps"] is None
        assert metrics["revenue_growth_yoy"] is None
        assert metrics["earnings_growth_yoy"] is None
        assert metrics["debt_to_equity"] is None
        assert metrics["current_ratio"] is None

    def test_extract_metrics_invalid_data(self, mock_fundamental_fetcher):
        """Test metrics extraction with invalid data."""
        analyst = FundamentalAnalyst(None, mock_fundamental_fetcher)
        overview = {
            "PERatio": "-",
            "EPS": "N/A",
            "QuarterlyRevenueGrowthYOY": "invalid",
            "DebtToEquity": None,
        }

        metrics = analyst._extract_metrics(overview)

        assert metrics["pe_ratio"] is None
        assert metrics["eps"] is None
        assert metrics["revenue_growth_yoy"] is None
        assert metrics["debt_to_equity"] is None

    def test_assess_valuation_undervalued(self, mock_fundamental_fetcher):
        """Test valuation assessment for undervalued stock (P/E < 15)."""
        analyst = FundamentalAnalyst(None, mock_fundamental_fetcher)
        metrics = {"pe_ratio": 12.0}

        valuation = analyst._assess_valuation(metrics)

        assert valuation == "UNDERVALUED"

    def test_assess_valuation_overvalued(self, mock_fundamental_fetcher):
        """Test valuation assessment for overvalued stock (P/E > 30)."""
        analyst = FundamentalAnalyst(None, mock_fundamental_fetcher)
        metrics = {"pe_ratio": 35.0}

        valuation = analyst._assess_valuation(metrics)

        assert valuation == "OVERVALUED"

    def test_assess_valuation_fairly_valued(self, mock_fundamental_fetcher):
        """Test valuation assessment for fairly valued stock (15 <= P/E <= 30)."""
        analyst = FundamentalAnalyst(None, mock_fundamental_fetcher)
        metrics = {"pe_ratio": 20.0}

        valuation = analyst._assess_valuation(metrics)

        assert valuation == "FAIRLY_VALUED"

    def test_assess_valuation_no_pe(self, mock_fundamental_fetcher):
        """Test valuation assessment with no P/E ratio."""
        analyst = FundamentalAnalyst(None, mock_fundamental_fetcher)
        metrics = {"pe_ratio": None}

        valuation = analyst._assess_valuation(metrics)

        assert valuation == "FAIRLY_VALUED"

    def test_calculate_confidence_high_completeness(self, mock_fundamental_fetcher):
        """Test confidence calculation with high data completeness."""
        analyst = FundamentalAnalyst(None, mock_fundamental_fetcher)
        metrics = {
            "pe_ratio": 28.5,
            "eps": 6.15,
            "revenue_growth_yoy": 0.062,
            "earnings_growth_yoy": 0.102,
            "debt_to_equity": 2.05,
            "current_ratio": 0.94,
        }
        interpretation = "Strong financial performance with high confidence."

        confidence = analyst._calculate_confidence(metrics, interpretation)

        assert confidence >= 0.8  # 0.5 base + 0.3 completeness + 0.1 signal

    def test_calculate_confidence_low_completeness(self, mock_fundamental_fetcher):
        """Test confidence calculation with low data completeness."""
        analyst = FundamentalAnalyst(None, mock_fundamental_fetcher)
        metrics = {
            "pe_ratio": None,
            "eps": None,
            "revenue_growth_yoy": None,
            "earnings_growth_yoy": None,
            "debt_to_equity": None,
            "current_ratio": None,
        }
        interpretation = "Limited data available."

        confidence = analyst._calculate_confidence(metrics, interpretation)

        assert confidence <= 0.5  # 0.5 base - 0.2 uncertainty signal

    def test_calculate_confidence_uncertain_signal(self, mock_fundamental_fetcher):
        """Test confidence calculation with uncertain LLM signal."""
        analyst = FundamentalAnalyst(None, mock_fundamental_fetcher)
        metrics = {
            "pe_ratio": 28.5,
            "eps": 6.15,
            "revenue_growth_yoy": None,
            "earnings_growth_yoy": None,
            "debt_to_equity": None,
            "current_ratio": None,
        }
        interpretation = "Analysis uncertain due to limited data and unclear trends."

        confidence = analyst._calculate_confidence(metrics, interpretation)

        # 0.5 base + 0.3 * (2/6) = 0.6, then -0.2 for "uncertain" = 0.4
        assert confidence < 0.5  # Should be reduced due to "uncertain" signal

    def test_parse_float_valid_string(self, mock_fundamental_fetcher):
        """Test float parsing with valid string."""
        analyst = FundamentalAnalyst(None, mock_fundamental_fetcher)

        result = analyst._parse_float("28.5")

        assert result == 28.5

    def test_parse_float_valid_float(self, mock_fundamental_fetcher):
        """Test float parsing with valid float."""
        analyst = FundamentalAnalyst(None, mock_fundamental_fetcher)

        result = analyst._parse_float(28.5)

        assert result == 28.5

    def test_parse_float_none(self, mock_fundamental_fetcher):
        """Test float parsing with None."""
        analyst = FundamentalAnalyst(None, mock_fundamental_fetcher)

        result = analyst._parse_float(None)

        assert result is None

    def test_parse_float_dash(self, mock_fundamental_fetcher):
        """Test float parsing with dash."""
        analyst = FundamentalAnalyst(None, mock_fundamental_fetcher)

        result = analyst._parse_float("-")

        assert result is None

    def test_parse_float_invalid_string(self, mock_fundamental_fetcher):
        """Test float parsing with invalid string."""
        analyst = FundamentalAnalyst(None, mock_fundamental_fetcher)

        result = analyst._parse_float("N/A")

        assert result is None

    def test_build_analysis_prompt_complete_data(self, mock_fundamental_fetcher):
        """Test prompt building with complete data."""
        analyst = FundamentalAnalyst(None, mock_fundamental_fetcher)
        metrics = {
            "pe_ratio": 28.5,
            "eps": 6.15,
            "revenue_growth_yoy": 0.062,
            "earnings_growth_yoy": 0.102,
            "debt_to_equity": 2.05,
            "current_ratio": 0.94,
        }

        prompt = analyst._build_metrics_section(metrics, "FAIRLY_VALUED", 150.0)

        assert "$150.00" in prompt
        assert "FAIRLY_VALUED" in prompt
        assert "28.5" in prompt
        assert "6.15" in prompt
        assert "6.2%" in prompt
        assert "10.2%" in prompt
        assert "2.05" in prompt
        assert "0.94" in prompt

    def test_build_analysis_prompt_partial_data(self, mock_fundamental_fetcher):
        """Test prompt building with partial data."""
        analyst = FundamentalAnalyst(None, mock_fundamental_fetcher)
        metrics = {
            "pe_ratio": 28.5,
            "eps": None,
            "revenue_growth_yoy": None,
            "earnings_growth_yoy": None,
            "debt_to_equity": None,
            "current_ratio": None,
        }

        prompt = analyst._build_metrics_section(metrics, "FAIRLY_VALUED", None)

        assert "FAIRLY_VALUED" in prompt
        assert "28.5" in prompt
        assert "$" not in prompt  # No price

    @pytest.mark.asyncio
    async def test_analyze_without_current_price(self, mock_llm_client, mock_fundamental_fetcher):
        """Test analyze without providing current price."""
        analyst = FundamentalAnalyst(mock_llm_client, mock_fundamental_fetcher)

        result = await analyst.analyze("AAPL")

        assert isinstance(result, FundamentalAnalysis)
        assert result.confidence > 0.0

    @pytest.mark.asyncio
    async def test_analyze_edge_case_negative_earnings(self, mock_llm_client, mock_fundamental_fetcher):
        """Test analyze with negative earnings."""
        mock_fundamental_fetcher.fetch_overview.return_value = {
            "Symbol": "TEST",
            "PERatio": "-10.0",
            "EPS": "-2.50",
            "QuarterlyEarningsGrowthYOY": "-0.15",
        }
        analyst = FundamentalAnalyst(mock_llm_client, mock_fundamental_fetcher)

        result = await analyst.analyze("TEST")

        assert isinstance(result, FundamentalAnalysis)
        assert result.eps is not None
        assert result.eps < 0

    @pytest.mark.asyncio
    async def test_analyze_raises_on_fetcher_error(self, mock_llm_client, mock_fundamental_fetcher):
        """Test analyze raises exception when fetcher fails."""
        mock_fundamental_fetcher.fetch_overview.side_effect = ValueError("API error")
        analyst = FundamentalAnalyst(mock_llm_client, mock_fundamental_fetcher)

        with pytest.raises(ValueError, match="API error"):
            await analyst.analyze("INVALID")

    def test_repr(self, mock_llm_client, mock_fundamental_fetcher):
        """Test string representation."""
        analyst = FundamentalAnalyst(mock_llm_client, mock_fundamental_fetcher)

        repr_str = repr(analyst)

        assert "FundamentalAnalyst" in repr_str
        assert "llm=" in repr_str
        assert "fetcher=" in repr_str
