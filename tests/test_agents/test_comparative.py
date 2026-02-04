"""Tests for comparative analyst agent."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.agents.comparative import (
    ComparativeAnalysis,
    ComparativeAnalyst,
    RelativeValuation,
)
from src.data.comparative import ComparativeData, PerformanceData, StockInfo


@pytest.fixture
def sample_comparative_data():
    """Sample comparative data for testing."""
    return ComparativeData(
        stock_info=StockInfo(
            symbol="AAPL",
            sector="Technology",
            industry="Consumer Electronics",
            pe_ratio=28.5,
            price_to_book=45.2,
        ),
        stock_performance=PerformanceData(
            ytd_return=15.0,
            three_month_return=8.0,
        ),
        sector_etf="XLK",
        sector_pe=32.0,
        sector_performance=PerformanceData(
            ytd_return=12.0,
            three_month_return=5.0,
        ),
        market_pe=22.0,
        market_performance=PerformanceData(
            ytd_return=10.0,
            three_month_return=4.0,
        ),
        fetched_at=datetime.now(),
    )


@pytest.fixture
def mock_comparative_fetcher(sample_comparative_data):
    """Mock comparative data fetcher."""
    mock = MagicMock()
    mock.fetch_comparative_data.return_value = sample_comparative_data
    return mock


@pytest.fixture
def comparative_analyst(mock_llm_client, mock_comparative_fetcher):
    """ComparativeAnalyst instance with mocks."""
    return ComparativeAnalyst(mock_llm_client, mock_comparative_fetcher)


class TestRelativeValuation:
    """Test relative valuation assessment."""

    def test_relatively_undervalued_pe(self, comparative_analyst, sample_comparative_data):
        """Stock with P/E < 0.8x sector is undervalued."""
        # P/E 20 vs sector 32 = 0.625x
        sample_comparative_data.stock_info.pe_ratio = 20.0
        metrics = comparative_analyst._calculate_relative_metrics(sample_comparative_data)

        valuation = comparative_analyst._assess_relative_valuation(sample_comparative_data, metrics)

        assert valuation == RelativeValuation.RELATIVELY_UNDERVALUED

    def test_relatively_overvalued_pe(self, comparative_analyst, sample_comparative_data):
        """Stock with P/E > 1.3x sector is overvalued."""
        # P/E 45 vs sector 32 = 1.4x
        sample_comparative_data.stock_info.pe_ratio = 45.0
        metrics = comparative_analyst._calculate_relative_metrics(sample_comparative_data)

        valuation = comparative_analyst._assess_relative_valuation(sample_comparative_data, metrics)

        assert valuation == RelativeValuation.RELATIVELY_OVERVALUED

    def test_fairly_valued(self, comparative_analyst, sample_comparative_data):
        """Stock with P/E between 0.8x-1.3x sector is fairly valued."""
        # P/E 28.5 vs sector 32 = 0.89x
        metrics = comparative_analyst._calculate_relative_metrics(sample_comparative_data)

        valuation = comparative_analyst._assess_relative_valuation(sample_comparative_data, metrics)

        assert valuation == RelativeValuation.FAIRLY_VALUED

    def test_outperforming_at_discount(self, comparative_analyst, sample_comparative_data):
        """Outperforming >10% at discount triggers undervalued."""
        # P/E 20 vs sector 32 = 0.625x (discount)
        # 3M perf +8% vs sector +5% = +3% (not enough alone)
        # But with 3M perf +20% vs sector +5% = +15% (outperforming)
        sample_comparative_data.stock_info.pe_ratio = 20.0
        sample_comparative_data.stock_performance.three_month_return = 20.0
        metrics = comparative_analyst._calculate_relative_metrics(sample_comparative_data)

        valuation = comparative_analyst._assess_relative_valuation(sample_comparative_data, metrics)

        assert valuation == RelativeValuation.RELATIVELY_UNDERVALUED


class TestRelativeMetrics:
    """Test relative metrics calculation."""

    def test_pe_vs_sector(self, comparative_analyst, sample_comparative_data):
        """Calculate P/E vs sector ratio."""
        metrics = comparative_analyst._calculate_relative_metrics(sample_comparative_data)

        # 28.5 / 32.0 = 0.890625
        assert metrics["pe_vs_sector"] == pytest.approx(0.890625, rel=0.01)

    def test_pe_vs_market(self, comparative_analyst, sample_comparative_data):
        """Calculate P/E vs market ratio."""
        metrics = comparative_analyst._calculate_relative_metrics(sample_comparative_data)

        # 28.5 / 22.0 = 1.295
        assert metrics["pe_vs_market"] == pytest.approx(1.295, rel=0.01)

    def test_performance_vs_sector(self, comparative_analyst, sample_comparative_data):
        """Calculate performance difference vs sector."""
        metrics = comparative_analyst._calculate_relative_metrics(sample_comparative_data)

        assert metrics["perf_vs_sector_ytd"] == pytest.approx(3.0, rel=0.01)
        assert metrics["perf_vs_sector_3m"] == pytest.approx(3.0, rel=0.01)

    def test_performance_vs_market(self, comparative_analyst, sample_comparative_data):
        """Calculate performance difference vs market."""
        metrics = comparative_analyst._calculate_relative_metrics(sample_comparative_data)

        assert metrics["perf_vs_market_ytd"] == pytest.approx(5.0, rel=0.01)
        assert metrics["perf_vs_market_3m"] == pytest.approx(4.0, rel=0.01)

    def test_none_handling(self, comparative_analyst, sample_comparative_data):
        """Handle None values gracefully."""
        sample_comparative_data.stock_info.pe_ratio = None
        sample_comparative_data.sector_pe = None

        metrics = comparative_analyst._calculate_relative_metrics(sample_comparative_data)

        assert metrics["pe_vs_sector"] is None
        assert metrics["pe_vs_market"] is None


class TestConfidenceCalculation:
    """Test confidence score calculation."""

    def test_high_confidence_complete_data(self, comparative_analyst, sample_comparative_data):
        """High confidence with complete data."""
        metrics = comparative_analyst._calculate_relative_metrics(sample_comparative_data)

        confidence = comparative_analyst._calculate_confidence(sample_comparative_data, metrics)

        # Should be > 0.7 with complete data
        assert confidence >= 0.7

    def test_lower_confidence_missing_data(self, comparative_analyst, sample_comparative_data):
        """Lower confidence with missing data."""
        sample_comparative_data.stock_info.pe_ratio = None
        sample_comparative_data.sector_pe = None
        sample_comparative_data.stock_performance.ytd_return = None
        metrics = comparative_analyst._calculate_relative_metrics(sample_comparative_data)

        confidence = comparative_analyst._calculate_confidence(sample_comparative_data, metrics)

        # Should be lower with missing data
        assert confidence < 0.7

    def test_sector_specific_etf_boost(self, comparative_analyst, sample_comparative_data):
        """Confidence boost for sector-specific ETF."""
        metrics = comparative_analyst._calculate_relative_metrics(sample_comparative_data)
        conf_with_sector = comparative_analyst._calculate_confidence(sample_comparative_data, metrics)

        # Change to fallback SPY
        sample_comparative_data.sector_etf = "SPY"
        conf_with_spy = comparative_analyst._calculate_confidence(sample_comparative_data, metrics)

        assert conf_with_sector > conf_with_spy


class TestAnalyze:
    """Test full analyze method."""

    @pytest.mark.asyncio
    async def test_analyze_returns_comparative_analysis(self, comparative_analyst, mock_comparative_fetcher):
        """Analyze returns ComparativeAnalysis model."""
        result = await comparative_analyst.analyze("AAPL")

        assert isinstance(result, ComparativeAnalysis)
        assert result.relative_valuation in list(RelativeValuation)
        assert 0.0 <= result.confidence <= 1.0
        mock_comparative_fetcher.fetch_comparative_data.assert_called_once_with("AAPL")

    @pytest.mark.asyncio
    async def test_analyze_calls_llm(self, comparative_analyst, mock_llm_client):
        """Analyze calls LLM for interpretation."""
        await comparative_analyst.analyze("AAPL")

        mock_llm_client.acomplete.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_includes_sector_etf(self, comparative_analyst):
        """Result includes sector ETF ticker."""
        result = await comparative_analyst.analyze("AAPL")

        assert result.sector_etf == "XLK"


class TestRepr:
    """Test string representations."""

    def test_comparative_analysis_repr(self):
        """ComparativeAnalysis has meaningful repr."""
        analysis = ComparativeAnalysis(
            relative_valuation=RelativeValuation.FAIRLY_VALUED,
            pe_vs_sector=0.89,
            pe_vs_market=1.29,
            perf_vs_sector_ytd=3.0,
            perf_vs_sector_3m=3.0,
            perf_vs_market_ytd=5.0,
            perf_vs_market_3m=4.0,
            sector_etf="XLK",
            interpretation="Test interpretation",
            confidence=0.75,
        )

        repr_str = repr(analysis)

        assert "FAIRLY_VALUED" in repr_str
        assert "0.89" in repr_str
        assert "0.75" in repr_str

    def test_comparative_analyst_repr(self, comparative_analyst):
        """ComparativeAnalyst has meaningful repr."""
        repr_str = repr(comparative_analyst)

        assert "ComparativeAnalyst" in repr_str
