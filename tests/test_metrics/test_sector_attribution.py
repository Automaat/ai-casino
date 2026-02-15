"""Tests for sector attribution analysis."""

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from src.daemon.positions import PositionRecord
from src.data.broker import BrokerPosition
from src.data.comparative import Sector
from src.metrics.sector_attribution import (
    SectorAttributionAnalysis,
    SectorAttributionAnalyzer,
    SectorContribution,
)


@pytest.fixture
def sample_positions() -> list[PositionRecord]:
    """Sample position records."""
    return [
        PositionRecord(
            symbol="AAPL",
            entry_timestamp=datetime.now(UTC),
            entry_price=150.0,
            entry_signal="BUY",
            entry_confidence=0.8,
            current_qty=10.0,
            current_stop_loss=145.0,
            initial_stop_loss=145.0,
            profit_targets=[160.0, 170.0],
            last_updated=datetime.now(UTC),
        ),
        PositionRecord(
            symbol="MSFT",
            entry_timestamp=datetime.now(UTC),
            entry_price=300.0,
            entry_signal="BUY",
            entry_confidence=0.75,
            current_qty=5.0,
            current_stop_loss=290.0,
            initial_stop_loss=290.0,
            profit_targets=[320.0, 340.0],
            last_updated=datetime.now(UTC),
        ),
        PositionRecord(
            symbol="JNJ",
            entry_timestamp=datetime.now(UTC),
            entry_price=160.0,
            entry_signal="BUY",
            entry_confidence=0.7,
            current_qty=8.0,
            current_stop_loss=155.0,
            initial_stop_loss=155.0,
            profit_targets=[170.0, 180.0],
            last_updated=datetime.now(UTC),
        ),
    ]


@pytest.fixture
def sample_broker_positions() -> dict[str, BrokerPosition]:
    """Sample broker positions with current prices."""
    return {
        "AAPL": BrokerPosition(
            symbol="AAPL",
            qty=10.0,
            market_value=1600.0,
            avg_entry_price=150.0,
            unrealized_pnl=100.0,
            unrealized_pnl_percent=6.67,
        ),
        "MSFT": BrokerPosition(
            symbol="MSFT",
            qty=5.0,
            market_value=1600.0,
            avg_entry_price=300.0,
            unrealized_pnl=100.0,
            unrealized_pnl_percent=6.67,
        ),
        "JNJ": BrokerPosition(
            symbol="JNJ",
            qty=8.0,
            market_value=1360.0,
            avg_entry_price=160.0,
            unrealized_pnl=80.0,
            unrealized_pnl_percent=6.25,
        ),
    }


class TestSectorAttributionAnalyzer:
    """Tests for SectorAttributionAnalyzer."""

    @pytest.mark.unit
    async def test_empty_positions(self):
        """Test analysis with no positions."""
        analyzer = SectorAttributionAnalyzer()
        result = await analyzer.analyze_attribution([], {})

        assert isinstance(result, SectorAttributionAnalysis)
        assert len(result.contributions) == 0
        assert result.total_portfolio_value == 0.0
        assert result.benchmark_name == "SPY"

    @pytest.mark.unit
    async def test_sector_lookup_caching(self):
        """Test sector lookup is cached."""
        analyzer = SectorAttributionAnalyzer()

        with patch("src.metrics.sector_attribution.yf.Ticker") as mock_ticker:
            mock_info = {
                "symbol": "AAPL",
                "sector": "Technology",
            }
            mock_ticker.return_value.info = mock_info

            sector1 = await analyzer._get_position_sector("AAPL")
            assert sector1 == Sector.TECHNOLOGY

            sector2 = await analyzer._get_position_sector("AAPL")
            assert sector2 == Sector.TECHNOLOGY

            assert mock_ticker.call_count == 1

    @pytest.mark.unit
    async def test_sector_lookup_fallback(self):
        """Test sector lookup fallback to UNKNOWN on error."""
        analyzer = SectorAttributionAnalyzer()

        with patch("src.metrics.sector_attribution.yf.Ticker") as mock_ticker:
            mock_ticker.side_effect = Exception("API error")

            sector = await analyzer._get_position_sector("INVALID")
            assert sector == Sector.UNKNOWN

    @pytest.mark.unit
    async def test_analyze_attribution(self, sample_positions, sample_broker_positions):
        """Test full attribution analysis."""
        analyzer = SectorAttributionAnalyzer()

        async def mock_sector_lookup(symbol: str) -> Sector:
            if symbol in ("AAPL", "MSFT"):
                return Sector.TECHNOLOGY
            if symbol == "JNJ":
                return Sector.HEALTHCARE
            return Sector.UNKNOWN

        with patch.object(analyzer, "_get_position_sector", side_effect=mock_sector_lookup):
            result = await analyzer.analyze_attribution(sample_positions, sample_broker_positions)

        assert isinstance(result, SectorAttributionAnalysis)
        assert len(result.contributions) == 2

        sectors = {c.sector for c in result.contributions}
        assert "TECHNOLOGY" in sectors
        assert "HEALTHCARE" in sectors

        tech = next(c for c in result.contributions if c.sector == "TECHNOLOGY")
        assert tech.position_count == 2
        assert tech.sector_etf == "XLK"
        assert tech.total_value > 0
        assert tech.pnl > 0

        healthcare = next(c for c in result.contributions if c.sector == "HEALTHCARE")
        assert healthcare.position_count == 1
        assert healthcare.sector_etf == "XLV"
        assert healthcare.total_value > 0
        assert healthcare.pnl > 0

        assert result.total_portfolio_value == pytest.approx(tech.total_value + healthcare.total_value)

    @pytest.mark.unit
    async def test_over_under_weight_calculation(self, sample_positions, sample_broker_positions):
        """Test over/underweight vs benchmark calculation."""
        analyzer = SectorAttributionAnalyzer()

        async def mock_all_tech(symbol: str) -> Sector:
            return Sector.TECHNOLOGY

        with patch.object(analyzer, "_get_position_sector", side_effect=mock_all_tech):
            result = await analyzer.analyze_attribution(sample_positions, sample_broker_positions)

        assert len(result.contributions) == 1
        tech = result.contributions[0]

        assert tech.portfolio_weight == pytest.approx(1.0)
        assert tech.benchmark_weight == 0.29
        assert tech.over_under_weight == pytest.approx(0.71, abs=0.01)

    @pytest.mark.unit
    async def test_missing_broker_position(self, sample_positions):
        """Test handling of missing broker position."""
        analyzer = SectorAttributionAnalyzer()

        broker_positions = {
            "AAPL": BrokerPosition(
                symbol="AAPL",
                qty=10.0,
                market_value=1600.0,
                avg_entry_price=150.0,
                unrealized_pnl=100.0,
                unrealized_pnl_percent=6.67,
            ),
        }

        async def mock_all_tech(symbol: str) -> Sector:
            return Sector.TECHNOLOGY

        with patch.object(analyzer, "_get_position_sector", side_effect=mock_all_tech):
            result = await analyzer.analyze_attribution(sample_positions, broker_positions)

        assert len(result.contributions) == 1
        tech = result.contributions[0]
        assert tech.position_count == 1

    @pytest.mark.unit
    def test_sector_contribution_repr(self):
        """Test SectorContribution string representation."""
        contrib = SectorContribution(
            sector="TECHNOLOGY",
            sector_etf="XLK",
            total_value=10000.0,
            portfolio_weight=0.5,
            benchmark_weight=0.29,
            over_under_weight=0.21,
            pnl=500.0,
            return_pct=5.0,
            position_count=3,
        )

        repr_str = repr(contrib)
        assert "TECHNOLOGY" in repr_str
        assert "50.00%" in repr_str or "50%" in repr_str
        assert "$500.00" in repr_str

    @pytest.mark.unit
    def test_sector_attribution_analysis_repr(self):
        """Test SectorAttributionAnalysis string representation."""
        analysis = SectorAttributionAnalysis(
            contributions=[],
            total_portfolio_value=25000.0,
        )

        repr_str = repr(analysis)
        assert "0" in repr_str
        assert "25,000" in repr_str or "25000" in repr_str
