"""Tests for sector rotation analysis."""

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from src.metrics.sector_rotation import (
    MOMENTUM_THRESHOLD,
    WEIGHT_1M,
    WEIGHT_1W,
    WEIGHT_3M,
    Momentum,
    SectorRotationAnalyzer,
    SectorStrength,
)


class TestCalculateRelativeReturn:
    def test_positive_relative_return(self):
        analyzer = SectorRotationAnalyzer()
        # Sector up 10%, SPY up 5% => relative = 5%
        sector_prices = [100.0] * 20 + [110.0]
        spy_prices = [100.0] * 20 + [105.0]
        result = analyzer._calculate_relative_return(sector_prices, spy_prices, 20)
        assert result == pytest.approx(5.0, abs=0.1)

    def test_negative_relative_return(self):
        analyzer = SectorRotationAnalyzer()
        # Sector up 2%, SPY up 8% => relative = -6%
        sector_prices = [100.0] * 20 + [102.0]
        spy_prices = [100.0] * 20 + [108.0]
        result = analyzer._calculate_relative_return(sector_prices, spy_prices, 20)
        assert result == pytest.approx(-6.0, abs=0.1)

    def test_insufficient_data_returns_zero(self):
        analyzer = SectorRotationAnalyzer()
        result = analyzer._calculate_relative_return([100.0, 110.0], [100.0, 105.0], 20)
        assert result == 0.0


class TestCalculateMomentum:
    def test_accelerating(self):
        analyzer = SectorRotationAnalyzer()
        # 1w significantly higher than 1m
        result = analyzer._calculate_momentum(3.0, 1.0)
        assert result == Momentum.ACCELERATING

    def test_decelerating(self):
        analyzer = SectorRotationAnalyzer()
        # 1w significantly lower than 1m
        result = analyzer._calculate_momentum(1.0, 3.0)
        assert result == Momentum.DECELERATING

    def test_neutral(self):
        analyzer = SectorRotationAnalyzer()
        # Difference within threshold
        result = analyzer._calculate_momentum(2.0, 2.0 + MOMENTUM_THRESHOLD * 0.5)
        assert result == Momentum.NEUTRAL

    def test_exact_threshold_is_neutral(self):
        analyzer = SectorRotationAnalyzer()
        result = analyzer._calculate_momentum(2.0 + MOMENTUM_THRESHOLD, 2.0)
        assert result == Momentum.NEUTRAL


class TestCalculateComposite:
    def test_weighting(self):
        analyzer = SectorRotationAnalyzer()
        result = analyzer._calculate_composite(10.0, 5.0, 3.0)
        expected = WEIGHT_1W * 10.0 + WEIGHT_1M * 5.0 + WEIGHT_3M * 3.0
        assert result == pytest.approx(expected)

    def test_all_zero(self):
        analyzer = SectorRotationAnalyzer()
        result = analyzer._calculate_composite(0.0, 0.0, 0.0)
        assert result == 0.0


class TestAnalyze:
    def test_ranking(self):
        """Test sectors are ranked by composite strength (1=strongest)."""
        analyzer = SectorRotationAnalyzer()

        # Mock _fetch_sector_data to return predictable prices
        mock_closes = {"SPY": [100.0] * 130 + [105.0]}
        # Create 11 sectors with varying returns
        from src.metrics.sector_rotation import SECTOR_ETFS

        for i, (_name, etf) in enumerate(SECTOR_ETFS):
            # Each sector returns slightly more than the previous
            end_price = 100.0 + (i + 1) * 2
            mock_closes[etf] = [100.0] * 130 + [end_price]

        with patch.object(analyzer, "_fetch_sector_data", return_value=mock_closes):
            analysis = analyzer.analyze()

        assert len(analysis.sectors) == 11
        # Check ranks are 1-11
        ranks = [s.rank for s in analysis.sectors]
        assert sorted(ranks) == list(range(1, 12))
        # First sector should have rank 1 (highest relative strength)
        assert analysis.sectors[0].rank == 1

    def test_leading_lagging(self):
        """Test leading=top 3, lagging=bottom 3."""
        analyzer = SectorRotationAnalyzer()

        from src.metrics.sector_rotation import SECTOR_ETFS

        mock_closes = {"SPY": [100.0] * 130 + [105.0]}
        for i, (_name, etf) in enumerate(SECTOR_ETFS):
            end_price = 100.0 + (i + 1) * 2
            mock_closes[etf] = [100.0] * 130 + [end_price]

        with patch.object(analyzer, "_fetch_sector_data", return_value=mock_closes):
            analysis = analyzer.analyze()

        assert len(analysis.leading_sectors) == 3
        assert len(analysis.lagging_sectors) == 3
        # Leading should be top 3 ranked
        assert all(s in [sec.sector for sec in analysis.sectors[:3]] for s in analysis.leading_sectors)

    def test_spy_returns_populated(self):
        """Test SPY returns are calculated."""
        analyzer = SectorRotationAnalyzer()

        from src.metrics.sector_rotation import SECTOR_ETFS

        mock_closes = {"SPY": [100.0] * 130 + [110.0]}
        for _, etf in SECTOR_ETFS:
            mock_closes[etf] = [100.0] * 130 + [105.0]

        with patch.object(analyzer, "_fetch_sector_data", return_value=mock_closes):
            analysis = analyzer.analyze()

        assert analysis.spy_return_1w == pytest.approx(10.0, abs=0.1)
        assert analysis.timestamp is not None

    def test_missing_etf_data_skipped(self):
        """Test sectors with missing data are skipped."""
        analyzer = SectorRotationAnalyzer()

        from src.metrics.sector_rotation import SECTOR_ETFS

        # Only provide SPY and first 5 sectors
        mock_closes = {"SPY": [100.0] * 130 + [105.0]}
        for _, etf in SECTOR_ETFS[:5]:
            mock_closes[etf] = [100.0] * 130 + [108.0]

        with patch.object(analyzer, "_fetch_sector_data", return_value=mock_closes):
            analysis = analyzer.analyze()

        assert len(analysis.sectors) == 5


class TestFormatContext:
    def test_format_includes_all_sectors(self):
        analyzer = SectorRotationAnalyzer()

        from src.metrics.sector_rotation import SectorRotationAnalysis

        sectors = [
            SectorStrength(
                sector="TECHNOLOGY",
                etf="XLK",
                return_1w=2.0,
                return_1m=3.0,
                return_3m=5.0,
                relative_strength=3.5,
                momentum=Momentum.ACCELERATING,
                rank=1,
            ),
            SectorStrength(
                sector="ENERGY",
                etf="XLE",
                return_1w=-1.0,
                return_1m=-2.0,
                return_3m=-3.0,
                relative_strength=-2.1,
                momentum=Momentum.DECELERATING,
                rank=2,
            ),
        ]
        analysis = SectorRotationAnalysis(
            sectors=sectors,
            leading_sectors=["TECHNOLOGY"],
            lagging_sectors=["ENERGY"],
            spy_return_1w=1.0,
            spy_return_1m=2.0,
            spy_return_3m=4.0,
            timestamp=datetime.now(UTC),
        )

        result = analyzer.format_context(analysis)

        assert "Leading Sectors: TECHNOLOGY" in result
        assert "Lagging Sectors: ENERGY" in result
        assert "XLK" in result
        assert "ACCELERATING" in result
        assert "DECELERATING" in result
