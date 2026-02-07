"""Tests for daemon sector rotation integration."""

from datetime import UTC, datetime
from unittest.mock import Mock, patch

from src.daemon.sector_rotation import DaemonSectorRotation
from src.metrics.sector_rotation import Momentum, SectorRotationAnalysis, SectorStrength
from src.screening.screener import ScreeningResult
from src.strategies.signal import Signal


def _make_analysis(
    leading: list[str] | None = None,
    lagging: list[str] | None = None,
    sectors: list[SectorStrength] | None = None,
) -> SectorRotationAnalysis:
    """Helper to build a SectorRotationAnalysis."""
    if sectors is None:
        sectors = [
            SectorStrength(
                sector="TECHNOLOGY",
                etf="XLK",
                return_1w=3.0,
                return_1m=4.0,
                return_3m=5.0,
                relative_strength=4.1,
                momentum=Momentum.ACCELERATING,
                rank=1,
            ),
            SectorStrength(
                sector="HEALTHCARE",
                etf="XLV",
                return_1w=1.0,
                return_1m=1.5,
                return_3m=2.0,
                relative_strength=1.5,
                momentum=Momentum.NEUTRAL,
                rank=2,
            ),
            SectorStrength(
                sector="ENERGY",
                etf="XLE",
                return_1w=-2.0,
                return_1m=-1.0,
                return_3m=-3.0,
                relative_strength=-1.7,
                momentum=Momentum.DECELERATING,
                rank=3,
            ),
        ]
    return SectorRotationAnalysis(
        sectors=sectors,
        leading_sectors=leading or ["TECHNOLOGY"],
        lagging_sectors=lagging or ["ENERGY"],
        spy_return_1w=1.0,
        spy_return_1m=2.0,
        spy_return_3m=3.0,
        timestamp=datetime.now(UTC),
    )


def _make_candidate(symbol: str, sector: str, score: float) -> ScreeningResult:
    """Helper to build a ScreeningResult."""
    return ScreeningResult(
        symbol=symbol,
        name=f"{symbol} Inc",
        sector=sector,
        score=score,
        signal=Signal.BUY,
        metrics={"rsi": 45.0},
        reason="Momentum signal",
    )


class TestWeightCandidates:
    def test_leading_sector_boosted(self):
        daemon = DaemonSectorRotation()
        analysis = _make_analysis(leading=["TECHNOLOGY", "HEALTHCARE", "FINANCIALS"])
        candidates = [
            _make_candidate("XYZ", "Energy", 80.0),
            _make_candidate("AAPL", "Technology", 75.0),
        ]

        result = daemon.weight_candidates(candidates, analysis, boost_factor=0.15)

        # AAPL in leading sector should be boosted: 75 * 1.15 = 86.25 > 80
        assert result[0].symbol == "AAPL"

    def test_lagging_sector_penalized(self):
        daemon = DaemonSectorRotation()
        analysis = _make_analysis(lagging=["ENERGY", "UTILITIES", "MATERIALS"])
        candidates = [
            _make_candidate("XOM", "Energy", 80.0),
            _make_candidate("ABC", "Healthcare", 78.0),
        ]

        result = daemon.weight_candidates(candidates, analysis, boost_factor=0.15)

        # XOM in lagging sector penalized: 80 * 0.85 = 68 < 78
        assert result[0].symbol == "ABC"

    def test_neutral_sector_unchanged(self):
        daemon = DaemonSectorRotation()
        analysis = _make_analysis(leading=["TECHNOLOGY"], lagging=["ENERGY"])
        candidates = [
            _make_candidate("JNJ", "Healthcare", 90.0),
            _make_candidate("PFE", "Healthcare", 80.0),
        ]

        result = daemon.weight_candidates(candidates, analysis, boost_factor=0.15)

        # Both neutral, original score order preserved
        assert result[0].symbol == "JNJ"
        assert result[1].symbol == "PFE"

    def test_empty_candidates(self):
        daemon = DaemonSectorRotation()
        analysis = _make_analysis()

        result = daemon.weight_candidates([], analysis)

        assert result == []

    def test_does_not_mutate_originals(self):
        daemon = DaemonSectorRotation()
        analysis = _make_analysis(leading=["TECHNOLOGY"])
        candidates = [_make_candidate("AAPL", "Technology", 75.0)]

        result = daemon.weight_candidates(candidates, analysis)

        # Original score unchanged
        assert candidates[0].score == 75.0
        assert result[0].score == 75.0


class TestFlagWeakPositions:
    def test_flags_decelerating_sector(self):
        daemon = DaemonSectorRotation()
        analysis = _make_analysis()

        # Mock yfinance to return Energy sector for XOM
        mock_ticker = Mock()
        mock_ticker.info = {"sector": "Energy"}

        with patch("src.daemon.sector_rotation.yf.Ticker", return_value=mock_ticker):
            flagged = daemon.flag_weak_positions(["XOM"], analysis)

        assert "XOM" in flagged

    def test_no_flags_for_strong_sector(self):
        daemon = DaemonSectorRotation()
        analysis = _make_analysis()

        mock_ticker = Mock()
        mock_ticker.info = {"sector": "Technology"}

        with patch("src.daemon.sector_rotation.yf.Ticker", return_value=mock_ticker):
            flagged = daemon.flag_weak_positions(["AAPL"], analysis)

        assert flagged == []

    def test_empty_positions(self):
        daemon = DaemonSectorRotation()
        analysis = _make_analysis()

        flagged = daemon.flag_weak_positions([], analysis)

        assert flagged == []

    def test_yfinance_failure_skips_symbol(self):
        daemon = DaemonSectorRotation()
        analysis = _make_analysis()

        with patch("src.daemon.sector_rotation.yf.Ticker", side_effect=RuntimeError("API error")):
            flagged = daemon.flag_weak_positions(["BAD"], analysis)

        assert flagged == []
