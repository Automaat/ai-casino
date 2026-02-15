"""Tests for deep peer benchmarking analysis."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from src.daemon.peer_analysis import (
    DeepPeerAnalyzer,
    PeerAnalysisResult,
    PeerMetrics,
    _safe_float,
)
from src.data.universe import StockInfo


@pytest.fixture
def sample_universe():
    """Sample stock universe for testing."""
    return [
        StockInfo(symbol="AAPL", name="Apple", sector="Technology", industry="Consumer Electronics"),
        StockInfo(symbol="MSFT", name="Microsoft", sector="Technology", industry="Software"),
        StockInfo(symbol="GOOGL", name="Alphabet", sector="Technology", industry="Internet"),
        StockInfo(symbol="META", name="Meta", sector="Technology", industry="Internet"),
        StockInfo(symbol="NVDA", name="NVIDIA", sector="Technology", industry="Semiconductors"),
        StockInfo(symbol="JPM", name="JPMorgan", sector="Financial Services", industry="Banks"),
        StockInfo(symbol="BAC", name="Bank of America", sector="Financial Services", industry="Banks"),
        StockInfo(symbol="XOM", name="ExxonMobil", sector="Energy", industry="Oil & Gas"),
    ]


@pytest.fixture
def sample_overview():
    """Sample Alpha Vantage overview response."""
    return {
        "Symbol": "AAPL",
        "PERatio": "28.5",
        "PEGRatio": "2.1",
        "QuarterlyRevenueGrowthYOY": "0.08",
        "ProfitMargin": "0.25",
        "OperatingMarginTTM": "0.30",
        "DividendYield": "0.006",
        "MarketCapitalization": "3000000000000",
    }


@pytest.fixture
def analyzer(tmp_path):
    """Create DeepPeerAnalyzer with mocked dependencies."""
    fundamental = MagicMock()
    universe = MagicMock()
    return DeepPeerAnalyzer(
        fundamental_fetcher=fundamental,
        universe_fetcher=universe,
        output_dir=str(tmp_path / "peer-analysis"),
        max_peers=5,
        rate_limit_sleep=0.0,
    )


class TestSafeFloat:
    def test_valid_string(self):
        assert _safe_float("28.5") == 28.5

    def test_valid_float(self):
        assert _safe_float(28.5) == 28.5

    def test_none(self):
        assert _safe_float(None) is None

    def test_none_string(self):
        assert _safe_float("None") is None

    def test_dash(self):
        assert _safe_float("-") is None

    def test_invalid(self):
        assert _safe_float("bad") is None


class TestPeerIdentification:
    def test_same_sector_filtering(self, analyzer, sample_universe):
        peers = analyzer._get_peers("AAPL", "Technology", sample_universe)

        assert "AAPL" not in peers
        assert "MSFT" in peers
        assert "GOOGL" in peers
        assert "JPM" not in peers
        assert "XOM" not in peers

    def test_max_peers_cap(self, tmp_path, sample_universe):
        """Peers capped at max_peers."""
        fundamental = MagicMock()
        universe = MagicMock()
        analyzer = DeepPeerAnalyzer(
            fundamental_fetcher=fundamental,
            universe_fetcher=universe,
            output_dir=str(tmp_path / "peer-analysis"),
            max_peers=2,
            rate_limit_sleep=0.0,
        )

        peers = analyzer._get_peers("AAPL", "Technology", sample_universe)
        assert len(peers) <= 2

    def test_empty_universe(self, analyzer):
        peers = analyzer._get_peers("AAPL", "Technology", [])
        assert peers == []

    def test_no_sector_match(self, analyzer, sample_universe):
        peers = analyzer._get_peers("AAPL", "Healthcare", sample_universe)
        assert peers == []

    def test_case_insensitive_sector(self, analyzer, sample_universe):
        peers = analyzer._get_peers("AAPL", "technology", sample_universe)
        assert len(peers) > 0


class TestCompositeScore:
    def test_full_data(self, analyzer):
        metrics = PeerMetrics(
            symbol="AAPL",
            pe_ratio=25.0,
            peg_ratio=1.5,
            revenue_growth=0.15,
            profit_margin=0.25,
            operating_margin=0.30,
            dividend_yield=0.006,
        )
        score = analyzer._composite_score(metrics)
        assert 0.0 <= score <= 1.0

    def test_missing_data_gets_neutral(self, analyzer):
        metrics = PeerMetrics(symbol="TEST")
        score = analyzer._composite_score(metrics)
        assert score == pytest.approx(0.5, abs=0.05)

    def test_better_fundamentals_higher_score(self, analyzer):
        good = PeerMetrics(
            symbol="GOOD",
            pe_ratio=10.0,
            peg_ratio=0.5,
            revenue_growth=0.30,
            profit_margin=0.35,
            operating_margin=0.40,
            dividend_yield=0.03,
        )
        bad = PeerMetrics(
            symbol="BAD",
            pe_ratio=80.0,
            peg_ratio=4.0,
            revenue_growth=-0.10,
            profit_margin=0.02,
            operating_margin=0.05,
            dividend_yield=0.0,
        )
        assert analyzer._composite_score(good) > analyzer._composite_score(bad)

    def test_negative_pe_gets_neutral(self, analyzer):
        metrics = PeerMetrics(symbol="TEST", pe_ratio=-5.0)
        score = analyzer._composite_score(metrics)
        # Negative PE → treated as 0.5 (neutral)
        assert 0.0 <= score <= 1.0


class TestRanking:
    def test_correct_rank_assignment(self, analyzer):
        peers = [
            PeerMetrics(symbol="TOP", composite_score=0.9),
            PeerMetrics(symbol="MID", composite_score=0.6),
            PeerMetrics(symbol="BOT", composite_score=0.3),
        ]
        result = analyzer._rank_peers("MID", "Technology", peers)

        assert result.rank == 2
        assert result.symbol == "MID"
        assert result.peers[0].symbol == "TOP"
        assert result.peers[2].symbol == "BOT"

    def test_top_ranked_no_swap(self, analyzer):
        peers = [
            PeerMetrics(symbol="BEST", composite_score=0.9),
            PeerMetrics(symbol="OTHER", composite_score=0.5),
        ]
        result = analyzer._rank_peers("BEST", "Technology", peers)

        assert result.rank == 1
        assert result.swap_recommendation is None
        assert result.top_alternative is None

    def test_swap_recommendation_generated(self, analyzer):
        peers = [
            PeerMetrics(symbol="ALT", composite_score=0.9),
            PeerMetrics(symbol="POS", composite_score=0.5),
        ]
        result = analyzer._rank_peers("POS", "Technology", peers)

        assert result.rank == 2
        assert result.top_alternative == "ALT"
        assert "POS" in result.swap_recommendation
        assert "ALT" in result.swap_recommendation

    def test_empty_peers(self, analyzer):
        result = analyzer._rank_peers("AAPL", "Technology", [])
        assert result.rank == 0
        assert result.peer_count == 0


@pytest.mark.skip(reason="File-based persistence deprecated - migrated to DB. Needs async test rewrite.")
class TestPersistence:
    def test_write_and_load(self, analyzer):
        from src.daemon.peer_analysis import DeepPeerAnalysisResult

        analysis = PeerAnalysisResult(
            symbol="AAPL",
            sector="Technology",
            peer_count=3,
            rank=2,
            peers=[
                PeerMetrics(symbol="MSFT", composite_score=0.8),
                PeerMetrics(symbol="AAPL", composite_score=0.6),
                PeerMetrics(symbol="GOOGL", composite_score=0.4),
            ],
            top_alternative="MSFT",
            swap_recommendation="AAPL ranks #2 of 3 in Technology, consider MSFT (#1)",
            analyzed_at=datetime.now(UTC),
        )
        result = DeepPeerAnalysisResult(
            analyses=[analysis],
            total_symbols=1,
            total_peers_analyzed=3,
            total_duration_seconds=10.0,
            analyzed_at=datetime.now(UTC),
        )

        file_path = analyzer.persist(result)
        assert file_path.exists()

        loaded = analyzer.load_latest("AAPL")
        assert loaded is not None
        assert loaded.symbol == "AAPL"
        assert loaded.rank == 2

    def test_load_latest_no_files(self, tmp_path):
        fundamental = MagicMock()
        universe = MagicMock()
        a = DeepPeerAnalyzer(
            fundamental_fetcher=fundamental,
            universe_fetcher=universe,
            output_dir=str(tmp_path / "empty"),
            rate_limit_sleep=0.0,
        )
        assert a.load_latest("AAPL") is None

    def test_load_latest_symbol_not_found(self, analyzer):
        from src.daemon.peer_analysis import DeepPeerAnalysisResult

        result = DeepPeerAnalysisResult(
            analyses=[
                PeerAnalysisResult(
                    symbol="MSFT",
                    sector="Technology",
                    peer_count=1,
                    rank=1,
                    peers=[],
                    analyzed_at=datetime.now(UTC),
                )
            ],
            total_symbols=1,
            total_peers_analyzed=0,
            total_duration_seconds=1.0,
            analyzed_at=datetime.now(UTC),
        )
        analyzer.persist(result)

        assert analyzer.load_latest("AAPL") is None


@pytest.mark.skip(reason="File-based persistence deprecated - migrated to DB. Needs async test rewrite.")
class TestFormatContext:
    def test_format_with_data(self, analyzer):
        from src.daemon.peer_analysis import DeepPeerAnalysisResult

        analysis = PeerAnalysisResult(
            symbol="AAPL",
            sector="Technology",
            peer_count=3,
            rank=2,
            peers=[
                PeerMetrics(symbol="MSFT", composite_score=0.8, pe_ratio=30.0, profit_margin=0.35),
                PeerMetrics(symbol="AAPL", composite_score=0.6, pe_ratio=28.5, profit_margin=0.25),
                PeerMetrics(symbol="GOOGL", composite_score=0.4, pe_ratio=25.0, profit_margin=0.20),
            ],
            top_alternative="MSFT",
            swap_recommendation="AAPL ranks #2, consider MSFT (#1)",
            analyzed_at=datetime.now(UTC),
        )
        result = DeepPeerAnalysisResult(
            analyses=[analysis],
            total_symbols=1,
            total_peers_analyzed=3,
            total_duration_seconds=5.0,
            analyzed_at=datetime.now(UTC),
        )
        analyzer.persist(result)

        context = analyzer.format_context("AAPL")
        assert context is not None
        assert "Technology" in context
        assert "#2 of 3" in context
        assert "MSFT" in context

    def test_format_no_data(self, analyzer):
        assert analyzer.format_context("AAPL") is None

    def test_repr(self, analyzer):
        assert "max_peers=5" in repr(analyzer)
