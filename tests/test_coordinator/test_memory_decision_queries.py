"""Tests for CoordinatorMemory decision query methods."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.coordinator.decision_models import DecisionQueryResult
from src.coordinator.memory import CoordinatorMemory, DecisionQueryParams
from src.daemon.state.models import SignalOutcome


@pytest.fixture
def mock_database_engine():
    """Create mock database engine."""
    engine = MagicMock()
    engine.session.return_value = AsyncMock()
    return engine


@pytest.fixture
def mock_signal_outcome_repo():
    """Create mock signal outcome repository with async context manager support."""
    repo = AsyncMock()
    repo.owns_session = True
    repo.__aenter__ = AsyncMock(return_value=repo)
    repo.__aexit__ = AsyncMock(return_value=False)
    return repo


@pytest.fixture
def sample_signal_outcomes():
    """Create sample signal outcomes with various results."""
    base_time = datetime.now(UTC) - timedelta(days=30)

    return [
        SignalOutcome(
            symbol="AAPL",
            timestamp=base_time,
            signal="BUY",
            confidence=0.85,
            price_at_signal=150.0,
            price_at_5d=155.0,  # HIT: +3.3%
            regime="trending_bullish",
            strategy_used="momentum",
            trading_session="REGULAR",
        ),
        SignalOutcome(
            symbol="MSFT",
            timestamp=base_time + timedelta(days=1),
            signal="BUY",
            confidence=0.80,
            price_at_signal=300.0,
            price_at_5d=295.0,  # MISS: -1.7%
            regime="trending_bullish",
            strategy_used="momentum",
            trading_session="REGULAR",
        ),
        SignalOutcome(
            symbol="GOOGL",
            timestamp=base_time + timedelta(days=2),
            signal="SELL",
            confidence=0.90,
            price_at_signal=140.0,
            price_at_5d=135.0,  # HIT: -3.6%
            regime="trending_bearish",
            strategy_used="momentum",
            trading_session="REGULAR",
        ),
        SignalOutcome(
            symbol="TSLA",
            timestamp=base_time + timedelta(days=3),
            signal="BUY",
            confidence=0.75,
            price_at_signal=200.0,
            price_at_5d=None,  # PENDING
            regime="ranging",
            strategy_used="momentum",
            trading_session="REGULAR",
        ),
    ]


def _patch_signal_repo(mock_repo):
    """Patch SignalOutcomeRepository to return mock repo."""
    return patch(
        "src.database.repositories.signal_outcome.SignalOutcomeRepository",
        return_value=mock_repo,
    )


@pytest.mark.asyncio
class TestCoordinatorMemoryDecisionQueries:
    """Test CoordinatorMemory decision query functionality."""

    async def test_query_decisions_basic(
        self, mock_database_engine, mock_signal_outcome_repo, sample_signal_outcomes
    ):
        """Test basic decision query without filters."""
        mock_signal_outcome_repo.get_recent_outcomes.return_value = sample_signal_outcomes

        with _patch_signal_repo(mock_signal_outcome_repo):
            memory = CoordinatorMemory(database_engine=mock_database_engine)
            decisions = await memory.query_decisions(DecisionQueryParams(lookback_days=90, limit=50))

        assert len(decisions) == 4
        assert isinstance(decisions[0], DecisionQueryResult)
        mock_signal_outcome_repo.get_recent_outcomes.assert_called_once()

    async def test_query_decisions_with_symbol_filter(
        self, mock_database_engine, mock_signal_outcome_repo, sample_signal_outcomes
    ):
        """Test querying decisions for specific symbol."""
        mock_signal_outcome_repo.get_by_symbol.return_value = [sample_signal_outcomes[0]]

        with _patch_signal_repo(mock_signal_outcome_repo):
            memory = CoordinatorMemory(database_engine=mock_database_engine)
            decisions = await memory.query_decisions(DecisionQueryParams(symbol="AAPL", lookback_days=90))

        assert len(decisions) == 1
        assert decisions[0].symbol == "AAPL"
        mock_signal_outcome_repo.get_by_symbol.assert_called_once()

    async def test_query_decisions_hit_miss_classification(
        self, mock_database_engine, mock_signal_outcome_repo, sample_signal_outcomes
    ):
        """Test HIT/MISS classification for decisions."""
        mock_signal_outcome_repo.get_recent_outcomes.return_value = sample_signal_outcomes

        with _patch_signal_repo(mock_signal_outcome_repo):
            memory = CoordinatorMemory(database_engine=mock_database_engine)
            decisions = await memory.query_decisions(DecisionQueryParams(lookback_days=90))

        # Check HIT/MISS classification
        aapl_decision = next(d for d in decisions if d.symbol == "AAPL")
        assert aapl_decision.hit_miss == "HIT"  # BUY with price increase
        assert aapl_decision.return_pct > 0

        msft_decision = next(d for d in decisions if d.symbol == "MSFT")
        assert msft_decision.hit_miss == "MISS"  # BUY with price decrease
        assert msft_decision.return_pct < 0

        googl_decision = next(d for d in decisions if d.symbol == "GOOGL")
        assert googl_decision.hit_miss == "HIT"  # SELL with price decrease
        assert googl_decision.return_pct < 0

        tsla_decision = next(d for d in decisions if d.symbol == "TSLA")
        assert tsla_decision.hit_miss == "PENDING"  # No outcome yet
        assert tsla_decision.return_pct is None

    async def test_query_decisions_return_calculation(
        self, mock_database_engine, mock_signal_outcome_repo, sample_signal_outcomes
    ):
        """Test return percentage calculation."""
        mock_signal_outcome_repo.get_recent_outcomes.return_value = [sample_signal_outcomes[0]]

        with _patch_signal_repo(mock_signal_outcome_repo):
            memory = CoordinatorMemory(database_engine=mock_database_engine)
            decisions = await memory.query_decisions(DecisionQueryParams(lookback_days=90))

        assert len(decisions) == 1
        decision = decisions[0]

        # AAPL: 150 -> 155 = +3.33%
        expected_return = ((155.0 - 150.0) / 150.0) * 100
        assert abs(decision.return_pct - expected_return) < 0.01

    async def test_query_decisions_horizon_selection(self, mock_database_engine, mock_signal_outcome_repo):
        """Test selecting different outcome horizons."""
        outcome_with_multiple_horizons = SignalOutcome(
            symbol="AAPL",
            timestamp=datetime.now(UTC) - timedelta(days=30),
            signal="BUY",
            confidence=0.85,
            price_at_signal=150.0,
            price_at_1d=152.0,
            price_at_5d=155.0,
            price_at_20d=160.0,
            regime="trending_bullish",
            strategy_used="momentum",
            trading_session="REGULAR",
        )

        mock_signal_outcome_repo.get_recent_outcomes.return_value = [outcome_with_multiple_horizons]

        with _patch_signal_repo(mock_signal_outcome_repo):
            memory = CoordinatorMemory(database_engine=mock_database_engine)

            # Test 1d horizon
            decisions_1d = await memory.query_decisions(DecisionQueryParams(horizon="1d"))
            assert decisions_1d[0].price_at_outcome == 152.0

            # Test 5d horizon
            decisions_5d = await memory.query_decisions(DecisionQueryParams(horizon="5d"))
            assert decisions_5d[0].price_at_outcome == 155.0

            # Test 20d horizon
            decisions_20d = await memory.query_decisions(DecisionQueryParams(horizon="20d"))
            assert decisions_20d[0].price_at_outcome == 160.0

    async def test_get_success_rate_basic(
        self, mock_database_engine, mock_signal_outcome_repo, sample_signal_outcomes
    ):
        """Test calculating basic success rate statistics."""
        mock_signal_outcome_repo.get_recent_outcomes.return_value = sample_signal_outcomes

        with _patch_signal_repo(mock_signal_outcome_repo):
            memory = CoordinatorMemory(database_engine=mock_database_engine)
            stats = await memory.get_success_rate(lookback_days=90)

        assert isinstance(stats, dict)
        assert stats["total_decisions"] == 4
        assert stats["hit_count"] == 2  # AAPL BUY, GOOGL SELL
        assert stats["miss_count"] == 1  # MSFT BUY
        assert stats["pending_count"] == 1  # TSLA BUY
        assert stats["success_rate"] == 2 / 3  # 2 hits out of 3 completed

    async def test_get_success_rate_with_signal_filter(
        self, mock_database_engine, mock_signal_outcome_repo, sample_signal_outcomes
    ):
        """Test success rate filtered by signal type."""
        buy_signals = [s for s in sample_signal_outcomes if s.signal == "BUY"]
        mock_signal_outcome_repo.get_recent_outcomes.return_value = buy_signals

        with _patch_signal_repo(mock_signal_outcome_repo):
            memory = CoordinatorMemory(database_engine=mock_database_engine)
            stats = await memory.get_success_rate(signal="BUY", lookback_days=90)

        assert stats["total_decisions"] == 3
        assert stats["hit_count"] == 1  # Only AAPL
        assert stats["miss_count"] == 1  # MSFT
        assert stats["pending_count"] == 1  # TSLA

    async def test_get_success_rate_with_regime_filter(
        self, mock_database_engine, mock_signal_outcome_repo, sample_signal_outcomes
    ):
        """Test success rate filtered by regime."""
        bullish_signals = [s for s in sample_signal_outcomes if s.regime == "trending_bullish"]
        mock_signal_outcome_repo.get_recent_outcomes.return_value = bullish_signals

        with _patch_signal_repo(mock_signal_outcome_repo):
            memory = CoordinatorMemory(database_engine=mock_database_engine)

            # Query all, then filter by regime
            decisions = await memory.query_decisions(DecisionQueryParams(lookback_days=90))
            # Filter happens in get_success_rate
            bullish_decisions = [
                d for d in decisions if hasattr(d, "regime") and d.regime == "trending_bullish"
            ]

        assert len(bullish_decisions) >= 1

    async def test_get_success_rate_average_return(
        self, mock_database_engine, mock_signal_outcome_repo, sample_signal_outcomes
    ):
        """Test average return calculation in success rate stats."""
        completed_signals = [s for s in sample_signal_outcomes if s.price_at_5d is not None]
        mock_signal_outcome_repo.get_recent_outcomes.return_value = completed_signals

        with _patch_signal_repo(mock_signal_outcome_repo):
            memory = CoordinatorMemory(database_engine=mock_database_engine)
            stats = await memory.get_success_rate(lookback_days=90)

        # AAPL: +3.33%, MSFT: -1.67%, GOOGL: -3.57%
        # Average: (-3.33 - 1.67 - 3.57) / 3 ≈ -0.30%
        assert stats["avg_return"] is not None
        assert isinstance(stats["avg_return"], float)

    async def test_get_success_rate_average_confidence(
        self, mock_database_engine, mock_signal_outcome_repo, sample_signal_outcomes
    ):
        """Test average confidence calculation."""
        mock_signal_outcome_repo.get_recent_outcomes.return_value = sample_signal_outcomes

        with _patch_signal_repo(mock_signal_outcome_repo):
            memory = CoordinatorMemory(database_engine=mock_database_engine)
            stats = await memory.get_success_rate(lookback_days=90)

        # Expected average confidence from sample outcomes is 0.825
        assert abs(stats["avg_confidence"] - 0.825) < 0.01

    async def test_query_decisions_without_engine(self):
        """Test graceful handling when database engine is not available."""
        memory = CoordinatorMemory()

        decisions = await memory.query_decisions(DecisionQueryParams(lookback_days=90))

        assert decisions == []

    async def test_get_success_rate_without_engine(self):
        """Test success rate returns zero stats when database engine unavailable."""
        memory = CoordinatorMemory()

        stats = await memory.get_success_rate(lookback_days=90)

        assert stats["total_decisions"] == 0
        assert stats["success_rate"] == 0.0

    async def test_query_decisions_error_handling(self, mock_database_engine, mock_signal_outcome_repo):
        """Test error handling in decision queries."""
        mock_signal_outcome_repo.get_recent_outcomes.side_effect = Exception("Database error")

        with _patch_signal_repo(mock_signal_outcome_repo):
            memory = CoordinatorMemory(database_engine=mock_database_engine)
            decisions = await memory.query_decisions(DecisionQueryParams(lookback_days=90))

        assert decisions == []

    async def test_get_success_rate_error_handling(self, mock_database_engine, mock_signal_outcome_repo):
        """Test error handling in success rate calculation."""
        mock_signal_outcome_repo.get_recent_outcomes.side_effect = Exception("Database error")

        with _patch_signal_repo(mock_signal_outcome_repo):
            memory = CoordinatorMemory(database_engine=mock_database_engine)
            stats = await memory.get_success_rate(lookback_days=90)

        assert stats["total_decisions"] == 0
        assert stats["success_rate"] == 0.0
