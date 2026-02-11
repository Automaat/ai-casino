"""Tests for signal outcome tracker."""

from unittest.mock import MagicMock

import pytest

from src.daemon.signal_tracker import SignalOutcomeTracker
from src.metrics.models import SignalUpdateRecord


@pytest.fixture
def mock_broker():
    """Mock broker for testing."""
    return MagicMock()


@pytest.fixture
def mock_historical_cache():
    """Mock historical cache for testing."""
    return MagicMock()


@pytest.fixture
def mock_market_fetcher():
    """Mock market fetcher for testing."""
    return MagicMock()


@pytest.fixture
def signal_tracker(mock_historical_cache, mock_market_fetcher, mock_broker):
    """SignalOutcomeTracker instance for testing."""
    return SignalOutcomeTracker(
        historical_cache=mock_historical_cache, market_fetcher=mock_market_fetcher, broker=mock_broker
    )


class TestSignalOutcomeTracker:
    """Tests for SignalOutcomeTracker."""

    def test_get_early_exits_skips_implementation(self, signal_tracker):
        """Verify early exit detection returns empty dict (not implemented)."""
        signals = [
            SignalUpdateRecord(
                id=1,
                symbol="AAPL",
                timestamp="2024-01-01T00:00:00",
                signal="BUY",
                price_at_signal=150.0,
            ),
            SignalUpdateRecord(
                id=2,
                symbol="MSFT",
                timestamp="2024-01-01T00:00:00",
                signal="SELL",
                price_at_signal=200.0,
            ),
        ]

        result = signal_tracker._get_early_exits(signals)

        # Should return empty dict (logging verified via captured stderr in test output)
        assert result == {}

    def test_get_early_exits_no_broker(self, mock_historical_cache, mock_market_fetcher):
        """Verify early exit detection returns empty dict when no broker."""
        tracker = SignalOutcomeTracker(
            historical_cache=mock_historical_cache, market_fetcher=mock_market_fetcher, broker=None
        )
        signals = [{"symbol": "AAPL", "signal": "BUY"}]

        result = tracker._get_early_exits(signals)

        # Should return empty dict without calling broker
        assert result == {}

    def test_get_early_exits_empty_signals(self, signal_tracker):
        """Verify early exit detection handles empty signals list."""
        result = signal_tracker._get_early_exits([])

        # Should return empty dict
        assert result == {}
