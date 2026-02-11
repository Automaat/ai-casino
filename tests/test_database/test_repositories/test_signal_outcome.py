"""Tests for SignalOutcomeRepository.

Note: These are unit tests using mocks. Integration tests with real database
should be added separately.
"""

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.daemon.state.models import SignalOutcome
from src.database.repositories.signal_outcome import SignalOutcomeRepository, SignalRecordInput


@pytest.fixture
def mock_session():
    """Create mock async session."""
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.execute = AsyncMock()
    return session


@pytest.fixture
def sample_signal_outcome() -> SignalOutcome:
    """Create sample signal outcome."""
    return SignalOutcome(
        symbol="AAPL",
        timestamp=datetime.now(UTC),
        signal="BUY",
        confidence=0.85,
        price_at_signal=150.0,
        strategy_used="momentum",
        regime="trending_bullish",
        trading_session="REGULAR",
        technical_signal="BUY",
        sentiment_signal="BUY",
        news_signal="HOLD",
    )


@pytest.mark.asyncio
class TestSignalOutcomeRepository:
    """Test SignalOutcomeRepository."""

    async def test_record_signal_creates_outcome(self, mock_session, sample_signal_outcome):
        """Test recording a new signal outcome."""
        repo = SignalOutcomeRepository(mock_session)

        input_data = SignalRecordInput(
            symbol=sample_signal_outcome.symbol,
            timestamp=sample_signal_outcome.timestamp,
            signal=sample_signal_outcome.signal,
            confidence=sample_signal_outcome.confidence,
            price_at_signal=sample_signal_outcome.price_at_signal,
            strategy_used=sample_signal_outcome.strategy_used,
            regime=sample_signal_outcome.regime,
            trading_session=sample_signal_outcome.trading_session,
            technical_signal=sample_signal_outcome.technical_signal,
            sentiment_signal=sample_signal_outcome.sentiment_signal,
            news_signal=sample_signal_outcome.news_signal,
        )
        outcome = await repo.record_signal(input_data)

        assert outcome.symbol == "AAPL"
        assert outcome.signal == "BUY"
        assert outcome.confidence == 0.85
        assert outcome.price_at_signal == 150.0
        assert outcome.regime == "trending_bullish"

        # Verify session methods were called
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    async def test_to_domain_converts_orm_to_model(self, mock_session):
        """Test ORM to domain model conversion."""
        repo = SignalOutcomeRepository(mock_session)

        # Create mock ORM object
        orm = MagicMock()
        orm.symbol = "AAPL"
        orm.timestamp = datetime.now(UTC)
        orm.signal = "BUY"
        orm.confidence = Decimal("0.85")
        orm.price_at_signal = Decimal("150.0")
        orm.strategy_used = "momentum"
        orm.regime = "trending_bullish"
        orm.trading_session = "REGULAR"
        orm.technical_signal = "BUY"
        orm.sentiment_signal = "BUY"
        orm.news_signal = "HOLD"
        orm.price_at_1d = None
        orm.price_at_5d = Decimal("155.0")
        orm.price_at_20d = None
        orm.actual_exit_price = None
        orm.actual_exit_date = None
        orm.outcome_updated_at = None

        domain = repo._to_domain(orm)

        assert domain.symbol == "AAPL"
        assert domain.signal == "BUY"
        assert domain.confidence == 0.85
        assert domain.price_at_5d == 155.0

    async def test_repr(self, mock_session):
        """Test string representation."""
        repo = SignalOutcomeRepository(mock_session)

        assert repr(repo) == "SignalOutcomeRepository()"
