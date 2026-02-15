"""Tests for signal analytics service."""

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from src.database.models import AnalysisRecordORM, SignalOutcomeORM
from src.metrics.signal_analytics import SignalAnalyticsService


@pytest.fixture
async def sample_signals(db_session):
    """Create sample signals for testing."""
    base_time = datetime.now(UTC)
    signals = []

    # Create BUY signals (5 total)
    for i in range(5):
        signal = SignalOutcomeORM(
            id=uuid.uuid4(),
            symbol="AAPL",
            timestamp=base_time - timedelta(days=i),
            signal="BUY",
            confidence=Decimal(f"0.{70 + i}"),
            price_at_signal=Decimal("150.00"),
            price_at_5d=Decimal("155.00") if i < 3 else Decimal("148.00"),  # 3 profitable, 2 unprofitable
            created_at=base_time,
        )
        db_session.add(signal)
        signals.append(signal)

    # Create SELL signals (3 total)
    for i in range(3):
        signal = SignalOutcomeORM(
            id=uuid.uuid4(),
            symbol="AAPL",
            timestamp=base_time - timedelta(days=i + 10),
            signal="SELL",
            confidence=Decimal(f"0.{80 + i}"),
            price_at_signal=Decimal("150.00"),
            price_at_5d=Decimal("145.00") if i < 2 else Decimal("152.00"),  # 2 profitable, 1 unprofitable
            created_at=base_time,
        )
        db_session.add(signal)
        signals.append(signal)

    await db_session.commit()
    return signals


@pytest.fixture
async def sample_analyses(db_session, sample_signals):
    """Create sample analysis records (executions) for testing."""
    analyses = []

    # Refresh signals to ensure they're attached to session
    await db_session.refresh(sample_signals[0])
    await db_session.refresh(sample_signals[1])
    await db_session.refresh(sample_signals[2])
    await db_session.refresh(sample_signals[5])
    await db_session.refresh(sample_signals[6])

    # Execute first 3 BUY signals and first 2 SELL signals
    signals_to_execute = sample_signals[:3] + sample_signals[5:7]

    for signal in signals_to_execute:
        analysis = AnalysisRecordORM(
            id=uuid.uuid4(),
            symbol=signal.symbol,
            timestamp=signal.timestamp + timedelta(minutes=2),  # 2 min after signal
            signal=signal.signal,
            confidence=signal.confidence,
            executed_trade=True,
            trading_session="REGULAR",
            is_paper_trade=True,
            reasoning=[],
            created_at=datetime.now(UTC),
        )
        db_session.add(analysis)
        analyses.append(analysis)

    await db_session.commit()
    return analyses


@pytest.mark.asyncio
async def test_flow_summary_calculation(sample_signals, sample_analyses):
    """Test flow summary calculation."""
    service = SignalAnalyticsService()

    start = datetime.now(UTC) - timedelta(days=30)
    end = datetime.now(UTC)

    summary = await service.get_flow_summary(start, end)

    # Debug: Check that 5 analyses were created
    assert len(sample_analyses) == 5

    assert summary.total_signals == 8  # 5 BUY + 3 SELL
    assert summary.total_buy_signals == 5
    assert summary.total_sell_signals == 3
    # Note: 5 executions matched (expanded window captures all analyses)
    assert summary.executed_count == 5
    assert summary.not_executed_count == 3
    assert summary.execution_rate == pytest.approx(5 / 8)


@pytest.mark.asyncio
async def test_sankey_data_structure(sample_signals, sample_analyses):
    """Test Sankey data structure."""
    service = SignalAnalyticsService()

    start = datetime.now(UTC) - timedelta(days=30)
    end = datetime.now(UTC)

    sankey = await service.get_sankey_data(start, end)

    # Check nodes
    assert len(sankey.nodes) == 6
    node_names = [n.name for n in sankey.nodes]
    assert "BUY" in node_names
    assert "SELL" in node_names
    assert "Executed" in node_names
    assert "Not Executed" in node_names
    assert "Profitable" in node_names
    assert "Unprofitable" in node_names

    # Check links exist and have values
    assert len(sankey.links) > 0
    for link in sankey.links:
        assert link.value > 0


@pytest.mark.asyncio
async def test_accuracy_by_type_5d(sample_signals, sample_analyses):
    """Test accuracy by type calculation for 5d horizon."""
    service = SignalAnalyticsService()

    start = datetime.now(UTC) - timedelta(days=30)
    end = datetime.now(UTC)

    accuracy = await service.get_accuracy_by_type(start, end, "5d")

    assert len(accuracy) == 2
    buy_accuracy = next(a for a in accuracy if a.signal_type == "BUY")
    sell_accuracy = next(a for a in accuracy if a.signal_type == "SELL")

    # BUY: 3 profitable out of 5 total = 60%
    assert buy_accuracy.hit_rate == pytest.approx(0.6)
    assert buy_accuracy.total_count == 5

    # SELL: 2 profitable out of 3 total = 66.67%
    assert sell_accuracy.hit_rate == pytest.approx(0.6666, abs=0.01)
    assert sell_accuracy.total_count == 3


@pytest.mark.asyncio
async def test_calibration_curve_buckets(sample_signals, sample_analyses):
    """Test calibration curve bucketing."""
    service = SignalAnalyticsService()

    start = datetime.now(UTC) - timedelta(days=30)
    end = datetime.now(UTC)

    calibration = await service.get_calibration_curves(start, end, "5d")

    # Should have buckets with data
    assert len(calibration.buckets) > 0

    for bucket in calibration.buckets:
        assert bucket.sample_count > 0
        assert 0.0 <= bucket.actual_accuracy <= 1.0
        assert 0.0 <= bucket.expected_confidence <= 1.0


@pytest.mark.asyncio
async def test_timing_analysis(sample_signals, sample_analyses):
    """Test timing analysis calculation."""
    service = SignalAnalyticsService()

    start = datetime.now(UTC) - timedelta(days=30)
    end = datetime.now(UTC)

    timing = await service.get_timing_analysis(start, end)

    # Should have average delay
    assert timing.avg_execution_delay_hours >= 0

    # Should have by-bucket data
    assert len(timing.by_confidence_bucket) > 0


@pytest.mark.asyncio
async def test_execution_rate_by_confidence(sample_signals, sample_analyses):
    """Test execution rate by confidence bucket."""
    service = SignalAnalyticsService()

    start = datetime.now(UTC) - timedelta(days=30)
    end = datetime.now(UTC)

    rates = await service.get_execution_rate_by_confidence(start, end)

    # Should have buckets
    assert len(rates) > 0

    for rate in rates:
        assert 0.0 <= rate.execution_rate <= 1.0
        assert rate.total_count >= rate.executed_count
