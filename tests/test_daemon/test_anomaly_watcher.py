"""Tests for anomaly watcher."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.cache.historical import HistoricalCache
from src.daemon.events import AnomalyEvent, Gap, PriceMove, VolumeSpike
from src.daemon.watchers.anomaly_watcher import AnomalyWatcher
from src.data.market import MarketData


@pytest.fixture
def historical_cache(tmp_path):
    """Create test historical cache."""
    return HistoricalCache(db_path=str(tmp_path / "cache" / "historical.db"))


@pytest.fixture
def mock_market_fetcher():
    """Mock market fetcher."""
    return MagicMock()


@pytest.fixture
def anomaly_watcher(historical_cache):
    """Create anomaly watcher with test config."""
    return AnomalyWatcher(
        historical_cache=historical_cache,
        poll_interval=900,
        relevance_threshold=0.7,
        cooldown_minutes=15,
        volume_spike_multiplier=2.0,
        price_move_threshold_pct=5.0,
        gap_threshold_pct=3.0,
        watchlist=["AAPL", "TSLA", "NVDA", "AMD", "GOOGL"],
        max_symbols_per_cycle=2,
        max_concurrent_analyses=2,
    )


def create_intraday_data(
    open_price: float = 100.0,
    close_price: float = 100.0,
    high: float = 100.0,
    low: float = 100.0,
    volume: float = 1000000.0,
) -> MarketData:
    """Create test intraday data."""
    now = datetime.now(UTC)
    df = pd.DataFrame(
        {
            "Open": [open_price],
            "High": [high],
            "Low": [low],
            "Close": [close_price],
            "Volume": [volume],
        },
        index=[now],
    )
    return MarketData(symbol="TEST", data=df, last_updated=now)


def create_daily_data(
    close_prices: list[float],
    volumes: list[float],
) -> MarketData:
    """Create test daily data with multiple days."""
    now = datetime.now(UTC)
    dates = [now - timedelta(days=i) for i in range(len(close_prices) - 1, -1, -1)]

    df = pd.DataFrame(
        {
            "Open": close_prices,
            "High": close_prices,
            "Low": close_prices,
            "Close": close_prices,
            "Volume": volumes,
        },
        index=dates,
    )
    return MarketData(symbol="TEST", data=df, last_updated=now)


@pytest.mark.asyncio
async def test_volume_spike_detection(anomaly_watcher, mock_market_fetcher):
    """Test volume spike detection (2-poll sequence: baseline → spike)."""
    with patch.object(anomaly_watcher, "_init_components"):
        anomaly_watcher._market_fetcher = mock_market_fetcher

        # Poll 1: Establish baseline (normal volume, no spike)
        mock_market_fetcher.fetch_intraday.return_value = create_intraday_data(
            volume=600000.0
        )  # 1.2x, below threshold
        mock_market_fetcher.fetch_daily.return_value = create_daily_data(
            close_prices=[100.0] * 30, volumes=[500000.0] * 30
        )

        events = await anomaly_watcher._fetch_events()
        assert len(events) == 0  # No spike yet, just baseline

        # Verify baseline established for both symbols
        assert "AAPL" in anomaly_watcher._volume_baselines
        assert "TSLA" in anomaly_watcher._volume_baselines
        assert anomaly_watcher._volume_baselines["AAPL"] == 500000.0

        # Poll 2: Volume spike (2x baseline)
        mock_market_fetcher.fetch_intraday.return_value = create_intraday_data(volume=1000000.0)

        events = await anomaly_watcher._fetch_events()
        assert len(events) >= 1  # At least one spike detected

        # Check first event
        event = events[0]
        assert isinstance(event, AnomalyEvent)
        assert "volume_spike" in event.anomaly_types
        assert event.volume_spike_data is not None
        assert event.volume_spike_data.spike_multiplier == 2.0


@pytest.mark.asyncio
async def test_price_move_detection(anomaly_watcher, mock_market_fetcher):
    """Test large intraday price move detection (>5%)."""
    with patch.object(anomaly_watcher, "_init_components"):
        anomaly_watcher._market_fetcher = mock_market_fetcher

        # 6% intraday move
        mock_market_fetcher.fetch_intraday.return_value = create_intraday_data(
            open_price=100.0, close_price=106.0, high=107.0, low=99.0
        )

        events = await anomaly_watcher._fetch_events()
        assert len(events) >= 1  # May return 2 events due to round-robin

        # Check first event
        event = events[0]
        assert "price_move" in event.anomaly_types
        assert event.price_move_data is not None
        assert event.price_move_data.change_pct == pytest.approx(6.0, abs=0.01)
        assert event.price_move_data.open_price == 100.0
        assert event.price_move_data.current_price == 106.0


@pytest.mark.asyncio
async def test_gap_up_detection(anomaly_watcher, mock_market_fetcher):
    """Test gap up detection (>3% from prev close)."""
    with patch.object(anomaly_watcher, "_init_components"):
        anomaly_watcher._market_fetcher = mock_market_fetcher

        # Gap up: prev close 100, open 105 (5% gap)
        mock_market_fetcher.fetch_intraday.return_value = create_intraday_data(open_price=105.0)
        mock_market_fetcher.fetch_daily.return_value = create_daily_data(
            close_prices=[100.0, 102.0],
            volumes=[500000.0, 500000.0],  # [-2] = 100.0 (prev close)
        )

        events = await anomaly_watcher._fetch_events()
        assert len(events) >= 1  # May return 2 events (AAPL + TSLA) due to round-robin

        # Check first event
        event = events[0]
        assert "gap" in event.anomaly_types
        assert event.gap_data is not None
        assert event.gap_data.gap_pct == pytest.approx(5.0, abs=0.01)
        assert event.gap_data.gap_direction == "up"
        assert event.gap_data.previous_close == 100.0
        assert event.gap_data.open_price == 105.0


@pytest.mark.asyncio
async def test_gap_down_detection(anomaly_watcher, mock_market_fetcher):
    """Test gap down detection (>3% from prev close)."""
    with patch.object(anomaly_watcher, "_init_components"):
        anomaly_watcher._market_fetcher = mock_market_fetcher

        # Gap down: prev close 100, open 95 (-5% gap)
        mock_market_fetcher.fetch_intraday.return_value = create_intraday_data(open_price=95.0)
        mock_market_fetcher.fetch_daily.return_value = create_daily_data(
            close_prices=[100.0, 102.0],
            volumes=[500000.0, 500000.0],  # [-2] = 100.0 (prev close)
        )

        events = await anomaly_watcher._fetch_events()
        assert len(events) >= 1  # May return 2 events due to round-robin

        # Check first event
        event = events[0]
        assert "gap" in event.anomaly_types
        assert event.gap_data is not None
        assert event.gap_data.gap_pct == pytest.approx(-5.0, abs=0.01)
        assert event.gap_data.gap_direction == "down"


@pytest.mark.asyncio
async def test_multiple_anomalies_single_event(anomaly_watcher, mock_market_fetcher):
    """Test multiple anomaly types detected in single event."""
    with patch.object(anomaly_watcher, "_init_components"):
        anomaly_watcher._market_fetcher = mock_market_fetcher

        # Establish volume baseline for both symbols
        anomaly_watcher._volume_baselines["AAPL"] = 500000.0
        anomaly_watcher._volume_baselines["TSLA"] = 500000.0
        anomaly_watcher._previous_close_cache["AAPL"] = 100.0
        anomaly_watcher._previous_close_cache["TSLA"] = 100.0

        # Volume spike (2x) + price move (6%) + gap (5%)
        mock_market_fetcher.fetch_intraday.return_value = create_intraday_data(
            open_price=105.0,  # 5% gap from 100
            close_price=111.3,  # 6% move from 105
            high=112.0,
            low=104.0,
            volume=1000000.0,  # 2x baseline
        )

        events = await anomaly_watcher._fetch_events()
        assert len(events) >= 1  # May return 2 events due to round-robin

        # Check first event has all 3 anomaly types
        event = events[0]
        assert len(event.anomaly_types) == 3
        assert "volume_spike" in event.anomaly_types
        assert "price_move" in event.anomaly_types
        assert "gap" in event.anomaly_types
        assert event.volume_spike_data is not None
        assert event.price_move_data is not None
        assert event.gap_data is not None


@pytest.mark.asyncio
async def test_round_robin_rotation(anomaly_watcher, mock_market_fetcher):
    """Test round-robin watchlist rotation."""
    anomaly_watcher._market_fetcher = mock_market_fetcher

    # Mock to always return normal data (no anomalies)
    mock_market_fetcher.fetch_intraday.return_value = create_intraday_data()

    # Poll 1: should check AAPL, TSLA (offset 0→2)
    symbols1 = anomaly_watcher._get_next_symbols()
    assert symbols1 == ["AAPL", "TSLA"]
    assert anomaly_watcher._rotation_offset == 2

    # Poll 2: should check NVDA, AMD (offset 2→4)
    symbols2 = anomaly_watcher._get_next_symbols()
    assert symbols2 == ["NVDA", "AMD"]
    assert anomaly_watcher._rotation_offset == 4

    # Poll 3: should check GOOGL, wrap to AAPL (offset 4→1)
    symbols3 = anomaly_watcher._get_next_symbols()
    assert symbols3 == ["GOOGL", "AAPL"]
    assert anomaly_watcher._rotation_offset == 1


@pytest.mark.asyncio
async def test_round_robin_wrap_around(anomaly_watcher, mock_market_fetcher):
    """Test round-robin wraps around watchlist."""
    anomaly_watcher._market_fetcher = mock_market_fetcher
    anomaly_watcher._rotation_offset = 4  # Start near end

    mock_market_fetcher.fetch_intraday.return_value = create_intraday_data()

    # Should get GOOGL (idx 4) + wrap to AAPL (idx 0)
    symbols = anomaly_watcher._get_next_symbols()
    assert symbols == ["GOOGL", "AAPL"]
    assert anomaly_watcher._rotation_offset == 1


@pytest.mark.asyncio
async def test_baseline_establishment(anomaly_watcher, mock_market_fetcher):
    """Test volume baseline establishment from daily data."""
    with patch.object(anomaly_watcher, "_init_components"):
        anomaly_watcher._market_fetcher = mock_market_fetcher

        # Return 30 days of data with avg volume 500k
        mock_market_fetcher.fetch_intraday.return_value = create_intraday_data()
        mock_market_fetcher.fetch_daily.return_value = create_daily_data(
            close_prices=[100.0] * 30, volumes=[500000.0] * 30
        )

        await anomaly_watcher._fetch_events()

        # Verify baseline established
        assert "AAPL" in anomaly_watcher._volume_baselines
        assert anomaly_watcher._volume_baselines["AAPL"] == 500000.0


@pytest.mark.asyncio
async def test_previous_close_cache_refresh(anomaly_watcher, mock_market_fetcher):
    """Test previous close cache clears on new day."""
    with patch.object(anomaly_watcher, "_init_components"):
        anomaly_watcher._market_fetcher = mock_market_fetcher

        # Set cache with old date
        old_date = datetime.now(UTC) - timedelta(days=2)
        anomaly_watcher._last_cache_refresh_date = old_date
        anomaly_watcher._previous_close_cache["AAPL"] = 100.0

        # Mock fetch to avoid errors
        mock_market_fetcher.fetch_intraday.return_value = create_intraday_data()

        # Run cycle (should clear cache)
        await anomaly_watcher._fetch_events()

        # Verify cache cleared
        assert len(anomaly_watcher._previous_close_cache) == 0


@pytest.mark.asyncio
async def test_lru_eviction(anomaly_watcher, mock_market_fetcher):
    """Test LRU eviction at 301st symbol."""
    anomaly_watcher._market_fetcher = mock_market_fetcher

    # Fill cache to 300 symbols
    for i in range(300):
        anomaly_watcher._update_volume_baseline(f"SYM{i}", 1000000.0)

    assert len(anomaly_watcher._volume_baselines) == 300

    # Add 301st symbol (should evict SYM0)
    anomaly_watcher._update_volume_baseline("SYM300", 1000000.0)

    assert len(anomaly_watcher._volume_baselines) == 300
    assert "SYM0" not in anomaly_watcher._volume_baselines
    assert "SYM300" in anomaly_watcher._volume_baselines


@pytest.mark.asyncio
async def test_empty_watchlist(anomaly_watcher, mock_market_fetcher):
    """Test empty watchlist returns no events."""
    with patch.object(anomaly_watcher, "_init_components"):
        anomaly_watcher._market_fetcher = mock_market_fetcher
        anomaly_watcher.watchlist = []

        events = await anomaly_watcher._fetch_events()
        assert len(events) == 0


@pytest.mark.asyncio
async def test_empty_intraday_data(anomaly_watcher, mock_market_fetcher):
    """Test handling of empty intraday data."""
    with patch.object(anomaly_watcher, "_init_components"):
        anomaly_watcher._market_fetcher = mock_market_fetcher

        # Return empty dataframe
        empty_data = MarketData(symbol="AAPL", data=pd.DataFrame(), last_updated=datetime.now(UTC))
        mock_market_fetcher.fetch_intraday.return_value = empty_data

        events = await anomaly_watcher._fetch_events()
        assert len(events) == 0


@pytest.mark.asyncio
async def test_api_failure_continues(anomaly_watcher, mock_market_fetcher):
    """Test API failure for one symbol doesn't stop processing."""
    with patch.object(anomaly_watcher, "_init_components"):
        anomaly_watcher._market_fetcher = mock_market_fetcher
        anomaly_watcher._volume_baselines["TSLA"] = 500000.0

        # AAPL fails, TSLA succeeds with volume spike
        def mock_fetch(symbol, interval):
            if symbol == "AAPL":
                msg = "API error"
                raise ValueError(msg)
            return create_intraday_data(volume=1000000.0)  # 2x spike

        mock_market_fetcher.fetch_intraday.side_effect = mock_fetch

        events = await anomaly_watcher._fetch_events()

        # Should get TSLA event despite AAPL failure
        assert len(events) == 1
        assert events[0].symbol == "TSLA"


@pytest.mark.asyncio
async def test_zero_volume_baseline_skipped(anomaly_watcher, mock_market_fetcher):
    """Test zero volume baseline doesn't cause division errors."""
    with patch.object(anomaly_watcher, "_init_components"):
        anomaly_watcher._market_fetcher = mock_market_fetcher
        anomaly_watcher._volume_baselines["AAPL"] = 0.0

        mock_market_fetcher.fetch_intraday.return_value = create_intraday_data(volume=1000000.0)

        events = await anomaly_watcher._fetch_events()

        # Should not detect volume spike (baseline is 0)
        for event in events:
            assert "volume_spike" not in event.anomaly_types


@pytest.mark.asyncio
async def test_zero_open_price_skipped(anomaly_watcher, mock_market_fetcher):
    """Test zero open price doesn't cause division errors."""
    with patch.object(anomaly_watcher, "_init_components"):
        anomaly_watcher._market_fetcher = mock_market_fetcher

        mock_market_fetcher.fetch_intraday.return_value = create_intraday_data(
            open_price=0.0, close_price=100.0
        )

        events = await anomaly_watcher._fetch_events()

        # Should not detect price move or gap (open is 0)
        for event in events:
            assert "price_move" not in event.anomaly_types
            assert "gap" not in event.anomaly_types


@pytest.mark.asyncio
async def test_zero_previous_close_skipped(anomaly_watcher, mock_market_fetcher):
    """Test zero previous close doesn't cause division errors."""
    with patch.object(anomaly_watcher, "_init_components"):
        anomaly_watcher._market_fetcher = mock_market_fetcher
        anomaly_watcher._previous_close_cache["AAPL"] = 0.0

        mock_market_fetcher.fetch_intraday.return_value = create_intraday_data(open_price=100.0)

        events = await anomaly_watcher._fetch_events()

        # Should not detect gap (prev close is 0)
        for event in events:
            assert "gap" not in event.anomaly_types


def test_volume_spike_repr():
    """Test VolumeSpike repr."""
    spike = VolumeSpike(current_volume=1000000.0, avg_volume_20d=500000.0, spike_multiplier=2.0)
    assert repr(spike) == "VolumeSpike(2.0x)"


def test_price_move_repr():
    """Test PriceMove repr."""
    move = PriceMove(open_price=100.0, current_price=106.0, change_pct=6.0, high=107.0, low=99.0)
    assert repr(move) == "PriceMove(+6.0%)"


def test_gap_repr():
    """Test Gap repr."""
    gap = Gap(previous_close=100.0, open_price=105.0, gap_pct=5.0, gap_direction="up")
    assert repr(gap) == "Gap(up 5.0%)"


def test_anomaly_event_repr():
    """Test AnomalyEvent repr."""
    event = AnomalyEvent(
        event_id="test",
        timestamp=datetime.now(UTC),
        symbol="AAPL",
        anomaly_types=["volume_spike", "price_move"],
    )
    assert "AAPL" in repr(event)
    assert "volume_spike+price_move" in repr(event)


def test_anomaly_event_to_prompt_text():
    """Test AnomalyEvent prompt text formatting."""
    event = AnomalyEvent(
        event_id="test",
        timestamp=datetime.now(UTC),
        symbol="AAPL",
        anomaly_types=["volume_spike", "price_move", "gap"],
        volume_spike_data=VolumeSpike(
            current_volume=1000000.0, avg_volume_20d=500000.0, spike_multiplier=2.0
        ),
        price_move_data=PriceMove(
            open_price=100.0, current_price=106.0, change_pct=6.0, high=107.0, low=99.0
        ),
        gap_data=Gap(previous_close=100.0, open_price=105.0, gap_pct=5.0, gap_direction="up"),
    )

    text = event.to_prompt_text()
    assert "MARKET ANOMALY" in text
    assert "AAPL" in text
    assert "volume_spike, price_move, gap" in text
    assert "1,000,000" in text
    assert "2.0x" in text
    assert "+6.0%" in text
    assert "up 5.0%" in text


def test_watcher_repr(anomaly_watcher):
    """Test AnomalyWatcher repr."""
    text = repr(anomaly_watcher)
    assert "AnomalyWatcher" in text
    assert "poll_interval=900s" in text
    assert "watchlist=5 symbols" in text
    assert "max_per_cycle=2" in text
