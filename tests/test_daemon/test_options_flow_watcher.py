"""Unit tests for OptionsFlowWatcher."""

import asyncio
from datetime import date
from unittest.mock import MagicMock

import pytest

from src.daemon.events import OptionsFlowDirection
from src.data.options_flow import OptionContract, OptionsChainSnapshot, OptionsFlowFetcher
from src.v1.watchers.options_flow_watcher import (
    OptionsFlowWatcher,
    OptionsFlowWatcherConfig,
)


def _contract(
    strike: float = 180.0,
    volume: int = 100,
    last_price: float = 5.0,
    bid: float = 4.8,
    ask: float = 5.2,
    option_type: str = "call",
    in_the_money: bool = False,
) -> OptionContract:
    """Helper to create OptionContract."""
    return OptionContract(
        strike=strike,
        last_price=last_price,
        bid=bid,
        ask=ask,
        volume=volume,
        open_interest=volume * 10,
        implied_volatility=0.3,
        in_the_money=in_the_money,
        expiry=date(2026, 3, 21),
        option_type=option_type,
    )


def _snapshot(
    symbol: str = "AAPL",
    calls: list[OptionContract] | None = None,
    puts: list[OptionContract] | None = None,
) -> OptionsChainSnapshot:
    """Helper to create OptionsChainSnapshot."""
    c = calls or []
    p = puts or []
    return OptionsChainSnapshot(
        symbol=symbol,
        total_call_volume=sum(x.volume for x in c),
        total_put_volume=sum(x.volume for x in p),
        total_call_oi=sum(x.open_interest for x in c),
        total_put_oi=sum(x.open_interest for x in p),
        near_term_expiry=date(2026, 3, 21),
        calls=c,
        puts=p,
    )


@pytest.fixture
def mock_fetcher() -> MagicMock:
    """Mock OptionsFlowFetcher."""
    return MagicMock(spec=OptionsFlowFetcher)


@pytest.fixture
def config() -> OptionsFlowWatcherConfig:
    """Default watcher config."""
    return OptionsFlowWatcherConfig(
        poll_interval_minutes=15,
        volume_spike_threshold=2.0,
        block_trade_threshold=100_000,
        symbols=["AAPL", "TSLA"],
    )


@pytest.fixture
def watcher(mock_fetcher: MagicMock, config: OptionsFlowWatcherConfig) -> OptionsFlowWatcher:
    """Create OptionsFlowWatcher with mocked fetcher."""
    return OptionsFlowWatcher(fetcher=mock_fetcher, config=config)


@pytest.mark.unit
def test_init(watcher: OptionsFlowWatcher) -> None:
    """get_signal returns None on init."""
    assert watcher.get_signal("AAPL") is None
    assert watcher.running is False


@pytest.mark.unit
def test_put_call_ratio_balanced(watcher: OptionsFlowWatcher) -> None:
    """P/C ratio with equal volume = 1.0."""
    snap = _snapshot(
        calls=[_contract(volume=100)],
        puts=[_contract(volume=100, option_type="put")],
    )
    assert watcher._compute_put_call_ratio(snap) == 1.0


@pytest.mark.unit
def test_put_call_ratio_bullish(watcher: OptionsFlowWatcher) -> None:
    """P/C ratio < 1 when call volume dominates."""
    snap = _snapshot(
        calls=[_contract(volume=200)],
        puts=[_contract(volume=50, option_type="put")],
    )
    assert watcher._compute_put_call_ratio(snap) == 0.25


@pytest.mark.unit
def test_put_call_ratio_zero_calls(watcher: OptionsFlowWatcher) -> None:
    """P/C ratio capped at 10.0 when no calls."""
    snap = _snapshot(
        calls=[],
        puts=[_contract(volume=100, option_type="put")],
    )
    assert watcher._compute_put_call_ratio(snap) == 10.0


@pytest.mark.unit
def test_put_call_ratio_zero_both(watcher: OptionsFlowWatcher) -> None:
    """P/C ratio 0.0 when no volume at all."""
    snap = _snapshot()
    assert watcher._compute_put_call_ratio(snap) == 0.0


@pytest.mark.unit
def test_block_trade_detection(watcher: OptionsFlowWatcher) -> None:
    """Detect contracts with premium > threshold."""
    # vol=200, midpoint=5.0, premium = 200*5*100 = 100,000 → at threshold
    big = _contract(volume=200, bid=4.8, ask=5.2)
    # vol=10, premium = 10*5*100 = 5,000 → below threshold
    small = _contract(strike=190, volume=10, bid=0.9, ask=1.1)

    snap = _snapshot(calls=[big, small])
    blocks = watcher._detect_block_trades(snap)

    assert len(blocks) == 1
    assert blocks[0].strike == 180.0
    assert blocks[0].premium >= 100_000


@pytest.mark.unit
def test_block_trade_lastprice_fallback(watcher: OptionsFlowWatcher) -> None:
    """Use lastPrice when bid/ask are 0 (after hours)."""
    # bid=0, ask=0 → midpoint=0, fallback to lastPrice=5.0
    # premium = 300 * 5.0 * 100 = 150,000
    contract = _contract(volume=300, bid=0, ask=0, last_price=5.0)
    snap = _snapshot(calls=[contract])
    blocks = watcher._detect_block_trades(snap)

    assert len(blocks) == 1
    assert blocks[0].premium == 150_000


@pytest.mark.unit
def test_volume_spike_first_observation(watcher: OptionsFlowWatcher) -> None:
    """First observation returns 1.0 (no history)."""
    spike = watcher._compute_volume_spike("AAPL", 1000)
    assert spike == 1.0


@pytest.mark.unit
def test_volume_spike_calculation(watcher: OptionsFlowWatcher) -> None:
    """Volume spike calculated vs rolling average."""
    # Build history: 3 sessions at 1000
    for _ in range(3):
        watcher._compute_volume_spike("AAPL", 1000)

    # Now 3000 volume → 3x spike
    spike = watcher._compute_volume_spike("AAPL", 3000)
    assert spike == pytest.approx(3.0, rel=0.1)


@pytest.mark.unit
def test_direction_bullish(watcher: OptionsFlowWatcher) -> None:
    """Direction BULLISH when call premium dominates."""
    snap = _snapshot(
        calls=[_contract(volume=500, last_price=5.0, bid=4.8, ask=5.2)],
        puts=[_contract(volume=50, last_price=2.0, bid=1.8, ask=2.2, option_type="put")],
    )
    direction = watcher._determine_direction(snap)
    assert direction == OptionsFlowDirection.BULLISH


@pytest.mark.unit
def test_direction_bearish(watcher: OptionsFlowWatcher) -> None:
    """Direction BEARISH when put premium dominates."""
    snap = _snapshot(
        calls=[_contract(volume=50, last_price=1.0, bid=0.8, ask=1.2)],
        puts=[_contract(volume=500, last_price=5.0, bid=4.8, ask=5.2, option_type="put")],
    )
    direction = watcher._determine_direction(snap)
    assert direction == OptionsFlowDirection.BEARISH


@pytest.mark.unit
def test_direction_neutral(watcher: OptionsFlowWatcher) -> None:
    """Direction NEUTRAL when premiums are balanced."""
    snap = _snapshot(
        calls=[_contract(volume=100, last_price=5.0, bid=4.8, ask=5.2)],
        puts=[_contract(volume=100, last_price=5.0, bid=4.8, ask=5.2, option_type="put")],
    )
    direction = watcher._determine_direction(snap)
    assert direction == OptionsFlowDirection.NEUTRAL


@pytest.mark.unit
def test_direction_empty(watcher: OptionsFlowWatcher) -> None:
    """Direction NEUTRAL when no data."""
    snap = _snapshot()
    direction = watcher._determine_direction(snap)
    assert direction == OptionsFlowDirection.NEUTRAL


@pytest.mark.unit
def test_significance_zero(watcher: OptionsFlowWatcher) -> None:
    """Significance 0.0 when no unusual activity."""
    score = watcher._compute_significance(spike=1.0, blocks=[], pcr=1.0)
    assert score == 0.0


@pytest.mark.unit
def test_significance_high(watcher: OptionsFlowWatcher) -> None:
    """Significance near 1.0 with extreme values."""
    from src.daemon.events import BlockTrade

    blocks = [
        BlockTrade(
            strike=180, expiry="2026-03-21", premium=200_000, volume=400, option_type="call", is_itm=True
        ),
        BlockTrade(
            strike=170, expiry="2026-03-21", premium=150_000, volume=300, option_type="call", is_itm=True
        ),
        BlockTrade(
            strike=190, expiry="2026-03-21", premium=120_000, volume=250, option_type="put", is_itm=False
        ),
    ]
    score = watcher._compute_significance(spike=4.0, blocks=blocks, pcr=2.5)
    assert score > 0.8


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fetch_and_assess_symbol(watcher: OptionsFlowWatcher, mock_fetcher: MagicMock) -> None:
    """_fetch_and_assess_symbol updates signal for symbol."""
    snap = _snapshot(
        symbol="AAPL",
        calls=[_contract(volume=500)],
        puts=[_contract(volume=200, option_type="put")],
    )
    mock_fetcher.fetch_options_chain.return_value = snap

    await watcher._fetch_and_assess_symbol("AAPL")

    signal = watcher.get_signal("AAPL")
    assert signal is not None
    assert signal.symbol == "AAPL"
    assert signal.put_call_ratio == pytest.approx(0.4)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fetch_and_assess_empty_snapshot(watcher: OptionsFlowWatcher, mock_fetcher: MagicMock) -> None:
    """Empty snapshot doesn't create a signal."""
    mock_fetcher.fetch_options_chain.return_value = OptionsChainSnapshot(symbol="AAPL")

    await watcher._fetch_and_assess_symbol("AAPL")

    assert watcher.get_signal("AAPL") is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fetch_and_assess_all(watcher: OptionsFlowWatcher, mock_fetcher: MagicMock) -> None:
    """_fetch_and_assess_all processes all configured symbols."""
    snap_aapl = _snapshot(
        symbol="AAPL",
        calls=[_contract(volume=100)],
        puts=[_contract(volume=50, option_type="put")],
    )
    snap_tsla = _snapshot(
        symbol="TSLA",
        calls=[_contract(volume=200)],
        puts=[_contract(volume=300, option_type="put")],
    )

    def side_effect(symbol: str) -> OptionsChainSnapshot:
        return snap_aapl if symbol == "AAPL" else snap_tsla

    mock_fetcher.fetch_options_chain.side_effect = side_effect

    await watcher._tick()

    assert watcher.get_signal("AAPL") is not None
    assert watcher.get_signal("TSLA") is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_updates_signals(watcher: OptionsFlowWatcher, mock_fetcher: MagicMock) -> None:
    """run() executes one cycle and sets signals."""
    snap = _snapshot(
        symbol="AAPL",
        calls=[_contract(volume=100)],
        puts=[_contract(volume=50, option_type="put")],
    )
    mock_fetcher.fetch_options_chain.return_value = snap

    async def stop_after_first_cycle() -> None:
        await asyncio.sleep(0.05)
        watcher.running = False

    await asyncio.gather(
        watcher.run(),
        stop_after_first_cycle(),
    )

    assert watcher.get_signal("AAPL") is not None
    mock_fetcher.fetch_options_chain.assert_called()
