"""Tests for Trump watcher daemon."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.trump import TrumpAnalysis
from src.daemon.trump_watcher import TrumpSignal, TrumpWatcher
from src.data.truth_social import TrumpPostData, TruthPost
from src.strategies.momentum import Signal


@pytest.fixture
def mock_trump_analysis():
    return TrumpAnalysis(
        market_relevant=True,
        signal=Signal.BUY,
        mentioned_tickers=["TSLA"],
        sentiment="positive",
        confidence=0.75,
        key_phrases=["great time to buy"],
        interpretation="Bullish sentiment detected",
        post_count=1,
    )


def test_trump_watcher_init():
    watcher = TrumpWatcher(poll_interval=30, max_analyses=3)

    assert watcher.poll_interval == 30
    assert watcher.max_analyses == 3
    assert watcher.running is False


def test_trump_watcher_repr():
    watcher = TrumpWatcher(poll_interval=60)
    repr_str = repr(watcher)

    assert "TrumpWatcher" in repr_str
    assert "60s" in repr_str


def test_sector_stocks_mapping():
    assert "tariff" in TrumpWatcher.SECTOR_STOCKS
    assert "CAT" in TrumpWatcher.SECTOR_STOCKS["tariff"]

    assert "crypto" in TrumpWatcher.SECTOR_STOCKS
    assert "COIN" in TrumpWatcher.SECTOR_STOCKS["crypto"]


@pytest.mark.asyncio
async def test_check_new_posts(sample_trump_posts):
    watcher = TrumpWatcher()

    with patch.object(watcher, "_init_components"):
        watcher._fetcher = MagicMock()
        watcher._fetcher.fetch_recent.return_value = TrumpPostData(
            posts=sample_trump_posts,
            total_count=3,
            latest_post_at=sample_trump_posts[0].created_at,
            fetched_at=datetime.now(UTC),
        )

        new_posts = await watcher._check_new_posts()

        assert len(new_posts) == 3
        assert watcher._last_post_id == sample_trump_posts[0].id


@pytest.mark.asyncio
async def test_identify_affected_stocks(sample_trump_posts):
    watcher = TrumpWatcher()

    with patch.object(watcher, "_init_components"):
        watcher._llm = MagicMock()
        watcher._llm.acomplete = AsyncMock(return_value="TSLA, AAPL")

        affected = await watcher._identify_affected_stocks(sample_trump_posts)

        # Should find TSLA from direct mention and AAPL from "Apple" mention
        assert "TSLA" in affected or "AAPL" in affected


@pytest.mark.asyncio
async def test_llm_identify_stocks():
    watcher = TrumpWatcher()

    watcher._llm = MagicMock()
    watcher._llm.acomplete = AsyncMock(return_value="TSLA, AAPL, BA")

    posts = [
        TruthPost(
            id="1",
            content="General market commentary",
            created_at=datetime.now(UTC),
            likes=1000,
            reposts=100,
            replies=50,
            url="https://example.com/1",
        )
    ]

    tickers = await watcher._llm_identify_stocks(posts)

    assert "TSLA" in tickers
    assert "AAPL" in tickers
    assert "BA" in tickers


@pytest.mark.asyncio
async def test_llm_identify_stocks_none():
    watcher = TrumpWatcher()

    watcher._llm = MagicMock()
    watcher._llm.acomplete = AsyncMock(return_value="NONE")

    posts = [
        TruthPost(
            id="1",
            content="Had a great golf game",
            created_at=datetime.now(UTC),
            likes=1000,
            reposts=100,
            replies=50,
            url="https://example.com/1",
        )
    ]

    tickers = await watcher._llm_identify_stocks(posts)

    assert len(tickers) == 0


def test_trump_signal_creation(sample_trump_posts, mock_trump_analysis):
    signal = TrumpSignal(
        post=sample_trump_posts[0],
        affected_symbols=["TSLA", "AAPL"],
        trump_analysis=mock_trump_analysis,
        analyses={},
        timestamp=datetime.now(UTC),
    )

    assert signal.post.id == sample_trump_posts[0].id
    assert "TSLA" in signal.affected_symbols
    assert signal.trump_analysis.market_relevant is True
