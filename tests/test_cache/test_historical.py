from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.cache.historical import FUNDAMENTALS_TTL_DAYS, HistoricalCache


@pytest.fixture
def cache(tmp_path):
    db_path = str(tmp_path / "test.db")
    c = HistoricalCache(db_path=db_path)
    yield c
    c.close()


@pytest.fixture
def sample_ohlcv_df():
    dates = pd.date_range("2025-01-01", periods=5, freq="B")
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "High": [105.0, 106.0, 107.0, 108.0, 109.0],
            "Low": [95.0, 96.0, 97.0, 98.0, 99.0],
            "Close": [102.0, 103.0, 104.0, 105.0, 106.0],
            "Volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
        },
        index=dates,
    )


class TestTableCreation:
    def test_tables_created(self, cache):
        tables = cache._conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = {t[0] for t in tables}
        assert "ohlcv_daily" in table_names
        assert "news_articles" in table_names
        assert "fundamentals" in table_names
        assert "order_fills" in table_names
        assert "truth_social_posts" in table_names
        assert "reddit_posts" in table_names

    def test_idempotent_creation(self, tmp_path):
        # Smoke test: verifies no exception when creating cache multiple times
        db_path = str(tmp_path / "test.db")
        c1 = HistoricalCache(db_path=db_path)
        c1.close()
        c2 = HistoricalCache(db_path=db_path)
        assert c2 is not None
        c2.close()


class TestOHLCV:
    def test_store_and_retrieve(self, cache, sample_ohlcv_df):
        inserted = cache.store_ohlcv("AAPL", sample_ohlcv_df)
        assert inserted == 5

        df = cache.get_ohlcv("AAPL")
        assert len(df) == 5
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert df.index.name == "Date"

    def test_dedup(self, cache, sample_ohlcv_df):
        cache.store_ohlcv("AAPL", sample_ohlcv_df)
        inserted = cache.store_ohlcv("AAPL", sample_ohlcv_df)
        assert inserted == 0
        assert len(cache.get_ohlcv("AAPL")) == 5

    def test_last_date(self, cache, sample_ohlcv_df):
        assert cache.get_last_ohlcv_date("AAPL") is None
        cache.store_ohlcv("AAPL", sample_ohlcv_df)
        last = cache.get_last_ohlcv_date("AAPL")
        assert last is not None
        assert last.year == 2025

    def test_count(self, cache, sample_ohlcv_df):
        assert cache.get_ohlcv_count("AAPL") == 0
        cache.store_ohlcv("AAPL", sample_ohlcv_df)
        assert cache.get_ohlcv_count("AAPL") == 5

    def test_empty_cache(self, cache):
        df = cache.get_ohlcv("MISSING")
        assert df.empty

    def test_empty_dataframe_noop(self, cache):
        assert cache.store_ohlcv("AAPL", pd.DataFrame()) == 0

    def test_symbols_isolated(self, cache, sample_ohlcv_df):
        cache.store_ohlcv("AAPL", sample_ohlcv_df)
        assert cache.get_ohlcv("TSLA").empty


class TestNews:
    def _make_article(self, url="https://example.com/1", title="Test"):
        article = MagicMock()
        article.url = url
        article.title = title
        article.description = "desc"
        article.published_at = datetime(2025, 1, 1, tzinfo=UTC)
        article.source = "test"
        return article

    def test_store_and_retrieve_urls(self, cache):
        articles = [self._make_article("https://a.com/1"), self._make_article("https://a.com/2")]
        inserted = cache.store_news_articles("AAPL", articles)
        assert inserted == 2

        urls = cache.get_cached_urls("AAPL")
        assert urls == {"https://a.com/1", "https://a.com/2"}

    def test_dedup_by_url(self, cache):
        article = self._make_article("https://a.com/1")
        cache.store_news_articles("AAPL", [article])
        inserted = cache.store_news_articles("AAPL", [article])
        assert inserted == 0

    def test_empty_list(self, cache):
        assert cache.store_news_articles("AAPL", []) == 0

    def test_empty_cache(self, cache):
        assert cache.get_cached_urls("MISSING") == set()


class TestFundamentals:
    def test_store_and_retrieve(self, cache):
        data = {"Symbol": "AAPL", "PE": 25.0}
        cache.store_fundamentals("AAPL", data)
        result = cache.get_fundamentals("AAPL")
        assert result == data

    def test_ttl_expiry(self, cache):
        data = {"Symbol": "AAPL", "PE": 25.0}
        cache.store_fundamentals("AAPL", data)

        # Manually set fetched_at to expired
        expired = (datetime.now() - timedelta(days=FUNDAMENTALS_TTL_DAYS + 1)).isoformat()
        cache._conn.execute(
            "UPDATE fundamentals SET fetched_at = ? WHERE symbol = ?",
            (expired, "AAPL"),
        )
        cache._conn.commit()

        assert cache.get_fundamentals("AAPL") is None

    def test_upsert(self, cache):
        cache.store_fundamentals("AAPL", {"PE": 25.0})
        cache.store_fundamentals("AAPL", {"PE": 30.0})
        assert cache.get_fundamentals("AAPL") == {"PE": 30.0}

    def test_missing(self, cache):
        assert cache.get_fundamentals("MISSING") is None


class TestOrderFills:
    def _make_order(self, order_id="ord-1"):
        order = MagicMock()
        order.order_id = order_id
        order.symbol = "AAPL"
        order.qty = 10.0
        order.filled_qty = 10.0
        order.side = "buy"
        order.status = "filled"
        order.submitted_at = datetime(2025, 1, 1, tzinfo=UTC)
        order.filled_at = datetime(2025, 1, 1, tzinfo=UTC)
        order.filled_avg_price = 150.0
        return order

    def test_store(self, cache):
        cache.store_order_fill(self._make_order())
        stats = cache.stats()
        assert stats["order_fills"] == 1

    def test_append_only(self, cache):
        cache.store_order_fill(self._make_order("ord-1"))
        cache.store_order_fill(self._make_order("ord-2"))
        assert cache.stats()["order_fills"] == 2

    def test_dedup(self, cache):
        cache.store_order_fill(self._make_order("ord-1"))
        cache.store_order_fill(self._make_order("ord-1"))
        assert cache.stats()["order_fills"] == 1


class TestTruthSocial:
    def _make_post(self, post_id="post-1"):
        post = MagicMock()
        post.id = post_id
        post.content = "test content"
        post.created_at = datetime(2025, 1, 1, tzinfo=UTC)
        post.likes = 100
        post.reposts = 50
        post.replies = 25
        post.url = f"https://truthsocial.com/posts/{post_id}"
        return post

    def test_store_and_get_ids(self, cache):
        posts = [self._make_post("p1"), self._make_post("p2")]
        inserted = cache.store_truth_social_posts(posts)
        assert inserted == 2

        ids = cache.get_cached_post_ids()
        assert ids == {"p1", "p2"}

    def test_dedup(self, cache):
        post = self._make_post("p1")
        cache.store_truth_social_posts([post])
        inserted = cache.store_truth_social_posts([post])
        assert inserted == 0

    def test_empty(self, cache):
        assert cache.get_cached_post_ids() == set()
        assert cache.store_truth_social_posts([]) == 0


class TestReddit:
    def _make_post(self, post_id="r1"):
        post = MagicMock()
        post.id = post_id
        post.title = "test"
        post.body = "body"
        post.subreddit = "wallstreetbets"
        post.score = 100
        post.upvote_ratio = 0.95
        post.url = f"https://reddit.com/r/wallstreetbets/{post_id}"
        post.created_utc = datetime(2025, 1, 1, tzinfo=UTC)
        post.num_comments = 50
        return post

    def test_store_and_get_ids(self, cache):
        posts = [self._make_post("r1"), self._make_post("r2")]
        inserted = cache.store_reddit_posts("AAPL", posts)
        assert inserted == 2

        ids = cache.get_cached_reddit_ids("AAPL")
        assert ids == {"r1", "r2"}

    def test_dedup(self, cache):
        post = self._make_post("r1")
        cache.store_reddit_posts("AAPL", [post])
        inserted = cache.store_reddit_posts("AAPL", [post])
        assert inserted == 0

    def test_symbol_isolation(self, cache):
        cache.store_reddit_posts("AAPL", [self._make_post("r1")])
        assert cache.get_cached_reddit_ids("TSLA") == set()

    def test_empty(self, cache):
        assert cache.get_cached_reddit_ids("MISSING") == set()
        assert cache.store_reddit_posts("AAPL", []) == 0


class TestStats:
    def test_empty_stats(self, cache):
        stats = cache.stats()
        assert all(v == 0 for v in stats.values())
        assert "ohlcv_daily" in stats
        assert "news_articles" in stats
        assert "fundamentals" in stats
        assert "order_fills" in stats
        assert "truth_social_posts" in stats
        assert "reddit_posts" in stats

    def test_stats_after_inserts(self, cache, sample_ohlcv_df):
        cache.store_ohlcv("AAPL", sample_ohlcv_df)
        stats = cache.stats()
        assert stats["ohlcv_daily"] == 5


class TestSignalOutcomes:
    def test_record_and_retrieve(self, cache):
        now = datetime.now(UTC)

        cache.record_signal_outcome(
            symbol="AAPL",
            timestamp=now,
            signal="BUY",
            confidence=0.85,
            price_at_signal=150.0,
            strategy_used="momentum",
            regime="BULL",
            trading_session="REGULAR",
            technical_signal="BUY",
            sentiment_signal="POSITIVE",
            news_signal="BULLISH",
        )

        signals = cache.get_signal_outcomes(window="all")
        assert len(signals) == 1
        assert signals[0]["symbol"] == "AAPL"
        assert signals[0]["signal"] == "BUY"
        assert signals[0]["confidence"] == 0.85
        assert signals[0]["price_at_signal"] == 150.0
        assert signals[0]["strategy_used"] == "momentum"
        assert signals[0]["regime"] == "BULL"

    def test_unique_constraint_symbol_timestamp(self, cache):
        now = datetime.now(UTC)

        cache.record_signal_outcome(
            symbol="AAPL",
            timestamp=now,
            signal="BUY",
            confidence=0.85,
            price_at_signal=150.0,
        )

        # Should ignore duplicate, not replace
        cache.record_signal_outcome(
            symbol="AAPL",
            timestamp=now,
            signal="SELL",
            confidence=0.90,
            price_at_signal=151.0,
        )

        signals = cache.get_signal_outcomes(window="all")
        assert len(signals) == 1
        assert signals[0]["signal"] == "BUY"
        assert signals[0]["confidence"] == 0.85

    def test_get_signals_needing_update_1d_5d_20d(self, cache):
        now = datetime.now(UTC)

        # Recent signal (should not appear in any horizon)
        cache.record_signal_outcome(
            symbol="AAPL",
            timestamp=now,
            signal="BUY",
            confidence=0.85,
            price_at_signal=150.0,
        )

        # Old signals (should appear based on horizon)
        from pandas.tseries.offsets import BDay

        old_2d = now - BDay(2)
        old_6d = now - BDay(6)
        old_21d = now - BDay(21)

        cache.record_signal_outcome(
            symbol="TSLA",
            timestamp=old_2d,
            signal="BUY",
            confidence=0.80,
            price_at_signal=200.0,
        )

        cache.record_signal_outcome(
            symbol="MSFT",
            timestamp=old_6d,
            signal="SELL",
            confidence=0.75,
            price_at_signal=300.0,
        )

        cache.record_signal_outcome(
            symbol="GOOGL",
            timestamp=old_21d,
            signal="BUY",
            confidence=0.70,
            price_at_signal=100.0,
        )

        # Check 1d horizon (should get 2d, 6d, 21d old)
        signals_1d = cache.get_signals_needing_update("1d")
        assert len(signals_1d) == 3
        symbols_1d = {s["symbol"] for s in signals_1d}
        assert symbols_1d == {"TSLA", "MSFT", "GOOGL"}

        # Check 5d horizon (should get 6d, 21d old)
        signals_5d = cache.get_signals_needing_update("5d")
        assert len(signals_5d) == 2
        symbols_5d = {s["symbol"] for s in signals_5d}
        assert symbols_5d == {"MSFT", "GOOGL"}

        # Check 20d horizon (should get only 21d old)
        signals_20d = cache.get_signals_needing_update("20d")
        assert len(signals_20d) == 1
        assert signals_20d[0]["symbol"] == "GOOGL"

    def test_update_signal_outcome(self, cache):
        now = datetime.now(UTC)

        cache.record_signal_outcome(
            symbol="AAPL",
            timestamp=now,
            signal="BUY",
            confidence=0.85,
            price_at_signal=150.0,
        )

        signals = cache.get_signal_outcomes(window="all")
        signal_id = signals[0]["id"]

        # Update with 1d price
        cache.update_signal_outcome(
            signal_id,
            price_at_1d=155.0,
            outcome_updated_at=now.isoformat(),
        )

        # Update with 5d and 20d prices
        cache.update_signal_outcome(
            signal_id,
            price_at_5d=160.0,
            price_at_20d=170.0,
            outcome_updated_at=now.isoformat(),
        )

        # Update with early exit
        cache.update_signal_outcome(
            signal_id,
            actual_exit_price=158.0,
            actual_exit_date=now.isoformat(),
            outcome_updated_at=now.isoformat(),
        )

        signals = cache.get_signal_outcomes(window="all")
        assert signals[0]["price_at_1d"] == 155.0
        assert signals[0]["price_at_5d"] == 160.0
        assert signals[0]["price_at_20d"] == 170.0
        assert signals[0]["actual_exit_price"] == 158.0
        assert signals[0]["actual_exit_date"] is not None

    def test_update_signal_outcome_rejects_invalid_fields(self, cache):
        now = datetime.now(UTC)

        cache.record_signal_outcome(
            symbol="AAPL",
            timestamp=now,
            signal="BUY",
            confidence=0.85,
            price_at_signal=150.0,
        )

        signals = cache.get_signal_outcomes(window="all")
        signal_id = signals[0]["id"]

        # Attempt to update with invalid field name (SQL injection attempt)
        with pytest.raises(ValueError, match=r"Invalid signal outcome fields: \{'malicious_field'\}"):
            cache.update_signal_outcome(signal_id, malicious_field="DROP TABLE signal_outcomes")

        # Attempt with multiple invalid fields
        with pytest.raises(ValueError, match="Invalid signal outcome fields:"):
            cache.update_signal_outcome(
                signal_id,
                price_at_1d=155.0,
                invalid_field="value",
                another_bad_field=123,
            )

        # Valid fields should still work
        cache.update_signal_outcome(signal_id, price_at_1d=155.0, outcome_updated_at=now.isoformat())
        signals = cache.get_signal_outcomes(window="all")
        assert signals[0]["price_at_1d"] == 155.0

    def test_get_signal_outcomes_window_filtering(self, cache):
        now = datetime.now(UTC)

        # Create signals at different ages
        cache.record_signal_outcome(
            symbol="AAPL",
            timestamp=now - timedelta(days=5),
            signal="BUY",
            confidence=0.85,
            price_at_signal=150.0,
        )

        cache.record_signal_outcome(
            symbol="TSLA",
            timestamp=now - timedelta(days=25),
            signal="SELL",
            confidence=0.80,
            price_at_signal=200.0,
        )

        cache.record_signal_outcome(
            symbol="MSFT",
            timestamp=now - timedelta(days=80),
            signal="BUY",
            confidence=0.75,
            price_at_signal=300.0,
        )

        # Test 7d window
        signals_7d = cache.get_signal_outcomes(window="7d")
        assert len(signals_7d) == 1
        assert signals_7d[0]["symbol"] == "AAPL"

        # Test 30d window
        signals_30d = cache.get_signal_outcomes(window="30d")
        assert len(signals_30d) == 2
        symbols_30d = {s["symbol"] for s in signals_30d}
        assert symbols_30d == {"AAPL", "TSLA"}

        # Test 90d window
        signals_90d = cache.get_signal_outcomes(window="90d")
        assert len(signals_90d) == 3

        # Test all window
        signals_all = cache.get_signal_outcomes(window="all")
        assert len(signals_all) == 3
