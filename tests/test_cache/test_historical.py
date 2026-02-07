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
        db_path = str(tmp_path / "test.db")
        c1 = HistoricalCache(db_path=db_path)
        c1.close()
        c2 = HistoricalCache(db_path=db_path)
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
