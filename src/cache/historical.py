"""SQLite-backed permanent cache for immutable historical data."""

import json
import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

# 90-day TTL for fundamentals
FUNDAMENTALS_TTL_DAYS = 90


class HistoricalCache:
    """Permanent SQLite cache for OHLCV, news, fundamentals, orders, and social posts."""

    def __init__(self, db_path: str | None = None) -> None:
        """Initialize historical cache.

        Args:
            db_path: Path to SQLite database file (defaults to ~/.ai-casino/cache/historical.db)
        """
        if db_path is None:
            db_dir = Path.home() / ".ai-casino" / "cache"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(db_dir / "historical.db")
        else:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()
        logger.info(f"HistoricalCache initialized (db={db_path})")

    def _create_tables(self) -> None:
        """Create tables if they don't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS ohlcv_daily (
                symbol TEXT NOT NULL,
                date    TEXT NOT NULL,
                open    REAL NOT NULL,
                high    REAL NOT NULL,
                low     REAL NOT NULL,
                close   REAL NOT NULL,
                volume  REAL NOT NULL,
                PRIMARY KEY (symbol, date)
            );
            CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol ON ohlcv_daily(symbol);

            CREATE TABLE IF NOT EXISTS news_articles (
                url          TEXT PRIMARY KEY,
                symbol       TEXT NOT NULL,
                title        TEXT NOT NULL,
                description  TEXT NOT NULL,
                published_at TEXT NOT NULL,
                source       TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_news_symbol ON news_articles(symbol);

            CREATE TABLE IF NOT EXISTS fundamentals (
                symbol     TEXT PRIMARY KEY,
                data       TEXT NOT NULL,
                fetched_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS order_fills (
                order_id     TEXT PRIMARY KEY,
                symbol       TEXT NOT NULL,
                qty          REAL NOT NULL,
                filled_qty   REAL NOT NULL,
                side         TEXT NOT NULL,
                status       TEXT NOT NULL,
                submitted_at TEXT NOT NULL,
                filled_at    TEXT,
                filled_avg_price REAL
            );
            CREATE INDEX IF NOT EXISTS idx_orders_symbol ON order_fills(symbol);

            CREATE TABLE IF NOT EXISTS truth_social_posts (
                id         TEXT PRIMARY KEY,
                content    TEXT NOT NULL,
                created_at TEXT NOT NULL,
                likes      INTEGER NOT NULL,
                reposts    INTEGER NOT NULL,
                replies    INTEGER NOT NULL,
                url        TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS reddit_posts (
                id            TEXT PRIMARY KEY,
                symbol        TEXT NOT NULL,
                title         TEXT NOT NULL,
                body          TEXT NOT NULL,
                subreddit     TEXT NOT NULL,
                score         INTEGER NOT NULL,
                upvote_ratio  REAL NOT NULL,
                url           TEXT NOT NULL,
                created_utc   TEXT NOT NULL,
                num_comments  INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_reddit_symbol ON reddit_posts(symbol);
        """)

    def get_ohlcv(self, symbol: str) -> pd.DataFrame:
        """Get all cached OHLCV rows for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            DataFrame with OHLCV data (empty if no cache)
        """
        rows = self._conn.execute(
            "SELECT date, open, high, low, close, volume FROM ohlcv_daily WHERE symbol = ? ORDER BY date",
            (symbol,),
        ).fetchall()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        df.index.name = "Date"
        return df

    def get_last_ohlcv_date(self, symbol: str) -> date | None:
        """Get the most recent cached OHLCV date for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Most recent date or None
        """
        row = self._conn.execute(
            "SELECT MAX(date) FROM ohlcv_daily WHERE symbol = ?",
            (symbol,),
        ).fetchone()

        if row and row[0]:
            return date.fromisoformat(row[0])
        return None

    def get_ohlcv_count(self, symbol: str) -> int:
        """Get number of cached OHLCV rows for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Row count
        """
        row = self._conn.execute(
            "SELECT COUNT(*) FROM ohlcv_daily WHERE symbol = ?",
            (symbol,),
        ).fetchone()
        return row[0] if row else 0

    def store_ohlcv(self, symbol: str, df: pd.DataFrame) -> int:
        """Store OHLCV rows (INSERT OR IGNORE for dedup).

        Args:
            symbol: Stock ticker symbol
            df: DataFrame with OHLCV data (index=Date, columns=Open/High/Low/Close/Volume)

        Returns:
            Number of new rows inserted
        """
        if df.empty:
            return 0

        rows = []
        for idx, row in df.iterrows():
            dt = idx.date() if hasattr(idx, "date") and callable(idx.date) else idx
            rows.append(
                (
                    symbol,
                    str(dt),
                    float(row["Open"]),
                    float(row["High"]),
                    float(row["Low"]),
                    float(row["Close"]),
                    float(row["Volume"]),
                )
            )

        cursor = self._conn.executemany(
            "INSERT OR IGNORE INTO ohlcv_daily (symbol, date, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        inserted = cursor.rowcount
        logger.debug(f"Stored {inserted} OHLCV rows for {symbol} ({len(rows)} total)")
        return inserted

    def get_cached_urls(self, symbol: str) -> set[str]:
        """Get URLs of cached news articles for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Set of cached article URLs
        """
        rows = self._conn.execute(
            "SELECT url FROM news_articles WHERE symbol = ?",
            (symbol,),
        ).fetchall()
        return {r[0] for r in rows}

    def store_news_articles(self, symbol: str, articles: list) -> int:
        """Store news articles (INSERT OR IGNORE on URL).

        Args:
            symbol: Stock ticker symbol
            articles: List of NewsArticle objects

        Returns:
            Number of new articles inserted
        """
        if not articles:
            return 0

        rows = []
        for article in articles:
            rows.append(
                (
                    article.url,
                    symbol,
                    article.title,
                    article.description,
                    article.published_at.isoformat(),
                    article.source,
                )
            )

        cursor = self._conn.executemany(
            "INSERT OR IGNORE INTO news_articles (url, symbol, title, description, published_at, source) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        inserted = cursor.rowcount
        logger.debug(f"Stored {inserted} news articles for {symbol}")
        return inserted

    def get_fundamentals(self, symbol: str) -> dict | None:
        """Get cached fundamentals if within TTL.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Fundamentals dict or None if missing/expired
        """
        row = self._conn.execute(
            "SELECT data, fetched_at FROM fundamentals WHERE symbol = ?",
            (symbol,),
        ).fetchone()

        if not row:
            return None

        fetched_at = datetime.fromisoformat(row[1])
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=UTC)
        if datetime.now(UTC) - fetched_at > timedelta(days=FUNDAMENTALS_TTL_DAYS):
            return None

        return json.loads(row[0])

    def store_fundamentals(self, symbol: str, data: dict) -> None:
        """Store or update fundamentals for a symbol.

        Args:
            symbol: Stock ticker symbol
            data: Fundamentals dictionary
        """
        self._conn.execute(
            "INSERT OR REPLACE INTO fundamentals (symbol, data, fetched_at) VALUES (?, ?, ?)",
            (symbol, json.dumps(data), datetime.now(UTC).isoformat()),
        )
        self._conn.commit()
        logger.debug(f"Stored fundamentals for {symbol}")

    def store_order_fill(self, order: object) -> None:
        """Store an order fill (INSERT OR IGNORE).

        Args:
            order: OrderStatus object
        """
        self._conn.execute(
            "INSERT OR IGNORE INTO order_fills "
            "(order_id, symbol, qty, filled_qty, side, status, submitted_at, filled_at, filled_avg_price) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                order.order_id,
                order.symbol,
                order.qty,
                order.filled_qty,
                order.side,
                order.status,
                order.submitted_at.isoformat(),
                order.filled_at.isoformat() if order.filled_at else None,
                order.filled_avg_price,
            ),
        )
        self._conn.commit()
        logger.debug(f"Stored order fill {order.order_id}")

    def get_cached_post_ids(self) -> set[str]:
        """Get IDs of cached Truth Social posts.

        Returns:
            Set of cached post IDs
        """
        rows = self._conn.execute("SELECT id FROM truth_social_posts").fetchall()
        return {r[0] for r in rows}

    def store_truth_social_posts(self, posts: list) -> int:
        """Store Truth Social posts (INSERT OR IGNORE on ID).

        Args:
            posts: List of TruthPost objects

        Returns:
            Number of new posts inserted
        """
        if not posts:
            return 0

        rows = []
        for post in posts:
            rows.append(
                (
                    post.id,
                    post.content,
                    post.created_at.isoformat(),
                    post.likes,
                    post.reposts,
                    post.replies,
                    post.url,
                )
            )

        cursor = self._conn.executemany(
            "INSERT OR IGNORE INTO truth_social_posts "
            "(id, content, created_at, likes, reposts, replies, url) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        inserted = cursor.rowcount
        logger.debug(f"Stored {inserted} Truth Social posts")
        return inserted

    def get_cached_reddit_ids(self, symbol: str) -> set[str]:
        """Get IDs of cached Reddit posts for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Set of cached post IDs
        """
        rows = self._conn.execute(
            "SELECT id FROM reddit_posts WHERE symbol = ?",
            (symbol,),
        ).fetchall()
        return {r[0] for r in rows}

    def store_reddit_posts(self, symbol: str, posts: list) -> int:
        """Store Reddit posts (INSERT OR IGNORE on ID).

        Args:
            symbol: Stock ticker symbol
            posts: List of RedditPost objects

        Returns:
            Number of new posts inserted
        """
        if not posts:
            return 0

        rows = []
        for post in posts:
            rows.append(
                (
                    post.id,
                    symbol,
                    post.title,
                    post.body,
                    post.subreddit,
                    post.score,
                    post.upvote_ratio,
                    post.url,
                    post.created_utc.isoformat(),
                    post.num_comments,
                )
            )

        cursor = self._conn.executemany(
            "INSERT OR IGNORE INTO reddit_posts "
            "(id, symbol, title, body, subreddit, score, upvote_ratio, url, created_utc, num_comments) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        inserted = cursor.rowcount
        logger.debug(f"Stored {inserted} Reddit posts for {symbol}")
        return inserted

    def stats(self) -> dict[str, int]:
        """Get row counts for all tables.

        Returns:
            Dict mapping table name to row count
        """
        tables = [
            "ohlcv_daily",
            "news_articles",
            "fundamentals",
            "order_fills",
            "truth_social_posts",
            "reddit_posts",
        ]
        counts = {}
        for table in tables:
            row = self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()  # noqa: S608
            counts[table] = row[0] if row else 0
        return counts

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
        logger.debug("HistoricalCache closed")

    def __repr__(self) -> str:
        """Return string representation."""
        return f"HistoricalCache(db={self._db_path})"
