"""SQLite-backed permanent cache for immutable historical data."""

import json
import sqlite3
import threading
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger
from pandas.tseries.offsets import BDay

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
        self._lock = threading.Lock()
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
                symbol       TEXT NOT NULL,
                url          TEXT NOT NULL,
                title        TEXT NOT NULL,
                description  TEXT NOT NULL,
                published_at TEXT NOT NULL,
                source       TEXT NOT NULL,
                PRIMARY KEY (symbol, url)
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
                symbol        TEXT NOT NULL,
                id            TEXT NOT NULL,
                title         TEXT NOT NULL,
                body          TEXT NOT NULL,
                subreddit     TEXT NOT NULL,
                score         INTEGER NOT NULL,
                upvote_ratio  REAL NOT NULL,
                url           TEXT NOT NULL,
                created_utc   TEXT NOT NULL,
                num_comments  INTEGER NOT NULL,
                PRIMARY KEY (symbol, id)
            );
            CREATE INDEX IF NOT EXISTS idx_reddit_symbol ON reddit_posts(symbol);

            CREATE TABLE IF NOT EXISTS signal_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                signal TEXT NOT NULL,
                confidence REAL NOT NULL,
                price_at_signal REAL NOT NULL,
                strategy_used TEXT,
                regime TEXT,
                trading_session TEXT,
                technical_signal TEXT,
                sentiment_signal TEXT,
                news_signal TEXT,
                price_at_1d REAL,
                price_at_5d REAL,
                price_at_20d REAL,
                actual_exit_price REAL,
                actual_exit_date TEXT,
                outcome_updated_at TEXT,
                UNIQUE(symbol, timestamp)
            );
            CREATE INDEX IF NOT EXISTS idx_signal_outcomes_symbol ON signal_outcomes(symbol);
            CREATE INDEX IF NOT EXISTS idx_signal_outcomes_timestamp ON signal_outcomes(timestamp);
        """)

    def get_ohlcv(self, symbol: str) -> pd.DataFrame:
        """Get all cached OHLCV rows for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            DataFrame with OHLCV data (empty if no cache)
        """
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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

        with self._lock:
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
        with self._lock:
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
                    symbol,
                    article.url,
                    article.title,
                    article.description,
                    article.published_at.isoformat(),
                    article.source,
                )
            )

        with self._lock:
            cursor = self._conn.executemany(
                "INSERT OR IGNORE INTO news_articles (symbol, url, title, description, published_at, source) "
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO order_fills (order_id, symbol, qty, filled_qty, side, status, "
                "submitted_at, filled_at, filled_avg_price) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
        with self._lock:
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

        with self._lock:
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
        with self._lock:
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
                    symbol,
                    post.id,
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

        with self._lock:
            cursor = self._conn.executemany(
                "INSERT OR IGNORE INTO reddit_posts "
                "(symbol, id, title, body, subreddit, score, upvote_ratio, url, created_utc, num_comments) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()
            inserted = cursor.rowcount
        logger.debug(f"Stored {inserted} Reddit posts for {symbol}")
        return inserted

    def record_signal_outcome(  # noqa: PLR0913
        self,
        symbol: str,
        timestamp: datetime,
        signal: str,
        confidence: float,
        price_at_signal: float,
        strategy_used: str | None = None,
        regime: str | None = None,
        trading_session: str | None = None,
        technical_signal: str | None = None,
        sentiment_signal: str | None = None,
        news_signal: str | None = None,
    ) -> None:
        """Record a signal outcome for accuracy tracking.

        Args:
            symbol: Stock ticker symbol
            timestamp: Signal generation timestamp (timezone-aware)
            signal: Trading signal (BUY/SELL/HOLD)
            confidence: Signal confidence (0.0-1.0)
            price_at_signal: Price when signal was generated
            strategy_used: Strategy name (e.g., "momentum")
            regime: Market regime (e.g., "trending_bullish")
            trading_session: Trading session (REGULAR/PRE_MARKET)
            technical_signal: Technical analysis signal
            sentiment_signal: Sentiment analysis signal
            news_signal: News analysis signal
        """
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO signal_outcomes "
                "(symbol, timestamp, signal, confidence, price_at_signal, "
                "strategy_used, regime, trading_session, technical_signal, sentiment_signal, news_signal) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    symbol,
                    timestamp.isoformat(),
                    signal,
                    confidence,
                    price_at_signal,
                    strategy_used,
                    regime,
                    trading_session,
                    technical_signal,
                    sentiment_signal,
                    news_signal,
                ),
            )
            self._conn.commit()
        logger.debug(f"Recorded signal outcome for {symbol} ({signal} @ {price_at_signal:.2f})")

    def get_signals_needing_update(self, horizon: str) -> list[dict]:
        """Get signals that need outcome price updates for a given horizon.

        Args:
            horizon: Time horizon ("1d", "5d", "20d")

        Returns:
            List of signal records missing outcome prices for the given horizon
        """
        field = f"price_at_{horizon}"
        cutoff_days = {"1d": 1, "5d": 5, "20d": 20}[horizon]

        cutoff_date = (datetime.now(UTC) - BDay(cutoff_days)).isoformat()

        with self._lock:
            rows = self._conn.execute(
                f"SELECT id, symbol, timestamp, signal, price_at_signal, actual_exit_price "  # noqa: S608
                f"FROM signal_outcomes "
                f"WHERE {field} IS NULL AND timestamp < ? "
                f"ORDER BY timestamp DESC",
                (cutoff_date,),
            ).fetchall()

        return [
            {
                "id": r[0],
                "symbol": r[1],
                "timestamp": r[2],
                "signal": r[3],
                "price_at_signal": r[4],
                "actual_exit_price": r[5],
            }
            for r in rows
        ]

    def update_signal_outcome(self, signal_id: int, **fields: float | str | None) -> None:
        """Update signal outcome fields.

        Args:
            signal_id: Signal outcome ID
            **fields: Fields to update (e.g., price_at_1d=150.5, outcome_updated_at="...")
        """
        if not fields:
            return

        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values())
        values.append(signal_id)

        with self._lock:
            self._conn.execute(
                f"UPDATE signal_outcomes SET {set_clause} WHERE id = ?",  # noqa: S608
                values,
            )
            self._conn.commit()

    def get_signal_outcomes(
        self,
        window: str = "all",
        signal_type: str | None = None,
    ) -> list[dict]:
        """Get signal outcomes for metrics calculation.

        Args:
            window: Time window ("7d", "30d", "90d", "all")
            signal_type: Filter by signal type (BUY/SELL/HOLD) or None for all

        Returns:
            List of signal outcome records
        """
        query = "SELECT * FROM signal_outcomes WHERE 1=1"
        params = []

        if window != "all":
            days = int(window.rstrip("d"))
            cutoff_date = (datetime.now(UTC) - timedelta(days=days)).isoformat()
            query += " AND timestamp >= ?"
            params.append(cutoff_date)

        if signal_type:
            query += " AND signal = ?"
            params.append(signal_type)

        query += " ORDER BY timestamp DESC"

        with self._lock:
            rows = self._conn.execute(query, params).fetchall()

        cols = [
            "id",
            "symbol",
            "timestamp",
            "signal",
            "confidence",
            "price_at_signal",
            "strategy_used",
            "regime",
            "trading_session",
            "technical_signal",
            "sentiment_signal",
            "news_signal",
            "price_at_1d",
            "price_at_5d",
            "price_at_20d",
            "actual_exit_price",
            "actual_exit_date",
            "outcome_updated_at",
        ]

        return [dict(zip(cols, row, strict=False)) for row in rows]

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
            "signal_outcomes",
        ]
        counts = {}
        with self._lock:
            for table in tables:
                row = self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()  # noqa: S608
                counts[table] = row[0] if row else 0
        return counts

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._conn.close()
        logger.debug("HistoricalCache closed")

    def __repr__(self) -> str:
        """Return string representation."""
        return f"HistoricalCache(db={self._db_path})"
