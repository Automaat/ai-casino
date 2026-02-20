"""Automated tearsheet generation for daemon."""

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pandas as pd
from loguru import logger
from pandas import DatetimeIndex

from src.daemon.state import AnalysisRecord
from src.data.market import MarketDataFetcher
from src.metrics.quantstats_reporter import QuantStatsReporter
from src.metrics.tracker import TearSheet, TradeRecord
from src.strategies.signal import Signal
from src.v1.trades.brokers import Broker


class DaemonTearsheetGenerator:
    """Generate performance tearsheets from daemon analysis history."""

    def __init__(
        self,
        risk_free_rate: float,
        broker: Broker | None = None,
        market_fetcher: MarketDataFetcher | None = None,
    ) -> None:
        """Initialize tearsheet generator.

        Args:
            risk_free_rate: Annual risk-free rate for metrics
            broker: Optional Alpaca broker for fetching closed trades
            market_fetcher: Optional market data fetcher for benchmark data
        """
        self.broker = broker
        self.market_fetcher = market_fetcher
        self.reporter = QuantStatsReporter(risk_free_rate)
        logger.info("Initialized DaemonTearsheetGenerator")

    def set_broker(self, broker: Broker) -> None:
        """Set broker after initialization (deferred to avoid event loop issues)."""
        self.broker = broker
        logger.debug("DaemonTearsheetGenerator broker updated")

    def generate_portfolio_tearsheet(
        self,
        analyses: list[AnalysisRecord],
        benchmark_symbol: str = "SPY",
    ) -> TearSheet | None:
        """Generate tearsheet from analysis records.

        Args:
            analyses: List of analysis records from daemon state
            benchmark_symbol: Benchmark symbol (default: SPY)

        Returns:
            TearSheet or None if insufficient data
        """
        logger.info(f"Generating portfolio tearsheet from {len(analyses)} analysis records")

        trades = self._convert_analyses_to_trades(analyses)
        if not trades:
            logger.warning("No closed trades available for tearsheet generation")
            return None

        try:
            benchmark_returns = self._fetch_benchmark_returns(benchmark_symbol, trades)
            tearsheet = self.reporter.generate_tearsheet(
                symbol="PORTFOLIO",
                trades=trades,
                benchmark_symbol=benchmark_symbol,
                benchmark_returns=benchmark_returns,
            )
            logger.info(f"Generated tearsheet: {tearsheet.html_report_path}")
            return tearsheet
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to generate tearsheet: {e}")
            raise

    def _convert_analyses_to_trades(self, analyses: list[AnalysisRecord]) -> list[TradeRecord]:
        """Convert analysis records to trade records.

        Uses Alpaca broker if available to fetch actual closed trades.
        Otherwise, simulates trades from analysis signals.

        Args:
            analyses: List of analysis records

        Returns:
            List of closed TradeRecord objects
        """
        if self.broker:
            logger.info("Fetching closed trades from Alpaca broker")
            broker_trades = self._fetch_broker_trades()
            if broker_trades:
                return broker_trades
            logger.info("No broker trades, falling back to simulation")

        logger.info("Simulating trades from analysis records (no broker available)")
        return self._simulate_trades_from_analyses(analyses)

    def _fetch_broker_trades(self) -> list[TradeRecord]:
        """Fetch closed trades from Alpaca broker.

        The current Broker implementation does not expose closed trades.
        Returns empty list; tearsheets rely on simulation instead.

        Returns:
            Empty list (broker-based closed trades not supported)
        """
        if self.broker:
            logger.info(
                "Broker configured, but closed-trade fetching not supported; "
                "will simulate trades from analysis records instead"
            )
        return []

    def _simulate_trades_from_analyses(self, analyses: list[AnalysisRecord]) -> list[TradeRecord]:
        """Simulate trades from analysis records using real market data.

        Fetches actual entry/exit prices from market data at analysis timestamps
        to produce accurate PnL metrics.

        Args:
            analyses: List of analysis records

        Returns:
            List of simulated closed TradeRecord objects with real prices
        """
        if not analyses:
            return []

        closed_trades: list[TradeRecord] = []
        open_positions: dict[str, AnalysisRecord] = {}
        executed_analyses = [a for a in analyses if a.executed_trade]

        for analysis in executed_analyses:
            symbol = analysis.symbol
            signal = analysis.signal

            if signal == "HOLD":
                continue

            if symbol not in open_positions:
                if signal in ("BUY", "SELL"):
                    open_positions[symbol] = analysis
            else:
                entry_analysis = open_positions[symbol]
                entry_signal = entry_analysis.signal

                if (entry_signal == "BUY" and signal == "SELL") or (
                    entry_signal == "SELL" and signal == "BUY"
                ):
                    # Fetch real prices from market data
                    entry_price = self._get_price_at_timestamp(symbol, entry_analysis.timestamp)
                    exit_price = self._get_price_at_timestamp(symbol, analysis.timestamp)

                    if entry_price is None or exit_price is None:
                        logger.warning(
                            f"Cannot fetch prices for {symbol}, skipping trade "
                            f"({entry_analysis.timestamp} -> {analysis.timestamp})"
                        )
                        del open_positions[symbol]
                        continue

                    # Calculate real PnL
                    shares = 100  # Standard lot size
                    pnl = (
                        (exit_price - entry_price) * shares
                        if entry_signal == "BUY"
                        else (entry_price - exit_price) * shares
                    )
                    pnl_percent = (
                        ((exit_price - entry_price) / entry_price) * 100
                        if entry_signal == "BUY"
                        else ((entry_price - exit_price) / entry_price) * 100
                    )

                    trade = TradeRecord(
                        timestamp=entry_analysis.timestamp,
                        symbol=symbol,
                        action=Signal.BUY if entry_signal == "BUY" else Signal.SELL,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        shares=shares,
                        stop_loss_price=0.0,
                        confidence=entry_analysis.confidence,
                        risk_level="MEDIUM",
                        status="CLOSED",
                        pnl=pnl,
                        pnl_percent=pnl_percent,
                        strategy_name="daemon",
                    )
                    closed_trades.append(trade)
                    del open_positions[symbol]

        logger.info(
            f"Simulated {len(closed_trades)} closed trades with real prices "
            f"from {len(executed_analyses)} executed analyses"
        )
        return closed_trades

    def _get_price_at_timestamp(self, symbol: str, timestamp: datetime) -> float | None:
        """Get closing price for symbol at given timestamp.

        Args:
            symbol: Stock ticker symbol
            timestamp: Timestamp to fetch price for

        Returns:
            Closing price or None if unavailable
        """
        if not self.market_fetcher:
            logger.warning("No market fetcher available, cannot fetch real prices")
            return None

        try:
            # Fetch 5 days around timestamp to handle weekends/holidays
            period_days = 5
            market_data = self.market_fetcher.fetch_daily(symbol, period_days=period_days)

            if market_data.data.empty:
                logger.warning(f"No market data for {symbol}")
                return None

            # Find closest date to timestamp
            target_date = timestamp.date()
            df = market_data.data.copy()

            # Try exact date match
            if target_date in df.index:
                close_val = df.loc[pd.Timestamp(target_date), "close"]
                return float(close_val)  # type: ignore[arg-type]

            # Fall back to nearest date (handles weekends/holidays)
            import numpy as np

            target_timestamp = pd.Timestamp(target_date)
            # Type narrowing for Index subtraction
            time_diffs_td = cast("DatetimeIndex", df.index) - target_timestamp  # TimedeltaIndex
            time_diffs = time_diffs_td.total_seconds()  # type: ignore[union-attr]
            date_diff_array = np.abs(time_diffs)
            df["date_diff"] = date_diff_array
            nearest_idx = cast("pd.Series", df["date_diff"]).idxmin()
            price_val = df.loc[nearest_idx, "close"]
            price = float(price_val)  # type: ignore[arg-type]

            nearest_date = nearest_idx.date() if hasattr(nearest_idx, "date") else nearest_idx
            logger.debug(f"Fetched {symbol} price {price} for {target_date} (nearest: {nearest_date})")
            return price

        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to fetch price for {symbol} at {timestamp}: {e}")
            return None

    def _fetch_benchmark_returns(self, benchmark_symbol: str, trades: list[TradeRecord]) -> pd.Series | None:
        """Fetch benchmark returns for comparison.

        Args:
            benchmark_symbol: Benchmark ticker (e.g., SPY)
            trades: List of trades to determine date range

        Returns:
            Pandas Series of daily returns or None
        """
        if not self.market_fetcher or not trades:
            return None

        try:
            start_date = min(t.timestamp for t in trades)
            end_date = max(t.timestamp for t in trades)
            days = (end_date - start_date).days + 1

            logger.info(f"Fetching {benchmark_symbol} benchmark data ({days} days)")
            market_data = self.market_fetcher.fetch_daily(benchmark_symbol, period_days=days)

            if market_data.data.empty:
                logger.warning(f"No benchmark data available for {benchmark_symbol}")
                return None

            returns = market_data.data["close"].pct_change().dropna()
            logger.info(f"Fetched {len(returns)} days of benchmark returns")
            return returns
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to fetch benchmark data: {e}")
            return None

    def cleanup_old_tearsheets(self, retention_days: int = 30) -> None:
        """Delete tearsheets older than retention_days.

        Args:
            retention_days: Number of days to retain tearsheets
        """
        tearsheet_dir = Path.home() / ".ai-casino" / "tearsheets"
        if not tearsheet_dir.exists():
            return

        from datetime import timedelta

        cutoff_time = datetime.now(UTC) - timedelta(days=retention_days)
        deleted_count = 0

        for tearsheet_file in tearsheet_dir.glob("*.html"):
            try:
                file_mtime = datetime.fromtimestamp(tearsheet_file.stat().st_mtime, tz=UTC)
                if file_mtime < cutoff_time:
                    tearsheet_file.unlink()
                    deleted_count += 1
                    logger.debug(f"Deleted old tearsheet: {tearsheet_file.name}")
            except Exception as e:
                logger.opt(exception=True).warning(f"Failed to delete {tearsheet_file}: {e}")

        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old tearsheets (retention: {retention_days} days)")

    def __repr__(self) -> str:
        """String representation."""
        broker_status = "with broker" if self.broker else "no broker"
        return f"DaemonTearsheetGenerator({broker_status})"
