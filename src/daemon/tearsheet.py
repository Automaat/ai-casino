"""Automated tearsheet generation for daemon."""

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from src.daemon.state import AnalysisRecord
from src.data.broker import AlpacaBroker
from src.data.market import MarketDataFetcher
from src.metrics.quantstats_reporter import QuantStatsReporter
from src.metrics.tracker import TearSheet, TradeRecord
from src.strategies.signal import Signal


class DaemonTearsheetGenerator:
    """Generate performance tearsheets from daemon analysis history."""

    def __init__(
        self,
        broker: AlpacaBroker | None = None,
        market_fetcher: MarketDataFetcher | None = None,
    ) -> None:
        """Initialize tearsheet generator.

        Args:
            broker: Optional Alpaca broker for fetching closed trades
            market_fetcher: Optional market data fetcher for benchmark data
        """
        self.broker = broker
        self.market_fetcher = market_fetcher
        self.reporter = QuantStatsReporter()
        logger.info("Initialized DaemonTearsheetGenerator")

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
            logger.error(f"Failed to generate tearsheet: {e}")
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
            return self._fetch_broker_trades()

        logger.info("Simulating trades from analysis records (no broker available)")
        return self._simulate_trades_from_analyses(analyses)

    def _fetch_broker_trades(self) -> list[TradeRecord]:
        """Fetch closed trades from Alpaca broker.

        Returns:
            List of closed TradeRecord objects
        """
        try:
            account_info = self.broker.get_account_info()
            closed_positions = [p for p in account_info.closed_positions if p.unrealized_pl is not None]

            if not closed_positions:
                logger.warning("No closed positions found in broker")
                return []

            trades: list[TradeRecord] = []
            for pos in closed_positions:
                if pos.avg_entry_price is None or pos.current_price is None:
                    continue

                trade = TradeRecord(
                    timestamp=pos.entry_time or datetime.now(UTC),
                    symbol=pos.symbol,
                    action=Signal.BUY if pos.qty > 0 else Signal.SELL,
                    entry_price=pos.avg_entry_price,
                    exit_price=pos.current_price,
                    shares=abs(pos.qty),
                    stop_loss_price=0.0,
                    confidence=0.0,
                    risk_level="UNKNOWN",
                    status="CLOSED",
                    pnl=pos.unrealized_pl,
                    pnl_percent=pos.unrealized_plpc * 100 if pos.unrealized_plpc else None,
                    strategy_name="daemon",
                )
                trades.append(trade)

            logger.info(f"Fetched {len(trades)} closed trades from broker")
            return trades
        except Exception as e:
            logger.warning(f"Failed to fetch broker trades: {e}, falling back to simulation")
            return []

    def _simulate_trades_from_analyses(self, analyses: list[AnalysisRecord]) -> list[TradeRecord]:
        """Simulate trades from analysis records.

        Creates synthetic closed trades from BUY/SELL signals.
        Pairs consecutive BUY->SELL or SELL->BUY for same symbol.

        Args:
            analyses: List of analysis records

        Returns:
            List of simulated closed TradeRecord objects
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
                    entry_price = 100.0
                    exit_price = 105.0 if entry_signal == "BUY" else 95.0

                    pnl = (
                        (exit_price - entry_price) * 100
                        if entry_signal == "BUY"
                        else (entry_price - exit_price) * 100
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
                        shares=100,
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
            f"Simulated {len(closed_trades)} closed trades from {len(executed_analyses)} executed analyses"
        )
        return closed_trades

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
            end_date = max((t.exit_price and t.timestamp) or datetime.now(UTC) for t in trades)
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
            logger.warning(f"Failed to fetch benchmark data: {e}")
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
                logger.warning(f"Failed to delete {tearsheet_file}: {e}")

        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old tearsheets (retention: {retention_days} days)")

    def __repr__(self) -> str:
        """String representation."""
        broker_status = "with broker" if self.broker else "no broker"
        return f"DaemonTearsheetGenerator({broker_status})"
