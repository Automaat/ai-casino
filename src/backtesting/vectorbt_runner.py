"""Vectorized backtesting runner using numpy/pandas for fast portfolio simulation."""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import pandas_ta_classic as ta  # type: ignore[import-untyped]
import yfinance as yf
from loguru import logger
from pydantic import BaseModel


class VectorBTResult(BaseModel):
    """Vectorized backtest result for a single symbol."""

    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    equity_curve: list[float]
    symbol: str
    start_date: datetime
    end_date: datetime


class MultiAssetBacktest(BaseModel):
    """Multi-asset portfolio backtest result."""

    symbols: list[str]
    results: list[VectorBTResult]
    portfolio_sharpe: float
    portfolio_return: float
    portfolio_max_drawdown: float
    correlation_matrix: dict[str, dict[str, float]]


@dataclass
class _SimulationInput:
    """Input parameters for portfolio simulation."""

    data: pd.DataFrame
    entries: pd.Series
    exits: pd.Series
    symbol: str
    start_date: datetime
    end_date: datetime


class VectorBTRunner:
    """Vectorized backtesting engine using numpy/pandas.

    Generates entry/exit signals from technical indicators and simulates
    portfolio performance using vectorized operations for 10-100x speedup
    over event-driven backtesting.
    """

    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70

    def __init__(self, cash: float = 100_000.0, commission: float = 0.002) -> None:
        """Initialize vectorized backtest runner.

        Args:
            cash: Initial cash balance
            commission: Commission rate (0.002 = 0.2%)
        """
        self.cash = cash
        self.commission = commission
        logger.info(f"Initialized VectorBTRunner (cash=${cash:,.2f}, commission={commission:.2%})")

    def run_backtest(
        self,
        symbol: str,
        start_date: str | datetime,
        end_date: str | datetime,
    ) -> VectorBTResult:
        """Run vectorized backtest for symbol.

        Args:
            symbol: Stock ticker
            start_date: Backtest start date (YYYY-MM-DD or datetime)
            end_date: Backtest end date (YYYY-MM-DD or datetime)

        Returns:
            VectorBTResult with metrics and equity curve
        """
        logger.info(f"Running vectorized backtest for {symbol} ({start_date} to {end_date})")

        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")  # noqa: DTZ007
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")  # noqa: DTZ007

        data = self._fetch_data(symbol, start_date, end_date)
        entries, exits = self._generate_signals(data)
        sim_input = _SimulationInput(data, entries, exits, symbol, start_date, end_date)
        result = self._simulate(sim_input)

        logger.info(
            f"Vectorized backtest complete: {result.total_trades} trades, "
            f"{result.total_return:.2%} return, {result.sharpe_ratio:.2f} Sharpe"
        )

        return result

    def run_portfolio_backtest(
        self,
        symbols: list[str],
        start_date: str | datetime,
        end_date: str | datetime,
    ) -> MultiAssetBacktest:
        """Run vectorized backtest across multiple symbols.

        Args:
            symbols: List of stock tickers
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            MultiAssetBacktest with per-symbol results and portfolio metrics
        """
        logger.info(f"Running portfolio backtest for {symbols}")

        results: list[VectorBTResult] = []
        returns_dict: dict[str, pd.Series] = {}

        for symbol in symbols:
            result = self.run_backtest(symbol, start_date, end_date)
            results.append(result)

            equity = pd.Series(result.equity_curve)
            daily_returns = equity.pct_change().dropna()
            returns_dict[symbol] = daily_returns

        returns_df = pd.DataFrame(returns_dict)
        correlation_matrix = {
            sym: {other: float(returns_df[sym].corr(returns_df[other])) for other in symbols}
            for sym in symbols
        }

        equal_weight_returns = returns_df.mean(axis=1)
        portfolio_return = float((1 + equal_weight_returns).prod() - 1)
        portfolio_sharpe = self._calc_sharpe(equal_weight_returns)
        portfolio_max_dd = self._calc_max_drawdown_from_returns(equal_weight_returns)

        return MultiAssetBacktest(
            symbols=symbols,
            results=results,
            portfolio_sharpe=portfolio_sharpe,
            portfolio_return=portfolio_return,
            portfolio_max_drawdown=portfolio_max_dd,
            correlation_matrix=correlation_matrix,
        )

    def _fetch_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch historical data via yfinance."""
        logger.info(f"Fetching data for {symbol} via yfinance")

        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)

        if data.empty:
            msg = f"No data available for {symbol} between {start_date} and {end_date}"
            raise ValueError(msg)

        data = data[["Open", "High", "Low", "Close", "Volume"]]
        logger.info(f"Fetched {len(data)} bars for {symbol}")
        return data

    def _generate_signals(self, data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Generate entry/exit signals from RSI + MACD indicators.

        Returns:
            Tuple of (entries, exits) boolean Series
        """
        close = data["Close"]

        rsi = ta.rsi(close, length=14)
        macd_result = ta.macd(close, fast=12, slow=26, signal=9)

        if macd_result is not None:
            macd_hist = macd_result["MACDh_12_26_9"]
        else:
            macd_hist = pd.Series(0.0, index=close.index)

        rsi = rsi.fillna(50.0)
        macd_hist = macd_hist.fillna(0.0)

        entries = (rsi < self.RSI_OVERSOLD) & (macd_hist > 0)
        exits = (rsi > self.RSI_OVERBOUGHT) & (macd_hist < 0)

        return entries, exits

    def _build_positions(
        self, entries_arr: np.ndarray, exits_arr: np.ndarray, n: int
    ) -> tuple[np.ndarray, list[int], list[int]]:
        """Build position array and trade entry/exit indices from signals."""
        position = np.zeros(n, dtype=np.int8)
        in_pos = False
        trade_entries: list[int] = []
        trade_exits: list[int] = []

        for i in range(n):
            if not in_pos and entries_arr[i]:
                in_pos = True
                trade_entries.append(i)
            elif in_pos and exits_arr[i]:
                in_pos = False
                trade_exits.append(i)
            position[i] = 1 if in_pos else 0

        return position, trade_entries, trade_exits

    def _simulate(self, sim: _SimulationInput) -> VectorBTResult:
        """Simulate portfolio from entry/exit signals using vectorized ops."""
        close = sim.data["Close"].values
        entries_arr = sim.entries.values.astype(bool)
        exits_arr = sim.exits.values.astype(bool)
        n = len(close)

        position, trade_entries, trade_exits = self._build_positions(entries_arr, exits_arr, n)

        # Compute equity curve
        daily_returns = np.diff(close) / close[:-1]
        strategy_returns = np.zeros(n)
        strategy_returns[1:] = daily_returns * position[:-1]

        # Apply commission at entry/exit points
        for idx in trade_entries:
            if idx < n:
                strategy_returns[idx] -= self.commission
        for idx in trade_exits:
            if idx < n:
                strategy_returns[idx] -= self.commission

        equity_curve = self.cash * np.cumprod(1 + strategy_returns)

        # Trade metrics
        trade_returns: list[float] = []
        for i in range(min(len(trade_entries), len(trade_exits))):
            entry_price = close[trade_entries[i]]
            exit_price = close[trade_exits[i]]
            ret = (exit_price - entry_price) / entry_price - 2 * self.commission
            trade_returns.append(ret)

        total_trades = len(trade_returns)
        wins = [r for r in trade_returns if r > 0]
        losses = [r for r in trade_returns if r <= 0]
        win_rate = len(wins) / total_trades if total_trades > 0 else 0.0

        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

        strategy_returns_series = pd.Series(strategy_returns)
        total_return = float(equity_curve[-1] / self.cash - 1) if n > 0 else 0.0
        sharpe = self._calc_sharpe(strategy_returns_series)
        sortino = self._calc_sortino(strategy_returns_series)
        max_dd = self._calc_max_drawdown(equity_curve)
        calmar = abs(total_return / max_dd) if max_dd != 0 else 0.0

        return VectorBTResult(
            total_return=total_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            equity_curve=equity_curve.tolist(),
            symbol=sim.symbol,
            start_date=sim.start_date,
            end_date=sim.end_date,
        )

    def _calc_sharpe(self, returns: pd.Series, trading_days: int = 252) -> float:
        """Annualized Sharpe ratio."""
        if returns.std() == 0:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(trading_days))

    def _calc_sortino(self, returns: pd.Series, trading_days: int = 252) -> float:
        """Annualized Sortino ratio."""
        downside = returns[returns < 0]
        if len(downside) == 0 or downside.std() == 0:
            return 0.0
        return float(returns.mean() / downside.std() * np.sqrt(trading_days))

    def _calc_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """Maximum drawdown from equity curve."""
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        return float(drawdown.min())

    def _calc_max_drawdown_from_returns(self, returns: pd.Series) -> float:
        """Maximum drawdown from returns series."""
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        return float(drawdown.min())

    def __repr__(self) -> str:
        """String representation."""
        return f"VectorBTRunner(cash=${self.cash:,.2f}, commission={self.commission:.2%})"
