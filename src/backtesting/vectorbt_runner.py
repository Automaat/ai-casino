"""Vectorized backtesting runner using numpy/pandas for fast portfolio simulation."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pandas_ta_classic as ta  # type: ignore[import-untyped]
import yfinance as yf
from loguru import logger
from pydantic import BaseModel

if TYPE_CHECKING:
    from src.strategies.ensemble import EnsembleStrategy
    from src.strategies.mean_reversion import MeanReversionStrategy
    from src.strategies.momentum import MomentumStrategy
    from src.strategies.trend_following import TrendFollowingStrategy


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
    equity_dates: list[datetime]
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
        strategy: (
            "MomentumStrategy | MeanReversionStrategy | TrendFollowingStrategy | EnsembleStrategy | None"
        ) = None,
    ) -> VectorBTResult:
        """Run vectorized backtest for symbol.

        Args:
            symbol: Stock ticker
            start_date: Backtest start date (YYYY-MM-DD or datetime)
            end_date: Backtest end date (YYYY-MM-DD or datetime)
            strategy: Trading strategy (defaults to momentum if None)

        Returns:
            VectorBTResult with metrics and equity curve
        """
        strategy_name = strategy.__class__.__name__ if strategy else "MomentumStrategy"
        logger.info(
            f"Running vectorized backtest for {symbol} ({start_date} to {end_date}) using {strategy_name}"
        )

        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)

        data = self._fetch_data(symbol, start_date, end_date)
        entries, exits = self._generate_signals_for_strategy(data, strategy)
        sim_input = _SimulationInput(data, entries, exits, symbol, start_date, end_date)
        result = self._simulate(sim_input)

        logger.info(
            f"Vectorized backtest complete ({strategy_name}): {result.total_trades} trades, "
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

            equity = pd.Series(result.equity_curve, index=result.equity_dates)
            daily_returns = equity.pct_change().dropna()
            returns_dict[symbol] = daily_returns

        # Align returns on common dates (intersection) before correlation
        returns_df = pd.DataFrame(returns_dict).dropna()
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

    def _generate_signals_for_strategy(
        self,
        data: pd.DataFrame,
        strategy: (
            "MomentumStrategy | MeanReversionStrategy | TrendFollowingStrategy | EnsembleStrategy | None"
        ),
    ) -> tuple[pd.Series, pd.Series]:
        """Generate entry/exit signals based on strategy type.

        Args:
            data: OHLCV dataframe
            strategy: Trading strategy instance (defaults to momentum if None)

        Returns:
            Tuple of (entries, exits) boolean Series
        """
        from src.strategies.ensemble import EnsembleStrategy
        from src.strategies.mean_reversion import MeanReversionStrategy
        from src.strategies.momentum import MomentumStrategy
        from src.strategies.trend_following import TrendFollowingStrategy

        if strategy is None or isinstance(strategy, MomentumStrategy):
            return self._generate_momentum_signals(data, strategy)
        if isinstance(strategy, MeanReversionStrategy):
            return self._generate_mean_reversion_signals(data, strategy)
        if isinstance(strategy, TrendFollowingStrategy):
            return self._generate_trend_following_signals(data, strategy)
        if isinstance(strategy, EnsembleStrategy):
            return self._generate_ensemble_signals(data, strategy)

        logger.warning(f"Unknown strategy type {type(strategy)}, defaulting to momentum")
        return self._generate_momentum_signals(data, None)

    def _generate_momentum_signals(
        self, data: pd.DataFrame, strategy: "MomentumStrategy | None" = None
    ) -> tuple[pd.Series, pd.Series]:
        """Generate entry/exit signals from RSI + MACD indicators (momentum strategy).

        Args:
            data: OHLCV dataframe
            strategy: MomentumStrategy instance (uses params if provided, else defaults)

        Returns:
            Tuple of (entries, exits) boolean Series
        """
        close = data["Close"]

        rsi_period = strategy.rsi_period if strategy else 14
        rsi_oversold = strategy.rsi_oversold if strategy else 30.0
        rsi_overbought = strategy.rsi_overbought if strategy else 70.0
        macd_fast = strategy.macd_fast if strategy else 12
        macd_slow = strategy.macd_slow if strategy else 26
        macd_signal = strategy.macd_signal if strategy else 9

        rsi = ta.rsi(close, length=rsi_period)
        macd_result = ta.macd(close, fast=macd_fast, slow=macd_slow, signal=macd_signal)

        if macd_result is not None:
            macd_hist = macd_result[f"MACDh_{macd_fast}_{macd_slow}_{macd_signal}"]
        else:
            macd_hist = pd.Series(0.0, index=close.index)

        rsi = rsi.fillna(50.0)
        macd_hist = macd_hist.fillna(0.0)

        entries = (rsi < rsi_oversold) & (macd_hist > 0)
        exits = (rsi > rsi_overbought) & (macd_hist < 0)

        return entries, exits

    def _generate_mean_reversion_signals(
        self, data: pd.DataFrame, strategy: "MeanReversionStrategy | None" = None
    ) -> tuple[pd.Series, pd.Series]:
        """Generate entry/exit signals from Bollinger Bands (mean reversion).

        Matches MeanReversionStrategy logic: pure BB-based, no RSI filter.

        Args:
            data: OHLCV dataframe
            strategy: MeanReversionStrategy instance (uses params if provided, else defaults)

        Returns:
            Tuple of (entries, exits) boolean Series
        """
        close = data["Close"]

        bb_period = strategy.bb_period if strategy else 20
        bb_std = strategy.bb_std if strategy else 2.0

        bb_result = ta.bbands(close, length=bb_period, std=bb_std)

        if bb_result is not None:
            std_str = f"{bb_std:.1f}"
            bb_lower = bb_result[f"BBL_{bb_period}_{std_str}"]
            bb_upper = bb_result[f"BBU_{bb_period}_{std_str}"]
        else:
            bb_lower = close * 0.98
            bb_upper = close * 1.02

        bb_lower = bb_lower.fillna(close * 0.98)
        bb_upper = bb_upper.fillna(close * 1.02)

        # Pure mean reversion: no RSI filter (aligns with MeanReversionStrategy)
        entries = close <= bb_lower
        exits = close >= bb_upper

        return entries, exits

    def _generate_trend_following_signals(
        self, data: pd.DataFrame, strategy: "TrendFollowingStrategy | None" = None
    ) -> tuple[pd.Series, pd.Series]:
        """Generate entry/exit signals from SMA crossover + ADX (trend following).

        Uses strategy-configured parameters including adx_threshold_weak for exits.

        Args:
            data: OHLCV dataframe
            strategy: TrendFollowingStrategy instance (uses params if provided, else defaults)

        Returns:
            Tuple of (entries, exits) boolean Series
        """
        close = data["Close"]
        high = data["High"]
        low = data["Low"]

        sma_fast_length = strategy.sma_fast if strategy else 50
        sma_slow_length = strategy.sma_slow if strategy else 200
        adx_length = strategy.adx_period if strategy else 14
        adx_strong_threshold = strategy.adx_threshold if strategy else 25.0

        # Use new adx_threshold_weak parameter (defaults to adx_threshold - 5)
        adx_weak_threshold = (
            strategy.adx_threshold_weak
            if strategy and hasattr(strategy, "adx_threshold_weak")
            else adx_strong_threshold - 5.0
        )

        sma_fast = ta.sma(close, length=sma_fast_length)
        sma_slow = ta.sma(close, length=sma_slow_length)
        adx = ta.adx(high, low, close, length=adx_length)

        adx_col = f"ADX_{adx_length}"
        adx_val = adx[adx_col] if adx is not None and adx_col in adx else pd.Series(0.0, index=close.index)

        sma_fast = sma_fast.fillna(close)
        sma_slow = sma_slow.fillna(close)
        adx_val = adx_val.fillna(0.0)

        entries = (sma_fast > sma_slow) & (adx_val > adx_strong_threshold)
        exits = (sma_fast < sma_slow) | (adx_val < adx_weak_threshold)

        return entries, exits

    def _generate_ensemble_signals(
        self,
        data: pd.DataFrame,
        strategy: "EnsembleStrategy",
    ) -> tuple[pd.Series, pd.Series]:
        """Generate entry/exit signals from weighted voting of all strategies.

        Args:
            data: OHLCV dataframe
            strategy: EnsembleStrategy instance with component strategies

        Returns:
            Tuple of (entries, exits) boolean Series
        """
        momentum_entries, momentum_exits = self._generate_momentum_signals(data)
        mean_rev_entries, mean_rev_exits = self._generate_mean_reversion_signals(data)
        trend_entries, trend_exits = self._generate_trend_following_signals(data)

        # Convert boolean signals to numeric (1/0) for weighted voting
        momentum_weight = strategy.weights["momentum"]
        mean_rev_weight = strategy.weights["mean_reversion"]
        trend_weight = strategy.weights["trend_following"]

        momentum_entry_score = momentum_entries.astype(int) * momentum_weight
        mean_rev_entry_score = mean_rev_entries.astype(int) * mean_rev_weight
        trend_entry_score = trend_entries.astype(int) * trend_weight

        momentum_exit_score = momentum_exits.astype(int) * momentum_weight
        mean_rev_exit_score = mean_rev_exits.astype(int) * mean_rev_weight
        trend_exit_score = trend_exits.astype(int) * trend_weight

        # Aggregate scores: entry if majority vote (>0.5 weighted sum)
        total_weight = momentum_weight + mean_rev_weight + trend_weight
        threshold = total_weight / 2

        entry_scores = momentum_entry_score + mean_rev_entry_score + trend_entry_score
        exit_scores = momentum_exit_score + mean_rev_exit_score + trend_exit_score

        entries = entry_scores > threshold
        exits = exit_scores > threshold

        return entries, exits

    def _build_positions(
        self, entries_arr: np.ndarray, exits_arr: np.ndarray, n: int
    ) -> tuple[np.ndarray, list[int], list[int]]:
        """Build position array and trade entry/exit indices from signals using vectorized ops."""
        # Create state change markers: +1 for entry, -1 for exit
        changes = np.zeros(n, dtype=np.int8)
        changes[entries_arr] = 1
        changes[exits_arr] = -1

        # Handle conflicts: if both entry and exit on same bar, prioritize exit
        conflicts = entries_arr & exits_arr
        changes[conflicts] = -1

        # Accumulate state changes to get position array (0 or 1)
        position_raw = np.cumsum(changes)
        position = np.clip(position_raw, 0, 1).astype(np.int8)

        # Extract trade indices: entries where position goes 0→1, exits where 1→0
        position_shift = np.roll(position, 1)
        position_shift[0] = 0  # First bar has no prior position

        trade_entries = np.where((position_shift == 0) & (position == 1))[0].tolist()
        trade_exits = np.where((position_shift == 1) & (position == 0))[0].tolist()

        return position, trade_entries, trade_exits

    def _simulate(self, sim: _SimulationInput) -> VectorBTResult:
        """Simulate portfolio from entry/exit signals using vectorized ops."""
        close = sim.data["Close"].values
        dates = sim.data.index.to_pydatetime().tolist()
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
        calmar = total_return / abs(max_dd) if max_dd != 0 else 0.0

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
            equity_dates=dates,
            symbol=sim.symbol,
            start_date=sim.start_date,
            end_date=sim.end_date,
        )

    def _calc_sharpe(self, returns: pd.Series, trading_days: int = 252) -> float:
        """Annualized Sharpe ratio."""
        mean = returns.mean()
        std = returns.std()
        if pd.isna(mean) or pd.isna(std) or std == 0:
            return 0.0
        return float(mean / std * np.sqrt(trading_days))

    def _calc_sortino(self, returns: pd.Series, trading_days: int = 252) -> float:
        """Annualized Sortino ratio."""
        downside = returns[returns < 0]
        mean = returns.mean()
        downside_std = downside.std()
        if len(downside) == 0 or pd.isna(mean) or pd.isna(downside_std) or downside_std == 0:
            return 0.0
        return float(mean / downside_std * np.sqrt(trading_days))

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
