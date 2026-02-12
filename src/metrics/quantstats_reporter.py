"""QuantStats-based performance reporting."""

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import SupportsFloat, cast

import numpy as np
import pandas as pd
import quantstats as qs
from loguru import logger

from src.metrics.performance import build_daily_equity_curve, equity_curve_to_returns
from src.metrics.tracker import TearSheet, TradeRecord

# Minimum data points required for statistical calculations
MIN_REQUIRED_RETURNS = 2


def _to_float(value: SupportsFloat | pd.Series) -> float:
    """Convert numpy/pandas scalar to Python float.

    Args:
        value: Numeric value (Python, numpy, or pandas scalar)

    Returns:
        Python float
    """
    if isinstance(value, np.generic):
        return float(value.item())
    if isinstance(value, pd.Series):
        return float(value.item())
    return float(value)


class QuantStatsReporter:
    """Generate QuantStats performance tearsheets."""

    def __init__(self, risk_free_rate: float | None = None) -> None:
        """Initialize QuantStats reporter.

        Args:
            risk_free_rate: Annual risk-free rate (default from env or 0.02)
        """
        self.risk_free_rate = risk_free_rate or float(os.getenv("RISK_FREE_RATE", "0.02"))
        logger.info(f"Initialized QuantStatsReporter (risk_free_rate={self.risk_free_rate:.4f})")

    def generate_tearsheet(
        self,
        symbol: str,
        trades: list[TradeRecord],
        benchmark_symbol: str | None = None,
        benchmark_returns: pd.Series | None = None,
    ) -> TearSheet:
        """Generate tearsheet from trades.

        Args:
            symbol: Stock ticker symbol
            trades: List of closed TradeRecord objects
            benchmark_symbol: Optional benchmark symbol (e.g., "SPY")
            benchmark_returns: Optional pre-fetched benchmark returns series

        Returns:
            TearSheet with metrics and HTML report path
        """
        logger.info(f"Generating tearsheet for {symbol} ({len(trades)} trades)")

        equity_curve = build_daily_equity_curve(trades)
        if equity_curve.empty:
            msg = f"Cannot generate tearsheet: no closed trades for {symbol}"
            raise ValueError(msg)

        returns = equity_curve_to_returns(equity_curve)

        metrics = self._calculate_metrics(returns, benchmark_returns)

        html_path = self._generate_html(symbol, returns, benchmark_returns, benchmark_symbol)

        start_date = equity_curve.index[0]
        end_date = equity_curve.index[-1]

        return TearSheet(
            symbol=symbol,
            start_date=pd.Timestamp(start_date).to_pydatetime().replace(tzinfo=UTC),
            end_date=pd.Timestamp(end_date).to_pydatetime().replace(tzinfo=UTC),
            benchmark_symbol=benchmark_symbol,
            html_report_path=html_path,
            generated_at=datetime.now(UTC),
            **metrics,
        )

    def _calculate_metrics(self, returns: pd.Series, benchmark_returns: pd.Series | None = None) -> dict:
        """Calculate QuantStats metrics.

        Args:
            returns: Daily returns series
            benchmark_returns: Optional benchmark returns series

        Returns:
            Dictionary of metrics
        """
        logger.debug("Calculating QuantStats metrics")

        cagr = qs.stats.cagr(returns, rf=self.risk_free_rate)
        sharpe = qs.stats.sharpe(returns, rf=self.risk_free_rate)
        sortino = qs.stats.sortino(returns, rf=self.risk_free_rate)
        calmar = qs.stats.calmar(returns)
        max_dd = qs.stats.max_drawdown(returns)
        volatility = qs.stats.volatility(returns)

        winning_days = returns[returns > 0]
        losing_days = returns[returns < 0]
        win_rate = len(winning_days) / len(returns[returns != 0]) if len(returns[returns != 0]) > 0 else 0.0
        avg_win = winning_days.mean() if len(winning_days) > 0 else 0.0
        avg_loss = losing_days.mean() if len(losing_days) > 0 else 0.0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

        best_day = returns.max()
        worst_day = returns.min()

        monthly_returns_series = qs.stats.monthly_returns(returns)
        monthly_returns_dict = {}
        for idx, val in monthly_returns_series.items():
            key = idx.strftime("%Y-%m") if hasattr(idx, "strftime") else str(idx)
            monthly_returns_dict[key] = _to_float(val)

        dd_details = qs.stats.to_drawdown_series(returns)
        dd_duration = self._calculate_max_dd_duration(dd_details)

        metrics = {
            "cagr": _to_float(cagr) if pd.notna(cagr) else None,
            "sharpe_ratio": _to_float(sharpe) if pd.notna(sharpe) else None,
            "sortino_ratio": _to_float(sortino) if pd.notna(sortino) else None,
            "calmar_ratio": _to_float(calmar) if pd.notna(calmar) else None,
            "max_drawdown": _to_float(max_dd) if pd.notna(max_dd) else None,
            "max_drawdown_duration_days": dd_duration,
            "volatility_annual": _to_float(volatility) if pd.notna(volatility) else None,
            "win_rate": _to_float(win_rate),
            "profit_factor": _to_float(profit_factor),
            "avg_win": _to_float(avg_win),
            "avg_loss": _to_float(avg_loss),
            "best_day": _to_float(best_day),
            "worst_day": _to_float(worst_day),
            "monthly_returns": monthly_returns_dict,
        }

        if benchmark_returns is not None:
            benchmark_cagr = qs.stats.cagr(benchmark_returns, rf=self.risk_free_rate)
            benchmark_sharpe = qs.stats.sharpe(benchmark_returns, rf=self.risk_free_rate)
            beta = self._calculate_beta(returns, benchmark_returns)
            alpha = self._calculate_alpha(returns, benchmark_returns, beta)

            metrics["benchmark_cagr"] = _to_float(benchmark_cagr) if pd.notna(benchmark_cagr) else None
            metrics["benchmark_sharpe"] = _to_float(benchmark_sharpe) if pd.notna(benchmark_sharpe) else None
            metrics["alpha"] = _to_float(alpha) if pd.notna(alpha) else None
            metrics["beta"] = _to_float(beta) if pd.notna(beta) else None
        else:
            metrics["benchmark_cagr"] = None
            metrics["benchmark_sharpe"] = None
            metrics["alpha"] = None
            metrics["beta"] = None

        cagr = metrics.get("cagr")
        sharpe_ratio = metrics.get("sharpe_ratio")
        cagr_str = f"{cagr:.4f}" if isinstance(cagr, (int, float)) else "N/A"
        sharpe_str = f"{sharpe_ratio:.4f}" if isinstance(sharpe_ratio, (int, float)) else "N/A"
        logger.debug(f"Calculated metrics: CAGR={cagr_str}, Sharpe={sharpe_str}")
        return metrics

    def _calculate_beta(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate beta (volatility relative to benchmark).

        Args:
            returns: Portfolio returns series
            benchmark_returns: Benchmark returns series

        Returns:
            Beta value
        """
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join="inner")
        if len(aligned_returns) < MIN_REQUIRED_RETURNS:
            return 0.0
        covariance = _to_float(aligned_returns.cov(aligned_benchmark))
        benchmark_variance = _to_float(cast("SupportsFloat", aligned_benchmark.var()))
        return covariance / benchmark_variance if benchmark_variance != 0 else 0.0

    def _calculate_alpha(self, returns: pd.Series, benchmark_returns: pd.Series, beta: float) -> float:
        """Calculate alpha (excess return vs benchmark).

        Args:
            returns: Portfolio returns series
            benchmark_returns: Benchmark returns series
            beta: Previously calculated beta value

        Returns:
            Annualized alpha
        """
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join="inner")
        if len(aligned_returns) < MIN_REQUIRED_RETURNS:
            return 0.0
        portfolio_mean = aligned_returns.mean()
        benchmark_mean = aligned_benchmark.mean()
        alpha = portfolio_mean - (
            self.risk_free_rate / 252 + beta * (benchmark_mean - self.risk_free_rate / 252)
        )
        return float(alpha * 252)

    def _calculate_max_dd_duration(self, dd_series: pd.Series) -> int | None:
        """Calculate maximum drawdown duration in days.

        Args:
            dd_series: Drawdown series from QuantStats

        Returns:
            Maximum drawdown duration in days
        """
        if dd_series.empty:
            return None

        current_duration = 0
        max_duration = 0

        for val in dd_series:
            if val < 0:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_duration if max_duration > 0 else None

    def _generate_html(
        self,
        symbol: str,
        returns: pd.Series,
        benchmark_returns: pd.Series | None = None,
        benchmark_symbol: str | None = None,
    ) -> str:
        """Generate HTML tearsheet report.

        Args:
            symbol: Stock ticker symbol
            returns: Daily returns series
            benchmark_returns: Optional benchmark returns series
            benchmark_symbol: Optional benchmark symbol for labeling

        Returns:
            Path to generated HTML file
        """
        output_dir = Path.home() / ".ai-casino" / "tearsheets"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{symbol}_{timestamp}.html"

        logger.info(f"Generating HTML tearsheet to {output_path}")

        title = f"{symbol} Performance Tearsheet"
        if benchmark_symbol:
            title += f" vs {benchmark_symbol}"

        try:
            qs.reports.html(
                returns,
                benchmark=benchmark_returns,
                output=str(output_path),
                title=title,
                periods_per_year=252,
                download_filename=f"{symbol}_tearsheet.html",
            )
            logger.info(f"Successfully generated HTML tearsheet: {output_path}")
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to generate HTML tearsheet: {e}")
            raise

        return str(output_path)

    def __repr__(self) -> str:
        """String representation."""
        return f"QuantStatsReporter(risk_free_rate={self.risk_free_rate})"
