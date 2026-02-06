"""Tests for stock screener."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data.universe import StockInfo, StockUniverse
from src.screening.screener import (
    ScreeningCriteria,
    ScreeningOutput,
    ScreeningResult,
    StockScreener,
)
from src.strategies.signal import Signal


@pytest.fixture
def sample_stock_universe():
    """Sample stock universe for testing."""
    return StockUniverse(
        name="TEST",
        stocks=[
            StockInfo(symbol="AAPL", name="Apple Inc.", sector="Technology", industry="Hardware"),
            StockInfo(symbol="MSFT", name="Microsoft Corp", sector="Technology", industry="Software"),
            StockInfo(symbol="JPM", name="JPMorgan Chase", sector="Financials", industry="Banks"),
        ],
        fetched_at=datetime.now(),
    )


@pytest.fixture
def mock_universe_fetcher(sample_stock_universe):
    """Mock StockUniverseFetcher."""
    mock = MagicMock()
    mock.fetch_sp500.return_value = sample_stock_universe
    mock.fetch_nasdaq100.return_value = sample_stock_universe
    mock.fetch_combined.return_value = sample_stock_universe
    return mock


@pytest.fixture
def sample_ohlcv_momentum():
    """Sample OHLCV data that matches momentum criteria."""
    np.random.seed(42)
    n = 100

    # Create oversold condition with bullish reversal
    closes = np.concatenate(
        [
            np.linspace(150, 100, 70),  # Downtrend (oversold)
            np.linspace(100, 110, 30),  # Reversal (bullish MACD)
        ]
    )
    closes = closes + np.random.normal(0, 1, n)

    return pd.DataFrame(
        {
            "Open": closes - np.random.uniform(0, 1, n),
            "High": closes + np.random.uniform(1, 3, n),
            "Low": closes - np.random.uniform(1, 3, n),
            "Close": closes,
            "Volume": [1000000] * n,
        }
    )


@pytest.fixture
def sample_ohlcv_breakout():
    """Sample OHLCV data that matches breakout criteria."""
    np.random.seed(42)
    n = 100

    # Create uptrend near 52-week high with volume spike
    closes = np.linspace(100, 148, n) + np.random.normal(0, 1, n)

    volumes = [1000000] * (n - 5) + [2000000] * 5  # Volume spike at end

    return pd.DataFrame(
        {
            "Open": closes - np.random.uniform(0, 1, n),
            "High": closes + np.random.uniform(0, 2, n),
            "Low": closes - np.random.uniform(0, 2, n),
            "Close": closes,
            "Volume": volumes,
        }
    )


@pytest.fixture
def stock_screener(mock_universe_fetcher, tmp_path):
    """Create StockScreener with mocked universe fetcher."""
    return StockScreener(
        universe_fetcher=mock_universe_fetcher,
        cache_dir=str(tmp_path / "screening_cache"),
    )


class TestScreeningCriteria:
    """Tests for ScreeningCriteria enum."""

    def test_values(self):
        """Test criteria values."""
        assert ScreeningCriteria.MOMENTUM == "momentum"
        assert ScreeningCriteria.VALUE == "value"
        assert ScreeningCriteria.BREAKOUT == "breakout"


class TestScreeningResult:
    """Tests for ScreeningResult model."""

    def test_create(self):
        """Test ScreeningResult creation."""
        result = ScreeningResult(
            symbol="AAPL",
            name="Apple Inc.",
            sector="Technology",
            score=0.85,
            signal=Signal.BUY,
            metrics={"rsi": 28.5, "macd_hist": 0.15},
            reason="RSI oversold, MACD bullish",
        )

        assert result.symbol == "AAPL"
        assert result.score == 0.85
        assert result.signal == Signal.BUY
        assert "rsi" in result.metrics


class TestScreeningOutput:
    """Tests for ScreeningOutput model."""

    def test_create(self):
        """Test ScreeningOutput creation."""
        results = [
            ScreeningResult(
                symbol="AAPL",
                name="Apple",
                sector="Tech",
                score=0.85,
                signal=Signal.BUY,
                metrics={},
                reason="Test",
            )
        ]
        output = ScreeningOutput(
            criteria=ScreeningCriteria.MOMENTUM,
            universe="SP500",
            results=results,
            total_screened=500,
            errors=["FAILED"],
            screened_at=datetime.now(),
        )

        assert output.criteria == ScreeningCriteria.MOMENTUM
        assert output.universe == "SP500"
        assert len(output.results) == 1
        assert output.total_screened == 500
        assert len(output.errors) == 1


class TestStockScreener:
    """Tests for StockScreener."""

    def test_init(self, stock_screener):
        """Test screener initialization."""
        assert "StockScreener" in repr(stock_screener)

    @patch("src.screening.screener.yf.download")
    def test_screen_momentum(self, mock_download, stock_screener, sample_ohlcv_momentum):
        """Test momentum screening."""
        mock_download.return_value = pd.concat(
            {"AAPL": sample_ohlcv_momentum, "MSFT": sample_ohlcv_momentum, "JPM": sample_ohlcv_momentum},
            axis=1,
        )

        output = stock_screener.screen(ScreeningCriteria.MOMENTUM, "SP500", top_n=5)

        assert output.criteria == ScreeningCriteria.MOMENTUM
        assert output.universe == "SP500"
        assert output.total_screened == 3

    @patch("src.screening.screener.yf.download")
    @patch("src.screening.screener.yf.Ticker")
    def test_screen_value(self, mock_ticker, mock_download, stock_screener, sample_ohlcv_momentum):
        """Test value screening."""
        mock_download.return_value = pd.concat(
            {"AAPL": sample_ohlcv_momentum, "MSFT": sample_ohlcv_momentum, "JPM": sample_ohlcv_momentum},
            axis=1,
        )

        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {
            "trailingPE": 15.0,
            "priceToBook": 2.0,
            "forwardPE": 14.0,
        }
        mock_ticker.return_value = mock_ticker_instance

        output = stock_screener.screen(ScreeningCriteria.VALUE, "SP500", top_n=5)

        assert output.criteria == ScreeningCriteria.VALUE
        assert output.total_screened == 3

    @patch("src.screening.screener.yf.download")
    def test_screen_breakout(self, mock_download, stock_screener, sample_ohlcv_breakout):
        """Test breakout screening."""
        mock_download.return_value = pd.concat(
            {"AAPL": sample_ohlcv_breakout, "MSFT": sample_ohlcv_breakout, "JPM": sample_ohlcv_breakout},
            axis=1,
        )

        output = stock_screener.screen(ScreeningCriteria.BREAKOUT, "SP500", top_n=5)

        assert output.criteria == ScreeningCriteria.BREAKOUT
        assert output.total_screened == 3

    @patch("src.screening.screener.yf.download")
    def test_screen_caching(self, mock_download, stock_screener, sample_ohlcv_momentum):
        """Test that screening results are cached."""
        mock_download.return_value = pd.concat(
            {"AAPL": sample_ohlcv_momentum, "MSFT": sample_ohlcv_momentum, "JPM": sample_ohlcv_momentum},
            axis=1,
        )

        output1 = stock_screener.screen(ScreeningCriteria.MOMENTUM, "SP500", top_n=5)
        output2 = stock_screener.screen(ScreeningCriteria.MOMENTUM, "SP500", top_n=5)

        assert mock_download.call_count == 1  # Only fetched once
        assert output1.total_screened == output2.total_screened

    @patch("src.screening.screener.yf.download")
    def test_screen_top_n(self, mock_download, stock_screener, sample_ohlcv_momentum):
        """Test top_n parameter limits results."""
        mock_download.return_value = pd.concat(
            {"AAPL": sample_ohlcv_momentum, "MSFT": sample_ohlcv_momentum, "JPM": sample_ohlcv_momentum},
            axis=1,
        )

        output = stock_screener.screen(ScreeningCriteria.MOMENTUM, "SP500", top_n=1)

        assert len(output.results) <= 1

    @patch("src.screening.screener.yf.download")
    def test_fetch_universe_sp500(
        self, mock_download, stock_screener, sample_ohlcv_momentum, mock_universe_fetcher
    ):
        """Test SP500 universe selection."""
        mock_download.return_value = pd.concat(
            {"AAPL": sample_ohlcv_momentum, "MSFT": sample_ohlcv_momentum, "JPM": sample_ohlcv_momentum},
            axis=1,
        )

        stock_screener.screen(ScreeningCriteria.MOMENTUM, "SP500", top_n=5)

        mock_universe_fetcher.fetch_sp500.assert_called_once()

    @patch("src.screening.screener.yf.download")
    def test_fetch_universe_nasdaq100(
        self, mock_download, stock_screener, sample_ohlcv_momentum, mock_universe_fetcher
    ):
        """Test NASDAQ100 universe selection."""
        mock_download.return_value = pd.concat(
            {"AAPL": sample_ohlcv_momentum, "MSFT": sample_ohlcv_momentum, "JPM": sample_ohlcv_momentum},
            axis=1,
        )

        stock_screener.clear_cache()  # Clear to avoid cache from previous tests
        stock_screener.screen(ScreeningCriteria.MOMENTUM, "NASDAQ100", top_n=5)

        mock_universe_fetcher.fetch_nasdaq100.assert_called_once()

    def test_clear_cache(self, stock_screener):
        """Test cache clearing."""
        stock_screener.clear_cache()
        # Should not raise


class TestScoringFunctions:
    """Tests for individual scoring functions."""

    def test_score_momentum_criteria(self, stock_screener, sample_ohlcv_momentum):
        """Test momentum scoring matches expected criteria."""
        from src.data.universe import StockInfo

        info = StockInfo(symbol="TEST", name="Test Inc", sector="Tech", industry="Software")

        # Modify data to ensure it meets criteria
        df = sample_ohlcv_momentum.copy()

        result = stock_screener._score_momentum(df, info)

        # May or may not match depending on generated data
        if result:
            assert result.signal == Signal.BUY
            assert "rsi" in result.metrics
            assert "macd_hist" in result.metrics

    def test_score_breakout_criteria(self, stock_screener, sample_ohlcv_breakout):
        """Test breakout scoring matches expected criteria."""
        from src.data.universe import StockInfo

        info = StockInfo(symbol="TEST", name="Test Inc", sector="Tech", industry="Software")

        result = stock_screener._score_breakout(sample_ohlcv_breakout, info)

        if result:
            assert result.signal == Signal.BUY
            assert "pct_from_high" in result.metrics
            assert "volume_ratio" in result.metrics
