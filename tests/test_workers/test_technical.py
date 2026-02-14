"""Tests for technical worker."""

import pandas as pd

from src.strategies.ensemble import EnsembleStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.signal import Signal
from src.strategies.trend_following import TrendFollowingStrategy
from src.tools.models import ToolDefinition
from src.workers.technical import TechnicalAnalysis


def test_technical_worker_init(test_container):
    """Test worker initialization."""
    worker = test_container.technical_worker()

    assert worker.llm_client is not None
    tool_def = worker.get_tool_definition()
    assert tool_def is not None


async def test_technical_worker_analyze_momentum(test_container, sample_ohlcv_data):
    """Test analysis with momentum strategy."""
    worker = test_container.technical_worker()
    strategy = MomentumStrategy()

    result = await worker.analyze("AAPL", sample_ohlcv_data, strategy)

    assert isinstance(result, TechnicalAnalysis)
    assert isinstance(result.signal, Signal)
    assert 0.0 <= result.confidence <= 1.0
    assert result.interpretation
    assert result.rsi is not None
    assert result.macd_hist is not None


async def test_technical_worker_analyze_trend_following(test_container):
    """Test analysis with trend following strategy."""
    # TrendFollowing needs 200+ rows for SMA_200
    large_ohlcv = pd.DataFrame(
        {
            "Open": [100 + i for i in range(250)],
            "High": [105 + i for i in range(250)],
            "Low": [99 + i for i in range(250)],
            "Close": [104 + i for i in range(250)],
            "Volume": [1000000] * 250,
        }
    )

    worker = test_container.technical_worker()
    strategy = TrendFollowingStrategy()

    result = await worker.analyze("AAPL", large_ohlcv, strategy)

    assert isinstance(result, TechnicalAnalysis)
    assert isinstance(result.signal, Signal)
    assert 0.0 <= result.confidence <= 1.0
    assert result.interpretation


async def test_technical_worker_analyze_mean_reversion(test_container, sample_ohlcv_data):
    """Test analysis with mean reversion strategy."""
    worker = test_container.technical_worker()
    strategy = MeanReversionStrategy()

    result = await worker.analyze("AAPL", sample_ohlcv_data, strategy)

    assert isinstance(result, TechnicalAnalysis)
    assert isinstance(result.signal, Signal)
    assert 0.0 <= result.confidence <= 1.0
    assert result.interpretation


async def test_technical_worker_analyze_ensemble(test_container, sample_ohlcv_data):
    """Test analysis with ensemble strategy."""
    worker = test_container.technical_worker()
    strategy = EnsembleStrategy()

    result = await worker.analyze("AAPL", sample_ohlcv_data, strategy)

    assert isinstance(result, TechnicalAnalysis)
    assert isinstance(result.signal, Signal)
    assert 0.0 <= result.confidence <= 1.0
    assert result.interpretation
    assert result.ensemble_result is not None


async def test_technical_worker_multi_timeframe(test_container):
    """Test multi-timeframe analysis."""
    from datetime import datetime

    from src.strategies.timeframe import MultiTimeframeData, Timeframe

    # Create larger datasets for TrendFollowingStrategy (needs 200+ rows for SMA_200)
    daily_data = pd.DataFrame(
        {
            "Open": [100 + i for i in range(250)],
            "High": [105 + i for i in range(250)],
            "Low": [99 + i for i in range(250)],
            "Close": [104 + i for i in range(250)],
            "Volume": [1000000] * 250,
        }
    )

    hourly_data = pd.DataFrame(
        {
            "Open": [140 + i * 0.1 for i in range(250)],
            "High": [141 + i * 0.1 for i in range(250)],
            "Low": [139 + i * 0.1 for i in range(250)],
            "Close": [140.5 + i * 0.1 for i in range(250)],
            "Volume": [500000] * 250,
        }
    )

    multi_data = MultiTimeframeData(
        symbol="AAPL",
        timeframes={Timeframe.DAILY: daily_data, Timeframe.HOURLY: hourly_data},
        last_updated=datetime.now(),
    )

    worker = test_container.technical_worker()
    strategy = TrendFollowingStrategy()

    result = await worker.analyze("AAPL", multi_data, strategy, enable_multi_timeframe=True)

    assert isinstance(result, TechnicalAnalysis)
    assert isinstance(result.signal, Signal)
    assert result.multi_timeframe is not None
    assert len(result.multi_timeframe.timeframe_results) > 0
    assert 0.0 <= result.multi_timeframe.confluence_score <= 1.0


async def test_multi_timeframe_5_timeframes(test_container, sample_ohlcv_data):
    """Test TechnicalWorker with all 5 timeframes."""
    import time
    from datetime import datetime

    from src.strategies.timeframe import MultiTimeframeData, Timeframe

    # Create MultiTimeframeData with 5 timeframes
    multi_data = MultiTimeframeData(
        symbol="AAPL",
        timeframes={
            Timeframe.DAILY: sample_ohlcv_data,
            Timeframe.HOURLY: sample_ohlcv_data,
            Timeframe.FIFTEEN_MIN: sample_ohlcv_data,
            Timeframe.FIVE_MIN: sample_ohlcv_data,
            Timeframe.ONE_MIN: sample_ohlcv_data,
        },
        last_updated=datetime.now(),
    )

    worker = test_container.technical_worker()
    strategy = MomentumStrategy()

    start = time.perf_counter()
    result = await worker.analyze("AAPL", multi_data, strategy, enable_multi_timeframe=True)
    duration = time.perf_counter() - start

    # Assertions
    assert result.multi_timeframe is not None
    assert len(result.multi_timeframe.timeframe_results) == 5
    assert 0.0 <= result.multi_timeframe.confluence_score <= 1.0
    assert result.multi_timeframe.primary_timeframe in Timeframe
    assert duration < 30.0  # Performance requirement (generous for LLM calls)

    # Verify all 5 timeframes are present
    assert Timeframe.DAILY in result.multi_timeframe.timeframe_results
    assert Timeframe.HOURLY in result.multi_timeframe.timeframe_results
    assert Timeframe.FIFTEEN_MIN in result.multi_timeframe.timeframe_results
    assert Timeframe.FIVE_MIN in result.multi_timeframe.timeframe_results
    assert Timeframe.ONE_MIN in result.multi_timeframe.timeframe_results


def test_technical_worker_tool_definition(test_container):
    """Test tool definition for supervisor integration."""
    worker = test_container.technical_worker()

    tool_def = worker.get_tool_definition()

    assert isinstance(tool_def, ToolDefinition)
    assert tool_def.type == "function"
    assert tool_def.function.name == "analyze_technical"
    assert "symbol" in tool_def.function.parameters.properties
    assert "strategy" in tool_def.function.parameters.properties
    assert "enable_multi_timeframe" in tool_def.function.parameters.properties


def test_technical_worker_repr(test_container):
    """Test string representation."""
    worker = test_container.technical_worker()

    repr_str = repr(worker)

    assert "TechnicalWorker" in repr_str
    assert "pydantic_ai" in repr_str
