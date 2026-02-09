"""Tests for MetaAgent strategy selection."""

from unittest.mock import MagicMock

import pytest

from src.agents.meta import (
    DEFAULT_WEIGHTS,
    STRATEGY_REGIME_MAP,
    MetaAgent,
    StrategySelection,
)
from src.strategies.ensemble import EnsembleStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.regime import MarketRegime, MarketRegimeDetector, RegimeAnalysis, RegimeIndicators
from src.strategies.trend_following import TrendFollowingStrategy


@pytest.fixture
def mock_regime_detector():
    """Mock regime detector."""
    mock = MagicMock(spec=MarketRegimeDetector)
    mock.adx_threshold = 25.0
    mock.atr_vol_ratio = 1.5
    return mock


@pytest.fixture
def sample_regime_analysis():
    """Sample high-confidence regime analysis."""
    return RegimeAnalysis(
        regime=MarketRegime.TRENDING_BULLISH,
        indicators=RegimeIndicators(
            adx=35.0,
            plus_di=30.0,
            minus_di=15.0,
            atr=2.5,
            atr_ratio=1.1,
            bb_width=5.0,
        ),
        confidence=0.75,
        reasoning="Strong bullish trend detected",
    )


@pytest.fixture
def low_confidence_regime_analysis():
    """Sample low-confidence regime analysis."""
    return RegimeAnalysis(
        regime=MarketRegime.RANGING,
        indicators=RegimeIndicators(
            adx=22.0,
            plus_di=18.0,
            minus_di=17.0,
            atr=2.0,
            atr_ratio=1.0,
            bb_width=4.0,
        ),
        confidence=0.4,
        reasoning="Weak ranging market, low confidence",
    )


class TestMetaAgent:
    """Tests for MetaAgent."""

    def test_meta_agent_init(self, mock_llm_client, mock_regime_detector) -> None:
        """Test meta agent initialization."""
        agent = MetaAgent(mock_llm_client, mock_regime_detector)

        assert agent.llm == mock_llm_client
        assert agent.regime_detector == mock_regime_detector
        assert agent.metrics_tracker is None

    def test_meta_agent_init_with_tracker(self, mock_llm_client, mock_regime_detector) -> None:
        """Test meta agent initialization with metrics tracker."""
        mock_tracker = MagicMock()
        agent = MetaAgent(mock_llm_client, mock_regime_detector, mock_tracker)

        assert agent.metrics_tracker == mock_tracker

    async def test_select_strategy_trending_market(
        self, mock_llm_client, mock_regime_detector, sample_ohlcv_trending_up, sample_regime_analysis
    ) -> None:
        """Test strategy selection for trending market."""
        mock_regime_detector.detect_regime.return_value = sample_regime_analysis

        agent = MetaAgent(mock_llm_client, mock_regime_detector)
        result = await agent.select_strategy("AAPL", sample_ohlcv_trending_up)

        assert isinstance(result, StrategySelection)
        assert result.strategy_name == "trend_following"
        assert isinstance(result.strategy_instance, TrendFollowingStrategy)
        assert result.regime == MarketRegime.TRENDING_BULLISH
        assert result.ensemble_weights is None
        assert result.confidence == sample_regime_analysis.confidence

    async def test_select_strategy_ranging_market(
        self, mock_llm_client, mock_regime_detector, sample_ohlcv_ranging
    ) -> None:
        """Test strategy selection for ranging market."""
        ranging_analysis = RegimeAnalysis(
            regime=MarketRegime.RANGING,
            indicators=RegimeIndicators(
                adx=18.0, plus_di=15.0, minus_di=14.0, atr=1.5, atr_ratio=0.9, bb_width=3.0
            ),
            confidence=0.7,
            reasoning="Ranging market detected",
        )
        mock_regime_detector.detect_regime.return_value = ranging_analysis

        agent = MetaAgent(mock_llm_client, mock_regime_detector)
        result = await agent.select_strategy("AAPL", sample_ohlcv_ranging)

        assert result.strategy_name == "mean_reversion"
        assert isinstance(result.strategy_instance, MeanReversionStrategy)
        assert result.regime == MarketRegime.RANGING

    async def test_select_strategy_volatile_market(
        self, mock_llm_client, mock_regime_detector, sample_ohlcv_volatile
    ) -> None:
        """Test strategy selection for volatile market."""
        volatile_analysis = RegimeAnalysis(
            regime=MarketRegime.HIGH_VOLATILITY,
            indicators=RegimeIndicators(
                adx=28.0, plus_di=22.0, minus_di=20.0, atr=5.0, atr_ratio=2.0, bb_width=8.0
            ),
            confidence=0.65,
            reasoning="High volatility detected",
        )
        mock_regime_detector.detect_regime.return_value = volatile_analysis

        agent = MetaAgent(mock_llm_client, mock_regime_detector)
        result = await agent.select_strategy("AAPL", sample_ohlcv_volatile)

        assert result.strategy_name == "momentum"
        assert isinstance(result.strategy_instance, MomentumStrategy)
        assert result.regime == MarketRegime.HIGH_VOLATILITY

    async def test_ensemble_fallback_on_low_confidence(
        self, mock_llm_client, mock_regime_detector, sample_ohlcv_data, low_confidence_regime_analysis
    ) -> None:
        """Test ensemble fallback when regime confidence is low."""
        mock_regime_detector.detect_regime.return_value = low_confidence_regime_analysis

        agent = MetaAgent(mock_llm_client, mock_regime_detector)
        result = await agent.select_strategy("AAPL", sample_ohlcv_data)

        assert result.strategy_name == "ensemble"
        assert isinstance(result.strategy_instance, EnsembleStrategy)
        assert result.ensemble_weights is not None
        assert sum(result.ensemble_weights.values()) == pytest.approx(1.0, abs=0.01)
        assert result.confidence == low_confidence_regime_analysis.confidence

    async def test_ensemble_weights_boost_regime_strategy(
        self, mock_llm_client, mock_regime_detector, sample_ohlcv_data
    ) -> None:
        """Test that ensemble weights boost the regime-matched strategy."""
        low_conf_bullish = RegimeAnalysis(
            regime=MarketRegime.TRENDING_BULLISH,
            indicators=RegimeIndicators(
                adx=26.0, plus_di=20.0, minus_di=18.0, atr=2.0, atr_ratio=1.0, bb_width=4.0
            ),
            confidence=0.45,  # Below threshold
            reasoning="Weak bullish trend",
        )
        mock_regime_detector.detect_regime.return_value = low_conf_bullish

        agent = MetaAgent(mock_llm_client, mock_regime_detector)
        result = await agent.select_strategy("AAPL", sample_ohlcv_data)

        assert result.ensemble_weights is not None
        # trend_following should have highest weight (boosted)
        assert result.ensemble_weights["trend_following"] > DEFAULT_WEIGHTS["trend_following"]

    async def test_performance_based_weight_adjustment(
        self, mock_llm_client, mock_regime_detector, sample_ohlcv_data
    ) -> None:
        """Test performance-based weight adjustment."""
        low_conf = RegimeAnalysis(
            regime=MarketRegime.RANGING,
            indicators=RegimeIndicators(
                adx=20.0, plus_di=15.0, minus_di=14.0, atr=1.5, atr_ratio=1.0, bb_width=3.0
            ),
            confidence=0.4,
            reasoning="Ranging",
        )
        mock_regime_detector.detect_regime.return_value = low_conf

        mock_tracker = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.total_decisions = 10
        mock_metrics.win_rate = 65.0  # Above 50%
        mock_tracker.calculate_metrics.return_value = mock_metrics

        agent = MetaAgent(mock_llm_client, mock_regime_detector, mock_tracker)
        result = await agent.select_strategy("AAPL", sample_ohlcv_data)

        assert result.ensemble_weights is not None
        mock_tracker.calculate_metrics.assert_called_once_with("30d")

    def test_create_strategy_momentum(self, mock_llm_client, mock_regime_detector) -> None:
        """Test creating momentum strategy."""
        agent = MetaAgent(mock_llm_client, mock_regime_detector)
        strategy = agent._create_strategy("momentum")

        assert isinstance(strategy, MomentumStrategy)

    def test_create_strategy_mean_reversion(self, mock_llm_client, mock_regime_detector) -> None:
        """Test creating mean reversion strategy."""
        agent = MetaAgent(mock_llm_client, mock_regime_detector)
        strategy = agent._create_strategy("mean_reversion")

        assert isinstance(strategy, MeanReversionStrategy)

    def test_create_strategy_trend_following(self, mock_llm_client, mock_regime_detector) -> None:
        """Test creating trend following strategy."""
        agent = MetaAgent(mock_llm_client, mock_regime_detector)
        strategy = agent._create_strategy("trend_following")

        assert isinstance(strategy, TrendFollowingStrategy)

    def test_create_strategy_unknown_raises(self, mock_llm_client, mock_regime_detector) -> None:
        """Test that unknown strategy raises ValueError."""
        agent = MetaAgent(mock_llm_client, mock_regime_detector)

        with pytest.raises(ValueError, match="Unknown strategy"):
            agent._create_strategy("unknown_strategy")

    def test_repr(self, mock_llm_client, mock_regime_detector) -> None:
        """Test string representation."""
        agent = MetaAgent(mock_llm_client, mock_regime_detector)
        repr_str = repr(agent)

        assert "MetaAgent" in repr_str
        assert "has_metrics_tracker=False" in repr_str


class TestStrategyRegimeMap:
    """Tests for STRATEGY_REGIME_MAP."""

    def test_all_regimes_mapped(self) -> None:
        """Test that all regimes have a strategy mapping."""
        for regime in MarketRegime:
            assert regime in STRATEGY_REGIME_MAP
            assert STRATEGY_REGIME_MAP[regime] in ["momentum", "mean_reversion", "trend_following"]

    def test_trending_maps_to_trend_following(self) -> None:
        """Test trending regimes map to trend following."""
        assert STRATEGY_REGIME_MAP[MarketRegime.TRENDING_BULLISH] == "trend_following"
        assert STRATEGY_REGIME_MAP[MarketRegime.TRENDING_BEARISH] == "trend_following"

    def test_ranging_maps_to_mean_reversion(self) -> None:
        """Test ranging regime maps to mean reversion."""
        assert STRATEGY_REGIME_MAP[MarketRegime.RANGING] == "mean_reversion"

    def test_volatile_maps_to_momentum(self) -> None:
        """Test high volatility maps to momentum."""
        assert STRATEGY_REGIME_MAP[MarketRegime.HIGH_VOLATILITY] == "momentum"


class TestStrategySelection:
    """Tests for StrategySelection model."""

    def test_strategy_selection_fields(self) -> None:
        """Test StrategySelection model fields."""
        regime_analysis = RegimeAnalysis(
            regime=MarketRegime.TRENDING_BULLISH,
            indicators=RegimeIndicators(
                adx=35.0, plus_di=30.0, minus_di=15.0, atr=2.5, atr_ratio=1.1, bb_width=5.0
            ),
            confidence=0.75,
            reasoning="Test reasoning",
        )
        selection = StrategySelection(
            strategy_name="trend_following",
            strategy_instance=TrendFollowingStrategy(sma_fast=20, sma_slow=50),
            ensemble_weights=None,
            regime=MarketRegime.TRENDING_BULLISH,
            regime_confidence=0.75,
            regime_analysis=regime_analysis,
            reasoning="Test reasoning",
            confidence=0.75,
        )

        assert selection.strategy_name == "trend_following"
        assert isinstance(selection.strategy_instance, TrendFollowingStrategy)
        assert selection.ensemble_weights is None
        assert selection.regime == MarketRegime.TRENDING_BULLISH
        assert selection.regime_confidence == 0.75
        assert selection.regime_analysis == regime_analysis
        assert selection.confidence == 0.75

    def test_strategy_selection_with_ensemble(self) -> None:
        """Test StrategySelection with ensemble weights."""
        weights = {"momentum": 0.4, "mean_reversion": 0.3, "trend_following": 0.3}
        regime_analysis = RegimeAnalysis(
            regime=MarketRegime.RANGING,
            indicators=RegimeIndicators(
                adx=18.0, plus_di=20.0, minus_di=22.0, atr=1.5, atr_ratio=1.0, bb_width=3.0
            ),
            confidence=0.4,
            reasoning="Low confidence",
        )
        selection = StrategySelection(
            strategy_name="ensemble",
            strategy_instance=EnsembleStrategy(weights=weights),
            ensemble_weights=weights,
            regime=MarketRegime.RANGING,
            regime_confidence=0.4,
            regime_analysis=regime_analysis,
            reasoning="Low confidence, using ensemble",
            confidence=0.4,
        )

        assert selection.strategy_name == "ensemble"
        assert selection.ensemble_weights == weights
        assert selection.ensemble_weights is not None
        assert sum(selection.ensemble_weights.values()) == pytest.approx(1.0)
