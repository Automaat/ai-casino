"""Tests for trading workflow."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from src.agents.bearish_researcher import BearishResearchAnalysis
from src.agents.bullish_researcher import BullishResearchAnalysis
from src.agents.fundamental import FundamentalAnalysis
from src.agents.news import NewsAnalysis
from src.agents.risk import AccountInfo, RiskAssessment
from src.agents.sentiment import SentimentAnalysis
from src.agents.technical import TechnicalAnalysis
from src.agents.trader import TradingDecision
from src.backtesting.vectorbt_runner import VectorBTResult
from src.daemon.config import PreTradeBacktestingConfig
from src.data.market import MarketData
from src.strategies.ensemble import EnsembleStrategy
from src.strategies.session import TradingSession
from src.strategies.signal import Signal
from src.workflows.trading import TradingState, TradingWorkflow
from src.workflows.types import TradingWorkflowResult


@pytest.fixture
def mock_workflow_dependencies(
    mock_llm_client, mock_finbert, mock_fundamental_fetcher, sample_ohlcv_data, sample_news_articles
):
    market_fetcher = MagicMock()
    market_data = MarketData(
        symbol="AAPL",
        data=sample_ohlcv_data,
        last_updated="2024-01-15T12:00:00",
    )
    market_fetcher.fetch_daily.return_value = market_data

    news_fetcher = MagicMock()
    news_fetcher.fetch_company_news.return_value = sample_news_articles

    return market_fetcher, news_fetcher, mock_llm_client, mock_finbert, mock_fundamental_fetcher


def test_trading_workflow_init(mock_workflow_dependencies):
    market_fetcher, news_fetcher, llm_client, finbert, fundamental_fetcher = mock_workflow_dependencies

    workflow = TradingWorkflow(
        llm_client, market_fetcher, news_fetcher, finbert, fundamental_fetcher, use_meta_agent=False
    )

    assert workflow.market_fetcher == market_fetcher
    assert workflow.news_fetcher == news_fetcher
    assert workflow.sentiment_analyst is not None
    assert workflow.news_analyst is not None
    assert workflow.fundamental_analyst is not None
    assert workflow.bullish_researcher is not None
    assert workflow.bearish_researcher is not None
    assert workflow.trader is not None
    assert workflow.risk_manager is not None


def test_trading_workflow_init_ensemble(mock_workflow_dependencies):
    market_fetcher, news_fetcher, llm_client, finbert, fundamental_fetcher = mock_workflow_dependencies

    workflow = TradingWorkflow(
        llm_client,
        market_fetcher,
        news_fetcher,
        finbert,
        fundamental_fetcher,
        use_ensemble=True,
        use_meta_agent=False,
    )

    assert workflow.use_ensemble is True
    assert isinstance(workflow._default_strategy, EnsembleStrategy)
    assert repr(workflow) == "TradingWorkflow(agents=9, mode=ensemble)"


def test_trading_workflow_init_meta_agent(mock_workflow_dependencies):
    market_fetcher, news_fetcher, llm_client, finbert, fundamental_fetcher = mock_workflow_dependencies

    workflow = TradingWorkflow(
        llm_client, market_fetcher, news_fetcher, finbert, fundamental_fetcher, use_meta_agent=True
    )

    assert workflow.use_meta_agent is True
    assert workflow.meta_agent is not None
    assert repr(workflow) == "TradingWorkflow(agents=9, mode=meta-agent)"


async def test_trading_workflow_analyze(mock_workflow_dependencies):
    market_fetcher, news_fetcher, llm_client, finbert, fundamental_fetcher = mock_workflow_dependencies

    workflow = TradingWorkflow(
        llm_client, market_fetcher, news_fetcher, finbert, fundamental_fetcher, use_meta_agent=False
    )

    result = await workflow.analyze("AAPL", period_days=90)

    assert isinstance(result, TradingWorkflowResult)
    assert result.symbol == "AAPL"
    assert isinstance(result.technical, TechnicalAnalysis)
    assert isinstance(result.sentiment, SentimentAnalysis)
    assert isinstance(result.news, NewsAnalysis)
    assert isinstance(result.fundamental, FundamentalAnalysis)
    assert isinstance(result.bullish, BullishResearchAnalysis)
    assert isinstance(result.bearish, BearishResearchAnalysis)
    assert isinstance(result.decision, TradingDecision)
    assert isinstance(result.risk, RiskAssessment)
    assert result.risk.validation is not None

    market_fetcher.fetch_daily.assert_called_once_with("AAPL", 90)
    news_fetcher.fetch_company_news.assert_called_once_with("AAPL", limit=10)


async def test_trading_workflow_analyze_with_meta_agent(mock_workflow_dependencies):
    """Test full analyze flow with meta-agent enabled."""
    market_fetcher, news_fetcher, llm_client, finbert, fundamental_fetcher = mock_workflow_dependencies

    workflow = TradingWorkflow(
        llm_client, market_fetcher, news_fetcher, finbert, fundamental_fetcher, use_meta_agent=True
    )

    result = await workflow.analyze("AAPL", period_days=90)

    assert isinstance(result, TradingWorkflowResult)
    assert result.symbol == "AAPL"
    assert result.regime is not None
    assert result.strategy_used is not None
    assert isinstance(result.technical, TechnicalAnalysis)
    assert isinstance(result.sentiment, SentimentAnalysis)
    assert isinstance(result.news, NewsAnalysis)
    assert isinstance(result.fundamental, FundamentalAnalysis)
    assert isinstance(result.bullish, BullishResearchAnalysis)
    assert isinstance(result.bearish, BearishResearchAnalysis)
    assert isinstance(result.decision, TradingDecision)
    assert isinstance(result.risk, RiskAssessment)


async def test_fetch_data(mock_workflow_dependencies):
    market_fetcher, news_fetcher, llm_client, finbert, fundamental_fetcher = mock_workflow_dependencies

    workflow = TradingWorkflow(
        llm_client, market_fetcher, news_fetcher, finbert, fundamental_fetcher, use_meta_agent=False
    )

    state = await workflow._fetch_data("AAPL", 90, TradingSession.REGULAR)

    assert state["symbol"] == "AAPL"
    assert state["market_data"] is not None
    assert state["news_articles"] is not None
    assert len(state["news_articles"]) > 0


async def test_run_technical_analysis(mock_workflow_dependencies, sample_ohlcv_data):
    _, _, llm_client, _, _ = mock_workflow_dependencies

    # Create technical analyst with default strategy for testing
    from src.agents.technical import TechnicalAnalyst
    from src.strategies.momentum import MomentumStrategy

    technical_analyst = TechnicalAnalyst(llm_client, MomentumStrategy())
    result = await technical_analyst.analyze("AAPL", sample_ohlcv_data)

    assert result is not None
    assert isinstance(result, TechnicalAnalysis)


async def test_run_sentiment_analysis(mock_workflow_dependencies, sample_news_articles):
    market_fetcher, news_fetcher, llm_client, finbert, fundamental_fetcher = mock_workflow_dependencies

    workflow = TradingWorkflow(
        llm_client, market_fetcher, news_fetcher, finbert, fundamental_fetcher, use_meta_agent=False
    )

    result = await workflow.sentiment_analyst.analyze("AAPL", sample_news_articles)

    assert result is not None
    assert isinstance(result, SentimentAnalysis)


async def test_make_decision(mock_workflow_dependencies, sample_bullish_research, sample_bearish_research):
    market_fetcher, news_fetcher, llm_client, finbert, fundamental_fetcher = mock_workflow_dependencies

    workflow = TradingWorkflow(
        llm_client, market_fetcher, news_fetcher, finbert, fundamental_fetcher, use_meta_agent=False
    )

    state: TradingState = {
        "symbol": "AAPL",
        "trading_session": TradingSession.REGULAR,
        "market_data": None,
        "enable_multi_timeframe": False,
        "news_articles": None,
        "trump_posts": None,
        "technical_analysis": TechnicalAnalysis(
            signal=Signal.BUY,
            rsi=35.0,
            macd_hist=0.5,
            interpretation="Bullish",
            confidence=0.8,
        ),
        "sentiment_analysis": SentimentAnalysis(
            overall_sentiment="positive",
            sentiment_score=0.6,
            positive_ratio=0.7,
            negative_ratio=0.1,
            neutral_ratio=0.2,
            article_count=10,
            summary="Positive",
        ),
        "news_analysis": NewsAnalysis(
            key_themes=["Growth"],
            impact_assessment="Positive",
            recommendation="Buy",
        ),
        "trump_analysis": None,
        "fundamental_analysis": FundamentalAnalysis(
            valuation="FAIRLY_VALUED",
            pe_ratio=28.5,
            eps=6.15,
            revenue_growth_yoy=0.062,
            earnings_growth_yoy=0.102,
            debt_to_equity=2.05,
            current_ratio=0.94,
            interpretation="Solid fundamentals",
            confidence=0.75,
        ),
        "comparative_analysis": None,
        "web_research": None,
        "social_sentiment_analysis": None,
        "bullish_research": sample_bullish_research,
        "bearish_research": sample_bearish_research,
        "final_decision": None,
        "risk_assessment": None,
        "account_info": AccountInfo(
            balance=100000.0,
            available_cash=100000.0,
            positions={},
            total_exposure=0.0,
        ),
        "order_status": None,
        "regime_analysis": None,
        "strategy_selection": None,
        "sector_rotation_context": None,
        "earnings_context": None,
        "peer_analysis_context": None,
        "game_plan_context": None,
        "position_context": None,
        "broker_positions": None,
        "portfolio_value": None,
        "backtest_validation": None,
        "degradation_context": None,
        "warnings": [],
        "broker_api_failed": False,
    }

    result_state = await workflow.make_decision(state)

    assert result_state["final_decision"] is not None
    assert isinstance(result_state["final_decision"], TradingDecision)


def test_repr(mock_workflow_dependencies):
    market_fetcher, news_fetcher, llm_client, finbert, fundamental_fetcher = mock_workflow_dependencies

    workflow = TradingWorkflow(
        llm_client, market_fetcher, news_fetcher, finbert, fundamental_fetcher, use_meta_agent=False
    )

    assert repr(workflow) == "TradingWorkflow(agents=9, mode=momentum)"


async def test_execute_trade_with_broker(mock_workflow_dependencies, sample_ohlcv_data):
    """Test trade execution when broker provided and risk approved."""
    market_fetcher, news_fetcher, llm_client, finbert, fundamental_fetcher = mock_workflow_dependencies
    mock_broker = MagicMock()

    mock_order = MagicMock()
    mock_order.qty = 10
    mock_broker.submit_order.return_value = mock_order

    workflow = TradingWorkflow(
        llm_client,
        market_fetcher,
        news_fetcher,
        finbert,
        fundamental_fetcher,
        broker=mock_broker,
        use_meta_agent=False,
    )

    state: TradingState = {
        "symbol": "AAPL",
        "trading_session": TradingSession.REGULAR,
        "market_data": sample_ohlcv_data,
        "enable_multi_timeframe": False,
        "news_articles": None,
        "trump_posts": None,
        "technical_analysis": None,
        "sentiment_analysis": None,
        "news_analysis": None,
        "trump_analysis": None,
        "fundamental_analysis": None,
        "comparative_analysis": None,
        "web_research": None,
        "social_sentiment_analysis": None,
        "bullish_research": None,
        "bearish_research": None,
        "final_decision": TradingDecision(
            action=Signal.BUY, confidence=0.85, reasoning=["Test"], risk_level="LOW"
        ),
        "risk_assessment": MagicMock(
            validation=MagicMock(approved=True),
            position_sizing=MagicMock(recommended_shares=10),
            stop_loss=MagicMock(stop_loss_price=140.0),
        ),
        "account_info": None,
        "order_status": None,
        "regime_analysis": None,
        "strategy_selection": None,
        "sector_rotation_context": None,
        "earnings_context": None,
        "peer_analysis_context": None,
        "game_plan_context": None,
        "position_context": None,
        "broker_positions": None,
        "portfolio_value": None,
        "backtest_validation": None,
        "degradation_context": None,
        "warnings": [],
        "broker_api_failed": False,
    }

    result_state = await workflow._execute_trade(state)

    assert result_state["order_status"] == mock_order
    mock_broker.submit_order.assert_called_once_with(
        symbol="AAPL",
        qty=10,
        side="buy",
        stop_loss_price=140.0,
    )


async def test_execute_trade_error_handling(mock_workflow_dependencies, sample_ohlcv_data):
    """Test trade execution handles broker errors gracefully."""
    market_fetcher, news_fetcher, llm_client, finbert, fundamental_fetcher = mock_workflow_dependencies
    mock_broker = MagicMock()
    mock_broker.submit_order.side_effect = Exception("API error")

    workflow = TradingWorkflow(
        llm_client,
        market_fetcher,
        news_fetcher,
        finbert,
        fundamental_fetcher,
        broker=mock_broker,
        use_meta_agent=False,
    )

    state: TradingState = {
        "symbol": "AAPL",
        "trading_session": TradingSession.REGULAR,
        "market_data": sample_ohlcv_data,
        "enable_multi_timeframe": False,
        "news_articles": None,
        "trump_posts": None,
        "technical_analysis": None,
        "sentiment_analysis": None,
        "news_analysis": None,
        "trump_analysis": None,
        "fundamental_analysis": None,
        "comparative_analysis": None,
        "web_research": None,
        "social_sentiment_analysis": None,
        "bullish_research": None,
        "bearish_research": None,
        "final_decision": TradingDecision(
            action=Signal.BUY, confidence=0.85, reasoning=["Test"], risk_level="LOW"
        ),
        "risk_assessment": MagicMock(
            validation=MagicMock(approved=True),
            position_sizing=MagicMock(recommended_shares=10),
            stop_loss=MagicMock(stop_loss_price=140.0),
        ),
        "account_info": None,
        "order_status": None,
        "regime_analysis": None,
        "strategy_selection": None,
        "sector_rotation_context": None,
        "earnings_context": None,
        "peer_analysis_context": None,
        "game_plan_context": None,
        "position_context": None,
        "broker_positions": None,
        "portfolio_value": None,
        "backtest_validation": None,
        "degradation_context": None,
        "warnings": [],
        "broker_api_failed": False,
    }

    result_state = await workflow._execute_trade(state)

    assert result_state["order_status"] is None
    mock_broker.submit_order.assert_called_once()


async def test_account_info_passed_to_trader(
    mock_workflow_dependencies, sample_bullish_research, sample_bearish_research
):
    """Test portfolio info passed to trader for context-aware decisions."""
    market_fetcher, news_fetcher, llm_client, finbert, fundamental_fetcher = mock_workflow_dependencies

    workflow = TradingWorkflow(
        llm_client, market_fetcher, news_fetcher, finbert, fundamental_fetcher, use_meta_agent=False
    )

    state = {
        "symbol": "AAPL",
        "market_data": None,
        "news_articles": None,
        "technical_analysis": TechnicalAnalysis(
            signal=Signal.HOLD,
            rsi=50.0,
            macd_hist=0.1,
            interpretation="Neutral",
            confidence=0.6,
        ),
        "sentiment_analysis": SentimentAnalysis(
            overall_sentiment="neutral",
            sentiment_score=0.0,
            positive_ratio=0.3,
            negative_ratio=0.3,
            neutral_ratio=0.4,
            article_count=5,
            summary="Mixed",
        ),
        "news_analysis": NewsAnalysis(
            key_themes=["Stable"],
            impact_assessment="Neutral",
            recommendation="Hold",
        ),
        "fundamental_analysis": FundamentalAnalysis(
            valuation="FAIRLY_VALUED",
            pe_ratio=25.0,
            eps=5.0,
            revenue_growth_yoy=0.05,
            earnings_growth_yoy=0.08,
            debt_to_equity=1.5,
            current_ratio=1.2,
            interpretation="Stable",
            confidence=0.7,
        ),
        "comparative_analysis": None,
        "bullish_research": sample_bullish_research,
        "bearish_research": sample_bearish_research,
        "final_decision": None,
        "risk_assessment": None,
        "account_info": AccountInfo(
            balance=100000.0,
            available_cash=50000.0,
            positions={"AAPL": 100.0},
            total_exposure=50000.0,
        ),
        "order_status": None,
    }

    result_state = await workflow.make_decision(state)  # type: ignore[arg-type]

    final_decision = result_state["final_decision"]
    assert final_decision is not None
    assert final_decision.owns_position is True
    assert final_decision.position_qty == 100.0


async def test_workflow_continues_when_fundamental_rate_limited(mock_workflow_dependencies):
    """Test workflow continues with fundamental=None and captures warning when rate limited."""
    market_fetcher, news_fetcher, llm_client, finbert, fundamental_fetcher = mock_workflow_dependencies

    # Make fundamental fetcher raise rate limit error
    fundamental_fetcher.fetch_overview.side_effect = Exception("API rate limit: 5 api calls per minute")

    workflow = TradingWorkflow(
        llm_client, market_fetcher, news_fetcher, finbert, fundamental_fetcher, use_meta_agent=False
    )

    result = await workflow.analyze("AAPL", period_days=90)

    assert result.fundamental is None
    assert isinstance(result.decision, TradingDecision)
    assert isinstance(result.technical, TechnicalAnalysis)
    assert isinstance(result.sentiment, SentimentAnalysis)
    assert isinstance(result.news, NewsAnalysis)
    # Verify warning captured in result
    assert result.has_incomplete_data
    assert len(result.warnings) >= 1
    assert any("rate limit" in w.lower() for w in result.warnings)


async def test_workflow_raises_when_fundamental_fails_non_rate_limit(mock_workflow_dependencies):
    """Test workflow re-raises non-rate-limit errors from fundamental analysis."""
    market_fetcher, news_fetcher, llm_client, finbert, fundamental_fetcher = mock_workflow_dependencies

    # Make fundamental fetcher raise non-rate-limit error
    fundamental_fetcher.fetch_overview.side_effect = Exception("Invalid API key")

    workflow = TradingWorkflow(
        llm_client, market_fetcher, news_fetcher, finbert, fundamental_fetcher, use_meta_agent=False
    )

    with pytest.raises(Exception, match="Invalid API key"):
        await workflow.analyze("AAPL", period_days=90)


async def test_backtest_validation_pass(mock_workflow_dependencies):
    """Test backtest validation passes with good metrics - confidence unchanged."""
    market_fetcher, news_fetcher, llm_client, finbert, fundamental_fetcher = mock_workflow_dependencies

    config = PreTradeBacktestingConfig(
        enabled=True,
        lookback_days=180,
        min_sharpe_threshold=0.5,
        max_drawdown_threshold=0.25,
        confidence_penalty_multiplier=0.7,
    )

    workflow = TradingWorkflow(
        llm_client,
        market_fetcher,
        news_fetcher,
        finbert,
        fundamental_fetcher,
        use_meta_agent=False,
        pre_trade_backtest_config=config,
    )

    mock_backtest_result = VectorBTResult(
        sharpe_ratio=1.2,
        sortino_ratio=1.5,
        max_drawdown=-0.15,
        total_return=0.25,
        win_rate=0.60,
        profit_factor=2.0,
        total_trades=30,
        calmar_ratio=1.67,
        equity_curve=[100000, 125000],
        equity_dates=[datetime.now(UTC), datetime.now(UTC)],
        symbol="AAPL",
        start_date=datetime.now(UTC),
        end_date=datetime.now(UTC),
    )

    with patch.object(workflow.vectorbt_runner, "run_backtest", return_value=mock_backtest_result):
        result = await workflow.analyze("AAPL", period_days=90)

    assert result.backtest_validation is not None
    assert result.backtest_validation.passed is True
    assert result.backtest_validation.sharpe_ratio == 1.2
    assert result.backtest_validation.max_drawdown == -0.15
    assert result.backtest_validation.confidence_adjustment == 1.0
    assert len(result.backtest_validation.failure_reasons) == 0


async def test_backtest_validation_fail_sharpe(mock_workflow_dependencies):
    """Test backtest validation fails on low Sharpe - confidence penalized."""
    market_fetcher, news_fetcher, llm_client, finbert, fundamental_fetcher = mock_workflow_dependencies

    config = PreTradeBacktestingConfig(
        enabled=True,
        lookback_days=180,
        min_sharpe_threshold=0.5,
        max_drawdown_threshold=0.25,
        confidence_penalty_multiplier=0.7,
    )

    workflow = TradingWorkflow(
        llm_client,
        market_fetcher,
        news_fetcher,
        finbert,
        fundamental_fetcher,
        use_meta_agent=False,
        pre_trade_backtest_config=config,
    )

    mock_backtest_result = VectorBTResult(
        sharpe_ratio=0.2,
        sortino_ratio=0.3,
        max_drawdown=-0.15,
        total_return=0.10,
        win_rate=0.52,
        profit_factor=1.1,
        total_trades=20,
        calmar_ratio=0.67,
        equity_curve=[100000, 110000],
        equity_dates=[datetime.now(UTC), datetime.now(UTC)],
        symbol="AAPL",
        start_date=datetime.now(UTC),
        end_date=datetime.now(UTC),
    )

    with patch.object(workflow.vectorbt_runner, "run_backtest", return_value=mock_backtest_result):
        result = await workflow.analyze("AAPL", period_days=90)

    assert result.backtest_validation is not None
    assert result.backtest_validation.passed is False
    assert result.backtest_validation.sharpe_ratio == 0.2
    assert result.backtest_validation.confidence_adjustment == 0.7
    assert len(result.backtest_validation.failure_reasons) == 1
    assert "Sharpe" in result.backtest_validation.failure_reasons[0]
    assert any("Backtest FAILED" in w for w in result.warnings)


async def test_backtest_validation_disabled(mock_workflow_dependencies):
    """Test backtest validation disabled - no validation runs."""
    market_fetcher, news_fetcher, llm_client, finbert, fundamental_fetcher = mock_workflow_dependencies

    config = PreTradeBacktestingConfig(enabled=False)

    workflow = TradingWorkflow(
        llm_client,
        market_fetcher,
        news_fetcher,
        finbert,
        fundamental_fetcher,
        use_meta_agent=False,
        pre_trade_backtest_config=config,
    )

    result = await workflow.analyze("AAPL", period_days=90)

    assert result.backtest_validation is None
    assert workflow.vectorbt_runner is None


async def test_backtest_validation_error(mock_workflow_dependencies):
    """Test backtest validation error handling - graceful degradation."""
    market_fetcher, news_fetcher, llm_client, finbert, fundamental_fetcher = mock_workflow_dependencies

    config = PreTradeBacktestingConfig(
        enabled=True,
        lookback_days=180,
        min_sharpe_threshold=0.5,
        max_drawdown_threshold=0.25,
    )

    workflow = TradingWorkflow(
        llm_client,
        market_fetcher,
        news_fetcher,
        finbert,
        fundamental_fetcher,
        use_meta_agent=False,
        pre_trade_backtest_config=config,
    )

    with patch.object(workflow.vectorbt_runner, "run_backtest", side_effect=ValueError("Insufficient data")):
        result = await workflow.analyze("AAPL", period_days=90)

    assert result.backtest_validation is None
    assert any("Backtest error" in w for w in result.warnings)


async def test_broker_api_failure_blocks_trade(mock_workflow_dependencies):
    """Broker API failure prevents trade execution."""
    from src.data.broker import BrokerAPIError

    market_fetcher, news_fetcher, llm_client, finbert, fundamental_fetcher = mock_workflow_dependencies

    mock_broker = MagicMock()
    mock_broker.get_account_info.side_effect = BrokerAPIError("API timeout")

    workflow = TradingWorkflow(
        llm_client,
        market_fetcher,
        news_fetcher,
        finbert,
        fundamental_fetcher,
        broker=mock_broker,
        use_meta_agent=False,
    )

    # Mock trader to force BUY signal
    mock_decision = TradingDecision(
        action=Signal.BUY,
        confidence=0.85,
        reasoning=["Strong buy signal"],
        risk_level="LOW",
        owns_position=False,
        position_qty=None,
    )
    with patch.object(workflow.trader, "decide", return_value=mock_decision):
        result = await workflow.analyze("AAPL")

    assert not result.risk.validation.approved
    assert "broker_available" in result.risk.validation.constraints_met
    assert not result.risk.validation.constraints_met["broker_available"]
    assert result.order is None
    assert any("Broker API unavailable" in w for w in result.warnings)


async def test_paper_trading_unaffected(mock_workflow_dependencies):
    """Paper trading (broker=None) still uses mock data."""
    market_fetcher, news_fetcher, llm_client, finbert, fundamental_fetcher = mock_workflow_dependencies

    workflow = TradingWorkflow(
        llm_client,
        market_fetcher,
        news_fetcher,
        finbert,
        fundamental_fetcher,
        broker=None,
        use_meta_agent=False,
    )

    result = await workflow.analyze("AAPL")

    assert result.risk.account_info.balance == 100000.0
    assert not any("Broker API" in w for w in result.warnings)


async def test_order_submission_failure_handled(mock_workflow_dependencies):
    """Order submission failures handled gracefully."""
    from src.data.broker import BrokerAccountInfo, BrokerAPIError

    market_fetcher, news_fetcher, llm_client, finbert, fundamental_fetcher = mock_workflow_dependencies

    mock_broker = MagicMock()
    mock_broker.get_account_info.return_value = BrokerAccountInfo(
        balance=50000.0,
        available_cash=30000.0,
        positions={},
        total_exposure=0.0,
        portfolio_value=50000.0,
    )
    mock_broker.submit_order.side_effect = BrokerAPIError("Order rejected")

    workflow = TradingWorkflow(
        llm_client,
        market_fetcher,
        news_fetcher,
        finbert,
        fundamental_fetcher,
        broker=mock_broker,
        use_meta_agent=False,
    )

    # Mock trader to force BUY signal
    mock_decision = TradingDecision(
        action=Signal.BUY,
        confidence=0.85,
        reasoning=["Strong buy signal"],
        risk_level="LOW",
        owns_position=False,
        position_qty=None,
    )
    with patch.object(workflow.trader, "decide", return_value=mock_decision):
        result = await workflow.analyze("AAPL")

    # Risk assessment should approve (broker was available for account info)
    assert result.risk.validation.approved
    assert result.order is None
    assert any("Order submission failed" in w for w in result.warnings)
