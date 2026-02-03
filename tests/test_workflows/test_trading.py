"""Tests for trading workflow."""

from unittest.mock import MagicMock

import pytest

from src.agents.bullish_researcher import BullishResearchAnalysis
from src.agents.fundamental import FundamentalAnalysis
from src.agents.news import NewsAnalysis
from src.agents.risk import AccountInfo, RiskAssessment
from src.agents.sentiment import SentimentAnalysis
from src.agents.technical import TechnicalAnalysis
from src.agents.trader import TradingDecision
from src.data.market import MarketData
from src.strategies.momentum import Signal
from src.workflows.trading import TradingWorkflow, TradingWorkflowResult


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

    workflow = TradingWorkflow(llm_client, market_fetcher, news_fetcher, finbert, fundamental_fetcher)

    assert workflow.market_fetcher == market_fetcher
    assert workflow.news_fetcher == news_fetcher
    assert workflow.technical_analyst is not None
    assert workflow.sentiment_analyst is not None
    assert workflow.news_analyst is not None
    assert workflow.fundamental_analyst is not None
    assert workflow.bullish_researcher is not None
    assert workflow.trader is not None
    assert workflow.risk_manager is not None


def test_trading_workflow_analyze(mock_workflow_dependencies):
    market_fetcher, news_fetcher, llm_client, finbert, fundamental_fetcher = mock_workflow_dependencies

    workflow = TradingWorkflow(llm_client, market_fetcher, news_fetcher, finbert, fundamental_fetcher)

    result = workflow.analyze("AAPL", period_days=90)

    assert isinstance(result, TradingWorkflowResult)
    assert result.symbol == "AAPL"
    assert isinstance(result.technical, TechnicalAnalysis)
    assert isinstance(result.sentiment, SentimentAnalysis)
    assert isinstance(result.news, NewsAnalysis)
    assert isinstance(result.fundamental, FundamentalAnalysis)
    assert isinstance(result.bullish, BullishResearchAnalysis)
    assert isinstance(result.decision, TradingDecision)
    assert isinstance(result.risk, RiskAssessment)
    assert result.risk.validation is not None

    market_fetcher.fetch_daily.assert_called_once_with("AAPL", 90)
    news_fetcher.fetch_company_news.assert_called_once_with("AAPL", limit=10)


def test_fetch_data(mock_workflow_dependencies):
    market_fetcher, news_fetcher, llm_client, finbert, fundamental_fetcher = mock_workflow_dependencies

    workflow = TradingWorkflow(llm_client, market_fetcher, news_fetcher, finbert, fundamental_fetcher)

    state = workflow._fetch_data("AAPL", 90)

    assert state["symbol"] == "AAPL"
    assert state["market_data"] is not None
    assert state["news_articles"] is not None
    assert len(state["news_articles"]) > 0


def test_run_technical_analysis(mock_workflow_dependencies, sample_ohlcv_data):
    market_fetcher, news_fetcher, llm_client, finbert, fundamental_fetcher = mock_workflow_dependencies

    workflow = TradingWorkflow(llm_client, market_fetcher, news_fetcher, finbert, fundamental_fetcher)

    state = {
        "symbol": "AAPL",
        "market_data": sample_ohlcv_data,
        "news_articles": [],
        "technical_analysis": None,
        "sentiment_analysis": None,
        "news_analysis": None,
        "final_decision": None,
        "risk_assessment": None,
        "account_info": None,
    }

    result_state = workflow._run_technical_analysis(state)

    assert result_state["technical_analysis"] is not None
    assert isinstance(result_state["technical_analysis"], TechnicalAnalysis)


def test_run_sentiment_analysis(mock_workflow_dependencies, sample_news_articles):
    market_fetcher, news_fetcher, llm_client, finbert, fundamental_fetcher = mock_workflow_dependencies

    workflow = TradingWorkflow(llm_client, market_fetcher, news_fetcher, finbert, fundamental_fetcher)

    state = {
        "symbol": "AAPL",
        "market_data": None,
        "news_articles": sample_news_articles,
        "technical_analysis": None,
        "sentiment_analysis": None,
        "news_analysis": None,
        "final_decision": None,
        "risk_assessment": None,
        "account_info": None,
    }

    result_state = workflow._run_sentiment_analysis(state)

    assert result_state["sentiment_analysis"] is not None
    assert isinstance(result_state["sentiment_analysis"], SentimentAnalysis)


def test_make_decision(mock_workflow_dependencies, sample_bullish_research):
    market_fetcher, news_fetcher, llm_client, finbert, fundamental_fetcher = mock_workflow_dependencies

    workflow = TradingWorkflow(llm_client, market_fetcher, news_fetcher, finbert, fundamental_fetcher)

    state = {
        "symbol": "AAPL",
        "market_data": None,
        "news_articles": None,
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
        "bullish_research": sample_bullish_research,
        "final_decision": None,
        "risk_assessment": None,
        "account_info": AccountInfo(
            balance=100000.0,
            available_cash=100000.0,
            positions={},
            total_exposure=0.0,
        ),
        "order_status": None,
    }

    result_state = workflow._make_decision(state)

    assert result_state["final_decision"] is not None
    assert isinstance(result_state["final_decision"], TradingDecision)


def test_repr(mock_workflow_dependencies):
    market_fetcher, news_fetcher, llm_client, finbert, fundamental_fetcher = mock_workflow_dependencies

    workflow = TradingWorkflow(llm_client, market_fetcher, news_fetcher, finbert, fundamental_fetcher)

    assert repr(workflow) == "TradingWorkflow(agents=7)"


def test_execute_trade_with_broker(mock_workflow_dependencies, sample_ohlcv_data):
    """Test trade execution when broker provided and risk approved."""
    market_fetcher, news_fetcher, llm_client, finbert, fundamental_fetcher = mock_workflow_dependencies
    mock_broker = MagicMock()

    mock_order = MagicMock()
    mock_order.qty = 10
    mock_broker.submit_order.return_value = mock_order

    workflow = TradingWorkflow(
        llm_client, market_fetcher, news_fetcher, finbert, fundamental_fetcher, broker=mock_broker
    )

    state = {
        "symbol": "AAPL",
        "market_data": sample_ohlcv_data,
        "news_articles": None,
        "technical_analysis": None,
        "sentiment_analysis": None,
        "news_analysis": None,
        "final_decision": TradingDecision(
            action=Signal.BUY, confidence=0.85, reasoning="Test", risk_level="LOW"
        ),
        "risk_assessment": MagicMock(
            validation=MagicMock(approved=True),
            position_sizing=MagicMock(recommended_shares=10),
            stop_loss=MagicMock(stop_loss_price=140.0),
        ),
        "account_info": None,
        "order_status": None,
    }

    result_state = workflow._execute_trade(state)

    assert result_state["order_status"] == mock_order
    mock_broker.submit_order.assert_called_once_with(
        symbol="AAPL",
        qty=10,
        side="buy",
        stop_loss_price=140.0,
    )


def test_execute_trade_error_handling(mock_workflow_dependencies, sample_ohlcv_data):
    """Test trade execution handles broker errors gracefully."""
    market_fetcher, news_fetcher, llm_client, finbert, fundamental_fetcher = mock_workflow_dependencies
    mock_broker = MagicMock()
    mock_broker.submit_order.side_effect = Exception("API error")

    workflow = TradingWorkflow(
        llm_client, market_fetcher, news_fetcher, finbert, fundamental_fetcher, broker=mock_broker
    )

    state = {
        "symbol": "AAPL",
        "market_data": sample_ohlcv_data,
        "news_articles": None,
        "technical_analysis": None,
        "sentiment_analysis": None,
        "news_analysis": None,
        "final_decision": TradingDecision(
            action=Signal.BUY, confidence=0.85, reasoning="Test", risk_level="LOW"
        ),
        "risk_assessment": MagicMock(
            validation=MagicMock(approved=True),
            position_sizing=MagicMock(recommended_shares=10),
            stop_loss=MagicMock(stop_loss_price=140.0),
        ),
        "account_info": None,
        "order_status": None,
    }

    result_state = workflow._execute_trade(state)

    assert result_state["order_status"] is None
    mock_broker.submit_order.assert_called_once()


def test_account_info_passed_to_trader(mock_workflow_dependencies, sample_bullish_research):
    """Test portfolio info passed to trader for context-aware decisions."""
    market_fetcher, news_fetcher, llm_client, finbert, fundamental_fetcher = mock_workflow_dependencies

    workflow = TradingWorkflow(llm_client, market_fetcher, news_fetcher, finbert, fundamental_fetcher)

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
        "bullish_research": sample_bullish_research,
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

    result_state = workflow._make_decision(state)

    assert result_state["final_decision"].owns_position is True
    assert result_state["final_decision"].position_qty == 100.0
