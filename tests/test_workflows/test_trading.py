"""Tests for trading workflow."""

from unittest.mock import MagicMock

import pytest

from src.agents.news import NewsAnalysis
from src.agents.sentiment import SentimentAnalysis
from src.agents.technical import TechnicalAnalysis
from src.agents.trader import TradingDecision
from src.data.market import MarketData
from src.strategies.momentum import Signal
from src.workflows.trading import TradingWorkflow, TradingWorkflowResult


@pytest.fixture
def mock_workflow_dependencies(
    mock_llm_client, mock_finbert, sample_ohlcv_data, sample_news_articles
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

    return market_fetcher, news_fetcher, mock_llm_client, mock_finbert


def test_trading_workflow_init(mock_workflow_dependencies):
    market_fetcher, news_fetcher, llm_client, finbert = mock_workflow_dependencies

    workflow = TradingWorkflow(llm_client, market_fetcher, news_fetcher, finbert)

    assert workflow.market_fetcher == market_fetcher
    assert workflow.news_fetcher == news_fetcher
    assert workflow.technical_analyst is not None
    assert workflow.sentiment_analyst is not None
    assert workflow.news_analyst is not None
    assert workflow.trader is not None


def test_trading_workflow_analyze(mock_workflow_dependencies):
    market_fetcher, news_fetcher, llm_client, finbert = mock_workflow_dependencies

    workflow = TradingWorkflow(llm_client, market_fetcher, news_fetcher, finbert)

    result = workflow.analyze("AAPL", period_days=90)

    assert isinstance(result, TradingWorkflowResult)
    assert result.symbol == "AAPL"
    assert isinstance(result.technical, TechnicalAnalysis)
    assert isinstance(result.sentiment, SentimentAnalysis)
    assert isinstance(result.news, NewsAnalysis)
    assert isinstance(result.decision, TradingDecision)

    market_fetcher.fetch_daily.assert_called_once_with("AAPL", 90)
    news_fetcher.fetch_company_news.assert_called_once_with("AAPL", limit=10)


def test_fetch_data(mock_workflow_dependencies):
    market_fetcher, news_fetcher, llm_client, finbert = mock_workflow_dependencies

    workflow = TradingWorkflow(llm_client, market_fetcher, news_fetcher, finbert)

    state = workflow._fetch_data("AAPL", 90)

    assert state["symbol"] == "AAPL"
    assert state["market_data"] is not None
    assert state["news_articles"] is not None
    assert len(state["news_articles"]) > 0


def test_run_technical_analysis(mock_workflow_dependencies, sample_ohlcv_data):
    market_fetcher, news_fetcher, llm_client, finbert = mock_workflow_dependencies

    workflow = TradingWorkflow(llm_client, market_fetcher, news_fetcher, finbert)

    state = {
        "symbol": "AAPL",
        "market_data": sample_ohlcv_data,
        "news_articles": [],
        "technical_analysis": None,
        "sentiment_analysis": None,
        "news_analysis": None,
        "final_decision": None,
    }

    result_state = workflow._run_technical_analysis(state)

    assert result_state["technical_analysis"] is not None
    assert isinstance(result_state["technical_analysis"], TechnicalAnalysis)


def test_run_sentiment_analysis(mock_workflow_dependencies, sample_news_articles):
    market_fetcher, news_fetcher, llm_client, finbert = mock_workflow_dependencies

    workflow = TradingWorkflow(llm_client, market_fetcher, news_fetcher, finbert)

    state = {
        "symbol": "AAPL",
        "market_data": None,
        "news_articles": sample_news_articles,
        "technical_analysis": None,
        "sentiment_analysis": None,
        "news_analysis": None,
        "final_decision": None,
    }

    result_state = workflow._run_sentiment_analysis(state)

    assert result_state["sentiment_analysis"] is not None
    assert isinstance(result_state["sentiment_analysis"], SentimentAnalysis)


def test_make_decision(mock_workflow_dependencies):
    market_fetcher, news_fetcher, llm_client, finbert = mock_workflow_dependencies

    workflow = TradingWorkflow(llm_client, market_fetcher, news_fetcher, finbert)

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
        "final_decision": None,
    }

    result_state = workflow._make_decision(state)

    assert result_state["final_decision"] is not None
    assert isinstance(result_state["final_decision"], TradingDecision)


def test_repr(mock_workflow_dependencies):
    market_fetcher, news_fetcher, llm_client, finbert = mock_workflow_dependencies

    workflow = TradingWorkflow(llm_client, market_fetcher, news_fetcher, finbert)

    assert repr(workflow) == "TradingWorkflow(agents=4)"
