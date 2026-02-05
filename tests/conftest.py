"""Shared pytest fixtures."""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from src.agents.bearish_researcher import BearishResearchAnalysis
from src.agents.bullish_researcher import BullishResearchAnalysis
from src.agents.comparative import ComparativeAnalysis, RelativeValuation
from src.agents.risk import AccountInfo
from src.agents.web_researcher import ResearchCategory, WebResearchAnalysis, WebResearchResult
from src.data.broker import BrokerAccountInfo, BrokerPosition, OrderStatus
from src.data.comparative import ComparativeData, PerformanceData
from src.data.comparative import StockInfo as ComparativeStockInfo
from src.data.news import NewsArticle
from src.data.truth_social import TruthPost
from src.data.universe import StockInfo as UniverseStockInfo
from src.data.universe import StockUniverse
from src.data.websearch import SearchType, WebSearchResponse
from src.data.websearch import WebSearchResult as SearchResult
from src.metrics.tracker import TradeRecord
from src.models.sentiment import SentimentScore
from src.screening.analyzer import ScreeningAnalysis
from src.screening.screener import ScreeningCriteria, ScreeningOutput, ScreeningResult
from src.strategies.momentum import Signal


@pytest.fixture
def sample_ohlcv_data():
    """Sample OHLCV data for testing."""
    return pd.DataFrame(
        {
            "Open": [100 + i for i in range(50)],
            "High": [105 + i for i in range(50)],
            "Low": [99 + i for i in range(50)],
            "Close": [104 + i for i in range(50)],
            "Volume": [1000000] * 50,
        }
    )


@pytest.fixture
def sample_news_articles():
    """Sample news articles for testing."""
    return [
        NewsArticle(
            title="Company reports strong earnings",
            description="Quarterly earnings exceed expectations",
            url="https://example.com/1",
            published_at=datetime(2024, 1, 15, 10, 0),
            source="Reuters",
        ),
        NewsArticle(
            title="New product launch announced",
            description="Company unveils innovative product line",
            url="https://example.com/2",
            published_at=datetime(2024, 1, 15, 12, 0),
            source="Bloomberg",
        ),
        NewsArticle(
            title="Market analysts upgrade rating",
            description="Analysts raise price target",
            url="https://example.com/3",
            published_at=datetime(2024, 1, 15, 14, 0),
            source="CNBC",
        ),
    ]


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    from src.models.providers.base import StructuredOutputError

    mock = MagicMock()
    mock.provider = "ollama"
    mock.model = "qwen3:14b"
    mock.complete.return_value = "Mock LLM response with analysis and high confidence."
    mock.acomplete = AsyncMock(return_value="Mock LLM response with analysis and high confidence.")

    # astructured raises StructuredOutputError to trigger fallback to acomplete
    async def astructured_side_effect(*args, **kwargs):
        msg = "Mock structured output not configured"
        raise StructuredOutputError(msg, raw_response=None)

    mock.astructured = AsyncMock(side_effect=astructured_side_effect)
    mock.supports_structured_output = True
    return mock


@pytest.fixture
def mock_finbert():
    """Mock FinBERT sentiment analyzer."""
    mock = MagicMock()
    mock.device = "cpu"
    mock.analyze.return_value = SentimentScore(
        positive=0.7,
        negative=0.1,
        neutral=0.2,
    )
    mock.analyze_batch.return_value = [
        SentimentScore(positive=0.7, negative=0.1, neutral=0.2),
        SentimentScore(positive=0.6, negative=0.2, neutral=0.2),
        SentimentScore(positive=0.8, negative=0.05, neutral=0.15),
    ]
    return mock


@pytest.fixture
def account_info():
    """Sample account info for risk testing."""
    return AccountInfo(
        balance=100000.0,
        available_cash=80000.0,
        positions={"SPY": 100.0},
        total_exposure=20000.0,
    )


@pytest.fixture
def mock_alpaca_broker():
    """Mock Alpaca broker for testing."""
    mock = MagicMock()
    mock.paper = True

    mock.get_account_info.return_value = BrokerAccountInfo(
        balance=100000.0,
        available_cash=80000.0,
        positions={
            "AAPL": BrokerPosition(
                symbol="AAPL",
                qty=10.0,
                market_value=1500.0,
                avg_entry_price=150.0,
                unrealized_pnl=50.0,
                unrealized_pnl_percent=0.033,
            )
        },
        total_exposure=1500.0,
        portfolio_value=100000.0,
    )

    mock.submit_order.return_value = OrderStatus(
        order_id="order-123",
        symbol="AAPL",
        qty=10.0,
        filled_qty=10.0,
        side="buy",
        status="filled",
        submitted_at=datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC),
        filled_at=datetime(2024, 1, 1, 10, 0, 5, tzinfo=UTC),
        filled_avg_price=150.0,
    )

    return mock


@pytest.fixture
def sample_fundamental_overview():
    """Sample fundamental overview data from Alpha Vantage."""
    return {
        "Symbol": "AAPL",
        "AssetType": "Common Stock",
        "Name": "Apple Inc",
        "PERatio": "28.5",
        "EPS": "6.15",
        "QuarterlyRevenueGrowthYOY": "0.062",
        "QuarterlyEarningsGrowthYOY": "0.102",
        "DebtToEquity": "2.05",
        "CurrentRatio": "0.94",
        "MarketCapitalization": "2850000000000",
        "EBITDA": "125000000000",
        "PriceToBookRatio": "45.2",
        "DividendYield": "0.0052",
    }


@pytest.fixture
def mock_fundamental_fetcher(sample_fundamental_overview):
    """Mock fundamental data fetcher."""
    mock = MagicMock()
    mock.api_key = "test_api_key"
    mock.fetch_overview.return_value = sample_fundamental_overview
    return mock


@pytest.fixture
def sample_bullish_research():
    """Sample bullish research analysis for testing."""
    return BullishResearchAnalysis(
        thesis=(
            "Stock shows strong momentum with improving fundamentals and positive market sentiment. "
            "Technical indicators suggest continued upward trajectory. Undervalued at current levels."
        ),
        key_strengths=[
            "Strong technical momentum with positive RSI and MACD",
            "Positive sentiment across recent news articles",
            "Undervalued relative to growth potential",
            "Strong revenue growth trajectory",
        ],
        target_upside=25.0,
        confidence=0.8,
    )


@pytest.fixture
def sample_bearish_research():
    """Sample bearish research analysis for testing."""
    return BearishResearchAnalysis(
        thesis=(
            "Stock faces headwinds with weakening fundamentals and deteriorating market sentiment. "
            "Technical indicators suggest downward pressure. Overvalued at current levels."
        ),
        key_weaknesses=[
            "Weak technical momentum with negative RSI and MACD",
            "Negative sentiment across recent news articles",
            "Overvalued relative to peers",
            "High debt-to-equity ratio",
        ],
        target_downside=20.0,
        confidence=0.7,
    )


@pytest.fixture
def sample_ohlcv_trending_up():
    """Sample OHLCV data with clear uptrend for regime testing (100 rows)."""
    import numpy as np

    np.random.seed(42)
    n = 100
    trend = np.linspace(100, 150, n)
    noise = np.random.normal(0, 1, n)
    close = trend + noise

    return pd.DataFrame(
        {
            "Open": close - np.abs(np.random.normal(0, 0.5, n)),
            "High": close + np.abs(np.random.normal(1, 0.5, n)),
            "Low": close - np.abs(np.random.normal(1, 0.5, n)),
            "Close": close,
            "Volume": [1000000 + int(np.random.normal(0, 100000)) for _ in range(n)],
        }
    )


@pytest.fixture
def sample_ohlcv_trending_down():
    """Sample OHLCV data with clear downtrend for regime testing (100 rows)."""
    import numpy as np

    np.random.seed(43)
    n = 100
    trend = np.linspace(150, 100, n)
    noise = np.random.normal(0, 1, n)
    close = trend + noise

    return pd.DataFrame(
        {
            "Open": close + np.abs(np.random.normal(0, 0.5, n)),
            "High": close + np.abs(np.random.normal(1, 0.5, n)),
            "Low": close - np.abs(np.random.normal(1, 0.5, n)),
            "Close": close,
            "Volume": [1000000 + int(np.random.normal(0, 100000)) for _ in range(n)],
        }
    )


@pytest.fixture
def sample_ohlcv_ranging():
    """Sample OHLCV data with sideways movement for regime testing (100 rows)."""
    import numpy as np

    np.random.seed(44)
    n = 100
    base = 120
    noise = np.random.normal(0, 2, n)
    close = base + noise

    return pd.DataFrame(
        {
            "Open": close - np.random.uniform(-0.5, 0.5, n),
            "High": close + np.abs(np.random.normal(0.5, 0.2, n)),
            "Low": close - np.abs(np.random.normal(0.5, 0.2, n)),
            "Close": close,
            "Volume": [1000000] * n,
        }
    )


@pytest.fixture
def sample_ohlcv_volatile():
    """Sample OHLCV data with high volatility for regime testing (100 rows)."""
    import numpy as np

    np.random.seed(45)
    n = 100
    base = 120
    # High volatility swings
    volatility = np.random.normal(0, 8, n)
    close = base + volatility.cumsum() * 0.1 + volatility

    return pd.DataFrame(
        {
            "Open": close - np.random.uniform(-2, 2, n),
            "High": close + np.abs(np.random.normal(3, 1, n)),
            "Low": close - np.abs(np.random.normal(3, 1, n)),
            "Close": close,
            "Volume": [2000000 + int(np.random.normal(0, 500000)) for _ in range(n)],
        }
    )


@pytest.fixture
def sample_trade_record():
    """Sample trade record for testing."""
    return TradeRecord(
        timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
        symbol="AAPL",
        action=Signal.BUY,
        entry_price=150.0,
        exit_price=None,
        shares=10,
        stop_loss_price=145.0,
        confidence=0.75,
        risk_level="MEDIUM",
        status="OPEN",
        pnl=None,
        pnl_percent=None,
        strategy_name="momentum",
    )


@pytest.fixture
def mock_trade_repository():
    """Mock trade repository for testing."""
    mock = MagicMock()

    trade_record = TradeRecord(
        timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
        symbol="AAPL",
        action=Signal.BUY,
        entry_price=150.0,
        exit_price=None,
        shares=10,
        stop_loss_price=145.0,
        confidence=0.75,
        risk_level="MEDIUM",
        status="OPEN",
        pnl=None,
        pnl_percent=None,
        strategy_name="momentum",
    )

    mock.create = AsyncMock(return_value=trade_record)
    mock.get_by_id = AsyncMock(return_value=trade_record)
    mock.get_open_trades = AsyncMock(return_value=[trade_record])
    mock.get_by_window = AsyncMock(return_value=[trade_record])
    mock.get_by_symbol = AsyncMock(return_value=[trade_record])
    mock.get_all = AsyncMock(return_value=[trade_record])
    mock.update = AsyncMock(return_value=trade_record)

    return mock


@pytest.fixture
def sample_comparative_data():
    """Sample comparative data for testing."""
    return ComparativeData(
        stock_info=ComparativeStockInfo(
            symbol="AAPL",
            sector="Technology",
            industry="Consumer Electronics",
            pe_ratio=28.5,
            price_to_book=45.2,
        ),
        stock_performance=PerformanceData(
            ytd_return=15.0,
            three_month_return=8.0,
        ),
        sector_etf="XLK",
        sector_pe=32.0,
        sector_performance=PerformanceData(
            ytd_return=12.0,
            three_month_return=5.0,
        ),
        market_pe=22.0,
        market_performance=PerformanceData(
            ytd_return=10.0,
            three_month_return=4.0,
        ),
        fetched_at=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
    )


@pytest.fixture
def sample_comparative_analysis():
    """Sample comparative analysis for testing."""
    return ComparativeAnalysis(
        relative_valuation=RelativeValuation.FAIRLY_VALUED,
        pe_vs_sector=0.89,
        pe_vs_market=1.29,
        perf_vs_sector_ytd=3.0,
        perf_vs_sector_3m=3.0,
        perf_vs_market_ytd=5.0,
        perf_vs_market_3m=4.0,
        sector_etf="XLK",
        interpretation="Stock is fairly valued relative to its technology sector peers.",
        confidence=0.75,
    )


@pytest.fixture
def mock_comparative_fetcher(sample_comparative_data):
    """Mock comparative data fetcher."""
    mock = MagicMock()
    mock.fetch_comparative_data.return_value = sample_comparative_data
    return mock


@pytest.fixture
def mock_snapshot_repository():
    """Mock portfolio snapshot repository for testing."""
    from src.database.repositories.snapshot import PortfolioSnapshot

    mock = MagicMock()

    snapshot = PortfolioSnapshot(
        id=str(uuid.uuid4()),
        timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
        balance=100000.0,
        available_cash=80000.0,
        total_exposure=20000.0,
        portfolio_value=100000.0,
        positions={"AAPL": 10.0},
        trigger="TRADE",
    )

    mock.create = AsyncMock(return_value=snapshot)
    mock.get_by_id = AsyncMock(return_value=snapshot)
    mock.get_latest = AsyncMock(return_value=snapshot)
    mock.get_by_date_range = AsyncMock(return_value=[snapshot])
    mock.get_by_trigger = AsyncMock(return_value=[snapshot])

    return mock


@pytest.fixture
def sample_web_search_response():
    """Sample web search response for testing."""
    return WebSearchResponse(
        query="AAPL stock news",
        search_type=SearchType.NEWS,
        results=[
            SearchResult(
                title="Apple Reports Strong Earnings",
                url="https://example.com/aapl-earnings",
                body="Apple Inc reported quarterly earnings that beat analyst expectations.",
                source="Reuters",
                published_at=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
            ),
            SearchResult(
                title="AAPL Stock Price Target Raised",
                url="https://example.com/aapl-target",
                body="Multiple analysts raised price targets for Apple stock.",
                source="Bloomberg",
                published_at=datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
            ),
        ],
        fetched_at=datetime(2024, 1, 15, 14, 0, 0, tzinfo=UTC),
    )


@pytest.fixture
def sample_web_research_analysis():
    """Sample web research analysis for testing."""
    return WebResearchAnalysis(
        symbol="AAPL",
        results=[
            WebResearchResult(
                category=ResearchCategory.LATEST_NEWS,
                summary="Apple reports strong earnings with revenue beating expectations.",
                key_findings=[
                    "Quarterly revenue exceeded analyst estimates",
                    "iPhone sales remain strong",
                    "Services revenue continues growth",
                ],
                sentiment_indication="Bullish",
                confidence=0.8,
                sources_count=5,
            ),
            WebResearchResult(
                category=ResearchCategory.MARKET_SENTIMENT,
                summary="Analysts maintain positive outlook on Apple stock.",
                key_findings=[
                    "Multiple price target upgrades",
                    "Strong institutional buying",
                ],
                sentiment_indication="Bullish",
                confidence=0.75,
                sources_count=4,
            ),
        ],
        overall_sentiment="Bullish",
        confidence=0.775,
    )


@pytest.fixture
def mock_web_search_fetcher(sample_web_search_response):
    """Mock WebSearchFetcher for testing."""
    mock = MagicMock()
    mock.search.return_value = sample_web_search_response
    mock.search_news.return_value = sample_web_search_response
    return mock


@pytest.fixture
def mock_web_search_tool(mock_web_search_fetcher):
    """Mock WebSearchTool for testing."""
    mock = MagicMock()
    mock.TOOL_NAME = "web_search"
    mock.fetcher = mock_web_search_fetcher
    mock.execute.return_value = """Search results for 'AAPL stock news' (news):

1. Apple Reports Strong Earnings
   URL: https://example.com/aapl-earnings
   Apple Inc reported quarterly earnings that beat analyst expectations.
   Source: Reuters

2. AAPL Stock Price Target Raised
   URL: https://example.com/aapl-target
   Multiple analysts raised price targets for Apple stock.
   Source: Bloomberg"""
    mock.get_tool_definition.return_value = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "search_type": {"type": "string", "enum": ["general", "news"]},
                },
                "required": ["query", "search_type"],
            },
        },
    }
    return mock


@pytest.fixture
def mock_web_researcher(sample_web_research_analysis):
    """Mock WebResearchAgent for testing."""
    mock = MagicMock()
    mock.research = AsyncMock(return_value=sample_web_research_analysis)
    return mock


@pytest.fixture
def sample_stock_universe():
    """Sample stock universe for testing."""
    return StockUniverse(
        name="TEST",
        stocks=[
            UniverseStockInfo(symbol="AAPL", name="Apple Inc.", sector="Technology", industry="Hardware"),
            UniverseStockInfo(symbol="MSFT", name="Microsoft Corp", sector="Technology", industry="Software"),
            UniverseStockInfo(symbol="JPM", name="JPMorgan Chase", sector="Financials", industry="Banks"),
        ],
        fetched_at=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
    )


@pytest.fixture
def mock_universe_fetcher(sample_stock_universe):
    """Mock StockUniverseFetcher for testing."""
    mock = MagicMock()
    mock.fetch_sp500.return_value = sample_stock_universe
    mock.fetch_nasdaq100.return_value = sample_stock_universe
    mock.fetch_combined.return_value = sample_stock_universe
    return mock


@pytest.fixture
def sample_screening_output():
    """Sample screening output for testing."""
    return ScreeningOutput(
        criteria=ScreeningCriteria.MOMENTUM,
        universe="SP500",
        results=[
            ScreeningResult(
                symbol="AAPL",
                name="Apple Inc.",
                sector="Technology",
                score=0.85,
                signal=Signal.BUY,
                metrics={"rsi": 28.5, "macd_hist": 0.15},
                reason="RSI oversold, MACD bullish",
            ),
            ScreeningResult(
                symbol="MSFT",
                name="Microsoft Corp",
                sector="Technology",
                score=0.78,
                signal=Signal.BUY,
                metrics={"rsi": 32.1, "macd_hist": 0.12},
                reason="RSI oversold, MACD bullish",
            ),
        ],
        total_screened=500,
        errors=["FAILED1"],
        screened_at=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
    )


@pytest.fixture
def sample_screening_analysis():
    """Sample screening analysis for testing."""
    return ScreeningAnalysis(
        summary="Found 2 momentum stocks with strong oversold signals.",
        top_picks=[
            "AAPL - Strongest RSI oversold signal",
            "MSFT - Solid momentum setup",
        ],
        sector_insights="Heavy concentration in Technology sector.",
        risk_factors="Correlation risk due to similar technical patterns.",
        next_steps="Research fundamental catalysts before entry.",
    )


@pytest.fixture
def sample_trump_posts():
    """Sample Trump Truth Social posts for testing."""
    return [
        TruthPost(
            id="123456",
            content="Great day for America! $TSLA going to the moon! This is a GREAT TIME TO BUY!",
            created_at=datetime(2025, 1, 15, 10, 30, tzinfo=UTC),
            likes=50000,
            reposts=10000,
            replies=5000,
            url="https://truthsocial.com/@realDonaldTrump/posts/123456",
        ),
        TruthPost(
            id="123455",
            content="Tariffs on China working great! Trade deal coming soon!",
            created_at=datetime(2025, 1, 14, 8, 0, tzinfo=UTC),
            likes=45000,
            reposts=8000,
            replies=3000,
            url="https://truthsocial.com/@realDonaldTrump/posts/123455",
        ),
        TruthPost(
            id="123454",
            content="Just had a great meeting with Apple CEO Tim Cook!",
            created_at=datetime(2025, 1, 13, 12, 0, tzinfo=UTC),
            likes=40000,
            reposts=7000,
            replies=2500,
            url="https://truthsocial.com/@realDonaldTrump/posts/123454",
        ),
    ]
