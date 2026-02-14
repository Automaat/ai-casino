"""Test container factory and mock utilities for DI testing."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pandas as pd

from src.daemon.config import ApiKeysConfig, DaemonConfig
from src.data.broker import BrokerAccountInfo, BrokerPosition, OrderStatus
from src.data.comparative import ComparativeData, PerformanceData
from src.data.comparative import StockInfo as ComparativeStockInfo
from src.data.market import MarketData
from src.di.container import AppContainer
from src.models.sentiment import SentimentScore


def create_test_config() -> DaemonConfig:
    """Create minimal test daemon config.

    Returns:
        DaemonConfig with test API keys
    """
    return DaemonConfig(
        api_keys=ApiKeysConfig(
            alpha_vantage_api_key="test_av_key",
            marketaux_api_key="test_marketaux_key",
        ),
    )


def create_test_container(
    config_overrides: DaemonConfig | None = None,
    temp_cache_path: Path | None = None,
    override_llm: bool = True,
    override_finbert: bool = True,
    override_fetchers: bool = True,
    override_broker: bool = False,
) -> AppContainer:
    """Create test container with common mock overrides.

    Args:
        config_overrides: Custom DaemonConfig (uses create_test_config() if None)
        temp_cache_path: Path for test cache DB (no cache override if None)
        override_llm: Override llm_client Factory with mock
        override_finbert: Override finbert_sentiment Singleton with mock
        override_fetchers: Override all data fetchers with mocks
        override_broker: Override alpaca_broker with mock

    Returns:
        AppContainer with specified overrides applied
    """
    container = AppContainer()

    # Config override
    config = config_overrides or create_test_config()
    container.daemon_config.override(config)

    # Cache override (if temp path provided)
    if temp_cache_path:
        from src.cache.historical import HistoricalCache

        test_cache = HistoricalCache(db_path=str(temp_cache_path))
        container.historical_cache.override(test_cache)

    # LLM client override (Factory pattern - override provider directly)
    if override_llm:
        from dependency_injector import providers

        mock_llm = create_mock_llm_client()
        container.llm_client.override(providers.Factory(lambda: mock_llm))

    # FinBERT override (Singleton pattern)
    if override_finbert:
        mock_finbert = create_mock_finbert()
        container.finbert_sentiment.override(mock_finbert)

    # Fetcher overrides (all Singleton pattern)
    if override_fetchers:
        container.market_fetcher.override(create_mock_market_fetcher())
        container.yfinance_market_fetcher.override(create_mock_market_fetcher())
        container.news_fetcher.override(create_mock_news_fetcher())
        container.fundamental_fetcher.override(create_mock_fundamental_fetcher())
        container.finnhub_fetcher.override(create_mock_finnhub_fetcher())
        container.reddit_fetcher.override(create_mock_reddit_fetcher())
        container.truth_social_fetcher.override(create_mock_truth_social_fetcher())
        container.websearch_fetcher.override(create_mock_web_search_fetcher())
        container.earnings_fetcher.override(create_mock_earnings_fetcher())
        container.comparative_fetcher.override(create_mock_comparative_fetcher())

    # Broker override (Singleton pattern)
    if override_broker:
        container.alpaca_broker.override(create_mock_alpaca_broker())

    # Tool component overrides (always override these for tool tests)
    if override_fetchers:  # Only override if we're in full test mode
        container.backtest_runner.override(create_mock_backtest_runner())
        container.optuna_optimizer.override(create_mock_optuna_optimizer())
        container.metrics_tracker.override(create_mock_metrics_tracker())
        container.quantstats_reporter.override(create_mock_quantstats_reporter())
        container.stock_screener.override(create_mock_stock_screener())

    # Risk audit repository override (mock for unit tests)
    container.risk_audit_repository.override(create_mock_risk_audit_repository())

    return container


def reset_test_container(container: AppContainer, providers: list[str] | None = None) -> None:
    """Reset provider overrides for test isolation.

    Args:
        container: AppContainer to reset
        providers: List of provider names to reset (resets all if None)
    """
    all_providers = [
        "daemon_config",
        "historical_cache",
        "database_engine",
        "risk_audit_repository",
        "llm_client",
        "finbert_sentiment",
        "market_fetcher",
        "yfinance_market_fetcher",
        "news_fetcher",
        "fundamental_fetcher",
        "finnhub_fetcher",
        "reddit_fetcher",
        "truth_social_fetcher",
        "websearch_fetcher",
        "earnings_fetcher",
        "comparative_fetcher",
        "alpaca_broker",
        "backtest_runner",
        "optuna_optimizer",
        "metrics_tracker",
        "quantstats_reporter",
        "stock_screener",
    ]

    providers_to_reset = providers or all_providers

    for provider_name in providers_to_reset:
        if hasattr(container, provider_name):
            provider = getattr(container, provider_name)
            provider.reset_override()


# Mock creation utilities


def create_mock_llm_client() -> MagicMock:
    """Create mock LLM client for testing.

    Returns:
        Mock with acomplete/astructured configured
    """
    from src.models.providers.base import StructuredOutputError

    mock = MagicMock()
    mock.provider = "ollama"
    mock.model = "qwen3:14b"
    mock.complete.return_value = "Mock LLM response with analysis and high confidence."
    mock.acomplete = AsyncMock(return_value="Mock LLM response with analysis and high confidence.")

    # astructured raises StructuredOutputError to trigger fallback
    async def astructured_side_effect(*args: Any, **kwargs: Any) -> None:
        msg = "Mock structured output not configured"
        raise StructuredOutputError(msg, raw_response=None)

    mock.astructured = AsyncMock(side_effect=astructured_side_effect)
    mock.supports_structured_output = True
    return mock


def create_mock_finbert() -> MagicMock:
    """Create mock FinBERT sentiment analyzer.

    Returns:
        Mock with analyze/analyze_batch methods
    """
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


def create_mock_market_fetcher() -> MagicMock:
    """Create mock MarketDataFetcher returning canned OHLCV data.

    Returns:
        Mock with fetch_daily method
    """
    mock = MagicMock()

    def fetch_daily(symbol: str, period_days: int = 90) -> MarketData:
        prices = {"AAPL": (150.0, 155.0), "TSLA": (200.0, 195.0), "GOOGL": (140.0, 140.5)}
        open_price, close_price = prices.get(symbol, (100.0, 101.0))
        # Generate 50 rows of data (minimum for MACD/regime detection)
        df = pd.DataFrame(
            {
                "Open": [open_price + i * 0.5 for i in range(50)],
                "High": [max(open_price, close_price) + 2 + i * 0.5 for i in range(50)],
                "Low": [min(open_price, close_price) - 2 + i * 0.5 for i in range(50)],
                "Close": [close_price + i * 0.5 for i in range(50)],
                "Volume": [1000000] * 50,
            }
        )
        return MarketData(symbol=symbol, data=df, last_updated=datetime(2024, 1, 15, 16, 0))

    mock.fetch_daily = MagicMock(side_effect=fetch_daily)
    mock.fetch_intraday = MagicMock(side_effect=fetch_daily)
    return mock


def create_mock_news_fetcher() -> MagicMock:
    """Create mock NewsDataFetcher.

    Returns:
        Mock with fetch_news and fetch_company_news methods
    """
    mock = MagicMock()
    mock.api_key = "test_news_key"
    mock.fetch_news.return_value = []
    mock.fetch_company_news.return_value = []
    return mock


def create_mock_fundamental_fetcher() -> MagicMock:
    """Create mock FundamentalDataFetcher.

    Returns:
        Mock with fetch_overview method
    """
    mock = MagicMock()
    mock.api_key = "test_fundamental_key"
    mock.fetch_overview.return_value = {
        "Symbol": "AAPL",
        "PERatio": "28.5",
        "EPS": "6.15",
        "MarketCapitalization": "2850000000000",
        "QuarterlyRevenueGrowthYOY": "0.062",
        "QuarterlyEarningsGrowthYOY": "0.102",
        "DebtToEquity": "2.05",
        "CurrentRatio": "0.94",
    }
    return mock


def create_mock_finnhub_fetcher() -> MagicMock:
    """Create mock FinnhubDataFetcher.

    Returns:
        Mock with fetch_social_sentiment method
    """
    mock = MagicMock()
    mock.api_key = "test_finnhub_key"
    mock.fetch_social_sentiment.return_value = {"reddit": 0.5, "twitter": 0.6}
    return mock


def create_mock_reddit_fetcher() -> MagicMock:
    """Create mock RedditDataFetcher.

    Returns:
        Mock with fetch_trending_stocks method
    """
    mock = MagicMock()
    mock.fetch_trending_stocks.return_value = []
    return mock


def create_mock_truth_social_fetcher() -> MagicMock:
    """Create mock TruthSocialFetcher.

    Returns:
        Mock with fetch_recent(hours=...) method returning TrumpPostData-like object
    """
    mock = MagicMock()
    # Maintain backward compatibility
    mock.fetch_recent_posts.return_value = []
    # Match real TruthSocialFetcher API: fetch_recent(hours=...) -> TrumpPostData
    mock.fetch_recent.return_value = MagicMock(posts=[])
    return mock


def create_mock_web_search_fetcher() -> MagicMock:
    """Create mock WebSearchFetcher.

    Returns:
        Mock with search/search_news methods
    """
    mock = MagicMock()
    mock.search.return_value = MagicMock(results=[])
    mock.search_news.return_value = MagicMock(results=[])
    return mock


def create_mock_earnings_fetcher() -> MagicMock:
    """Create mock EarningsDataFetcher.

    Returns:
        Mock with fetch_earnings_calendar method
    """
    mock = MagicMock()
    mock.fetch_earnings_calendar.return_value = []
    return mock


def create_mock_comparative_fetcher() -> MagicMock:
    """Create mock ComparativeDataFetcher.

    Returns:
        Mock with fetch_comparative_data method
    """
    mock = MagicMock()
    mock.fetch_comparative_data.return_value = ComparativeData(
        stock_info=ComparativeStockInfo(
            symbol="AAPL",
            sector="Technology",
            industry="Consumer Electronics",
            pe_ratio=28.5,
            price_to_book=45.2,
        ),
        stock_performance=PerformanceData(ytd_return=15.0, three_month_return=8.0),
        sector_etf="XLK",
        sector_pe=32.0,
        sector_performance=PerformanceData(ytd_return=12.0, three_month_return=5.0),
        market_pe=22.0,
        market_performance=PerformanceData(ytd_return=10.0, three_month_return=4.0),
        fetched_at=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
    )
    return mock


def create_mock_alpaca_broker() -> MagicMock:
    """Create mock AlpacaBroker.

    Returns:
        Mock with get_account_info/submit_order methods
    """
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


def create_mock_backtest_runner() -> MagicMock:
    """Create mock BacktestRunner.

    Returns:
        Mock with run_backtest method
    """
    from src.backtesting.runner import BacktestResult
    from src.metrics.tracker import TradeRecord
    from src.strategies.signal import Signal

    mock = MagicMock()
    mock.cash = 100000.0
    mock.commission = 0.002

    mock.run_backtest.return_value = BacktestResult(
        symbol="AAPL",
        start_date=datetime(2023, 1, 1, tzinfo=UTC),
        end_date=datetime(2024, 1, 1, tzinfo=UTC),
        total_return=0.2534,
        sharpe_ratio=1.45,
        max_drawdown=-0.0823,
        win_rate=0.62,
        total_trades=48,
        avg_return_per_trade=0.0053,
        trades=[
            TradeRecord(
                timestamp=datetime(2023, 1, 5, tzinfo=UTC),
                symbol="AAPL",
                action=Signal.BUY,
                entry_price=150.0,
                exit_price=155.0,
                shares=10,
                stop_loss_price=145.0,
                confidence=0.8,
                risk_level="LOW",
                status="CLOSED",
                pnl=50.0,
                pnl_percent=0.033,
            )
        ],
    )

    return mock


def create_mock_optuna_optimizer() -> MagicMock:
    """Create mock OptunaOptimizer.

    Returns:
        Mock with optimize method
    """
    from src.optimization.results import OptimizationResult

    mock = MagicMock()
    mock.n_trials = 50

    mock.optimize.return_value = OptimizationResult(
        strategy_name="momentum",
        symbol="AAPL",
        best_params={"rsi_period": 14, "macd_fast": 12, "macd_slow": 26},
        best_metrics={"sharpe_ratio": 1.87, "total_return": 0.3245, "max_drawdown": 0.0912},
        total_trials=50,
        optimization_time_seconds=42.3,
    )

    return mock


def create_mock_metrics_tracker() -> MagicMock:
    """Create mock MetricsTracker.

    Returns:
        Mock with trades property
    """
    mock = MagicMock()
    mock.risk_free_rate = 0.02
    mock.trades = []

    return mock


def create_mock_quantstats_reporter() -> MagicMock:
    """Create mock QuantStatsReporter.

    Returns:
        Mock with generate_tearsheet method
    """
    from src.metrics.tracker import TearSheet

    mock = MagicMock()
    mock.risk_free_rate = 0.02

    mock.generate_tearsheet.return_value = TearSheet(
        symbol="AAPL",
        start_date=datetime(2023, 1, 1, tzinfo=UTC),
        end_date=datetime(2024, 1, 1, tzinfo=UTC),
        cagr=0.1523,
        sharpe_ratio=1.34,
        sortino_ratio=1.89,
        calmar_ratio=2.15,
        max_drawdown=-0.0712,
        max_drawdown_duration_days=15,
        volatility_annual=0.1845,
        win_rate=0.58,
        profit_factor=1.67,
        avg_win=0.025,
        avg_loss=-0.018,
        best_day=0.045,
        worst_day=-0.038,
        monthly_returns={"2023-01": 0.05, "2023-02": 0.03},
        benchmark_symbol="SPY",
        benchmark_cagr=0.1234,
        benchmark_sharpe=1.12,
        alpha=0.0289,
        beta=0.85,
        html_report_path="/home/user/.ai-casino/tearsheets/AAPL_20240101.html",
        generated_at=datetime(2024, 1, 1, tzinfo=UTC),
    )

    return mock


def create_mock_stock_screener() -> MagicMock:
    """Create mock StockScreener.

    Returns:
        Mock with screen method
    """
    from src.screening.screener import ScreeningCriteria, ScreeningOutput, ScreeningResult
    from src.strategies.signal import Signal

    mock = MagicMock()

    mock.screen.return_value = ScreeningOutput(
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
            )
        ],
        total_screened=500,
        errors=[],
        screened_at=datetime.now(UTC),
    )

    return mock


def create_mock_risk_audit_repository() -> MagicMock:
    """Create mock RiskAuditRepository.

    Returns:
        Mock with log_decision method
    """
    mock = MagicMock()
    mock.log_decision = AsyncMock(return_value=None)
    return mock
