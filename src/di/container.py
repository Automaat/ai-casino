"""DI container for AI Casino application."""

from pathlib import Path

from dependency_injector import containers, providers

from src.agents.base_researcher import ResearchDirection
from src.di.config import load_daemon_config
from src.di.providers import agents as agent_providers
from src.di.providers import daemon as daemon_providers
from src.di.providers import data as data_providers
from src.di.providers import database as database_providers
from src.di.providers import models as model_providers
from src.di.providers import workflows as workflow_providers


class AppContainer(containers.DeclarativeContainer):
    """Application DI container.

    Provides config, a historical cache, multiple data fetchers, and a broker.
    """

    # Config path storage
    config = providers.Configuration()

    # DaemonConfig singleton - loaded via utility
    daemon_config = providers.Singleton(
        load_daemon_config,
        config_path=config.config_path,
    )

    # Historical cache singleton - shared across all fetchers
    historical_cache = providers.Singleton(
        data_providers.create_historical_cache,
    )

    # Database engine singleton
    database_engine = providers.Singleton(
        database_providers.create_database_engine,
        daemon_config=daemon_config,
    )

    # Database repositories - Factory pattern for per-request sessions
    analysis_repository = providers.Factory(
        database_providers.create_analysis_repository,
        database_engine=database_engine,
    )

    position_repository = providers.Factory(
        database_providers.create_position_repository,
        database_engine=database_engine,
    )

    position_action_repository = providers.Factory(
        database_providers.create_position_action_repository,
        database_engine=database_engine,
    )

    discovery_repository = providers.Factory(
        database_providers.create_discovery_repository,
        database_engine=database_engine,
    )

    snapshot_repository = providers.Factory(
        database_providers.create_snapshot_repository,
        database_engine=database_engine,
    )

    trade_repository = providers.Factory(
        database_providers.create_trade_repository,
        database_engine=database_engine,
    )

    # Data fetchers - all Singleton
    market_fetcher = providers.Singleton(
        data_providers.create_market_fetcher,
        daemon_config=daemon_config,
        historical_cache=historical_cache,
    )

    yfinance_market_fetcher = providers.Singleton(
        data_providers.create_yfinance_market_fetcher,
        historical_cache=historical_cache,
    )

    news_fetcher = providers.Singleton(
        data_providers.create_news_fetcher,
        daemon_config=daemon_config,
        historical_cache=historical_cache,
    )

    fundamental_fetcher = providers.Singleton(
        data_providers.create_fundamental_fetcher,
        daemon_config=daemon_config,
        historical_cache=historical_cache,
    )

    finnhub_fetcher = providers.Singleton(
        data_providers.create_finnhub_fetcher,
        daemon_config=daemon_config,
    )

    reddit_fetcher = providers.Singleton(
        data_providers.create_reddit_fetcher,
        daemon_config=daemon_config,
        historical_cache=historical_cache,
    )

    truth_social_fetcher = providers.Singleton(
        data_providers.create_truth_social_fetcher,
        historical_cache=historical_cache,
    )

    stock_universe_fetcher = providers.Singleton(
        data_providers.create_stock_universe_fetcher,
    )

    websearch_fetcher = providers.Singleton(
        data_providers.create_websearch_fetcher,
    )

    earnings_fetcher = providers.Singleton(
        data_providers.create_earnings_fetcher,
    )

    comparative_fetcher = providers.Singleton(
        data_providers.create_comparative_fetcher,
    )

    alpaca_broker = providers.Singleton(
        data_providers.create_alpaca_broker,
        daemon_config=daemon_config,
        historical_cache=historical_cache,
    )

    # Model providers
    llm_client = providers.Factory(
        model_providers.create_llm_client,
        daemon_config=daemon_config,
        metrics_collector=None,
    )

    finbert_sentiment = providers.Singleton(
        model_providers.create_finbert_sentiment,
        device=None,
    )

    risk_metrics_calculator = providers.Singleton(
        model_providers.create_risk_metrics_calculator,
    )

    portfolio_var_calculator = providers.Singleton(
        model_providers.create_portfolio_var_calculator,
        risk_calculator=risk_metrics_calculator,
        market_fetcher=market_fetcher,
    )

    web_search_tool = providers.Singleton(
        model_providers.create_web_search_tool,
        container=providers.Self(),
    )

    market_regime_detector = providers.Singleton(
        model_providers.create_market_regime_detector,
    )

    backtest_runner = providers.Factory(
        model_providers.create_backtest_runner,
        cash=10000.0,
    )

    optuna_optimizer = providers.Factory(
        model_providers.create_optuna_optimizer,
        n_trials=50,
    )

    metrics_tracker = providers.Singleton(
        model_providers.create_metrics_tracker,
    )

    quantstats_reporter = providers.Singleton(
        model_providers.create_quantstats_reporter,
    )

    stock_screener = providers.Singleton(
        model_providers.create_stock_screener,
        universe_fetcher=stock_universe_fetcher,
        daemon_config=daemon_config,
    )

    coordinator_tool_registry = providers.Singleton(
        model_providers.create_coordinator_tool_registry,
        container=providers.Self(),
        daemon_state=None,
    )

    # Agent providers
    news_analyst = providers.Factory(
        agent_providers.create_news_analyst,
        llm_client=llm_client,
    )

    sentiment_analyst = providers.Factory(
        agent_providers.create_sentiment_analyst,
        finbert_sentiment=finbert_sentiment,
    )

    trump_analyst = providers.Factory(
        agent_providers.create_trump_analyst,
        llm_client=llm_client,
    )

    fundamental_analyst = providers.Factory(
        agent_providers.create_fundamental_analyst,
        llm_client=llm_client,
        fundamental_fetcher=fundamental_fetcher,
    )

    social_sentiment_analyst = providers.Factory(
        agent_providers.create_social_sentiment_analyst,
        llm_client=llm_client,
        finnhub_fetcher=finnhub_fetcher,
        reddit_fetcher=reddit_fetcher,
        finbert_sentiment=finbert_sentiment,
    )

    trader_agent = providers.Factory(
        agent_providers.create_trader_agent,
        llm_client=llm_client,
    )

    bullish_researcher = providers.Factory(
        agent_providers.create_thesis_researcher,
        llm_client=llm_client,
        direction=providers.Object(ResearchDirection.BULLISH),
    )

    bearish_researcher = providers.Factory(
        agent_providers.create_thesis_researcher,
        llm_client=llm_client,
        direction=providers.Object(ResearchDirection.BEARISH),
    )

    event_triage_agent = providers.Factory(
        agent_providers.create_event_triage_agent,
        llm_client=llm_client,
    )

    comparative_analyst = providers.Factory(
        agent_providers.create_comparative_analyst,
        llm_client=llm_client,
        comparative_fetcher=comparative_fetcher,
    )

    game_plan_agent = providers.Factory(
        agent_providers.create_game_plan_agent,
        llm_client=llm_client,
        market_fetcher=market_fetcher,
    )

    trade_journal_agent = providers.Factory(
        agent_providers.create_trade_journal_agent,
        llm_client=llm_client,
        market_fetcher=market_fetcher,
    )

    technical_analyst = providers.Factory(
        agent_providers.create_technical_analyst,
        llm_client=llm_client,
    )

    web_research_agent = providers.Factory(
        agent_providers.create_web_research_agent,
        llm_client=llm_client,
        search_tool=web_search_tool,
    )

    meta_agent = providers.Factory(
        agent_providers.create_meta_agent,
        llm_client=llm_client,
        regime_detector=market_regime_detector,
    )

    risk_management_agent = providers.Factory(
        agent_providers.create_risk_management_agent,
        llm_client=llm_client,
        daemon_config=daemon_config,
        portfolio_var_calculator=portfolio_var_calculator,
    )

    # Coordinator agent - Factory pattern to support runtime daemon_state override
    # Note: container must be passed explicitly when calling this factory
    # (providers.Self() doesn't work reliably with Factory providers)
    coordinator_agent = providers.Factory(
        agent_providers.create_trading_coordinator,
        llm_client=llm_client,
        daemon_config=daemon_config,
    )

    # Workflow providers (4 named variants) - Factory pattern to support runtime overrides
    # Note: container must be passed explicitly when calling these factories
    # (providers.Self() doesn't work reliably with Factory providers)
    workflow_meta = providers.Factory(
        workflow_providers.create_workflow_meta,
        llm_client=llm_client,
        market_fetcher=market_fetcher,
        news_fetcher=news_fetcher,
        finbert_sentiment=finbert_sentiment,
        fundamental_fetcher=fundamental_fetcher,
        historical_cache=historical_cache,
        portfolio_var_calculator=portfolio_var_calculator,
        daemon_config=daemon_config,
    )

    workflow_momentum = providers.Factory(
        workflow_providers.create_workflow_momentum,
        llm_client=llm_client,
        market_fetcher=market_fetcher,
        news_fetcher=news_fetcher,
        finbert_sentiment=finbert_sentiment,
        fundamental_fetcher=fundamental_fetcher,
        historical_cache=historical_cache,
        portfolio_var_calculator=portfolio_var_calculator,
        daemon_config=daemon_config,
    )

    workflow_trump = providers.Factory(
        workflow_providers.create_workflow_trump,
        llm_client=llm_client,
        market_fetcher=market_fetcher,
        news_fetcher=news_fetcher,
        finbert_sentiment=finbert_sentiment,
        fundamental_fetcher=fundamental_fetcher,
        historical_cache=historical_cache,
        portfolio_var_calculator=portfolio_var_calculator,
        daemon_config=daemon_config,
    )

    workflow_full = providers.Factory(
        workflow_providers.create_workflow_full,
        llm_client=llm_client,
        market_fetcher=market_fetcher,
        news_fetcher=news_fetcher,
        finbert_sentiment=finbert_sentiment,
        fundamental_fetcher=fundamental_fetcher,
        historical_cache=historical_cache,
        portfolio_var_calculator=portfolio_var_calculator,
        daemon_config=daemon_config,
    )

    # Daemon component providers
    # Note: container must be passed explicitly when calling these factories
    # (providers.Self() doesn't work reliably with Factory providers)
    daemon_factory = providers.Factory(
        daemon_providers.create_daemon_factory,
        daemon_config=daemon_config,
    )

    context_builder = providers.Factory(
        daemon_providers.create_context_builder,
    )

    task_service = providers.Factory(
        daemon_providers.create_task_service,
    )


def create_container(config_path: str | Path | None = None) -> AppContainer:
    """Create dependency injection container.

    Args:
        config_path: Optional path to daemon.yaml (supports ~ expansion)

    Returns:
        Configured Container instance
    """
    container = AppContainer()

    if config_path:
        # Expand ~ and resolve to absolute path
        normalized_path = Path(config_path).expanduser().resolve()
        container.config.from_dict({"config_path": normalized_path})

    return container
