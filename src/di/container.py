"""DI container for AI Casino application."""

from pathlib import Path

from dependency_injector import containers, providers

from src.di.config import load_daemon_config
from src.di.providers import agents as agent_providers
from src.di.providers import circuit_breaker as circuit_breaker_providers
from src.di.providers import daemon as daemon_providers
from src.di.providers import data as data_providers
from src.di.providers import database as database_providers
from src.di.providers import models as model_providers
from src.di.providers import watchers as watcher_providers
from src.di.providers import workers as worker_providers
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

    # Market event queue singleton
    market_event_queue = providers.Singleton(
        database_providers.create_market_event_queue,
        database_engine=database_engine,
    )

    # Repository factories - create repos with fresh sessions per call
    analysis_repository = providers.Factory(
        database_providers.create_analysis_repository,
        database_engine=database_engine,
    )

    trade_repository = providers.Factory(
        database_providers.create_trade_repository,
        database_engine=database_engine,
    )

    signal_outcome_repository = providers.Factory(
        database_providers.create_signal_outcome_repository,
        database_engine=database_engine,
    )

    coordinator_metrics_repository = providers.Factory(
        database_providers.create_coordinator_metrics_repository,
        database_engine=database_engine,
    )

    # Circuit breaker registry - Singleton
    circuit_breaker_registry = providers.Singleton(
        circuit_breaker_providers.create_circuit_breaker_registry,
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
        circuit_breaker_registry=circuit_breaker_registry,
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

    newsdata_fetcher = providers.Singleton(
        data_providers.create_newsdata_fetcher,
        daemon_config=daemon_config,
        historical_cache=historical_cache,
    )

    finnhub_news_fetcher = providers.Singleton(
        data_providers.create_finnhub_news_fetcher,
        daemon_config=daemon_config,
        historical_cache=historical_cache,
    )

    duckduckgo_news_fetcher = providers.Singleton(
        data_providers.create_duckduckgo_news_fetcher,
        historical_cache=historical_cache,
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

    notification_service = providers.Singleton(
        data_providers.create_notification_service,
        daemon_config=daemon_config,
    )

    # Model providers — shared default LLM client (used by components without overrides)
    llm_client = providers.Factory(
        model_providers.create_llm_client,
        daemon_config=daemon_config,
        metrics_collector=None,
    )

    # Per-agent LLM clients (support model_overrides from config)
    _event_triage_llm = providers.Factory(
        model_providers.create_llm_client_for_agent, daemon_config=daemon_config, agent_name="event_triage"
    )
    _game_plan_llm = providers.Factory(
        model_providers.create_llm_client_for_agent, daemon_config=daemon_config, agent_name="game_plan"
    )
    _critic_llm = providers.Factory(
        model_providers.create_llm_client_for_agent, daemon_config=daemon_config, agent_name="critic"
    )
    _trader_llm = providers.Factory(
        model_providers.create_llm_client_for_agent, daemon_config=daemon_config, agent_name="trader"
    )
    _supervisor_llm = providers.Factory(
        model_providers.create_llm_client_for_agent, daemon_config=daemon_config, agent_name="supervisor"
    )
    _coordinator_llm = providers.Factory(
        model_providers.create_llm_client_for_agent, daemon_config=daemon_config, agent_name="coordinator"
    )
    _journal_llm = providers.Factory(
        model_providers.create_llm_client_for_agent, daemon_config=daemon_config, agent_name="journal"
    )
    _technical_llm = providers.Factory(
        model_providers.create_llm_client_for_agent, daemon_config=daemon_config, agent_name="technical"
    )
    _news_llm = providers.Factory(
        model_providers.create_llm_client_for_agent, daemon_config=daemon_config, agent_name="news"
    )
    _fundamental_llm = providers.Factory(
        model_providers.create_llm_client_for_agent, daemon_config=daemon_config, agent_name="fundamental"
    )
    _comparative_llm = providers.Factory(
        model_providers.create_llm_client_for_agent, daemon_config=daemon_config, agent_name="comparative"
    )
    _trump_llm = providers.Factory(
        model_providers.create_llm_client_for_agent, daemon_config=daemon_config, agent_name="trump"
    )
    _social_sentiment_llm = providers.Factory(
        model_providers.create_llm_client_for_agent,
        daemon_config=daemon_config,
        agent_name="social_sentiment",
    )
    _thesis_research_llm = providers.Factory(
        model_providers.create_llm_client_for_agent, daemon_config=daemon_config, agent_name="thesis_research"
    )

    finbert_sentiment = providers.Singleton(
        model_providers.create_finbert_sentiment,
        daemon_config=daemon_config,
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
        websearch_fetcher=websearch_fetcher,
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
        daemon_config=daemon_config,
        database_engine=database_engine,
    )

    quantstats_reporter = providers.Singleton(
        model_providers.create_quantstats_reporter,
        daemon_config=daemon_config,
    )

    stock_screener = providers.Singleton(
        model_providers.create_stock_screener,
        universe_fetcher=stock_universe_fetcher,
        daemon_config=daemon_config,
    )

    pre_market_screener = providers.Singleton(
        model_providers.create_pre_market_screener,
        universe_fetcher=stock_universe_fetcher,
        news_fetcher=news_fetcher,
        earnings_fetcher=earnings_fetcher,
    )

    coordinator_tool_registry = providers.Singleton(
        model_providers.create_coordinator_tool_registry,
        container=providers.Self(),
        daemon_state=None,
    )

    # Agent providers (using per-agent LLM clients for model routing)
    critic_agent = providers.Factory(
        agent_providers.create_critic_agent,
        llm_client=_critic_llm,
    )

    trader_agent = providers.Factory(
        agent_providers.create_trader_agent,
        llm_client=_trader_llm,
    )

    supervisor = providers.Singleton(
        agent_providers.create_supervisor,
        llm_client=_supervisor_llm,
    )

    event_triage_agent = providers.Factory(
        agent_providers.create_event_triage_agent,
        llm_client=_event_triage_llm,
    )

    game_plan_agent = providers.Factory(
        agent_providers.create_game_plan_agent,
        llm_client=_game_plan_llm,
        market_fetcher=market_fetcher,
    )

    trade_journal_agent = providers.Factory(
        agent_providers.create_trade_journal_agent,
        llm_client=_journal_llm,
        market_fetcher=market_fetcher,
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
        audit_repository=None,
    )

    # Coordinator agent - Factory pattern to support runtime daemon_state override
    # Note: container must be passed explicitly when calling this factory
    # (providers.Self() doesn't work reliably with Factory providers)
    coordinator_agent = providers.Factory(
        agent_providers.create_trading_coordinator,
        llm_client=_coordinator_llm,
        daemon_config=daemon_config,
    )

    # Workers (using per-agent LLM clients for model routing)
    technical_worker = providers.Factory(
        worker_providers.create_technical_worker,
        llm_client=_technical_llm,
    )

    sentiment_worker = providers.Factory(
        worker_providers.create_sentiment_worker,
        finbert=finbert_sentiment,
    )

    news_worker = providers.Factory(
        worker_providers.create_news_worker,
        llm_client=_news_llm,
    )

    fundamental_worker = providers.Singleton(
        worker_providers.create_fundamental_worker,
        llm_client=_fundamental_llm,
        fundamental_fetcher=fundamental_fetcher,
        earnings_fetcher=earnings_fetcher,
    )

    comparative_worker = providers.Factory(
        worker_providers.create_comparative_worker,
        llm_client=_comparative_llm,
        comparative_fetcher=comparative_fetcher,
    )

    web_research_worker = providers.Factory(
        worker_providers.create_web_research_worker,
        llm_client=llm_client,
        search_tool=web_search_tool,
    )

    trump_worker = providers.Factory(
        worker_providers.create_trump_worker,
        llm_client=_trump_llm,
    )

    bullish_thesis_worker = providers.Factory(
        worker_providers.create_thesis_worker,
        llm_client=_thesis_research_llm,
        direction=providers.Object("bullish"),
    )

    bearish_thesis_worker = providers.Factory(
        worker_providers.create_thesis_worker,
        llm_client=_thesis_research_llm,
        direction=providers.Object("bearish"),
    )

    social_sentiment_worker = providers.Factory(
        worker_providers.create_social_sentiment_worker,
        llm_client=_social_sentiment_llm,
        finnhub_fetcher=finnhub_fetcher,
        reddit_fetcher=reddit_fetcher,
        finbert=finbert_sentiment,
    )

    # Workflow providers (4 named variants) - Factory pattern to support runtime overrides
    # Note: container must be passed explicitly when calling these factories
    # (providers.Self() doesn't work reliably with Factory providers)
    workflow_meta = providers.Factory(
        workflow_providers.create_workflow_meta_wrapper,
        llm_client=llm_client,
        market_fetcher=market_fetcher,
        news_fetcher=news_fetcher,
        finbert_sentiment=finbert_sentiment,
        fundamental_fetcher=fundamental_fetcher,
        historical_cache=historical_cache,
        portfolio_var_calculator=portfolio_var_calculator,
        daemon_config=daemon_config,
        web_search_fetcher=websearch_fetcher,
    )

    workflow_momentum = providers.Factory(
        workflow_providers.create_workflow_momentum_wrapper,
        llm_client=llm_client,
        market_fetcher=market_fetcher,
        news_fetcher=news_fetcher,
        finbert_sentiment=finbert_sentiment,
        fundamental_fetcher=fundamental_fetcher,
        historical_cache=historical_cache,
        portfolio_var_calculator=portfolio_var_calculator,
        daemon_config=daemon_config,
        web_search_fetcher=websearch_fetcher,
    )

    workflow_trump = providers.Factory(
        workflow_providers.create_workflow_trump_wrapper,
        llm_client=llm_client,
        market_fetcher=market_fetcher,
        news_fetcher=news_fetcher,
        finbert_sentiment=finbert_sentiment,
        fundamental_fetcher=fundamental_fetcher,
        historical_cache=historical_cache,
        portfolio_var_calculator=portfolio_var_calculator,
        daemon_config=daemon_config,
        web_search_fetcher=websearch_fetcher,
    )

    workflow_full = providers.Factory(
        workflow_providers.create_workflow_full_wrapper,
        llm_client=llm_client,
        market_fetcher=market_fetcher,
        news_fetcher=news_fetcher,
        finbert_sentiment=finbert_sentiment,
        fundamental_fetcher=fundamental_fetcher,
        historical_cache=historical_cache,
        portfolio_var_calculator=portfolio_var_calculator,
        daemon_config=daemon_config,
        web_search_fetcher=websearch_fetcher,
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

    # Event watchers - Singleton (stateful background tasks)
    news_watcher = providers.Singleton(
        watcher_providers.create_news_watcher,
        historical_cache=historical_cache,
        daemon_config=daemon_config,
        container=providers.Self(),
    )

    social_watcher = providers.Singleton(
        watcher_providers.create_social_watcher,
        historical_cache=historical_cache,
        daemon_config=daemon_config,
        container=providers.Self(),
    )

    economic_calendar_watcher = providers.Singleton(
        watcher_providers.create_economic_calendar_watcher,
        daemon_config=daemon_config,
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
