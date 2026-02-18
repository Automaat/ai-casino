"""Agent providers for DI container."""

from typing import TYPE_CHECKING

from src.daemon.config import DaemonConfig
from src.data.market import MarketDataFetcher
from src.models.llm import LLMClient

if TYPE_CHECKING:
    from src.agents.critic import CriticAgent
    from src.agents.event_triage import EventTriageAgent
    from src.agents.game_plan import GamePlanAgent
    from src.agents.journal import TradeJournalAgent
    from src.agents.meta import MetaAgent
    from src.agents.risk import RiskManagementAgent
    from src.agents.supervisor import TradingSupervisor
    from src.agents.trader import TraderAgent
    from src.coordinator.agent import TradingCoordinator
    from src.coordinator.confirmation import TradeConfirmationHandler
    from src.coordinator.pattern_analyzer import PatternAnalyzer
    from src.daemon.notification_channels import TelegramChannel
    from src.daemon.state import DaemonState
    from src.daemon.threshold_adapter import AdaptiveThresholdManager
    from src.database.repositories.risk_audit import RiskAuditRepository
    from src.di.container import AppContainer
    from src.metrics.portfolio_var import PortfolioVaRCalculator
    from src.strategies.regime import MarketRegimeDetector


def create_critic_agent(llm_client: LLMClient) -> CriticAgent:
    """Create CriticAgent with LLM client.

    Args:
        llm_client: LLM client for decision evaluation

    Returns:
        Configured CriticAgent
    """
    from src.agents.critic import CriticAgent

    return CriticAgent(llm_client)


def create_trader_agent(llm_client: LLMClient) -> TraderAgent:
    """Create TraderAgent with LLM client.

    Args:
        llm_client: LLM client for trading decisions

    Returns:
        Configured TraderAgent
    """
    from src.agents.trader import TraderAgent

    return TraderAgent(llm_client)


def create_event_triage_agent(llm_client: LLMClient) -> EventTriageAgent:
    """Create EventTriageAgent with LLM client.

    Args:
        llm_client: LLM client for event triage

    Returns:
        Configured EventTriageAgent
    """
    from src.agents.event_triage import EventTriageAgent

    return EventTriageAgent(llm_client)


def create_game_plan_agent(
    llm_client: LLMClient,
    market_fetcher: MarketDataFetcher,
) -> GamePlanAgent:
    """Create GamePlanAgent with LLM client and market fetcher.

    Args:
        llm_client: LLM client for game plan generation
        market_fetcher: Market data fetcher

    Returns:
        Configured GamePlanAgent
    """
    from src.agents.game_plan import GamePlanAgent

    return GamePlanAgent(llm_client, market_fetcher)


def create_trade_journal_agent(
    llm_client: LLMClient,
    market_fetcher: MarketDataFetcher,
) -> TradeJournalAgent:
    """Create TradeJournalAgent with LLM client and market fetcher.

    Args:
        llm_client: LLM client for trade journal analysis
        market_fetcher: Market data fetcher

    Returns:
        Configured TradeJournalAgent
    """
    from src.agents.journal import TradeJournalAgent

    return TradeJournalAgent(llm_client, market_fetcher)


def create_meta_agent(llm_client: LLMClient, regime_detector: MarketRegimeDetector) -> MetaAgent:
    """Create MetaAgent with LLM client and regime detector.

    Optional dependencies (metrics_tracker, param_store) passed as None.

    Args:
        llm_client: LLM client for meta-agent decisions
        regime_detector: Market regime detector

    Returns:
        Configured MetaAgent
    """
    from src.agents.meta import MetaAgent

    return MetaAgent(llm_client, regime_detector, metrics_tracker=None, param_store=None)


def create_risk_management_agent(
    llm_client: LLMClient,
    daemon_config: DaemonConfig,
    portfolio_var_calculator: PortfolioVaRCalculator | None = None,
    audit_repository: RiskAuditRepository | None = None,
) -> RiskManagementAgent:
    """Create RiskManagementAgent with config extraction.

    Extracts position_sizing and portfolio_var configs from daemon_config.

    Args:
        llm_client: LLM client for risk analysis
        daemon_config: Daemon configuration
        portfolio_var_calculator: Optional PortfolioVaRCalculator
        audit_repository: Optional repository for database audit logging

    Returns:
        Configured RiskManagementAgent
    """
    from src.agents.risk import PortfolioVaRConfig, RiskManagementAgent

    position_sizing_config = getattr(daemon_config, "position_sizing", None)
    portfolio_var_config = None
    risk_limits = getattr(daemon_config, "risk_limits", None)
    if risk_limits is not None:
        enabled = getattr(risk_limits, "enabled", True)
        if enabled:
            if hasattr(risk_limits, "model_dump"):
                portfolio_var_config = PortfolioVaRConfig(**risk_limits.model_dump())
            else:
                portfolio_var_config = PortfolioVaRConfig(**risk_limits)

    return RiskManagementAgent(
        llm_client,
        portfolio_var_calculator=portfolio_var_calculator,
        portfolio_var_config=portfolio_var_config,
        position_sizing_config=position_sizing_config,
        audit_repository=audit_repository,
    )


def create_confirmation_handler(
    daemon_config: DaemonConfig,
    telegram_channel: TelegramChannel | None = None,
) -> TradeConfirmationHandler | None:
    """Create confirmation handler if Telegram enabled.

    Args:
        daemon_config: Daemon configuration
        telegram_channel: Optional Telegram channel

    Returns:
        TradeConfirmationHandler if Telegram configured, None otherwise
    """
    if not telegram_channel or not telegram_channel.is_configured():
        return None

    from src.coordinator.confirmation import TradeConfirmationHandler

    timeout = daemon_config.coordinator.approval_timeout_seconds
    return TradeConfirmationHandler(
        telegram_channel=telegram_channel,
        approval_timeout_seconds=timeout,
    )


def create_pattern_analyzer(
    daemon_config: DaemonConfig,
    container: AppContainer,
) -> PatternAnalyzer | None:
    """Create pattern analyzer if enabled.

    Args:
        daemon_config: Daemon configuration
        container: DI container for repositories

    Returns:
        PatternAnalyzer if pattern detection enabled, None otherwise
    """
    from src.coordinator.pattern_analyzer import PatternAnalyzer

    if not daemon_config.coordinator.pattern_detection.enabled:
        return None

    database_engine = container.database_engine()
    min_sample_size = daemon_config.coordinator.pattern_detection.min_sample_size

    return PatternAnalyzer(
        database_engine=database_engine,
        memory=None,
        min_sample_size=min_sample_size,
    )


def create_adaptive_threshold_manager(
    daemon_config: DaemonConfig,
    container: AppContainer,
) -> AdaptiveThresholdManager | None:
    """Create adaptive threshold manager if enabled.

    Args:
        daemon_config: Daemon configuration
        container: DI container for repositories

    Returns:
        AdaptiveThresholdManager if enabled, None otherwise
    """
    from src.daemon.threshold_adapter import AdaptiveThresholdManager

    if not daemon_config.coordinator.adaptive_thresholds.enabled:
        return None

    return AdaptiveThresholdManager(
        config=daemon_config.coordinator.adaptive_thresholds,
        database_engine=container.database_engine(),
    )


def create_supervisor(llm_client: LLMClient) -> TradingSupervisor:
    """Create TradingSupervisor with LLM client.

    Args:
        llm_client: LLM client for planning and synthesis

    Returns:
        Configured TradingSupervisor
    """
    from src.agents.supervisor import TradingSupervisor

    return TradingSupervisor(llm_client)


def create_trading_coordinator(
    llm_client: LLMClient,
    daemon_config: DaemonConfig,
    container: AppContainer,
    daemon_state: DaemonState | None = None,
) -> TradingCoordinator:
    """Create TradingCoordinator with all dependencies.

    If coordinator.model_override is set, creates dedicated LLM client with that model.
    Otherwise uses default llm_client parameter.

    Args:
        llm_client: LLM client for tool calling
        daemon_config: Daemon config for coordinator settings
        container: DI container for tool registry
        daemon_state: Optional daemon state for today's data access

    Returns:
        Configured TradingCoordinator
    """
    from src.coordinator.agent import TradingCoordinator
    from src.coordinator.memory import CoordinatorMemory
    from src.coordinator.tools import build_coordinator_registry

    # Get dependencies for enhanced memory
    broker = container.alpaca_broker()

    # Get database engine for per-request repo creation (avoids session leaks)
    database_engine = None
    if daemon_config.database.enable_persistence:
        try:
            database_engine = container.database_engine()
        except Exception as e:
            from loguru import logger

            logger.opt(exception=True).warning(f"Failed to get database_engine for memory: {e}")

    # Create enhanced memory with multi-tier context
    memory = CoordinatorMemory(
        daemon_state=daemon_state,
        database_engine=database_engine,
        broker=broker,
    )

    # Build temp tool registry without coordinator (for initial creation)
    # Note: adaptive_threshold_manager created after registry, pass None initially
    tool_registry_temp = build_coordinator_registry(
        container, memory, coordinator=None, adaptive_threshold_manager=None
    )

    # Extract coordinator config
    coordinator_config = daemon_config.coordinator

    # Apply model override: coordinator_config.model_override (legacy) > llm.model_overrides > default
    if coordinator_config.model_override:
        from loguru import logger

        logger.warning(
            "coordinator.model_override is deprecated, use llm.model_overrides.coordinator instead"
        )
        provider = daemon_config.llm.provider
        api_key = None
        if provider == "anthropic":
            api_key = daemon_config.api_keys.anthropic_api_key
        elif provider == "openai":
            api_key = daemon_config.api_keys.openai_api_key

        coordinator_llm = LLMClient(
            provider=provider,
            model=coordinator_config.model_override,
            api_key=api_key,
            openai_base_url=daemon_config.api_keys.openai_api_base,
            cache_ttl=daemon_config.llm.response_cache_ttl_seconds,
        )
    else:
        coordinator_llm = llm_client

    # Get critic agent
    critic_agent = container.critic_agent()

    # Get adaptive threshold manager if enabled
    adaptive_threshold_manager = None
    if coordinator_config.adaptive_thresholds.enabled:
        try:
            adaptive_threshold_manager = create_adaptive_threshold_manager(daemon_config, container)
        except Exception as e:
            from loguru import logger

            logger.opt(exception=True).warning(f"Failed to create adaptive_threshold_manager: {e}")

    # Create coordinator with temp registry
    coordinator = TradingCoordinator(
        llm_client=coordinator_llm,
        tool_registry=tool_registry_temp,
        memory=memory,
        config=coordinator_config,
        broker=broker,
        critic_agent=critic_agent,
        adaptive_threshold_manager=adaptive_threshold_manager,
    )

    # Rebuild registry with coordinator reference for reflection tool
    # Pass critic_agent and adaptive_threshold_manager to avoid creating duplicate instances
    tool_registry = build_coordinator_registry(
        container, memory, coordinator, critic_agent, adaptive_threshold_manager
    )
    coordinator._tools = tool_registry  # noqa: SLF001

    return coordinator
