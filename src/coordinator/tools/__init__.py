"""Coordinator tools package."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.critic import CriticAgent
    from src.coordinator.agent import TradingCoordinator
    from src.coordinator.memory import CoordinatorMemory
    from src.di.container import AppContainer
    from src.tools.registry import ToolRegistry


def build_coordinator_registry(
    container: AppContainer,
    memory: CoordinatorMemory | None = None,
    coordinator: TradingCoordinator | None = None,
    critic_agent: CriticAgent | None = None,
) -> ToolRegistry:
    """Create coordinator tool registry with all tools.

    Includes 7 coordinator-specific tools + 2 reused tools from src/tools.

    Args:
        container: DI container for dependency resolution
        memory: Optional shared memory (creates new if None)
        coordinator: Optional coordinator for reflection tool
        critic_agent: Optional critic agent to reuse (avoids duplicate instances)

    Returns:
        ToolRegistry with all coordinator tools registered
    """
    # Lazy imports to avoid circular dependencies
    from src.coordinator.memory import CoordinatorMemory
    from src.coordinator.tools.analyze import AnalyzeSymbolTool
    from src.coordinator.tools.decision_history import QueryPastDecisionsTool
    from src.coordinator.tools.execute_trade import ExecuteTradeTool
    from src.coordinator.tools.generate_game_plan import GenerateGamePlanTool
    from src.coordinator.tools.history import AnalysisHistoryTool
    from src.coordinator.tools.market_overview import MarketOverviewTool
    from src.coordinator.tools.observation import SaveObservationTool
    from src.coordinator.tools.portfolio import PortfolioStatusTool
    from src.tools import GetMarketDataTool, ScreenStocksTool
    from src.tools.notification import NotificationTool
    from src.tools.registry import ToolRegistry

    registry = ToolRegistry()

    # Reused tools from src/tools/
    registry.register(GetMarketDataTool(container=container))
    registry.register(ScreenStocksTool(container=container))

    # Coordinator-specific tools
    game_plan_agent = container.game_plan_agent()
    market_fetcher = container.market_fetcher()
    broker = container.alpaca_broker()
    daemon_config = container.daemon_config()
    notification_service = container.notification_service()

    # Create confirmation handler if Telegram configured
    confirmation_handler = None
    if daemon_config.coordinator.confirmation_mode == "manual":
        from src.coordinator.confirmation import TradeConfirmationHandler
        from src.daemon.notification_channels import TelegramChannel

        # Create Telegram channel if configured
        telegram_channel = TelegramChannel(daemon_config.notifications.telegram)
        if telegram_channel.is_configured():
            confirmation_handler = TradeConfirmationHandler(
                telegram_channel=telegram_channel,
                approval_timeout_seconds=daemon_config.coordinator.approval_timeout_seconds,
            )
        else:
            # Log warning if manual mode but no Telegram
            from loguru import logger

            logger.warning("Manual confirmation mode enabled but Telegram not configured")

    registry.register(GenerateGamePlanTool(game_plan_agent))
    registry.register(MarketOverviewTool(market_fetcher))
    registry.register(AnalyzeSymbolTool(container, coordinator))
    registry.register(PortfolioStatusTool(broker))
    registry.register(ExecuteTradeTool(broker, daemon_config, confirmation_handler))
    registry.register(NotificationTool(notification_service))

    # Use provided memory or create new
    if memory is None:
        memory = CoordinatorMemory()

    # Register analysis history tool with memory (always registered)
    registry.register(AnalysisHistoryTool(memory))
    registry.register(QueryPastDecisionsTool(memory))
    registry.register(SaveObservationTool(memory))

    # Register reflection tool if coordinator provided
    if coordinator:
        from src.coordinator.tools.reflect import ReflectOnDecisionTool

        # Reuse provided critic_agent to avoid creating duplicate instance
        if critic_agent is None:
            critic_agent = container.critic_agent()
        registry.register(ReflectOnDecisionTool(coordinator, critic_agent))

    return registry


__all__ = [
    "build_coordinator_registry",
]
