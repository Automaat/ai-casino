"""Coordinator tools package."""

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from src.agents.critic import CriticAgent
    from src.daemon.config import DaemonConfig
    from src.database.engine import DatabaseEngine
    from src.di.container import AppContainer
    from src.tools.registry import ToolRegistry
    from src.v1.coordinator.agent import TradingCoordinator
    from src.v1.coordinator.confirmation import TradeConfirmationHandler
    from src.v1.coordinator.memory import CoordinatorMemory


def _create_confirmation_handler(daemon_config: DaemonConfig) -> TradeConfirmationHandler | None:
    """Create confirmation handler if manual mode with Telegram configured."""
    if daemon_config.coordinator.confirmation_mode != "manual":
        return None

    from src.v1.coordinator.confirmation import TradeConfirmationHandler
    from src.v1.notifications.channels.telegram import TelegramChannel

    telegram_channel = TelegramChannel(daemon_config.notifications.telegram)
    if telegram_channel.is_configured():
        return TradeConfirmationHandler(
            telegram_channel=telegram_channel,
            approval_timeout_seconds=daemon_config.coordinator.approval_timeout_seconds,
        )

    logger.warning("Manual confirmation mode enabled but Telegram not configured")
    return None


def _resolve_database_engine(daemon_config: DaemonConfig, container: AppContainer) -> DatabaseEngine | None:
    """Resolve database engine if persistence enabled."""
    if not daemon_config.database.enable_persistence:
        return None
    try:
        return container.database_engine()
    except Exception:
        logger.warning("Database engine unavailable, trade persistence disabled")
        return None


def build_coordinator_registry(
    container: AppContainer,
    memory: CoordinatorMemory | None = None,
    coordinator: TradingCoordinator | None = None,
    critic_agent: CriticAgent | None = None,
) -> ToolRegistry:
    """Create coordinator tool registry with all tools.

    Args:
        container: DI container for dependency resolution
        memory: Optional shared memory (creates new if None)
        coordinator: Optional coordinator for reflection tool
        critic_agent: Optional critic agent to reuse (avoids duplicate instances)

    Returns:
        ToolRegistry with all coordinator tools registered
    """
    # Lazy imports to avoid circular dependencies
    from src.tools import GetMarketDataTool, ScreenStocksTool
    from src.tools.news import GetNewsTool
    from src.tools.notification import NotificationTool
    from src.tools.registry import ToolRegistry
    from src.tools.risk_metrics import GetRiskMetricsTool
    from src.tools.social_sentiment import GetSocialSentimentTool
    from src.tools.trump_analysis import TrumpAnalysisTool
    from src.tools.websearch import WebSearchTool
    from src.v1.coordinator.memory import CoordinatorMemory
    from src.v1.coordinator.tools.analyze import AnalyzeSymbolTool
    from src.v1.coordinator.tools.decision_history import QueryPastDecisionsTool
    from src.v1.coordinator.tools.execute_trade import ExecuteTradeTool
    from src.v1.coordinator.tools.history import AnalysisHistoryTool
    from src.v1.coordinator.tools.market_overview import MarketOverviewTool
    from src.v1.coordinator.tools.observation import SaveObservationTool
    from src.v1.coordinator.tools.portfolio import PortfolioStatusTool
    from src.v1.trades.service import TradingService

    registry = ToolRegistry()
    registry.register(GetMarketDataTool(container=container))
    registry.register(ScreenStocksTool(container=container))

    broker = container.alpaca_broker()
    daemon_config = container.daemon_config()
    notification_service = container.notification_service()

    registry.register(MarketOverviewTool(container.market_fetcher()))
    registry.register(AnalyzeSymbolTool(container, coordinator))
    registry.register(PortfolioStatusTool(broker))

    trading_service = TradingService(
        broker=broker,
        daemon_config=daemon_config,
        database_engine=_resolve_database_engine(daemon_config, container),
        notification_service=notification_service,
        confirmation_handler=_create_confirmation_handler(daemon_config),
    )
    registry.register(ExecuteTradeTool(trading_service, daemon_config))

    registry.register(NotificationTool(notification_service))
    registry.register(WebSearchTool(container.websearch_fetcher()))
    registry.register(GetNewsTool(container=container))
    registry.register(GetRiskMetricsTool(container=container))
    registry.register(GetSocialSentimentTool(container=container))
    registry.register(TrumpAnalysisTool(container=container))

    if memory is None:
        memory = CoordinatorMemory()

    registry.register(AnalysisHistoryTool(memory))
    registry.register(QueryPastDecisionsTool(memory))
    registry.register(SaveObservationTool(memory))

    if coordinator:
        from src.v1.coordinator.tools.reflect import ReflectOnDecisionTool

        if critic_agent is None:
            critic_agent = container.critic_agent()
        registry.register(ReflectOnDecisionTool(coordinator, critic_agent))

    return registry


__all__ = [
    "build_coordinator_registry",
]
