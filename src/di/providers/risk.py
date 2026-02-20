"""Risk service DI provider."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.risk.agent import RiskManagementAgent
    from src.daemon.config import DaemonConfig
    from src.data.market import MarketDataFetcher
    from src.v1.risk.service import RiskService
    from src.v1.trades.brokers import Broker


def create_risk_service(
    risk_agent: RiskManagementAgent,
    broker: Broker,
    market_fetcher: MarketDataFetcher,
    daemon_config: DaemonConfig,
) -> RiskService:
    """Create RiskService facade.

    Args:
        risk_agent: Risk management agent
        broker: Broker for account data
        market_fetcher: Market data fetcher
        daemon_config: Daemon configuration

    Returns:
        Configured RiskService
    """
    from src.v1.risk.service import RiskService

    return RiskService(risk_agent, broker, market_fetcher, daemon_config)
