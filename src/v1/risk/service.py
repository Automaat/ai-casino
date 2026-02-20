"""Risk service facade wrapping RiskManagementAgent for trade execution."""

import asyncio
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from src.agents.risk.models import AccountInfo
from src.strategies.signal import Signal
from src.v1.risk.models import RiskDecision

if TYPE_CHECKING:
    from src.agents.risk.agent import RiskManagementAgent
    from src.daemon.config import DaemonConfig
    from src.data.market import MarketDataFetcher
    from src.v1.trades.brokers import Broker


class RiskService:
    """Async facade for risk assessment at trade execution time."""

    def __init__(
        self,
        risk_agent: RiskManagementAgent,
        broker: Broker,
        market_fetcher: MarketDataFetcher,
        daemon_config: DaemonConfig,
    ) -> None:
        """Initialize risk service.

        Args:
            risk_agent: Risk management agent for calculations
            broker: Broker for fresh account data
            market_fetcher: Market data fetcher for OHLCV
            daemon_config: Daemon configuration
        """
        self._risk_agent = risk_agent
        self._broker = broker
        self._market_fetcher = market_fetcher
        self._daemon_config = daemon_config

    async def assess_trade(
        self,
        symbol: str,
        action: Signal,
        confidence: float,
        current_price: float | None = None,
    ) -> RiskDecision:
        """Assess trade risk with fresh account and market data.

        Args:
            symbol: Stock ticker
            action: Proposed trading action (BUY/SELL/HOLD)
            confidence: Decision confidence (0.0-1.0)
            current_price: Optional current price override

        Returns:
            RiskDecision with approval status and limits
        """
        if action == Signal.HOLD:
            return RiskDecision(
                approved=True,
                risk_level="LOW",
                recommended_shares=0,
                stop_loss_price=current_price or 0.0,
                position_value=0.0,
                risk_percent=0.0,
                warnings=[],
                reasoning="No risk — HOLD action",
            )

        try:
            account_info, market_data, broker_api_failed = await self._fetch_data(symbol)
        except Exception as e:
            logger.opt(exception=True).error(f"Risk data fetch failed for {symbol}: {e}")
            return RiskDecision(
                approved=False,
                risk_level="HIGH",
                recommended_shares=0,
                stop_loss_price=0.0,
                position_value=0.0,
                risk_percent=0.0,
                warnings=[f"Data fetch failed: {e}"],
                reasoning=f"REJECTED: cannot assess risk — {e}",
            )

        price = current_price or self._get_latest_price(market_data)
        if price <= 0:
            return RiskDecision(
                approved=False,
                risk_level="HIGH",
                recommended_shares=0,
                stop_loss_price=0.0,
                position_value=0.0,
                risk_percent=0.0,
                warnings=["Cannot determine current price"],
                reasoning="REJECTED: no valid price available",
            )

        assessment = await asyncio.to_thread(
            self._risk_agent.assess,
            symbol=symbol,
            action=action,
            current_price=price,
            account_info=account_info,
            market_data=market_data,
            decision_confidence=confidence,
            broker_api_failed=broker_api_failed,
        )

        return RiskDecision(
            approved=assessment.validation.approved,
            risk_level=assessment.validation.risk_level,
            recommended_shares=assessment.position_sizing.recommended_shares,
            stop_loss_price=assessment.stop_loss.stop_loss_price,
            take_profit_price=(assessment.take_profit.take_profit_price if assessment.take_profit else None),
            position_value=assessment.position_sizing.position_value,
            risk_percent=assessment.position_sizing.risk_percent,
            warnings=assessment.validation.warnings,
            reasoning=assessment.validation.reasoning,
        )

    async def _fetch_data(self, symbol: str) -> tuple[AccountInfo, pd.DataFrame, bool]:
        """Fetch fresh account info and market data concurrently.

        Args:
            symbol: Stock ticker for market data

        Returns:
            Tuple of (account_info, market_data, broker_api_failed)
        """
        broker_api_failed = False
        account_info = AccountInfo(balance=0.0, available_cash=0.0, positions={}, total_exposure=0.0)

        async def fetch_account() -> None:
            nonlocal account_info, broker_api_failed
            try:
                broker_info = await asyncio.to_thread(self._broker.get_account_info)
                account_info = AccountInfo(
                    balance=broker_info.balance,
                    available_cash=broker_info.available_cash,
                    positions={s: p.market_value for s, p in broker_info.positions.items()},
                    total_exposure=broker_info.total_exposure,
                )
            except Exception as e:
                logger.opt(exception=True).warning(f"Broker API failed: {e}")
                broker_api_failed = True

        async def fetch_market() -> pd.DataFrame:
            result = await asyncio.to_thread(self._market_fetcher.fetch_daily, symbol, 90)
            return result.data

        market_task = asyncio.create_task(fetch_market())
        account_task = asyncio.create_task(fetch_account())

        market_data, _ = await asyncio.gather(market_task, account_task)

        return account_info, market_data, broker_api_failed

    @staticmethod
    def _get_latest_price(market_data: pd.DataFrame) -> float:
        """Extract latest close price from market data."""
        if market_data is not None and not market_data.empty and "Close" in market_data.columns:
            return float(market_data["Close"].iloc[-1])
        return 0.0

    def __repr__(self) -> str:
        """String representation."""
        return "RiskService()"
