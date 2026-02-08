"""Analysis orchestration for daemon - extracts watchlist analysis logic."""

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel

from src.daemon.config import AnalysisOrchestratorConfig
from src.daemon.event_bus import DashboardEvent, EventType

if TYPE_CHECKING:
    from src.cache.historical import HistoricalCache
    from src.daemon.degradation import DegradationContext
    from src.daemon.event_bus import EventBus
    from src.daemon.notifications import NotificationService
    from src.daemon.position_manager import PositionManager
    from src.daemon.scheduler import MarketScheduler
    from src.daemon.state import DaemonState
    from src.data.broker import AlpacaBroker
    from src.workflows.trading import TradingWorkflow
from src.agents.news import NewsAnalysis
from src.agents.sentiment import SentimentAnalysis
from src.workflows.types import TradingWorkflowResult


class AnalysisOrchestrationResult(BaseModel):
    """Structured result from watchlist orchestration."""

    timestamp: datetime
    total_symbols: int
    successful: int
    failed: int
    position_actions: int
    results: list[TradingWorkflowResult]
    failed_symbols: list[str]
    duration_seconds: float
    position_sync_performed: bool


class AnalysisOrchestrator:
    """Orchestrate watchlist analysis with concurrency control."""

    def __init__(  # noqa: PLR0913
        self,
        workflow: "TradingWorkflow",
        state: "DaemonState",
        scheduler: "MarketScheduler",
        config: AnalysisOrchestratorConfig,
        trading_mode: str = "paper",
        broker: "AlpacaBroker | None" = None,
        position_manager: "PositionManager | None" = None,
        event_bus: "EventBus | None" = None,
        historical_cache: "HistoricalCache | None" = None,
        notification_service: "NotificationService | None" = None,
        context_builder: object | None = None,
    ) -> None:
        """Initialize analysis orchestrator.

        Args:
            workflow: Trading workflow instance
            state: Daemon state
            scheduler: Market scheduler
            config: Orchestrator configuration
            trading_mode: Trading mode (paper/live)
            broker: Optional broker for position fetching
            position_manager: Optional position manager
            event_bus: Optional event bus for publishing
            historical_cache: Optional historical cache
            notification_service: Optional notification service
            context_builder: Object with _build_analysis_contexts method
        """
        self.workflow = workflow
        self.state = state
        self.scheduler = scheduler
        self.config = config
        self.trading_mode = trading_mode
        self.broker = broker
        self.position_manager = position_manager
        self.event_bus = event_bus
        self.historical_cache = historical_cache
        self.notification_service = notification_service
        self._context_builder = context_builder
        logger.info("AnalysisOrchestrator initialized")

    def _sync_positions_with_broker(self) -> bool:
        """Sync positions with broker and update state.

        Returns:
            True if sync was performed successfully, False otherwise
        """
        if not (self.config.enable_position_sync and self.position_manager):
            return False

        try:
            new_positions, updated_positions, closed_symbols = self.position_manager.sync_with_broker(
                {sym: self.state.get_position(sym) for sym in self.state.active_positions}
            )
            for pos in new_positions:
                self.state.add_position(pos)
            for pos in updated_positions:
                self.state.update_position(pos)
            for symbol in closed_symbols:
                self.state.remove_position(symbol)
            return True
        except Exception as e:
            logger.error(f"Failed to sync positions: {e}")
            return False

    def _prefetch_broker_positions(self) -> dict | None:
        """Prefetch broker positions for analysis context.

        Returns:
            Dict of broker positions or None if unavailable
        """
        if not (self.position_manager and self.broker):
            return None

        try:
            broker_info = self.broker.get_account_info()
            return broker_info.positions
        except Exception as e:
            logger.warning(f"Failed to prefetch account info: {e}")
            return None

    def _apply_position_management(self, results: list["TradingWorkflowResult"]) -> int:
        """Apply position management rules to results.

        Args:
            results: Analysis results to process

        Returns:
            Number of position actions executed
        """
        if not self.position_manager:
            return 0

        position_actions = 0
        for result in results:
            if result.symbol in self.state.active_positions:
                try:
                    pos = self.state.get_position(result.symbol)
                    if pos:
                        actions = self.position_manager.review_position(
                            pos, result.risk.current_price, result
                        )
                        self.state.update_position(pos)
                        for action in actions:
                            self.state.record_position_action(action)
                            position_actions += 1
                            logger.info(
                                f"Position action: {action.action_type} {action.symbol} - {action.reason}"
                            )
                except Exception as e:
                    logger.error(f"Failed to review position {result.symbol}: {e}")

        return position_actions

    async def orchestrate(
        self,
        watchlist: list[str],
        target_allocations: dict[str, float] | None = None,
        degradation_context: "DegradationContext | None" = None,
    ) -> AnalysisOrchestrationResult:
        """Orchestrate watchlist analysis.

        Args:
            watchlist: Symbols to analyze
            target_allocations: Optional target allocations from rebalancing
            degradation_context: Optional degradation context

        Returns:
            AnalysisOrchestrationResult with stats and results
        """
        start_time = datetime.now(UTC)

        # Step 1: Sync positions with broker (if enabled)
        position_sync_performed = self._sync_positions_with_broker()

        # Step 2: Prefetch broker positions once
        broker_positions = self._prefetch_broker_positions()

        # Step 3: Concurrent analysis with semaphore
        results: list[TradingWorkflowResult] = []
        failed_symbols: list[str] = []
        semaphore = asyncio.Semaphore(self.config.max_concurrent_analyses)

        async def analyze_with_limit(symbol: str) -> TradingWorkflowResult | None:
            # Build position context if holding
            position_context = None
            if symbol in self.state.active_positions:
                pos = self.state.get_position(symbol)
                if pos:
                    # Get current price from prefetched broker positions
                    current_price = 0.0
                    if broker_positions and symbol in broker_positions:
                        broker_pos = broker_positions[symbol]
                        qty = broker_pos.qty
                        market_value = broker_pos.market_value
                        if qty > 0:
                            current_price = market_value / qty

                    unrealized_pnl_pct = 0.0
                    if current_price > 0 and pos.entry_price > 0:
                        unrealized_pnl_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100

                    position_context = {
                        "has_position": True,
                        "symbol": symbol,
                        "entry_price": pos.entry_price,
                        "entry_confidence": pos.entry_confidence,
                        "unrealized_pnl_percent": unrealized_pnl_pct,
                        "days_held": pos.days_held,
                        "current_qty": pos.current_qty,
                    }

            async with semaphore:
                return await self._analyze_symbol_with_context(
                    symbol, position_context, target_allocations, degradation_context
                )

        tasks = [analyze_with_limit(s) for s in watchlist]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Step 4: Filter exceptions and None results
        for i, result in enumerate(raw_results):
            symbol = watchlist[i]
            if isinstance(result, Exception):
                logger.error(f"Analysis failed for {symbol}: {result}")
                self.state.record_error(f"{symbol}: {result}")
                failed_symbols.append(symbol)
            elif result is None:
                logger.warning(f"Analysis returned None for {symbol}")
                failed_symbols.append(symbol)
            else:
                results.append(result)

        # Step 5: Apply position management rules
        position_actions = self._apply_position_management(results)

        # Step 6: Build result
        duration = (datetime.now(UTC) - start_time).total_seconds()
        return AnalysisOrchestrationResult(
            timestamp=start_time,
            total_symbols=len(watchlist),
            successful=len(results),
            failed=len(failed_symbols),
            position_actions=position_actions,
            results=results,
            failed_symbols=failed_symbols,
            duration_seconds=duration,
            position_sync_performed=position_sync_performed,
        )

    async def _analyze_symbol_with_context(
        self,
        symbol: str,
        position_context: dict[str, object] | None,
        target_allocations: dict[str, float] | None,
        degradation_context: "DegradationContext | None",
    ) -> TradingWorkflowResult | None:
        """Analyze single symbol with full context.

        Args:
            symbol: Stock ticker
            position_context: Position context
            target_allocations: Target allocations from rebalancing
            degradation_context: Degradation context

        Returns:
            TradingWorkflowResult or None on error
        """
        from src.strategies.session import TradingSession

        try:
            session = self.scheduler.get_trading_session() or TradingSession.REGULAR
            await self._publish_event("ANALYSIS_START", {"symbol": symbol, "trading_session": session.value})

            # Build contexts via delegated method
            sector_ctx, earnings_ctx, peer_ctx, game_plan_ctx = None, None, None, None
            if self._context_builder and hasattr(self._context_builder, "build_analysis_contexts"):
                sector_ctx, earnings_ctx, peer_ctx, game_plan_ctx = (
                    self._context_builder.build_analysis_contexts(symbol)
                )

            if target_allocations is not None:
                self.workflow.set_target_allocations(target_allocations)

            try:
                result = await self.workflow.analyze(
                    symbol,
                    period_days=90,
                    trading_session=session,
                    position_context=position_context,
                    sector_context=sector_ctx,
                    earnings_context=earnings_ctx,
                    peer_analysis_context=peer_ctx,
                    game_plan_context=game_plan_ctx,
                    degradation_context=degradation_context,
                )
            finally:
                if target_allocations is not None:
                    self.workflow.set_target_allocations(None)

            if self.notification_service:
                await self._maybe_notify_signal(result)

            rsi = result.technical.rsi if result.technical else None
            macd_hist = result.technical.macd_hist if result.technical else None

            self.state.record_analysis(
                symbol=symbol,
                signal=result.decision.action.value,
                confidence=result.decision.confidence,
                executed=result.order is not None,
                trading_session=result.trading_session.value,
                is_paper_trade=self.trading_mode == "paper",
                rsi=rsi,
                macd_hist=macd_hist,
            )

            # Record signal outcome in historical cache
            if self.historical_cache:
                try:
                    self.historical_cache.record_signal_outcome(
                        symbol=symbol,
                        timestamp=datetime.now(UTC),
                        signal=result.decision.action.value,
                        confidence=result.decision.confidence,
                        price_at_signal=result.risk.current_price,
                        strategy_used=result.strategy_used,
                        regime=result.regime.regime.value if result.regime else None,
                        trading_session=result.trading_session.value,
                        technical_signal=result.technical.signal.value,
                        sentiment_signal=self._extract_sentiment_signal(result.sentiment),
                        news_signal=self._extract_news_signal(result.news),
                    )
                except Exception as e:
                    logger.warning(f"Failed to record signal outcome for accuracy tracking: {e}")

            await self._publish_event(
                "ANALYSIS_COMPLETE",
                {
                    "symbol": symbol,
                    "signal": result.decision.action.value,
                    "confidence": result.decision.confidence,
                    "executed": result.order is not None,
                },
            )

            return result
        except Exception as e:
            error_msg = f"Failed to analyze {symbol}: {e}"
            logger.error(error_msg)
            self.state.record_error(error_msg)
            await self._publish_event("ANALYSIS_ERROR", {"symbol": symbol, "error": str(e)})
            return None

    async def _publish_event(self, event_type: str, data: dict[str, object]) -> None:
        """Publish event to EventBus.

        Args:
            event_type: Event type
            data: Event data
        """
        if self.event_bus:
            try:
                await self.event_bus.publish(DashboardEvent(event_type=EventType[event_type], data=data))
            except Exception as e:
                logger.warning(f"Event publish failed: {e}")

    async def _maybe_notify_signal(self, result: TradingWorkflowResult) -> None:
        """Send notification for signal.

        Args:
            result: Analysis result
        """
        if not self.notification_service:
            return

        from datetime import UTC, datetime

        from src.daemon.notifications import NotificationMessage, NotificationTrigger

        try:
            message = NotificationMessage(
                trigger=NotificationTrigger.SIGNAL,
                title=f"{result.symbol} {result.decision.action.value}",
                body=f"Confidence: {result.decision.confidence:.2f}, Risk: {result.risk.validation.risk_level}",
                metadata={
                    "symbol": result.symbol,
                    "signal": result.decision.action.value,
                    "confidence": result.decision.confidence,
                    "risk_level": result.risk.validation.risk_level,
                },
                timestamp=datetime.now(UTC),
            )
            await self.notification_service.notify(NotificationTrigger.SIGNAL, message)
        except Exception as e:
            logger.warning(f"Notification failed for {result.symbol}: {e}")

    def _extract_sentiment_signal(self, sentiment: SentimentAnalysis | None) -> str | None:
        """Extract sentiment signal.

        Args:
            sentiment: Sentiment analysis

        Returns:
            Signal string or None
        """
        if not sentiment:
            return None
        if sentiment.overall_sentiment == "positive":
            return "BUY"
        if sentiment.overall_sentiment == "negative":
            return "SELL"
        return "HOLD"

    def _extract_news_signal(self, news: NewsAnalysis | None) -> str | None:
        """Extract news signal.

        Args:
            news: News analysis

        Returns:
            Signal string or None
        """
        if not news:
            return None
        recommendation = news.recommendation.upper()
        if "BUY" in recommendation:
            return "BUY"
        if "SELL" in recommendation:
            return "SELL"
        return "HOLD"
