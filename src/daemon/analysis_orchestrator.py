"""Analysis orchestration for daemon - extracts watchlist analysis logic."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

from loguru import logger
from pydantic import BaseModel

from src.agents.news import NewsAnalysis
from src.agents.sentiment import SentimentAnalysis
from src.cache.historical import SignalOutcomeInput
from src.daemon.config import AnalysisOrchestratorConfig
from src.daemon.event_bus import DashboardEvent, EventType
from src.daemon.notification_helper import DaemonNotificationHelper
from src.workflows.types import TradingWorkflowResult

if TYPE_CHECKING:
    from src.daemon.degradation import DegradationContext
    from src.daemon.factory import DaemonComponents


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

    def __init__(
        self,
        config: AnalysisOrchestratorConfig,
        components: DaemonComponents,
        trading_mode: str = "paper",
        **deprecated_kwargs: object,
    ) -> None:
        """Initialize analysis orchestrator.

        Args:
            config: Orchestrator configuration
            components: Daemon components (required)
            trading_mode: Trading mode (paper/live)
            **deprecated_kwargs: Deprecated params (workflow, state, scheduler, broker, position_manager,
                                event_bus, historical_cache, notification_service, context_builder).
                                Use components instead.
        """
        self.config = config
        self.trading_mode = trading_mode
        self._components = components

        # Extract from components (with backward compat for deprecated kwargs)
        workflow = deprecated_kwargs.get("workflow", components.workflow)
        state = deprecated_kwargs.get("state", components.state)
        scheduler = deprecated_kwargs.get("scheduler", components.scheduler)

        if workflow is None:
            msg = "workflow must be provided in components"
            raise ValueError(msg)
        if state is None:
            msg = "state must be provided in components"
            raise ValueError(msg)
        if scheduler is None:
            msg = "scheduler must be provided in components"
            raise ValueError(msg)

        # Type-narrow after None checks
        from src.cache.historical import HistoricalCache
        from src.daemon.context_builder import DaemonContextBuilder
        from src.daemon.event_bus import EventBus
        from src.daemon.notifications import NotificationService
        from src.daemon.positions import PositionManager
        from src.daemon.scheduler import MarketScheduler
        from src.daemon.state import DaemonState
        from src.data.broker import AlpacaBroker
        from src.workflows import TradingWorkflow

        self.workflow: TradingWorkflow = cast("TradingWorkflow", workflow)
        self.state: DaemonState = cast("DaemonState", state)
        self.scheduler: MarketScheduler = cast("MarketScheduler", scheduler)
        self.broker: AlpacaBroker | None = cast(
            "AlpacaBroker | None", deprecated_kwargs.get("broker", components.broker)
        )
        self.position_manager: PositionManager | None = cast(
            "PositionManager | None", deprecated_kwargs.get("position_manager", components.position_manager)
        )
        self.event_bus: EventBus | None = cast(
            "EventBus | None", deprecated_kwargs.get("event_bus", components.event_bus)
        )
        self.historical_cache: HistoricalCache | None = cast(
            "HistoricalCache | None", deprecated_kwargs.get("historical_cache", components.historical_cache)
        )
        self.notification_service: NotificationService | None = cast(
            "NotificationService | None",
            deprecated_kwargs.get("notification_service", components.notification_service),
        )
        self._context_builder: DaemonContextBuilder | None = cast(
            "DaemonContextBuilder | None", deprecated_kwargs.get("context_builder")
        )
        self._notification_helper = DaemonNotificationHelper()
        logger.info("AnalysisOrchestrator initialized")

    def __repr__(self) -> str:
        """Return string representation."""
        return f"AnalysisOrchestrator(max_concurrent={self.config.max_concurrent_analyses})"

    def _sync_positions_with_broker(self) -> bool:
        """Sync positions with broker and update state.

        Returns:
            True if sync was performed successfully, False otherwise
        """
        position_manager = self.position_manager
        if not (self.config.enable_position_sync and position_manager):
            return False

        try:
            # Filter out None positions for type safety
            state_positions = {
                sym: pos
                for sym in self.state.active_positions
                if (pos := self.state.get_position(sym)) is not None
            }
            new_positions, updated_positions, closed_symbols = position_manager.sync_with_broker(
                state_positions
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
        broker = self.broker
        if not (self.position_manager and broker):
            return None

        try:
            broker_info = broker.get_account_info()
            return broker_info.positions
        except Exception as e:
            logger.warning(f"Failed to prefetch account info: {e}")
            return None

    def _apply_position_management(self, results: list[TradingWorkflowResult]) -> int:
        """Apply position management rules to results.

        Args:
            results: Analysis results to process

        Returns:
            Number of position actions executed
        """
        position_manager = self.position_manager
        if not position_manager:
            return 0

        position_actions = 0
        for result in results:
            if result.symbol in self.state.active_positions:
                try:
                    pos = self.state.get_position(result.symbol)
                    if pos:
                        actions = position_manager.review_position(pos, result.risk.current_price, result)
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

    def _filter_analysis_results(
        self,
        raw_results: list[TradingWorkflowResult | BaseException | None],
        watchlist: list[str],
    ) -> tuple[list[TradingWorkflowResult], list[str]]:
        """Filter analysis results and propagate control-flow exceptions.

        Args:
            raw_results: Raw results from asyncio.gather
            watchlist: Symbols that were analyzed

        Returns:
            Tuple of (successful_results, failed_symbols)

        Raises:
            CancelledError, KeyboardInterrupt, SystemExit: Propagated immediately
        """
        results: list[TradingWorkflowResult] = []
        failed_symbols: list[str] = []

        for i, result in enumerate(raw_results):
            symbol = watchlist[i]
            # Propagate cancellation and control-flow exceptions immediately
            if isinstance(result, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                raise result
            if isinstance(result, Exception):
                logger.error(f"Analysis failed for {symbol}: {result}")
                self.state.record_error(f"{symbol}: {result}")
                failed_symbols.append(symbol)
            elif result is None:
                logger.warning(f"Analysis returned None for {symbol}")
                failed_symbols.append(symbol)
            elif isinstance(result, TradingWorkflowResult):
                # Type narrowing: result is TradingWorkflowResult here
                results.append(result)
            else:
                logger.warning(f"Unexpected result type for {symbol}: {type(result)}")
                failed_symbols.append(symbol)

        return results, failed_symbols

    async def orchestrate(
        self,
        watchlist: list[str],
        target_allocations: dict[str, float] | None = None,
        degradation_context: DegradationContext | None = None,
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

        # Step 1: Sync positions with broker (if enabled) - offload to thread
        position_sync_performed = await asyncio.to_thread(self._sync_positions_with_broker)

        # Step 2: Prefetch broker positions once - offload to thread
        broker_positions = await asyncio.to_thread(self._prefetch_broker_positions)

        # Step 3: Concurrent analysis with semaphore
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

        # Wrap tasks to handle exceptions without canceling siblings
        async def safe_analyze(symbol: str) -> TradingWorkflowResult | BaseException | None:
            try:
                return await analyze_with_limit(symbol)
            except BaseException as e:
                # Re-raise control-flow exceptions so TaskGroup can cancel siblings promptly
                if isinstance(e, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                    raise
                # Return exception to be processed by _filter_analysis_results
                return e

        # Run analyses in parallel using TaskGroup
        async with asyncio.TaskGroup() as tg:
            task_results = [tg.create_task(safe_analyze(s)) for s in watchlist]

        # Extract results from tasks
        raw_results = [task.result() for task in task_results]

        # Step 4: Filter exceptions and None results
        results, failed_symbols = self._filter_analysis_results(raw_results, watchlist)

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

    async def _analyze_symbol(
        self,
        symbol: str,
        position_context: dict[str, object] | None = None,
        degradation_context: DegradationContext | None = None,
    ) -> TradingWorkflowResult | None:
        """Analyze a single symbol.

        Args:
            symbol: Stock ticker symbol
            position_context: Position context (entry price, P&L, days held) (optional)
            degradation_context: Optional degradation context

        Returns:
            TradingWorkflowResult or None on error
        """
        return await self._analyze_symbol_with_context(
            symbol, position_context, target_allocations=None, degradation_context=degradation_context
        )

    async def _analyze_symbol_with_context(
        self,
        symbol: str,
        position_context: dict[str, object] | None,
        target_allocations: dict[str, float] | None,
        degradation_context: DegradationContext | None,
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
            context_builder = self._context_builder
            if context_builder:
                sector_ctx, earnings_ctx, peer_ctx, game_plan_ctx = context_builder.build_analysis_contexts(
                    symbol
                )

            if target_allocations is not None:
                self.workflow.set_target_allocations(target_allocations)

            try:
                from src.workflows.types import WorkflowExtraContext

                extra_context = WorkflowExtraContext(
                    sector_rotation_context=sector_ctx,
                    earnings_context=earnings_ctx,
                    peer_analysis_context=peer_ctx,
                    game_plan_context=game_plan_ctx,
                    position_context=position_context,
                    degradation_context=degradation_context,
                )
                result = await self.workflow.analyze(
                    symbol,
                    period_days=90,
                    trading_session=session,
                    extra_context=extra_context,
                )
            finally:
                if target_allocations is not None:
                    self.workflow.set_target_allocations(None)

            if self.notification_service and self._components:
                await self._notification_helper.maybe_notify_signal(result, self._components)

            rsi = result.technical.rsi if result.technical else None
            macd_hist = result.technical.macd_hist if result.technical else None

            self.state.record_analysis(
                symbol=symbol,
                signal=result.decision.action.value,
                confidence=result.decision.confidence,
                executed=result.order is not None,
                trading_session=result.trading_session,
                is_paper_trade=self.trading_mode == "paper",
                rsi=rsi,
                macd_hist=macd_hist,
                reasoning=result.decision.reasoning,
            )

            # Record signal outcome in historical cache
            if self.historical_cache:
                try:
                    signal_input = SignalOutcomeInput(
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
                    historical_cache = self.historical_cache
                    if historical_cache:
                        historical_cache.record_signal_outcome(signal_input)
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
        event_bus = self.event_bus
        if event_bus:
            try:
                await event_bus.publish(DashboardEvent(event_type=EventType[event_type], data=data))
            except Exception as e:
                logger.warning(f"Event publish failed: {e}")

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
