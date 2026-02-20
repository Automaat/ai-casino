"""Analysis orchestration for daemon - extracts watchlist analysis logic."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Final, cast

from loguru import logger
from pydantic import BaseModel

from src.agents.news import NewsAnalysis
from src.agents.sentiment import SentimentAnalysis
from src.cache.historical import SignalOutcomeInput
from src.daemon.config import AnalysisOrchestratorConfig
from src.daemon.event_bus import DashboardEvent, EventType
from src.daemon.events import Sentiment, SignalEvent, TriageResult, Urgency
from src.daemon.notification_helper import DaemonNotificationHelper
from src.daemon.state.managers.trading import AnalysisRecordInput
from src.event_queue.service import MarketEventQueue
from src.strategies.signal import Signal
from src.workflows.types import TradingWorkflowResult, WorkflowExtraContext

_QUEUE_MIN_CONFIDENCE: Final = 0.5

if TYPE_CHECKING:
    from src.daemon.degradation import DegradationContext
    from src.daemon.factory import DaemonComponents
    from src.strategies.session import TradingSession


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
        from src.daemon.positions import PositionManager
        from src.daemon.scheduler import MarketScheduler
        from src.daemon.state import DaemonState
        from src.data.broker import AlpacaBroker
        from src.v1.notifications.service import NotificationService
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

        self._economic_watcher = components.economic_calendar_watcher
        self._options_flow_watcher = components.options_flow_watcher
        self._social_sentiment_watcher = components.social_sentiment_watcher

        self._notification_helper = DaemonNotificationHelper()
        self.market_event_queue: MarketEventQueue | None = None  # wired by runner
        logger.info("AnalysisOrchestrator initialized")

    def __repr__(self) -> str:
        """Return string representation."""
        return f"AnalysisOrchestrator(max_concurrent={self.config.max_concurrent_analyses})"

    async def _sync_positions_with_broker(self) -> bool:
        """Sync positions with broker and update state.

        Returns:
            True if sync was performed successfully, False otherwise
        """
        position_manager = self.position_manager
        if not (self.config.enable_position_sync and position_manager):
            return False

        try:
            state_positions = await self._fetch_all_positions()
            new_positions, updated_positions, closed_symbols = position_manager.sync_with_broker(
                state_positions
            )
            for pos in new_positions:
                await self.state.add_position(pos)
            for pos in updated_positions:
                await self.state.update_position(pos)
            for symbol in closed_symbols:
                await self.state.remove_position(symbol)
            return True
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to sync positions: {e}")
            return False

    async def _fetch_all_positions(self) -> dict:
        """Fetch all positions in a single query (avoids N+1).

        Returns:
            Dict mapping symbol to PositionRecord
        """
        all_positions = await self.state.get_all_positions()
        return {pos.symbol: pos for pos in all_positions}

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
            logger.opt(exception=True).warning(f"Failed to prefetch account info: {e}")
            return None

    async def _apply_position_management(self, results: list[TradingWorkflowResult]) -> int:
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
        active_positions = await self._fetch_all_positions()
        for result in results:
            pos = active_positions.get(result.symbol)
            if pos:
                try:
                    actions = position_manager.review_position(pos, result.risk.current_price, result)
                    await self.state.update_position(pos)
                    for action in actions:
                        await self.state.record_position_action(action)
                        position_actions += 1
                        logger.info(
                            f"Position action: {action.action_type} {action.symbol} - {action.reason}"
                        )
                except Exception as e:
                    logger.opt(exception=True).error(f"Failed to review position {result.symbol}: {e}")

        return position_actions

    async def _filter_analysis_results(
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
                logger.opt(exception=True).error(f"Analysis failed for {symbol}: {result}")
                await self.state.record_error(f"{symbol}: {result}")
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

        # Step 1: Sync positions with broker (if enabled) - already async
        position_sync_performed = await self._sync_positions_with_broker()

        # Step 2: Prefetch broker positions + active positions once (avoids N+1)
        broker_positions = await asyncio.to_thread(self._prefetch_broker_positions)
        active_positions = await self._fetch_all_positions()

        # Step 3: Concurrent analysis with semaphore
        semaphore = asyncio.Semaphore(self.config.max_concurrent_analyses)

        async def analyze_with_limit(symbol: str) -> TradingWorkflowResult | None:
            # Build position context from prefetched data (no per-symbol queries)
            position_context = None
            pos = active_positions.get(symbol)
            if pos:
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
                    "conviction_history": pos.conviction_history,
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
        results, failed_symbols = await self._filter_analysis_results(raw_results, watchlist)

        # Step 5: Apply position management rules
        position_actions = await self._apply_position_management(results)

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

            extra_context = await self._build_extra_context(symbol, position_context, degradation_context)
            result = await self._run_workflow_analysis(symbol, session, target_allocations, extra_context)

            await self._handle_notifications(result)
            await self._record_analysis_result(symbol, result)
            await self._record_signal_outcome(symbol, result)

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
            if isinstance(e, ValueError) and "No data returned" in str(e):
                self._components.broker_manager.config.remove_watchlist_symbol(symbol)
            error_msg = f"Failed to analyze {symbol}: {e}"
            logger.opt(exception=True).error(error_msg)
            await self.state.record_error(error_msg)
            await self._publish_event("ANALYSIS_ERROR", {"symbol": symbol, "error": str(e)})
            return None

    async def _build_extra_context(
        self,
        symbol: str,
        position_context: dict[str, object] | None,
        degradation_context: DegradationContext | None,
    ) -> WorkflowExtraContext:
        """Build extra context for workflow analysis.

        Args:
            symbol: Stock ticker
            position_context: Position context
            degradation_context: Degradation context

        Returns:
            WorkflowExtraContext instance
        """
        sector_ctx, earnings_ctx, peer_ctx, game_plan_ctx = None, None, None, None
        context_builder = self._context_builder
        if context_builder:
            sector_ctx, earnings_ctx, peer_ctx, game_plan_ctx = context_builder.build_analysis_contexts(
                symbol
            )

        economic_ctx = None
        if self._economic_watcher:
            from src.daemon.events import EconomicRiskLevel

            signal = self._economic_watcher.current_signal
            if signal and signal.risk_level != EconomicRiskLevel.LOW:
                economic_ctx = (
                    f"ECONOMIC RISK: {signal.risk_level} | "
                    f"{signal.recommendation} | {signal.reason} | "
                    f"Events: {', '.join(e.event for e in signal.upcoming_events[:3])}"
                )

        options_flow_ctx = None
        if self._options_flow_watcher:
            sig = self._options_flow_watcher.get_signal(symbol)
            if sig and sig.significance_score >= 0.3:
                options_flow_ctx = (
                    f"OPTIONS FLOW: {sig.net_premium_direction} | "
                    f"P/C={sig.put_call_ratio:.2f} | Vol {sig.volume_vs_avg:.1f}x | "
                    f"Score={sig.significance_score:.2f} | "
                    f"Blocks={len(sig.block_trades)} | {sig.reason}"
                )

        # Fetch active portfolio health constraints
        portfolio_health_ctx = None
        try:
            constraints = await self.state.get_active_constraints()
            if constraints:
                portfolio_health_ctx = f"ACTIVE CONSTRAINTS: {', '.join(constraints)}"
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to fetch portfolio health constraints: {e}")

        social_sentiment_ctx = None
        if self._social_sentiment_watcher:
            sig = self._social_sentiment_watcher.get_signal(symbol)
            if sig and sig.significance_score >= 0.3:
                platforms = ", ".join(f"{p.platform}({p.mention_count})" for p in sig.platform_breakdown)
                social_sentiment_ctx = (
                    f"SOCIAL: {sig.direction} | Buzz={sig.buzz_score:.2f} | "
                    f"Score={sig.significance_score:.2f} | "
                    f"Trending={sig.is_trending} | {platforms} | {sig.reason}"
                )

        recent_trades_ctx = await self._build_recent_trades_context(symbol)
        cooldown_symbols = await self._get_re_entry_cooldown_symbols()

        return WorkflowExtraContext(
            sector_rotation_context=sector_ctx,
            earnings_context=earnings_ctx,
            peer_analysis_context=peer_ctx,
            game_plan_context=game_plan_ctx,
            position_context=position_context,
            degradation_context=degradation_context,
            economic_calendar_context=economic_ctx,
            options_flow_context=options_flow_ctx,
            portfolio_health_context=portfolio_health_ctx,
            social_sentiment_context=social_sentiment_ctx,
            recent_trades_context=recent_trades_ctx,
            re_entry_cooldown_symbols=cooldown_symbols,
        )

    async def _build_recent_trades_context(self, symbol: str) -> str | None:
        """Build recent trade feedback context for trader prompt.

        Args:
            symbol: Stock ticker

        Returns:
            Formatted recent trades string or None
        """
        container = self._components.container if self._components else None
        if not container:
            return None

        try:
            repo = container.trade_repository()
            async with repo:
                recent_trades = await repo.get_recent_closed_by_symbol(symbol, limit=5)
                aggregate = await repo.get_aggregate_stats(days=30)

            if not recent_trades and aggregate.get("total_trades", 0) == 0:
                return None

            lines: list[str] = []
            for trade in recent_trades:
                pnl_str = f"{trade.pnl_percent:+.1f}%" if trade.pnl_percent is not None else "N/A"
                lines.append(f"  {trade.action.value} {trade.symbol} → {pnl_str} ({trade.strategy_name})")

            summary = "\n".join(lines) if lines else "No recent closed trades for this symbol."
            win_rate = aggregate.get("win_rate", 0.0)
            avg_gain = aggregate.get("avg_gain", 0.0)

            return (
                f"## RECENT TRADE HISTORY\n\n{summary}\n\n"
                f"Portfolio aggregate (30d, all symbols): Win Rate {win_rate:.0f}%, Avg Gain {avg_gain:.1f}%"
            )
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to build recent trades context: {e}")
            return None

    async def _get_re_entry_cooldown_symbols(self) -> list[str]:
        """Get symbols with active re-entry cooldown (recently closed).

        Returns:
            List of symbols in cooldown period
        """
        container = self._components.container if self._components else None
        if not container:
            return []

        try:
            from src.daemon.config import PositionManagementConfig

            pos_config = self._components.config.position_management
            if (
                not isinstance(pos_config, PositionManagementConfig)
                or not pos_config.whipsaw_prevention_enabled
            ):
                return []

            cutoff = datetime.now(UTC) - timedelta(hours=pos_config.re_entry_cooldown_hours)
            repo = container.trade_repository()
            async with repo:
                closed_trades = await repo.get_closed_since(cutoff)

            return list({t.symbol for t in closed_trades})
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to check re-entry cooldowns: {e}")
            return []

    async def _run_workflow_analysis(
        self,
        symbol: str,
        session: TradingSession,
        target_allocations: dict[str, float] | None,
        extra_context: WorkflowExtraContext,
    ) -> TradingWorkflowResult:
        """Run workflow analysis with target allocations if provided.

        Args:
            symbol: Stock ticker
            session: Trading session
            target_allocations: Target allocations (optional)
            extra_context: Extra context for analysis

        Returns:
            TradingWorkflowResult
        """
        if target_allocations is not None:
            self.workflow.set_target_allocations(target_allocations)

        try:
            return await self.workflow.analyze(
                symbol,
                period_days=90,
                trading_session=session,
                extra_context=extra_context,
            )
        finally:
            if target_allocations is not None:
                self.workflow.set_target_allocations(None)

    async def _handle_notifications(self, result: TradingWorkflowResult) -> None:
        """Handle notifications for analysis result.

        Args:
            result: TradingWorkflowResult
        """
        if self.notification_service and self._components:
            await self._notification_helper.maybe_notify_signal(result, self._components)
        if (
            self.market_event_queue is not None
            and result.decision.action in (Signal.BUY, Signal.SELL)
            and result.decision.confidence >= _QUEUE_MIN_CONFIDENCE
        ):
            await self._emit_signal_event(result)

    async def _emit_signal_event(self, result: TradingWorkflowResult) -> None:
        """Enqueue SignalEvent for coordinator processing."""
        try:
            from src.strategies.session import TradingSession

            process_after = (
                self.scheduler.next_regular_open()
                if result.trading_session == TradingSession.PRE_MARKET
                else datetime.now(UTC)
            )
            event_id = f"{result.symbol}_{result.decision.action.value}_{process_after.date().isoformat()}"
            signal_event = SignalEvent(
                event_id=event_id,
                symbol=result.symbol,
                signal=result.decision.action.value,
                confidence=result.decision.confidence,
                session=result.trading_session.value,
                reasoning=" ".join(result.decision.reasoning),
            )
            sentiment = Sentiment.BULLISH if result.decision.action == Signal.BUY else Sentiment.BEARISH
            triage = TriageResult(
                event_id=signal_event.event_id,
                event_type=signal_event.event_type,
                relevance=result.decision.confidence,
                symbols=[result.symbol],
                urgency=Urgency.IMMEDIATE,
                sentiment=sentiment,
                confidence=result.decision.confidence,
                reasoning=" ".join(result.decision.reasoning),
            )

            seconds_until = (process_after - datetime.now(UTC)).total_seconds()
            ttl_hours = max(8, int(seconds_until / 3600) + 2)
            queue = self.market_event_queue
            if queue is None:
                return
            from src.daemon.events import BaseEvent

            await queue.enqueue(
                cast("BaseEvent", signal_event), triage, ttl_hours=ttl_hours, process_after=process_after
            )
            logger.info(
                "Queued {} for {}, process_after={}", signal_event.signal, result.symbol, process_after
            )
        except Exception:
            logger.opt(exception=True).warning("Failed to enqueue signal event for {}", result.symbol)

    async def _record_analysis_result(self, symbol: str, result: TradingWorkflowResult) -> None:
        """Record analysis result to daemon state.

        Args:
            symbol: Stock ticker
            result: TradingWorkflowResult
        """
        rsi = result.technical.rsi if result.technical else None
        macd_hist = result.technical.macd_hist if result.technical else None

        technical_reasoning = result.technical.interpretation if result.technical else None
        sentiment_reasoning = result.sentiment.summary if result.sentiment else None
        news_reasoning = (
            f"{result.news.impact_assessment}\n\nRecommendation: {result.news.recommendation}"
            if result.news
            else None
        )

        input_data = AnalysisRecordInput(
            symbol=symbol,
            signal=result.decision.action.value,
            confidence=result.decision.confidence,
            executed=result.order is not None,
            trading_session=result.trading_session,
            is_paper_trade=self.trading_mode == "paper",
            rsi=rsi,
            macd_hist=macd_hist,
            reasoning=result.decision.reasoning,
            technical_analysis_reasoning=technical_reasoning,
            sentiment_analysis_reasoning=sentiment_reasoning,
            news_analysis_reasoning=news_reasoning,
        )
        await self.state.record_analysis(input_data)

    async def _record_signal_outcome(self, symbol: str, result: TradingWorkflowResult) -> None:
        """Record signal outcome to PostgreSQL or SQLite.

        Args:
            symbol: Stock ticker
            result: TradingWorkflowResult
        """
        container = self._components.container if self._components else None
        if container:
            await self._record_signal_to_postgres(symbol, result)
        elif self.historical_cache:
            self._record_signal_to_sqlite(symbol, result)

    async def _record_signal_to_postgres(self, symbol: str, result: TradingWorkflowResult) -> None:
        """Record signal to PostgreSQL with per-request repo.

        Args:
            symbol: Stock ticker
            result: TradingWorkflowResult
        """
        from src.database.repositories.signal_outcome import SignalRecordInput

        container = self._components.container if self._components else None
        if not container:
            return

        try:
            input_data = SignalRecordInput(
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
                technical_reasoning=result.technical.interpretation if result.technical else None,
                sentiment_reasoning=result.sentiment.summary if result.sentiment else None,
                news_reasoning=(
                    f"{result.news.impact_assessment}\n\nRecommendation: {result.news.recommendation}"
                    if result.news
                    else None
                ),
            )
            repo = container.signal_outcome_repository()
            async with repo:
                await repo.record_signal(input_data)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to record signal outcome: {e}")

    def _record_signal_to_sqlite(self, symbol: str, result: TradingWorkflowResult) -> None:
        """Record signal to SQLite historical cache.

        Args:
            symbol: Stock ticker
            result: TradingWorkflowResult
        """
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
                technical_reasoning=result.technical.interpretation if result.technical else None,
                sentiment_reasoning=result.sentiment.summary if result.sentiment else None,
                news_reasoning=(
                    f"{result.news.impact_assessment}\n\nRecommendation: {result.news.recommendation}"
                    if result.news
                    else None
                ),
            )
            historical_cache = self.historical_cache
            if historical_cache:
                historical_cache.record_signal_outcome(signal_input)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to record signal outcome for accuracy tracking: {e}")

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
                logger.opt(exception=True).warning(f"Event publish failed: {e}")

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
