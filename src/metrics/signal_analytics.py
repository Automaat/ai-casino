"""Signal analytics service for tracking signal accuracy and execution."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

from sqlalchemy import and_, select

from src.database.connection import get_db_engine
from src.database.models import AnalysisRecordORM, SignalOutcomeORM

# Confidence bucket thresholds
CONF_BUCKET_0_5 = 0.5
CONF_BUCKET_0_6 = 0.6
CONF_BUCKET_0_7 = 0.7
CONF_BUCKET_0_8 = 0.8
CONF_BUCKET_0_9 = 0.9
CONF_BUCKET_1_0 = 1.0


@dataclass
class SignalFlowSummary:
    """Aggregated signal flow metrics."""

    total_signals: int
    total_buy_signals: int
    total_sell_signals: int
    execution_rate: float
    executed_count: int
    not_executed_count: int
    profitable_count: int
    unprofitable_count: int
    overall_accuracy: float
    avg_confidence: float
    date_range: tuple[datetime, datetime]


@dataclass
class SankeyFlowData:
    """Sankey diagram data structure."""

    nodes: list[dict[str, str | dict[str, str]]]
    links: list[dict[str, str | int]]


@dataclass
class _CategorizedSignals:
    """Categorized signals for Sankey diagram."""

    buy_executed: list[SignalOutcomeORM]
    buy_not_executed: list[SignalOutcomeORM]
    sell_executed: list[SignalOutcomeORM]
    sell_not_executed: list[SignalOutcomeORM]
    buy_profitable: list[SignalOutcomeORM]
    buy_unprofitable: list[SignalOutcomeORM]
    sell_profitable: list[SignalOutcomeORM]
    sell_unprofitable: list[SignalOutcomeORM]


@dataclass
class AccuracyByType:
    """Signal accuracy by signal type."""

    signal_type: str
    horizon: str
    hit_rate: float
    executed_count: int
    total_count: int


@dataclass
class CalibrationBucket:
    """Calibration curve bucket."""

    confidence_bucket: str
    expected_confidence: float
    actual_accuracy: float
    sample_count: int


@dataclass
class CalibrationCurveData:
    """Calibration curve data."""

    buckets: list[CalibrationBucket]


@dataclass
class TimingAnalysis:
    """Signal timing analysis."""

    avg_execution_delay_hours: float
    by_confidence_bucket: dict[str, float]


@dataclass
class ExecutionRate:
    """Execution rate by confidence bucket."""

    confidence_bucket: str
    execution_rate: float
    executed_count: int
    total_count: int


class SignalAnalyticsService:
    """Service for analyzing signal accuracy and execution patterns."""

    def __init__(self) -> None:
        """Initialize signal analytics service."""
        self._cache: dict[str, tuple[datetime, object]] = {}
        self._cache_ttl = 300  # 5 minutes
        self._correlation_window_seconds = 300  # 5 minutes

    def __repr__(self) -> str:
        """Return string representation."""
        return "SignalAnalyticsService()"

    def _was_signal_executed(
        self,
        signal: SignalOutcomeORM,
        analyses: Sequence[AnalysisRecordORM],
    ) -> bool:
        """Check if signal was executed based on correlation with analyses.

        Args:
            signal: Signal to check
            analyses: List of analysis records

        Returns:
            True if signal was executed
        """
        for analysis in analyses:
            if (
                signal.symbol == analysis.symbol
                and signal.signal == analysis.signal
                and abs((signal.timestamp - analysis.timestamp).total_seconds())
                < self._correlation_window_seconds
                and analysis.executed_trade
            ):
                return True
        return False

    async def get_flow_summary(self, start: datetime, end: datetime) -> SignalFlowSummary:
        """Get signal flow summary for date range.

        Args:
            start: Start timestamp (inclusive)
            end: End timestamp (inclusive)

        Returns:
            Signal flow summary metrics
        """
        cache_key = f"flow_summary:{start.isoformat()}:{end.isoformat()}"
        cached = self._get_cached(cache_key)
        if cached:
            return cast("SignalFlowSummary", cached)

        engine = get_db_engine()
        async with engine.session() as session:
            # Get all signals in date range (exclude HOLD)
            signals_stmt = select(SignalOutcomeORM).where(
                and_(
                    SignalOutcomeORM.timestamp >= start,
                    SignalOutcomeORM.timestamp <= end,
                    SignalOutcomeORM.signal.in_(["BUY", "SELL"]),
                )
            )
            signals_result = await session.execute(signals_stmt)
            signals = signals_result.scalars().all()

            total_signals = len(signals)
            total_buy = sum(1 for s in signals if s.signal == "BUY")
            total_sell = sum(1 for s in signals if s.signal == "SELL")

            # Get analysis records (executions) in expanded window for correlation
            analyses_stmt = select(AnalysisRecordORM).where(
                and_(
                    AnalysisRecordORM.timestamp >= start,
                    AnalysisRecordORM.timestamp <= end,
                    AnalysisRecordORM.signal.in_(["BUY", "SELL"]),
                )
            )
            analyses_result = await session.execute(analyses_stmt)
            analyses = analyses_result.scalars().all()

            # Correlate signals with executions (5-min window)
            executed_signals = set()
            for signal in signals:
                for analysis in analyses:
                    if (
                        signal.symbol == analysis.symbol
                        and signal.signal == analysis.signal
                        and abs((signal.timestamp - analysis.timestamp).total_seconds())
                        < self._correlation_window_seconds
                        and analysis.executed_trade
                    ):
                        executed_signals.add(str(signal.id))
                        break

            executed_count = len(executed_signals)
            not_executed_count = total_signals - executed_count
            execution_rate = (executed_count / total_signals) if total_signals > 0 else 0.0

            # Calculate profitability for executed signals with outcomes (5d horizon)
            profitable_count = 0
            unprofitable_count = 0

            for signal in signals:
                if str(signal.id) not in executed_signals:
                    continue

                # Check if we have outcome data (5d)
                if signal.price_at_5d is None:
                    continue

                is_correct = self._is_signal_correct(signal, "5d")
                if is_correct is True:
                    profitable_count += 1
                elif is_correct is False:
                    unprofitable_count += 1

            # Overall accuracy (5d horizon for executed signals with data)
            total_with_outcome = profitable_count + unprofitable_count
            overall_accuracy = (profitable_count / total_with_outcome) if total_with_outcome > 0 else 0.0

            # Average confidence
            avg_confidence = (
                (sum(float(s.confidence) for s in signals) / total_signals) if total_signals > 0 else 0.0
            )

            summary = SignalFlowSummary(
                total_signals=total_signals,
                total_buy_signals=total_buy,
                total_sell_signals=total_sell,
                execution_rate=execution_rate,
                executed_count=executed_count,
                not_executed_count=not_executed_count,
                profitable_count=profitable_count,
                unprofitable_count=unprofitable_count,
                overall_accuracy=overall_accuracy,
                avg_confidence=avg_confidence,
                date_range=(start, end),
            )

            self._cache[cache_key] = (datetime.now(UTC), summary)
            return summary

    def _categorize_by_profitability(
        self, executed_signals: list[SignalOutcomeORM], horizon: str = "5d"
    ) -> tuple[list[SignalOutcomeORM], list[SignalOutcomeORM]]:
        """Categorize executed signals into profitable and unprofitable.

        Args:
            executed_signals: List of executed signals
            horizon: Time horizon for profitability check

        Returns:
            Tuple of (profitable_signals, unprofitable_signals)
        """
        profitable = []
        unprofitable = []

        for signal in executed_signals:
            price_field = f"price_at_{horizon}"
            if getattr(signal, price_field) is None:
                continue
            is_correct = self._is_signal_correct(signal, horizon)
            if is_correct is True:
                profitable.append(signal)
            elif is_correct is False:
                unprofitable.append(signal)

        return profitable, unprofitable

    def _build_sankey_nodes(self) -> list[dict[str, str | dict[str, str]]]:
        """Build Sankey diagram nodes with colors."""
        return [
            {"name": "BUY", "itemStyle": {"color": "#10b981"}},
            {"name": "SELL", "itemStyle": {"color": "#ef4444"}},
            {"name": "Executed", "itemStyle": {"color": "#3b82f6"}},
            {"name": "Not Executed", "itemStyle": {"color": "#9ca3af"}},
            {"name": "Profitable", "itemStyle": {"color": "#059669"}},
            {"name": "Unprofitable", "itemStyle": {"color": "#dc2626"}},
        ]

    def _build_sankey_links(self, categorized: _CategorizedSignals) -> list[dict[str, str | int]]:
        """Build Sankey diagram links with values.

        Args:
            categorized: Categorized signals data

        Returns:
            List of Sankey links
        """
        links = [
            {"source": "BUY", "target": "Executed", "value": len(categorized.buy_executed)},
            {"source": "BUY", "target": "Not Executed", "value": len(categorized.buy_not_executed)},
            {"source": "SELL", "target": "Executed", "value": len(categorized.sell_executed)},
            {"source": "SELL", "target": "Not Executed", "value": len(categorized.sell_not_executed)},
            {
                "source": "Executed",
                "target": "Profitable",
                "value": len(categorized.buy_profitable) + len(categorized.sell_profitable),
            },
            {
                "source": "Executed",
                "target": "Unprofitable",
                "value": len(categorized.buy_unprofitable) + len(categorized.sell_unprofitable),
            },
        ]
        return [link for link in links if cast("int", link["value"]) > 0]

    async def get_sankey_data(self, start: datetime, end: datetime) -> SankeyFlowData:
        """Get Sankey diagram data for signal flow.

        Flow: BUY/SELL → Executed/Not Executed → Profitable/Unprofitable

        Args:
            start: Start timestamp
            end: End timestamp

        Returns:
            Sankey flow data with nodes and links
        """
        cache_key = f"sankey:{start.isoformat()}:{end.isoformat()}"
        cached = self._get_cached(cache_key)
        if cached:
            return cast("SankeyFlowData", cached)

        engine = get_db_engine()
        async with engine.session() as session:
            # Get signals (BUY/SELL only)
            signals_stmt = select(SignalOutcomeORM).where(
                and_(
                    SignalOutcomeORM.timestamp >= start,
                    SignalOutcomeORM.timestamp <= end,
                    SignalOutcomeORM.signal.in_(["BUY", "SELL"]),
                )
            )
            signals_result = await session.execute(signals_stmt)
            signals = signals_result.scalars().all()

            # Get analysis records
            analyses_stmt = select(AnalysisRecordORM).where(
                and_(
                    AnalysisRecordORM.timestamp >= start,
                    AnalysisRecordORM.timestamp <= end,
                    AnalysisRecordORM.signal.in_(["BUY", "SELL"]),
                )
            )
            analyses_result = await session.execute(analyses_stmt)
            analyses = analyses_result.scalars().all()

            # Correlate and categorize
            buy_executed = []
            buy_not_executed = []
            sell_executed = []
            sell_not_executed = []

            for signal in signals:
                was_executed = self._was_signal_executed(signal, analyses)

                if signal.signal == "BUY":
                    target = buy_executed if was_executed else buy_not_executed
                else:
                    target = sell_executed if was_executed else sell_not_executed
                target.append(signal)

            # Further categorize executed signals by profitability
            buy_profitable, buy_unprofitable = self._categorize_by_profitability(buy_executed)
            sell_profitable, sell_unprofitable = self._categorize_by_profitability(sell_executed)

            # Build nodes and links
            categorized = _CategorizedSignals(
                buy_executed=buy_executed,
                buy_not_executed=buy_not_executed,
                sell_executed=sell_executed,
                sell_not_executed=sell_not_executed,
                buy_profitable=buy_profitable,
                buy_unprofitable=buy_unprofitable,
                sell_profitable=sell_profitable,
                sell_unprofitable=sell_unprofitable,
            )
            nodes = self._build_sankey_nodes()
            links = self._build_sankey_links(categorized)

            data = SankeyFlowData(nodes=nodes, links=links)
            self._cache[cache_key] = (datetime.now(UTC), data)
            return data

    def _find_executed_signals(
        self,
        signals: Sequence[SignalOutcomeORM],
        analyses: Sequence[AnalysisRecordORM],
    ) -> set[str]:
        """Find signals that were executed based on correlation with analyses.

        Args:
            signals: List of signals
            analyses: List of analysis records

        Returns:
            Set of executed signal IDs
        """
        executed_signal_ids = set()
        for signal in signals:
            if self._was_signal_executed(signal, analyses):
                executed_signal_ids.add(str(signal.id))
        return executed_signal_ids

    def _calculate_type_accuracy(
        self,
        signals: Sequence[SignalOutcomeORM],
        executed_signal_ids: set[str],
        horizon: str,
        signal_type: str,
    ) -> tuple[int, int, int]:
        """Calculate accuracy metrics for a specific signal type.

        Args:
            signals: List of all signals
            executed_signal_ids: Set of executed signal IDs
            horizon: Time horizon
            signal_type: Signal type (BUY/SELL)

        Returns:
            Tuple of (hits, total, executed_count)
        """
        hits = 0
        total = 0
        executed_count = 0

        for signal in signals:
            if signal.signal != signal_type:
                continue

            is_correct = self._is_signal_correct(signal, horizon)
            if is_correct is None:
                continue

            total += 1
            if is_correct:
                hits += 1
            if str(signal.id) in executed_signal_ids:
                executed_count += 1

        return hits, total, executed_count

    async def get_accuracy_by_type(
        self,
        start: datetime,
        end: datetime,
        horizon: str = "5d",
    ) -> list[AccuracyByType]:
        """Get accuracy by signal type (BUY/SELL) at horizon.

        Args:
            start: Start timestamp
            end: End timestamp
            horizon: Time horizon (1d/5d/20d)

        Returns:
            List of accuracy by type
        """
        cache_key = f"accuracy_by_type:{start.isoformat()}:{end.isoformat()}:{horizon}"
        cached = self._get_cached(cache_key)
        if cached:
            return cast("list[AccuracyByType]", cached)

        if horizon not in ["1d", "5d", "20d"]:
            msg = f"Invalid horizon: {horizon}. Must be one of: 1d, 5d, 20d"
            raise ValueError(msg)

        engine = get_db_engine()
        async with engine.session() as session:
            # Get signals with outcome data
            price_field = f"price_at_{horizon}"
            signals_stmt = select(SignalOutcomeORM).where(
                and_(
                    SignalOutcomeORM.timestamp >= start,
                    SignalOutcomeORM.timestamp <= end,
                    SignalOutcomeORM.signal.in_(["BUY", "SELL"]),
                    getattr(SignalOutcomeORM, price_field).is_not(None),
                )
            )
            signals_result = await session.execute(signals_stmt)
            signals = signals_result.scalars().all()

            # Get executions
            analyses_stmt = select(AnalysisRecordORM).where(
                and_(
                    AnalysisRecordORM.timestamp >= start,
                    AnalysisRecordORM.timestamp <= end,
                    AnalysisRecordORM.signal.in_(["BUY", "SELL"]),
                )
            )
            analyses_result = await session.execute(analyses_stmt)
            analyses = analyses_result.scalars().all()

            # Correlate to find executed signals
            executed_signal_ids = self._find_executed_signals(signals, analyses)

            # Calculate accuracy by type
            buy_hits, buy_total, buy_executed = self._calculate_type_accuracy(
                signals, executed_signal_ids, horizon, "BUY"
            )
            sell_hits, sell_total, sell_executed = self._calculate_type_accuracy(
                signals, executed_signal_ids, horizon, "SELL"
            )

            result = [
                AccuracyByType(
                    signal_type="BUY",
                    horizon=horizon,
                    hit_rate=(buy_hits / buy_total) if buy_total > 0 else 0.0,
                    executed_count=buy_executed,
                    total_count=buy_total,
                ),
                AccuracyByType(
                    signal_type="SELL",
                    horizon=horizon,
                    hit_rate=(sell_hits / sell_total) if sell_total > 0 else 0.0,
                    executed_count=sell_executed,
                    total_count=sell_total,
                ),
            ]

            self._cache[cache_key] = (datetime.now(UTC), result)
            return result

    async def get_calibration_curves(
        self,
        start: datetime,
        end: datetime,
        horizon: str = "5d",
    ) -> CalibrationCurveData:
        """Get calibration curve data (confidence vs actual accuracy).

        Args:
            start: Start timestamp
            end: End timestamp
            horizon: Time horizon (1d/5d/20d)

        Returns:
            Calibration curve data
        """
        cache_key = f"calibration:{start.isoformat()}:{end.isoformat()}:{horizon}"
        cached = self._get_cached(cache_key)
        if cached:
            return cast("CalibrationCurveData", cached)

        if horizon not in ["1d", "5d", "20d"]:
            msg = f"Invalid horizon: {horizon}. Must be one of: 1d, 5d, 20d"
            raise ValueError(msg)

        engine = get_db_engine()
        async with engine.session() as session:
            # Get signals with outcome data
            price_field = f"price_at_{horizon}"
            signals_stmt = select(SignalOutcomeORM).where(
                and_(
                    SignalOutcomeORM.timestamp >= start,
                    SignalOutcomeORM.timestamp <= end,
                    SignalOutcomeORM.signal.in_(["BUY", "SELL"]),
                    getattr(SignalOutcomeORM, price_field).is_not(None),
                )
            )
            signals_result = await session.execute(signals_stmt)
            signals = signals_result.scalars().all()

            # Bucket by confidence and calculate hit rate
            buckets_data: dict[str, list[bool]] = {
                "0.5-0.6": [],
                "0.6-0.7": [],
                "0.7-0.8": [],
                "0.8-0.9": [],
                "0.9-1.0": [],
            }

            for signal in signals:
                confidence = float(signal.confidence)
                bucket = self._get_confidence_bucket(confidence)
                if bucket is None:
                    continue

                is_correct = self._is_signal_correct(signal, horizon)
                if is_correct is not None:
                    buckets_data[bucket].append(is_correct)

            # Calculate accuracy per bucket
            calibration_buckets = []
            for bucket_name, results in buckets_data.items():
                if not results:
                    continue

                # Expected confidence is midpoint of bucket
                bucket_min = float(bucket_name.split("-")[0])
                bucket_max = float(bucket_name.split("-")[1])
                expected_conf = (bucket_min + bucket_max) / 2

                actual_accuracy = sum(1 for r in results if r) / len(results)

                calibration_buckets.append(
                    CalibrationBucket(
                        confidence_bucket=bucket_name,
                        expected_confidence=expected_conf,
                        actual_accuracy=actual_accuracy,
                        sample_count=len(results),
                    )
                )

            # Sort by expected confidence
            calibration_buckets.sort(key=lambda b: b.expected_confidence)

            data = CalibrationCurveData(buckets=calibration_buckets)
            self._cache[cache_key] = (datetime.now(UTC), data)
            return data

    async def get_timing_analysis(self, start: datetime, end: datetime) -> TimingAnalysis:
        """Get signal timing analysis (signal → execution delay).

        Args:
            start: Start timestamp
            end: End timestamp

        Returns:
            Timing analysis with avg delay by confidence bucket
        """
        cache_key = f"timing:{start.isoformat()}:{end.isoformat()}"
        cached = self._get_cached(cache_key)
        if cached:
            return cast("TimingAnalysis", cached)

        engine = get_db_engine()
        async with engine.session() as session:
            # Get signals
            signals_stmt = select(SignalOutcomeORM).where(
                and_(
                    SignalOutcomeORM.timestamp >= start,
                    SignalOutcomeORM.timestamp <= end,
                    SignalOutcomeORM.signal.in_(["BUY", "SELL"]),
                )
            )
            signals_result = await session.execute(signals_stmt)
            signals = signals_result.scalars().all()

            # Get executions
            analyses_stmt = select(AnalysisRecordORM).where(
                and_(
                    AnalysisRecordORM.timestamp >= start,
                    AnalysisRecordORM.timestamp <= end,
                    AnalysisRecordORM.signal.in_(["BUY", "SELL"]),
                )
            )
            analyses_result = await session.execute(analyses_stmt)
            analyses = analyses_result.scalars().all()

            # Calculate delays
            delays_by_bucket: dict[str, list[float]] = {
                "0.5-0.6": [],
                "0.6-0.7": [],
                "0.7-0.8": [],
                "0.8-0.9": [],
                "0.9-1.0": [],
            }
            all_delays = []

            for signal in signals:
                for analysis in analyses:
                    if (
                        signal.symbol == analysis.symbol
                        and signal.signal == analysis.signal
                        and abs((signal.timestamp - analysis.timestamp).total_seconds())
                        < self._correlation_window_seconds
                        and analysis.executed_trade
                    ):
                        # Calculate delay in hours
                        delay_seconds = abs((analysis.timestamp - signal.timestamp).total_seconds())
                        delay_hours = delay_seconds / 3600

                        all_delays.append(delay_hours)

                        # Add to confidence bucket
                        confidence = float(signal.confidence)
                        bucket = self._get_confidence_bucket(confidence)
                        if bucket:
                            delays_by_bucket[bucket].append(delay_hours)
                        break

            # Calculate averages
            avg_delay = (sum(all_delays) / len(all_delays)) if all_delays else 0.0

            by_bucket = {}
            for bucket, delays in delays_by_bucket.items():
                by_bucket[bucket] = (sum(delays) / len(delays)) if delays else 0.0

            timing = TimingAnalysis(
                avg_execution_delay_hours=avg_delay,
                by_confidence_bucket=by_bucket,
            )

            self._cache[cache_key] = (datetime.now(UTC), timing)
            return timing

    async def get_execution_rate_by_confidence(
        self,
        start: datetime,
        end: datetime,
    ) -> list[ExecutionRate]:
        """Get execution rate by confidence bucket.

        Args:
            start: Start timestamp
            end: End timestamp

        Returns:
            List of execution rates by confidence bucket
        """
        cache_key = f"execution_rate:{start.isoformat()}:{end.isoformat()}"
        cached = self._get_cached(cache_key)
        if cached:
            return cast("list[ExecutionRate]", cached)

        engine = get_db_engine()
        async with engine.session() as session:
            # Get signals
            signals_stmt = select(SignalOutcomeORM).where(
                and_(
                    SignalOutcomeORM.timestamp >= start,
                    SignalOutcomeORM.timestamp <= end,
                    SignalOutcomeORM.signal.in_(["BUY", "SELL"]),
                )
            )
            signals_result = await session.execute(signals_stmt)
            signals = signals_result.scalars().all()

            # Get executions
            analyses_stmt = select(AnalysisRecordORM).where(
                and_(
                    AnalysisRecordORM.timestamp >= start,
                    AnalysisRecordORM.timestamp <= end,
                    AnalysisRecordORM.signal.in_(["BUY", "SELL"]),
                )
            )
            analyses_result = await session.execute(analyses_stmt)
            analyses = analyses_result.scalars().all()

            # Categorize by bucket
            bucket_data: dict[str, dict[str, int]] = {
                "0.5-0.6": {"total": 0, "executed": 0},
                "0.6-0.7": {"total": 0, "executed": 0},
                "0.7-0.8": {"total": 0, "executed": 0},
                "0.8-0.9": {"total": 0, "executed": 0},
                "0.9-1.0": {"total": 0, "executed": 0},
            }

            for signal in signals:
                confidence = float(signal.confidence)
                bucket = self._get_confidence_bucket(confidence)
                if bucket is None:
                    continue

                bucket_data[bucket]["total"] += 1

                # Check if executed
                for analysis in analyses:
                    if (
                        signal.symbol == analysis.symbol
                        and signal.signal == analysis.signal
                        and abs((signal.timestamp - analysis.timestamp).total_seconds())
                        < self._correlation_window_seconds
                        and analysis.executed_trade
                    ):
                        bucket_data[bucket]["executed"] += 1
                        break

            # Calculate execution rates
            result = []
            for bucket, data in bucket_data.items():
                total = data["total"]
                executed = data["executed"]
                rate = (executed / total) if total > 0 else 0.0

                result.append(
                    ExecutionRate(
                        confidence_bucket=bucket,
                        execution_rate=rate,
                        executed_count=executed,
                        total_count=total,
                    )
                )

            # Sort by bucket
            result.sort(key=lambda r: float(r.confidence_bucket.split("-")[0]))

            self._cache[cache_key] = (datetime.now(UTC), result)
            return result

    def _is_signal_correct(self, signal: SignalOutcomeORM, horizon: str) -> bool | None:
        """Check if signal prediction was correct.

        Args:
            signal: Signal outcome ORM
            horizon: Time horizon (1d/5d/20d)

        Returns:
            True if correct, False if incorrect, None if no data
        """
        price_future = signal.actual_exit_price or getattr(signal, f"price_at_{horizon}")
        if price_future is None:
            return None

        signal_type = signal.signal
        if signal_type == "BUY":
            return float(price_future) > float(signal.price_at_signal)
        if signal_type == "SELL":
            return float(price_future) < float(signal.price_at_signal)
        return None

    def _get_confidence_bucket(self, confidence: float) -> str | None:
        """Get confidence bucket label for a confidence value.

        Args:
            confidence: Confidence value (0.0-1.0)

        Returns:
            Bucket label or None if out of range
        """
        if CONF_BUCKET_0_5 <= confidence < CONF_BUCKET_0_6:
            return "0.5-0.6"
        if CONF_BUCKET_0_6 <= confidence < CONF_BUCKET_0_7:
            return "0.6-0.7"
        if CONF_BUCKET_0_7 <= confidence < CONF_BUCKET_0_8:
            return "0.7-0.8"
        if CONF_BUCKET_0_8 <= confidence < CONF_BUCKET_0_9:
            return "0.8-0.9"
        if CONF_BUCKET_0_9 <= confidence <= CONF_BUCKET_1_0:
            return "0.9-1.0"
        return None

    def _get_cached(self, key: str) -> object | None:
        """Get cached value if not expired.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        if key not in self._cache:
            return None

        cached_time, value = self._cache[key]
        if (datetime.now(UTC) - cached_time).total_seconds() > self._cache_ttl:
            del self._cache[key]
            return None

        return value
