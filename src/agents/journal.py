"""Trade Journal Agent for after-hours signal review."""

import asyncio
from datetime import date
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from src.daemon.state import AnalysisRecord
from src.data.market import MarketDataFetcher
from src.models.llm import LLMClient
from src.models.providers.base import StructuredOutputError
from src.prompts import PromptLoader


class JournalLLMResponse(BaseModel):
    """LLM structured output for journal analysis."""

    winners: list[str] = Field(description="Symbols/factors that worked well today")
    losers: list[str] = Field(description="Symbols/factors that underperformed")
    lessons: list[str] = Field(description="Key lessons learned from today's signals")
    tomorrows_focus: list[str] = Field(description="Areas to focus on tomorrow")
    overall_assessment: str = Field(description="Overall assessment of today's signal accuracy")


class SignalOutcome(BaseModel):
    """Per-symbol signal vs actual price outcome."""

    symbol: str
    signal: str
    confidence: float
    price_open: float
    price_close: float
    price_change_pct: float
    signal_correct: bool


class DailyJournal(BaseModel):
    """Daily trade journal entry."""

    date: date
    outcomes: list[SignalOutcome]
    winners: list[str]
    losers: list[str]
    lessons: list[str]
    tomorrows_focus: list[str]
    overall_assessment: str


def _filter_outcomes(raw_outcomes: list[SignalOutcome | BaseException | None]) -> list[SignalOutcome]:
    """Filter gather results, logging exceptions."""
    outcomes: list[SignalOutcome] = []
    for o in raw_outcomes:
        if isinstance(o, BaseException):
            # Re-raise cancellation/shutdown exceptions
            if isinstance(o, (asyncio.CancelledError, KeyboardInterrupt)):
                raise o
            logger.error(f"Outcome fetch failed: {o}")
        elif o is not None:
            outcomes.append(o)
    return outcomes


class TradeJournalAgent:
    """Agent that reviews day's trading signals against actual price movement."""

    def __init__(self, llm_client: LLMClient, market_fetcher: MarketDataFetcher) -> None:
        """Initialize trade journal agent.

        Args:
            llm_client: LLM client for analysis
            market_fetcher: Market data fetcher for closing prices
        """
        self.llm = llm_client
        self.market_fetcher = market_fetcher
        self._prompts = PromptLoader("journal")
        logger.info("Initialized TradeJournalAgent")

    async def generate(self, journal_date: date, records: list[AnalysisRecord]) -> DailyJournal:
        """Generate daily journal from analysis records.

        Args:
            journal_date: Date to generate journal for
            records: Analysis records from the day

        Returns:
            DailyJournal with outcomes and LLM assessment
        """
        if not records:
            return self._create_empty_journal(journal_date, "No signals generated today")

        latest_by_symbol = self._deduplicate_records(records)
        outcomes = await self._build_outcomes(latest_by_symbol)

        if not outcomes:
            return self._create_empty_journal(journal_date, "Could not fetch market data for any symbols")

        correct_count = sum(1 for o in outcomes if o.signal_correct)
        accuracy_pct = round((correct_count / len(outcomes)) * 100, 1)

        journal_data = await self._get_llm_assessment(journal_date, outcomes, accuracy_pct)

        logger.info(f"Journal generated: {accuracy_pct}% accuracy across {len(outcomes)} symbols")

        return DailyJournal(date=journal_date, outcomes=outcomes, **journal_data)

    def _evaluate_signal(self, signal: str, price_change_pct: float) -> bool:
        """Evaluate if signal direction matched actual price movement.

        Args:
            signal: Trading signal (BUY/SELL/HOLD)
            price_change_pct: Actual price change percentage

        Returns:
            True if signal was directionally correct
        """
        if signal == "BUY":
            return price_change_pct > 0
        if signal == "SELL":
            return price_change_pct < 0
        # HOLD is correct if price didn't move much
        return abs(price_change_pct) < 1.0

    def _format_outcomes(self, outcomes: list[SignalOutcome]) -> str:
        """Format outcomes for LLM prompt.

        Args:
            outcomes: List of signal outcomes

        Returns:
            Formatted table text
        """
        lines = ["Symbol | Signal | Confidence | Open | Close | Change% | Correct"]
        lines.append("-------|--------|------------|------|-------|---------|--------")
        for o in outcomes:
            correct_mark = "YES" if o.signal_correct else "NO"
            lines.append(
                f"{o.symbol} | {o.signal} | {o.confidence:.2f} | "
                f"${o.price_open:.2f} | ${o.price_close:.2f} | "
                f"{o.price_change_pct:+.2f}% | {correct_mark}"
            )
        return "\n".join(lines)

    def persist(self, journal: DailyJournal, journal_dir: str) -> Path:
        """Write journal entry as markdown file.

        Args:
            journal: Daily journal to persist
            journal_dir: Directory path (supports ~ expansion)

        Returns:
            Path to written file
        """
        dir_path = Path(journal_dir).expanduser()
        dir_path.mkdir(parents=True, exist_ok=True)

        file_path = dir_path / f"{journal.date}.md"
        content = self._render_markdown(journal)
        file_path.write_text(content, encoding="utf-8")
        logger.info(f"Journal persisted to {file_path}")
        return file_path

    def _render_markdown(self, journal: DailyJournal) -> str:
        """Render journal as markdown.

        Args:
            journal: Daily journal to render

        Returns:
            Markdown string
        """
        lines = [f"# Trade Journal — {journal.date}", ""]

        if journal.outcomes:
            correct = sum(1 for o in journal.outcomes if o.signal_correct)
            total = len(journal.outcomes)
            lines.append(f"**Accuracy:** {correct}/{total} ({(correct / total) * 100:.1f}%)")
            lines.append("")
            lines.append("## Signal Outcomes")
            lines.append("")
            lines.append("| Symbol | Signal | Confidence | Open | Close | Change% | Correct |")
            lines.append("|--------|--------|------------|------|-------|---------|---------|")
            for o in journal.outcomes:
                correct_mark = "YES" if o.signal_correct else "NO"
                lines.append(
                    f"| {o.symbol} | {o.signal} | {o.confidence:.2f} | "
                    f"${o.price_open:.2f} | ${o.price_close:.2f} | "
                    f"{o.price_change_pct:+.2f}% | {correct_mark} |"
                )
            lines.append("")

        for heading, items in [
            ("Winners", journal.winners),
            ("Losers", journal.losers),
            ("Lessons", journal.lessons),
            ("Tomorrow's Focus", journal.tomorrows_focus),
        ]:
            if items:
                lines.append(f"## {heading}")
                lines.append("")
                lines.extend(f"- {item}" for item in items)
                lines.append("")

        lines.append("## Overall Assessment")
        lines.append("")
        lines.append(journal.overall_assessment)
        lines.append("")

        return "\n".join(lines)

    def _create_empty_journal(self, journal_date: date, reason: str) -> DailyJournal:
        """Create empty journal when no data available."""
        logger.warning(f"Creating empty journal: {reason}")
        return DailyJournal(
            date=journal_date,
            outcomes=[],
            winners=[],
            losers=[],
            lessons=[reason],
            tomorrows_focus=["Generate trading signals" if "signals" in reason else "Check market data"],
            overall_assessment=f"No signals to evaluate — {reason.lower()}",
        )

    def _deduplicate_records(self, records: list[AnalysisRecord]) -> dict[str, AnalysisRecord]:
        """Keep latest signal per symbol."""
        latest_by_symbol: dict[str, AnalysisRecord] = {}
        for record in records:
            if (
                record.symbol not in latest_by_symbol
                or record.timestamp > latest_by_symbol[record.symbol].timestamp
            ):
                latest_by_symbol[record.symbol] = record
        return latest_by_symbol

    async def _build_outcomes(self, latest_by_symbol: dict[str, AnalysisRecord]) -> list[SignalOutcome]:
        """Build outcomes by fetching closing prices (async with concurrency limit)."""
        semaphore = asyncio.Semaphore(5)

        async def fetch_outcome(symbol: str, record: AnalysisRecord) -> SignalOutcome | None:
            async with semaphore:
                try:
                    market_data = await asyncio.to_thread(self.market_fetcher.fetch_daily, symbol, 1)
                    df = market_data.data
                    if df.empty:
                        logger.warning(f"No market data for {symbol}, skipping")
                        return None

                    price_open = float(df["Open"].iloc[-1])
                    price_close = float(df["Close"].iloc[-1])
                    price_change_pct = ((price_close - price_open) / price_open) * 100

                    return SignalOutcome(
                        symbol=symbol,
                        signal=record.signal,
                        confidence=record.confidence,
                        price_open=price_open,
                        price_close=price_close,
                        price_change_pct=round(price_change_pct, 2),
                        signal_correct=self._evaluate_signal(record.signal, price_change_pct),
                    )
                except Exception as e:
                    logger.warning(f"Failed to fetch closing price for {symbol}: {e}")
                    return None

        async def safe_fetch_outcome(
            symbol: str, record: AnalysisRecord
        ) -> SignalOutcome | BaseException | None:
            try:
                return await fetch_outcome(symbol, record)
            except BaseException as e:
                return e

        async with asyncio.TaskGroup() as tg:
            task_results = [
                tg.create_task(safe_fetch_outcome(symbol, record))
                for symbol, record in latest_by_symbol.items()
            ]

        raw_outcomes = [task.result() for task in task_results]
        return _filter_outcomes(raw_outcomes)

    async def _get_llm_assessment(
        self, journal_date: date, outcomes: list[SignalOutcome], accuracy_pct: float
    ) -> dict[str, Any]:
        """Get LLM assessment of outcomes."""
        outcomes_text = self._format_outcomes(outcomes)
        prompt = self._prompts.load(
            "user", date=str(journal_date), outcomes_text=outcomes_text, accuracy_pct=str(accuracy_pct)
        )
        system_prompt = self._prompts.load("system")

        try:
            llm_response = await self.llm.astructured(
                prompt, JournalLLMResponse, system=system_prompt, temperature=0.5
            )
            return {
                "winners": llm_response.winners,
                "losers": llm_response.losers,
                "lessons": llm_response.lessons,
                "tomorrows_focus": llm_response.tomorrows_focus,
                "overall_assessment": llm_response.overall_assessment,
            }
        except StructuredOutputError as e:
            logger.warning(f"Structured output failed, falling back to text: {e}")
            response = await self.llm.acomplete(prompt, system=system_prompt, temperature=0.5)
            return {
                "winners": [o.symbol for o in outcomes if o.signal_correct],
                "losers": [o.symbol for o in outcomes if not o.signal_correct],
                "lessons": [response[:500]],
                "tomorrows_focus": [o.symbol for o in outcomes if not o.signal_correct],
                "overall_assessment": response[:300],
            }

    def __repr__(self) -> str:
        """String representation."""
        return f"TradeJournalAgent(llm={self.llm.provider})"
