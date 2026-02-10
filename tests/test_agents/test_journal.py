"""Tests for trade journal agent."""

from datetime import date, datetime
from unittest.mock import MagicMock

from src.agents.journal import DailyJournal, SignalOutcome, TradeJournalAgent
from src.daemon.state import AnalysisRecord


class TestEvaluateSignal:
    def setup_method(self):
        self.agent = TradeJournalAgent.__new__(TradeJournalAgent)

    def test_buy_correct_price_up(self):
        assert self.agent._evaluate_signal("BUY", 2.5) is True

    def test_buy_incorrect_price_down(self):
        assert self.agent._evaluate_signal("BUY", -1.5) is False

    def test_sell_correct_price_down(self):
        assert self.agent._evaluate_signal("SELL", -2.0) is True

    def test_sell_incorrect_price_up(self):
        assert self.agent._evaluate_signal("SELL", 1.0) is False

    def test_hold_correct_small_change(self):
        assert self.agent._evaluate_signal("HOLD", 0.3) is True

    def test_hold_correct_negative_small(self):
        assert self.agent._evaluate_signal("HOLD", -0.5) is True

    def test_hold_incorrect_large_change(self):
        assert self.agent._evaluate_signal("HOLD", 2.0) is False


class TestTradeJournalAgent:
    def test_init(self, test_container):
        market_fetcher = MagicMock()
        agent = TradeJournalAgent(test_container, market_fetcher)

        assert agent.llm == test_container
        assert agent.market_fetcher == market_fetcher

    def test_repr(self, test_container):
        agent = test_container.trade_journal_agent()
        repr_str = repr(agent)

        assert "TradeJournalAgent" in repr_str
        assert "ollama" in repr_str

    async def test_generate_empty_records(self, test_container):
        agent = test_container.trade_journal_agent()
        journal = await agent.generate(date(2024, 1, 15), [])

        assert journal.date == date(2024, 1, 15)
        assert journal.outcomes == []
        assert journal.overall_assessment == "No signals to evaluate"

    async def test_generate_journal(self, test_container, sample_analysis_records, mock_market_fetcher):
        agent = test_container.trade_journal_agent()
        journal = await agent.generate(date(2024, 1, 15), sample_analysis_records)

        assert isinstance(journal, DailyJournal)
        assert journal.date == date(2024, 1, 15)
        assert len(journal.outcomes) > 0

        for outcome in journal.outcomes:
            assert isinstance(outcome, SignalOutcome)
            assert outcome.price_open > 0
            assert outcome.price_close > 0

    async def test_generate_deduplicates_symbols(self, test_container, mock_market_fetcher):
        """Latest signal per symbol is used when duplicates exist."""
        records = [
            AnalysisRecord(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 15, 10, 0),
                signal="BUY",
                confidence=0.7,
            ),
            AnalysisRecord(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 15, 14, 0),
                signal="SELL",
                confidence=0.8,
            ),
        ]

        agent = test_container.trade_journal_agent()
        journal = await agent.generate(date(2024, 1, 15), records)

        # Should have 1 outcome for AAPL (latest signal SELL)
        assert len(journal.outcomes) == 1
        assert journal.outcomes[0].signal == "SELL"

    def test_persist_journal(self, tmp_path):
        journal = DailyJournal(
            date=date(2024, 1, 15),
            outcomes=[
                SignalOutcome(
                    symbol="AAPL",
                    signal="BUY",
                    confidence=0.8,
                    price_open=150.0,
                    price_close=155.0,
                    price_change_pct=3.33,
                    signal_correct=True,
                ),
            ],
            winners=["AAPL — strong momentum"],
            losers=[],
            lessons=["Momentum signals accurate in trending markets"],
            tomorrows_focus=["Watch TSLA for breakout"],
            overall_assessment="Good day with accurate signals.",
        )

        agent = TradeJournalAgent.__new__(TradeJournalAgent)
        file_path = agent.persist(journal, str(tmp_path))

        assert file_path.exists()
        assert file_path.name == "2024-01-15.md"

        content = file_path.read_text()
        assert "# Trade Journal" in content
        assert "AAPL" in content
        assert "BUY" in content
        assert "Winners" in content
        assert "Lessons" in content
        assert "Tomorrow's Focus" in content
        assert "Overall Assessment" in content
        assert "1/1" in content  # accuracy

    def test_persist_creates_directory(self, tmp_path):
        journal = DailyJournal(
            date=date(2024, 1, 15),
            outcomes=[],
            winners=[],
            losers=[],
            lessons=["No data"],
            tomorrows_focus=[],
            overall_assessment="N/A",
        )

        nested_dir = tmp_path / "nested" / "journal"
        agent = TradeJournalAgent.__new__(TradeJournalAgent)
        file_path = agent.persist(journal, str(nested_dir))

        assert file_path.exists()
        assert nested_dir.exists()
