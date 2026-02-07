"""Tests for daemon state."""

import tempfile

from src.daemon.state import DaemonState, EarningsEventRecord


class TestDaemonState:
    def test_default_state(self):
        state = DaemonState()

        assert state.last_run is None
        assert state.analyses == []
        assert state.errors == []
        assert state.total_analyses == 0
        assert state.total_trades == 0

    def test_record_analysis(self):
        state = DaemonState()

        state.record_analysis("AAPL", "BUY", 0.85, executed=False)

        assert state.total_analyses == 1
        assert state.total_trades == 0
        assert len(state.analyses) == 1
        assert state.analyses[0].symbol == "AAPL"
        assert state.analyses[0].signal == "BUY"
        assert state.analyses[0].confidence == 0.85
        assert state.analyses[0].executed_trade is False
        assert state.last_run is not None

    def test_record_analysis_with_trade(self):
        state = DaemonState()

        state.record_analysis("TSLA", "SELL", 0.9, executed=True)

        assert state.total_analyses == 1
        assert state.total_trades == 1
        assert state.analyses[0].executed_trade is True

    def test_record_error(self):
        state = DaemonState()

        state.record_error("Test error")

        assert len(state.errors) == 1
        assert "Test error" in state.errors[0]

    def test_analyses_pruning(self):
        state = DaemonState()

        for i in range(1100):
            state.record_analysis(f"SYM{i}", "HOLD", 0.5)

        assert len(state.analyses) < 1100
        assert state.total_analyses == 1100

    def test_errors_pruning(self):
        state = DaemonState()

        for i in range(150):
            state.record_error(f"Error {i}")

        assert len(state.errors) < 150

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/state.json"

            state = DaemonState()
            state.record_analysis("AAPL", "BUY", 0.85)
            state.record_error("Test error")
            state.save(path)

            loaded = DaemonState.load(path)

            assert loaded.total_analyses == 1
            assert len(loaded.analyses) == 1
            assert loaded.analyses[0].symbol == "AAPL"
            assert len(loaded.errors) == 1

    def test_load_nonexistent(self):
        state = DaemonState.load("/nonexistent/path.json")

        assert state.total_analyses == 0
        assert state.analyses == []

    def test_record_sector_rotation(self):
        state = DaemonState()

        state.record_sector_rotation(
            leading_sectors=["TECHNOLOGY", "HEALTHCARE", "FINANCIALS"],
            lagging_sectors=["ENERGY", "UTILITIES", "MATERIALS"],
            sector_strengths={"TECHNOLOGY": 4.1, "ENERGY": -1.7},
            sector_momenta={"TECHNOLOGY": "ACCELERATING", "ENERGY": "DECELERATING"},
            flagged_positions=["XOM"],
        )

        assert len(state.sector_rotation_history) == 1
        assert state.last_sector_rotation is not None
        record = state.sector_rotation_history[0]
        assert record.leading_sectors == ["TECHNOLOGY", "HEALTHCARE", "FINANCIALS"]
        assert record.lagging_sectors == ["ENERGY", "UTILITIES", "MATERIALS"]
        assert record.sector_strengths["TECHNOLOGY"] == 4.1
        assert record.sector_momenta["TECHNOLOGY"] == "ACCELERATING"
        assert record.flagged_positions == ["XOM"]

    def test_sector_rotation_pruning(self):
        state = DaemonState()

        for i in range(35):
            state.record_sector_rotation(
                leading_sectors=[f"S{i}"],
                lagging_sectors=[f"L{i}"],
                sector_strengths={f"S{i}": float(i)},
                sector_momenta={f"S{i}": "NEUTRAL"},
            )

        assert len(state.sector_rotation_history) == 30

    def test_sector_rotation_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/state.json"

            state = DaemonState()
            state.record_sector_rotation(
                leading_sectors=["TECHNOLOGY"],
                lagging_sectors=["ENERGY"],
                sector_strengths={"TECHNOLOGY": 4.1},
                sector_momenta={"TECHNOLOGY": "ACCELERATING"},
            )
            state.save(path)

            loaded = DaemonState.load(path)

            assert len(loaded.sector_rotation_history) == 1
            assert loaded.last_sector_rotation is not None
            assert loaded.sector_rotation_history[0].leading_sectors == ["TECHNOLOGY"]

    def test_record_earnings_fetch(self):
        state = DaemonState()

        events = [
            EarningsEventRecord(symbol="AAPL", earnings_date="2024-07-25", estimate_eps=1.35),
            EarningsEventRecord(symbol="MSFT", earnings_date="2024-07-30"),
        ]
        state.record_earnings_fetch(events=events, symbols_fetched=2, symbols_failed=1)

        assert len(state.earnings_calendar_history) == 1
        assert state.last_earnings_fetch is not None
        record = state.earnings_calendar_history[0]
        assert record.symbols_fetched == 2
        assert record.symbols_failed == 1
        assert len(record.events) == 2
        assert record.events[0].symbol == "AAPL"
        assert record.events[0].estimate_eps == 1.35

    def test_earnings_history_pruning(self):
        state = DaemonState()

        for i in range(15):
            state.record_earnings_fetch(
                events=[EarningsEventRecord(symbol=f"SYM{i}", earnings_date="2024-07-25")],
                symbols_fetched=1,
                symbols_failed=0,
            )

        assert len(state.earnings_calendar_history) == 10

    def test_earnings_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/state.json"

            state = DaemonState()
            events = [EarningsEventRecord(symbol="AAPL", earnings_date="2024-07-25")]
            state.record_earnings_fetch(events=events, symbols_fetched=1, symbols_failed=0)
            state.save(path)

            loaded = DaemonState.load(path)

            assert len(loaded.earnings_calendar_history) == 1
            assert loaded.last_earnings_fetch is not None
            assert loaded.earnings_calendar_history[0].events[0].symbol == "AAPL"

    def test_repr(self):
        state = DaemonState()
        state.record_analysis("AAPL", "BUY", 0.85, executed=True)

        repr_str = repr(state)

        assert "analyses=1" in repr_str
        assert "trades=1" in repr_str
