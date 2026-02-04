"""Tests for daemon state."""

import tempfile

from src.daemon.state import DaemonState


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

    def test_repr(self):
        state = DaemonState()
        state.record_analysis("AAPL", "BUY", 0.85, executed=True)

        repr_str = repr(state)

        assert "analyses=1" in repr_str
        assert "trades=1" in repr_str
