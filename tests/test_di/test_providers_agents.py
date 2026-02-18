"""Tests for agent providers."""

from unittest.mock import MagicMock, patch

from src.di import create_container


def test_trader_agent_provider():
    """Test TraderAgent provider is accessible."""
    container = create_container()
    assert hasattr(container, "trader_agent")

    with patch("src.agents.trader.TraderAgent") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        agent = container.trader_agent()
        assert agent is not None


def test_trader_agent_factory():
    """Test TraderAgent is factory (new instance per call)."""
    container = create_container()

    with patch("src.agents.trader.TraderAgent") as mock_class:
        mock_instance1 = MagicMock()
        mock_instance2 = MagicMock()
        mock_class.side_effect = [mock_instance1, mock_instance2]

        agent1 = container.trader_agent()
        agent2 = container.trader_agent()

        assert agent1 is not agent2
        assert mock_class.call_count == 2


def test_meta_agent_provider():
    """Test MetaAgent provider is accessible."""
    container = create_container()
    assert hasattr(container, "meta_agent")

    with patch("src.agents.meta.MetaAgent") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        agent = container.meta_agent()
        assert agent is not None


def test_meta_agent_factory():
    """Test MetaAgent is factory (new instance per call)."""
    container = create_container()

    with patch("src.agents.meta.MetaAgent") as mock_class:
        mock_instance1 = MagicMock()
        mock_instance2 = MagicMock()
        mock_class.side_effect = [mock_instance1, mock_instance2]

        agent1 = container.meta_agent()
        agent2 = container.meta_agent()

        assert agent1 is not agent2
        assert mock_class.call_count == 2


def test_risk_management_agent_provider(monkeypatch):
    """Test RiskManagementAgent provider is accessible."""
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test_key")
    container = create_container()
    assert hasattr(container, "risk_management_agent")

    # Mock database dependencies to avoid migration issues in unit tests
    mock_db_engine = MagicMock()
    container.database_engine.override(mock_db_engine)

    with patch("src.agents.risk.RiskManagementAgent") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        agent = container.risk_management_agent()
        assert agent is not None


def test_risk_management_agent_factory(monkeypatch):
    """Test RiskManagementAgent is factory (new instance per call)."""
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test_key")
    container = create_container()

    # Mock database dependencies to avoid migration issues in unit tests
    mock_db_engine = MagicMock()
    container.database_engine.override(mock_db_engine)

    with patch("src.agents.risk.RiskManagementAgent") as mock_class:
        mock_instance1 = MagicMock()
        mock_instance2 = MagicMock()
        mock_class.side_effect = [mock_instance1, mock_instance2]

        agent1 = container.risk_management_agent()
        agent2 = container.risk_management_agent()

        assert agent1 is not agent2
        assert mock_class.call_count == 2


def test_event_triage_agent_provider():
    """Test EventTriageAgent provider is accessible."""
    container = create_container()
    assert hasattr(container, "event_triage_agent")

    with patch("src.agents.event_triage.EventTriageAgent") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        agent = container.event_triage_agent()
        assert agent is not None


def test_event_triage_agent_factory():
    """Test EventTriageAgent is factory (new instance per call)."""
    container = create_container()

    with patch("src.agents.event_triage.EventTriageAgent") as mock_class:
        mock_instance1 = MagicMock()
        mock_instance2 = MagicMock()
        mock_class.side_effect = [mock_instance1, mock_instance2]

        agent1 = container.event_triage_agent()
        agent2 = container.event_triage_agent()

        assert agent1 is not agent2
        assert mock_class.call_count == 2


def test_game_plan_agent_provider(monkeypatch):
    """Test GamePlanAgent provider is accessible."""
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test_key")
    container = create_container()
    assert hasattr(container, "game_plan_agent")

    with patch("src.agents.game_plan.GamePlanAgent") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        agent = container.game_plan_agent()
        assert agent is not None


def test_game_plan_agent_factory(monkeypatch):
    """Test GamePlanAgent is factory (new instance per call)."""
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test_key")
    container = create_container()

    with patch("src.agents.game_plan.GamePlanAgent") as mock_class:
        mock_instance1 = MagicMock()
        mock_instance2 = MagicMock()
        mock_class.side_effect = [mock_instance1, mock_instance2]

        agent1 = container.game_plan_agent()
        agent2 = container.game_plan_agent()

        assert agent1 is not agent2
        assert mock_class.call_count == 2


def test_trade_journal_agent_provider(monkeypatch):
    """Test TradeJournalAgent provider is accessible."""
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test_key")
    container = create_container()
    assert hasattr(container, "trade_journal_agent")

    with patch("src.agents.journal.TradeJournalAgent") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        agent = container.trade_journal_agent()
        assert agent is not None


def test_trade_journal_agent_factory(monkeypatch):
    """Test TradeJournalAgent is factory (new instance per call)."""
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test_key")
    container = create_container()

    with patch("src.agents.journal.TradeJournalAgent") as mock_class:
        mock_instance1 = MagicMock()
        mock_instance2 = MagicMock()
        mock_class.side_effect = [mock_instance1, mock_instance2]

        agent1 = container.trade_journal_agent()
        agent2 = container.trade_journal_agent()

        assert agent1 is not agent2
        assert mock_class.call_count == 2
