"""Tests for agent providers."""

from unittest.mock import MagicMock, patch

from src.di import create_container


def test_news_analyst_provider():
    """Test NewsAnalyst provider is accessible."""
    container = create_container()
    assert hasattr(container, "news_analyst")

    with patch("src.agents.news.NewsAnalyst") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance

        analyst = container.news_analyst()
        assert analyst is not None


def test_sentiment_analyst_provider():
    """Test SentimentAnalyst provider is accessible."""
    container = create_container()
    assert hasattr(container, "sentiment_analyst")

    with patch("src.agents.sentiment.SentimentAnalyst") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance

        analyst = container.sentiment_analyst()
        assert analyst is not None


def test_trump_analyst_provider():
    """Test TrumpAnalyst provider is accessible."""
    container = create_container()
    assert hasattr(container, "trump_analyst")

    with patch("src.agents.trump.TrumpAnalyst") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance

        analyst = container.trump_analyst()
        assert analyst is not None


def test_fundamental_analyst_provider(monkeypatch):
    """Test FundamentalAnalyst provider is accessible."""
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test_key")
    container = create_container()
    assert hasattr(container, "fundamental_analyst")

    with patch("src.agents.fundamental.FundamentalAnalyst") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance

        analyst = container.fundamental_analyst()
        assert analyst is not None


def test_social_sentiment_analyst_provider():
    """Test SocialSentimentAnalyst provider is accessible."""
    container = create_container()
    assert hasattr(container, "social_sentiment_analyst")

    with patch("src.agents.social.SocialSentimentAnalyst") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance

        analyst = container.social_sentiment_analyst()
        assert analyst is not None


def test_news_analyst_factory():
    """Test NewsAnalyst is factory (new instance per call)."""
    container = create_container()

    with patch("src.agents.news.NewsAnalyst") as mock_class:
        mock_instance1 = MagicMock()
        mock_instance2 = MagicMock()
        mock_class.side_effect = [mock_instance1, mock_instance2]

        analyst1 = container.news_analyst()
        analyst2 = container.news_analyst()

        assert analyst1 is not analyst2
        assert mock_class.call_count == 2


def test_sentiment_analyst_factory():
    """Test SentimentAnalyst is factory (new instance per call)."""
    container = create_container()

    with patch("src.agents.sentiment.SentimentAnalyst") as mock_class:
        mock_instance1 = MagicMock()
        mock_instance2 = MagicMock()
        mock_class.side_effect = [mock_instance1, mock_instance2]

        analyst1 = container.sentiment_analyst()
        analyst2 = container.sentiment_analyst()

        assert analyst1 is not analyst2
        assert mock_class.call_count == 2


def test_trump_analyst_factory():
    """Test TrumpAnalyst is factory (new instance per call)."""
    container = create_container()

    with patch("src.agents.trump.TrumpAnalyst") as mock_class:
        mock_instance1 = MagicMock()
        mock_instance2 = MagicMock()
        mock_class.side_effect = [mock_instance1, mock_instance2]

        analyst1 = container.trump_analyst()
        analyst2 = container.trump_analyst()

        assert analyst1 is not analyst2
        assert mock_class.call_count == 2


def test_fundamental_analyst_factory(monkeypatch):
    """Test FundamentalAnalyst is factory (new instance per call)."""
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test_key")
    container = create_container()

    with patch("src.agents.fundamental.FundamentalAnalyst") as mock_class:
        mock_instance1 = MagicMock()
        mock_instance2 = MagicMock()
        mock_class.side_effect = [mock_instance1, mock_instance2]

        analyst1 = container.fundamental_analyst()
        analyst2 = container.fundamental_analyst()

        assert analyst1 is not analyst2
        assert mock_class.call_count == 2


def test_social_sentiment_analyst_factory():
    """Test SocialSentimentAnalyst is factory (new instance per call)."""
    container = create_container()

    with patch("src.agents.social.SocialSentimentAnalyst") as mock_class:
        mock_instance1 = MagicMock()
        mock_instance2 = MagicMock()
        mock_class.side_effect = [mock_instance1, mock_instance2]

        analyst1 = container.social_sentiment_analyst()
        analyst2 = container.social_sentiment_analyst()

        assert analyst1 is not analyst2
        assert mock_class.call_count == 2


def test_shared_finbert_singleton():
    """Test FinBERT singleton shared between sentiment_analyst and social_sentiment_analyst."""
    container = create_container()

    with (
        patch("src.models.sentiment.get_finbert_sentiment") as mock_finbert_factory,
        patch("src.agents.sentiment.SentimentAnalyst") as mock_sentiment_class,
        patch("src.agents.social.SocialSentimentAnalyst") as mock_social_class,
    ):
        mock_finbert = MagicMock()
        mock_finbert_factory.return_value = mock_finbert

        mock_sentiment_instance = MagicMock()
        mock_social_instance = MagicMock()
        mock_sentiment_class.return_value = mock_sentiment_instance
        mock_social_class.return_value = mock_social_instance

        # Create both analysts
        container.sentiment_analyst()
        container.social_sentiment_analyst()

        # FinBERT factory called once (singleton)
        assert mock_finbert_factory.call_count == 1

        # Both analysts initialized with same FinBERT instance
        mock_sentiment_class.assert_called_once()
        sentiment_finbert_arg = mock_sentiment_class.call_args[0][0]

        mock_social_class.assert_called_once()
        social_finbert_arg = mock_social_class.call_args[0][3]  # 4th arg

        assert sentiment_finbert_arg is mock_finbert
        assert social_finbert_arg is mock_finbert
        assert sentiment_finbert_arg is social_finbert_arg


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


def test_bullish_researcher_provider():
    """Test BullishResearcher provider is accessible."""
    container = create_container()
    assert hasattr(container, "bullish_researcher")

    with patch("src.agents.thesis_researcher.ThesisResearcher") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        agent = container.bullish_researcher()
        assert agent is not None


def test_bullish_researcher_factory():
    """Test BullishResearcher is factory (new instance per call)."""
    container = create_container()

    with patch("src.agents.thesis_researcher.ThesisResearcher") as mock_class:
        mock_instance1 = MagicMock()
        mock_instance2 = MagicMock()
        mock_class.side_effect = [mock_instance1, mock_instance2]

        agent1 = container.bullish_researcher()
        agent2 = container.bullish_researcher()

        assert agent1 is not agent2
        assert mock_class.call_count == 2


def test_bearish_researcher_provider():
    """Test BearishResearcher provider is accessible."""
    container = create_container()
    assert hasattr(container, "bearish_researcher")

    with patch("src.agents.thesis_researcher.ThesisResearcher") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        agent = container.bearish_researcher()
        assert agent is not None


def test_bearish_researcher_factory():
    """Test BearishResearcher is factory (new instance per call)."""
    container = create_container()

    with patch("src.agents.thesis_researcher.ThesisResearcher") as mock_class:
        mock_instance1 = MagicMock()
        mock_instance2 = MagicMock()
        mock_class.side_effect = [mock_instance1, mock_instance2]

        agent1 = container.bearish_researcher()
        agent2 = container.bearish_researcher()

        assert agent1 is not agent2
        assert mock_class.call_count == 2


def test_comparative_analyst_provider():
    """Test ComparativeAnalyst provider is accessible."""
    container = create_container()
    assert hasattr(container, "comparative_analyst")

    with patch("src.agents.comparative.ComparativeAnalyst") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        agent = container.comparative_analyst()
        assert agent is not None


def test_comparative_analyst_factory():
    """Test ComparativeAnalyst is factory (new instance per call)."""
    container = create_container()

    with patch("src.agents.comparative.ComparativeAnalyst") as mock_class:
        mock_instance1 = MagicMock()
        mock_instance2 = MagicMock()
        mock_class.side_effect = [mock_instance1, mock_instance2]

        agent1 = container.comparative_analyst()
        agent2 = container.comparative_analyst()

        assert agent1 is not agent2
        assert mock_class.call_count == 2


def test_web_research_agent_provider():
    """Test WebResearchAgent provider is accessible."""
    container = create_container()
    assert hasattr(container, "web_research_agent")

    with patch("src.agents.web_researcher.WebResearchAgent") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        agent = container.web_research_agent()
        assert agent is not None


def test_web_research_agent_factory():
    """Test WebResearchAgent is factory (new instance per call)."""
    container = create_container()

    with patch("src.agents.web_researcher.WebResearchAgent") as mock_class:
        mock_instance1 = MagicMock()
        mock_instance2 = MagicMock()
        mock_class.side_effect = [mock_instance1, mock_instance2]

        agent1 = container.web_research_agent()
        agent2 = container.web_research_agent()

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


def test_technical_analyst_factory_callable():
    """Test technical_analyst returns factory callable accepting strategy."""
    container = create_container()

    with patch("src.di.providers.agents.create_technical_analyst") as mock_factory_creator:
        mock_factory = MagicMock()
        mock_factory_creator.return_value = mock_factory

        factory = container.technical_analyst()
        assert callable(factory) or factory is mock_factory


def test_technical_analyst_factory_multiple_strategies():
    """Test technical_analyst factory creates different instances per strategy."""
    container = create_container()

    with patch("src.agents.technical.TechnicalAnalyst") as mock_class:
        from src.strategies.momentum import MomentumStrategy

        factory = container.technical_analyst()
        assert callable(factory)

        strategy1 = MomentumStrategy()
        strategy2 = MomentumStrategy()

        mock_instance1 = MagicMock()
        mock_instance2 = MagicMock()
        mock_class.side_effect = [mock_instance1, mock_instance2]

        agent1 = factory(strategy1)
        agent2 = factory(strategy2)

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
