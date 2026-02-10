"""Tests for news analyst agent."""

from src.agents.news import NewsAnalysis


def test_news_analyst_init(test_container):
    analyst = test_container.news_analyst()

    assert analyst.llm is not None


async def test_news_analyst_analyze(test_container, sample_news_articles):
    analyst = test_container.news_analyst()

    result = await analyst.analyze("AAPL", sample_news_articles)

    assert isinstance(result, NewsAnalysis)
    assert len(result.key_themes) > 0
    assert result.impact_assessment
    assert result.recommendation


async def test_news_analyst_analyze_empty(test_container):
    analyst = test_container.news_analyst()

    result = await analyst.analyze("AAPL", [])

    assert result.key_themes == ["No recent news"]
    assert "Insufficient" in result.impact_assessment


def test_format_articles(test_container, sample_news_articles):
    analyst = test_container.news_analyst()

    formatted = analyst._format_articles(sample_news_articles)

    assert "strong earnings" in formatted
    assert "2024-01-15" in formatted
    assert len(formatted) > 0


def test_format_articles_limit_10(test_container, sample_news_articles):
    analyst = test_container.news_analyst()

    many_articles = sample_news_articles * 10
    formatted = analyst._format_articles(many_articles)

    assert formatted.count("1.") == 1
    assert formatted.count("10.") == 1
    assert "11." not in formatted


def test_extract_themes(test_container):
    analyst = test_container.news_analyst()

    response = """
Key themes:
1. Strong earnings growth
2. New product innovation
3. Market expansion
"""

    themes = analyst._extract_themes(response)

    assert len(themes) > 0
    assert any("earnings" in t.lower() for t in themes)


def test_extract_section(test_container):
    analyst = test_container.news_analyst()

    response = """
Themes: xyz
Impact Assessment: Positive outlook due to strong fundamentals
Recommendation: Consider buying
"""

    impact = analyst._extract_section(response, "impact")

    assert "Positive" in impact or "strong" in impact


def test_repr(test_container):
    analyst = test_container.news_analyst()

    repr_str = repr(analyst)

    assert "NewsAnalyst" in repr_str
    assert "ollama" in repr_str
