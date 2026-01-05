"""News Analysis Agent."""

from loguru import logger
from pydantic import BaseModel

from src.data.news import NewsArticle
from src.models.llm import LLMClient


class NewsAnalysis(BaseModel):
    """News analysis result."""

    key_themes: list[str]
    impact_assessment: str
    recommendation: str


class NewsAnalyst:
    """Agent for analyzing news headlines and content."""

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize news analyst.

        Args:
            llm_client: LLM client for analysis
        """
        self.llm = llm_client
        logger.info("Initialized NewsAnalyst")

    def analyze(self, symbol: str, articles: list[NewsArticle]) -> NewsAnalysis:
        """Analyze news articles for trading implications.

        Args:
            symbol: Stock ticker symbol
            articles: List of news articles

        Returns:
            NewsAnalysis with themes and assessment
        """
        logger.info(f"Analyzing {len(articles)} news articles for {symbol}")

        if not articles:
            logger.warning("No articles provided for news analysis")
            return NewsAnalysis(
                key_themes=["No recent news"],
                impact_assessment="Insufficient data for assessment",
                recommendation="Wait for more information",
            )

        headlines_text = self._format_articles(articles)

        prompt = f"""Analyze these recent news articles for {symbol}:

{headlines_text}

Provide:
1. Key themes (3-5 main topics)
2. Impact assessment (how news affects stock outlook)
3. Trading recommendation based on news

Be concise and focus on actionable insights.
"""

        system_prompt = (
            "You are a financial news analyst. Extract key themes and assess "
            "their potential impact on stock price and trading decisions."
        )

        response = self.llm.complete(prompt, system=system_prompt, temperature=0.4)

        key_themes = self._extract_themes(response)
        impact = self._extract_section(response, "impact")
        recommendation = self._extract_section(response, "recommendation")

        logger.info(f"News analysis complete: {len(key_themes)} themes identified")

        return NewsAnalysis(
            key_themes=key_themes,
            impact_assessment=impact,
            recommendation=recommendation,
        )

    def _format_articles(self, articles: list[NewsArticle]) -> str:
        """Format articles for LLM prompt.

        Args:
            articles: List of news articles

        Returns:
            Formatted text
        """
        lines = []
        for i, article in enumerate(articles[:10], 1):
            date_str = article.published_at.strftime("%Y-%m-%d")
            lines.append(f"{i}. [{date_str}] {article.title}")
            if article.description:
                lines.append(f"   {article.description[:200]}")

        return "\n".join(lines)

    def _extract_themes(self, response: str) -> list[str]:
        """Extract key themes from response.

        Args:
            response: LLM response text

        Returns:
            List of themes
        """
        min_theme_length = 5
        max_theme_length = 100
        max_themes = 5

        themes = []
        lines = response.split("\n")

        for raw_line in lines:
            line = raw_line.strip()
            if any(keyword in line.lower() for keyword in ["theme", "topic", "key", "-", "•"]):
                cleaned = line.lstrip("0123456789.-•* ").strip()
                if min_theme_length < len(cleaned) < max_theme_length:
                    themes.append(cleaned)

        return themes[:max_themes] if themes else ["Market activity", "Company developments"]

    def _extract_section(self, response: str, section_name: str) -> str:
        """Extract specific section from response.

        Args:
            response: LLM response text
            section_name: Section to extract

        Returns:
            Extracted text
        """
        lines = response.split("\n")
        section_lines = []
        in_section = False

        for line in lines:
            if section_name.lower() in line.lower():
                in_section = True
                continue

            if in_section:
                if line.strip() and not any(
                    keyword in line.lower() for keyword in ["theme", "key", "1.", "2.", "3."]
                ):
                    section_lines.append(line.strip())
                elif section_lines:
                    break

        return " ".join(section_lines) if section_lines else response[:200]

    def __repr__(self) -> str:
        """String representation."""
        return f"NewsAnalyst(llm={self.llm.provider})"
