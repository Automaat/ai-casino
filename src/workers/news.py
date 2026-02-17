"""News Analysis Worker - Pydantic AI migration POC."""

from loguru import logger
from pydantic import BaseModel, Field

from src.agents.news import NewsAnalysis
from src.data.news import NewsArticle
from src.models.llm import LLMClient
from src.models.providers.base import StructuredOutputError
from src.prompts import PromptLoader
from src.tools.models import ToolDefinition, ToolFunction, ToolParameter, ToolParametersSchema


class NewsLLMResponse(BaseModel):
    """LLM response structure for news analysis."""

    key_themes: list[str] = Field(description="Top 3-5 key themes from the news")
    impact_assessment: str = Field(description="Assessment of market impact")
    recommendation: str = Field(description="Trading recommendation based on news")


class NewsWorker:
    """News analysis worker - Pydantic AI migration POC."""

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize news worker.

        Args:
            llm_client: LLM client for analysis
        """
        self.llm = llm_client
        self._prompts = PromptLoader("news")
        logger.info("Initialized NewsWorker (POC)")

    async def analyze(self, symbol: str, articles: list[NewsArticle]) -> NewsAnalysis:
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
                confidence=0.0,
            )

        headlines_text = self._format_articles(articles)

        prompt = self._prompts.load("user", symbol=symbol, headlines_text=headlines_text)
        system_prompt = self._prompts.load("system")

        # 0.6: structured output gives reliable schema-validated response, but LLM
        # reasoning quality is not guaranteed — moderate confidence baseline
        confidence = 0.6
        try:
            llm_response = await self.llm.astructured(
                prompt, NewsLLMResponse, system=system_prompt, temperature=0.4
            )
            key_themes = llm_response.key_themes
            impact = llm_response.impact_assessment
            recommendation = llm_response.recommendation
        except StructuredOutputError as e:
            logger.opt(exception=True).warning(f"Structured output failed, falling back to text parsing: {e}")
            response = await self.llm.acomplete(prompt, system=system_prompt, temperature=0.4)
            key_themes = self._extract_themes(response)
            impact = self._extract_section(response, "impact")
            recommendation = self._extract_section(response, "recommendation")
            # 0.4: regex-based text parsing is lossy and fragile — lower confidence
            confidence = 0.4

        logger.info(f"News analysis complete: {len(key_themes)} themes identified")

        return NewsAnalysis(
            key_themes=key_themes,
            impact_assessment=impact,
            recommendation=recommendation,
            confidence=confidence,
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
            # Check for theme keywords or numbered/bulleted lists
            if any(keyword in line.lower() for keyword in ["theme", "topic", "key", "-", "•"]) or (
                line and line[0].isdigit()
            ):
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
                # Extract text from same line if present (after colon)
                if ":" in line:
                    text_after_colon = line.split(":", 1)[1].strip()
                    if text_after_colon:
                        section_lines.append(text_after_colon)
                continue

            if in_section:
                if line.strip() and not any(
                    keyword in line.lower() for keyword in ["theme", "key", "1.", "2.", "3."]
                ):
                    section_lines.append(line.strip())
                elif section_lines:
                    break

        return " ".join(section_lines) if section_lines else response[:200]

    def get_tool_definition(self) -> ToolDefinition:
        """Get tool definition for supervisor integration.

        Returns:
            Tool definition
        """
        return ToolDefinition(
            type="function",
            function=ToolFunction(
                name="analyze_news",
                description="Analyze news articles for trading implications using LLM",
                parameters=ToolParametersSchema(
                    type="object",
                    properties={
                        "symbol": ToolParameter(
                            type="string",
                            description="Stock ticker symbol",
                        ),
                    },
                    required=["symbol"],
                ),
            ),
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"NewsWorker(llm={self.llm.provider})"
