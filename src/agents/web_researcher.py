"""Web research agent for gathering real-time market intelligence."""

import re
from enum import StrEnum

from loguru import logger
from pydantic import BaseModel, Field

from src.models.llm import LLMClient
from src.tools.websearch import WebSearchTool


class ResearchCategory(StrEnum):
    """Categories of web research."""

    LATEST_NEWS = "latest_news"
    MARKET_SENTIMENT = "market_sentiment"
    COMPANY_INFO = "company_info"
    COMPETITOR_ANALYSIS = "competitor_analysis"


class WebResearchResult(BaseModel):
    """Result from web research for a single category."""

    category: ResearchCategory
    summary: str = Field(description="Summary of findings (2-3 sentences)")
    key_findings: list[str] = Field(description="3-5 key findings")
    sentiment_indication: str = Field(description="Bullish, Bearish, or Neutral")
    confidence: float = Field(description="Confidence in findings (0.0-1.0)", ge=0.0, le=1.0)
    sources_count: int = Field(description="Number of sources consulted", default=0)


class WebResearchAnalysis(BaseModel):
    """Complete web research analysis."""

    symbol: str
    results: list[WebResearchResult]
    overall_sentiment: str = Field(description="Aggregated sentiment: Bullish, Bearish, or Neutral")
    confidence: float = Field(description="Overall confidence (0.0-1.0)", ge=0.0, le=1.0)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"WebResearchAnalysis(symbol={self.symbol}, categories={len(self.results)}, "
            f"sentiment={self.overall_sentiment}, confidence={self.confidence:.2f})"
        )


# Predefined query templates for Ollama fallback (no tool calling)
QUERY_TEMPLATES = {
    ResearchCategory.LATEST_NEWS: "{symbol} stock latest news today",
    ResearchCategory.MARKET_SENTIMENT: "{symbol} stock market sentiment analysis",
    ResearchCategory.COMPANY_INFO: "{symbol} company recent developments announcements",
    ResearchCategory.COMPETITOR_ANALYSIS: "{symbol} stock competitors comparison",
}


class WebResearchAgent:
    """Agent for gathering web-based research on stocks."""

    def __init__(self, llm_client: LLMClient, search_tool: WebSearchTool | None = None) -> None:
        """Initialize web research agent.

        Args:
            llm_client: LLM client for analysis
            search_tool: Web search tool. Creates default if not provided.
        """
        self.llm = llm_client
        self.search_tool = search_tool or WebSearchTool()
        logger.info(f"Initialized WebResearchAgent (tools_enabled={llm_client.supports_tools})")

    async def research(
        self,
        symbol: str,
        categories: list[ResearchCategory] | None = None,
    ) -> WebResearchAnalysis:
        """Perform web research for a stock.

        Args:
            symbol: Stock ticker symbol
            categories: Research categories to include. Defaults to all.

        Returns:
            WebResearchAnalysis with findings from all categories
        """
        if categories is None:
            categories = list(ResearchCategory)

        logger.info(f"Starting web research for {symbol} ({len(categories)} categories)")

        if self.llm.supports_tools:
            results = await self._research_with_tools(symbol, categories)
        else:
            results = await self._research_with_templates(symbol, categories)

        overall_sentiment = self._aggregate_sentiment(results)
        overall_confidence = sum(r.confidence for r in results) / len(results) if results else 0.5

        logger.info(
            f"Web research complete for {symbol}: sentiment={overall_sentiment}, "
            f"confidence={overall_confidence:.2f}"
        )

        return WebResearchAnalysis(
            symbol=symbol,
            results=results,
            overall_sentiment=overall_sentiment,
            confidence=overall_confidence,
        )

    async def _research_with_tools(
        self,
        symbol: str,
        categories: list[ResearchCategory],
    ) -> list[WebResearchResult]:
        """Research using LLM tool calling (Claude/OpenAI).

        Args:
            symbol: Stock ticker symbol
            categories: Categories to research

        Returns:
            List of WebResearchResult
        """
        results = []

        for category in categories:
            prompt = self._build_tool_prompt(symbol, category)
            system = (
                "You are a financial research analyst. Use the web_search tool to gather "
                "information, then analyze the results and provide your findings."
            )

            tools = [self.search_tool.get_tool_definition()]

            def tool_executor(name: str, args: dict) -> str:
                if name == WebSearchTool.TOOL_NAME:
                    return self.search_tool.execute(
                        query=args.get("query", ""),
                        search_type=args.get("search_type", "general"),
                    )
                return f"Unknown tool: {name}"

            response = await self.llm.acomplete_with_tools(
                prompt=prompt,
                tools=tools,
                tool_executor=tool_executor,
                system=system,
                temperature=0.3,
            )

            result = self._parse_research_response(category, response)
            results.append(result)

        return results

    async def _research_with_templates(
        self,
        symbol: str,
        categories: list[ResearchCategory],
    ) -> list[WebResearchResult]:
        """Research using predefined queries (Ollama fallback).

        Args:
            symbol: Stock ticker symbol
            categories: Categories to research

        Returns:
            List of WebResearchResult
        """
        results = []

        for category in categories:
            query = QUERY_TEMPLATES[category].format(symbol=symbol)
            search_type = "news" if category == ResearchCategory.LATEST_NEWS else "general"

            search_results = self.search_tool.execute(query, search_type=search_type, max_results=5)

            prompt = f"""Analyze these search results for {symbol} regarding {category.value}:

{search_results}

Provide:
1. Summary (2-3 sentences)
2. Key findings (3-5 bullet points starting with '- ')
3. Sentiment indication (Bullish, Bearish, or Neutral)

Format:
SUMMARY: [your summary]
FINDINGS:
- [finding 1]
- [finding 2]
- [finding 3]
SENTIMENT: [Bullish/Bearish/Neutral]"""

            system = "You are a financial research analyst summarizing web search results."
            response = await self.llm.acomplete(prompt, system=system, temperature=0.3)

            result = self._parse_research_response(category, response, sources_count=5)
            results.append(result)

        return results

    def _build_tool_prompt(self, symbol: str, category: ResearchCategory) -> str:
        """Build prompt for tool-calling research.

        Args:
            symbol: Stock ticker symbol
            category: Research category

        Returns:
            Prompt string
        """
        category_instructions = {
            ResearchCategory.LATEST_NEWS: (
                f"Search for the latest news about {symbol} stock. "
                "Focus on breaking news, earnings reports, and significant announcements."
            ),
            ResearchCategory.MARKET_SENTIMENT: (
                f"Search for market sentiment and analyst opinions on {symbol} stock. "
                "Look for price targets, ratings changes, and institutional sentiment."
            ),
            ResearchCategory.COMPANY_INFO: (
                f"Search for recent company information about {symbol}. "
                "Focus on product launches, partnerships, management changes, and business developments."
            ),
            ResearchCategory.COMPETITOR_ANALYSIS: (
                f"Search for information comparing {symbol} to its competitors. "
                "Look for market share data, competitive advantages, and industry positioning."
            ),
        }

        return f"""{category_instructions[category]}

After searching, analyze the results and provide:
1. Summary (2-3 sentences)
2. Key findings (3-5 bullet points starting with '- ')
3. Sentiment indication (Bullish, Bearish, or Neutral)

Format your response as:
SUMMARY: [your summary]
FINDINGS:
- [finding 1]
- [finding 2]
- [finding 3]
SENTIMENT: [Bullish/Bearish/Neutral]"""

    def _parse_research_response(
        self,
        category: ResearchCategory,
        response: str,
        sources_count: int = 0,
    ) -> WebResearchResult:
        """Parse LLM response into WebResearchResult.

        Args:
            category: Research category
            response: LLM response text
            sources_count: Number of sources (for template mode)

        Returns:
            WebResearchResult
        """
        summary = self._extract_section(response, "SUMMARY")
        findings = self._extract_findings(response)
        sentiment = self._extract_sentiment(response)

        confidence = self._calculate_confidence(findings, sentiment)

        return WebResearchResult(
            category=category,
            summary=summary or "Unable to extract summary from research.",
            key_findings=findings or ["No specific findings extracted."],
            sentiment_indication=sentiment,
            confidence=confidence,
            sources_count=sources_count,
        )

    def _extract_section(self, response: str, section: str) -> str | None:
        """Extract a section from LLM response.

        Args:
            response: LLM response text
            section: Section name (SUMMARY, SENTIMENT, etc.)

        Returns:
            Extracted text or None
        """
        pattern = rf"{section}:\s*(.+?)(?=\n[A-Z]+:|$)"
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else None

    def _extract_findings(self, response: str) -> list[str]:
        """Extract findings from LLM response.

        Args:
            response: LLM response text

        Returns:
            List of finding strings
        """
        match = re.search(r"FINDINGS:\s*(.+?)(?=SENTIMENT:|$)", response, re.DOTALL | re.IGNORECASE)
        if not match:
            bullets = re.findall(r"[-•]\s*(.+)", response)
            return [b.strip() for b in bullets[:5]] if bullets else []

        findings_text = match.group(1).strip()
        bullets = re.findall(r"[-•]\s*(.+)", findings_text)
        return [b.strip() for b in bullets[:5]] if bullets else []

    def _extract_sentiment(self, response: str) -> str:
        """Extract sentiment from LLM response.

        Args:
            response: LLM response text

        Returns:
            Sentiment string (Bullish, Bearish, or Neutral)
        """
        match = re.search(r"SENTIMENT:\s*(\w+)", response, re.IGNORECASE)
        if match:
            sentiment = match.group(1).lower()
            if "bull" in sentiment:
                return "Bullish"
            if "bear" in sentiment:
                return "Bearish"
        return "Neutral"

    def _calculate_confidence(self, findings: list[str], sentiment: str) -> float:
        """Calculate confidence based on findings quality.

        Args:
            findings: List of findings
            sentiment: Extracted sentiment

        Returns:
            Confidence score (0.0-1.0)
        """
        confidence = 0.5

        if len(findings) >= 3:
            confidence += 0.1
        if len(findings) >= 5:
            confidence += 0.1

        if sentiment != "Neutral":
            confidence += 0.1

        avg_finding_length = sum(len(f) for f in findings) / len(findings) if findings else 0
        if avg_finding_length > 50:
            confidence += 0.1

        return min(1.0, confidence)

    def _aggregate_sentiment(self, results: list[WebResearchResult]) -> str:
        """Aggregate sentiment from all results.

        Args:
            results: List of research results

        Returns:
            Overall sentiment (Bullish, Bearish, or Neutral)
        """
        if not results:
            return "Neutral"

        bullish = sum(1 for r in results if r.sentiment_indication == "Bullish")
        bearish = sum(1 for r in results if r.sentiment_indication == "Bearish")

        if bullish > bearish:
            return "Bullish"
        if bearish > bullish:
            return "Bearish"
        return "Neutral"

    def __repr__(self) -> str:
        """String representation."""
        return f"WebResearchAgent(llm={self.llm.provider}, tools_enabled={self.llm.supports_tools})"
