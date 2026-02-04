"""LLM-powered analysis of screening results."""

from loguru import logger
from pydantic import BaseModel

from src.models.llm import LLMClient
from src.screening.screener import ScreeningOutput


class ScreeningAnalysis(BaseModel):
    """LLM analysis of screening results."""

    summary: str
    top_picks: list[str]
    sector_insights: str
    risk_factors: str
    next_steps: str


SYSTEM_PROMPT = """You are a financial analyst reviewing stock screening results.
Provide clear, actionable insights without investment advice disclaimers.
Be concise and focus on the data presented."""


class ScreeningAnalyzer:
    """Analyze screening results using LLM."""

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize screening analyzer.

        Args:
            llm_client: LLM client for analysis
        """
        self._llm = llm_client
        logger.info("Initialized ScreeningAnalyzer")

    def analyze(
        self,
        screening_output: ScreeningOutput,
        market_context: str | None = None,
    ) -> ScreeningAnalysis:
        """Analyze screening results.

        Args:
            screening_output: Screening output to analyze
            market_context: Optional market context

        Returns:
            ScreeningAnalysis with insights
        """
        logger.info(f"Analyzing {len(screening_output.results)} screening results")

        prompt = self._build_prompt(screening_output, market_context)
        response = self._llm.complete(prompt, system=SYSTEM_PROMPT, temperature=0.5)

        return self._parse_response(response)

    def _build_prompt(self, output: ScreeningOutput, market_context: str | None) -> str:
        """Build analysis prompt.

        Args:
            output: Screening output
            market_context: Optional context

        Returns:
            Formatted prompt string
        """
        results_text = []
        for i, r in enumerate(output.results[:10], 1):
            metrics_str = ", ".join(f"{k}={v}" for k, v in r.metrics.items())
            results_text.append(
                f"{i}. {r.symbol} ({r.name}) - Score: {r.score:.2f}, Sector: {r.sector}\n"
                f"   Metrics: {metrics_str}\n"
                f"   Reason: {r.reason}"
            )

        results_block = "\n".join(results_text)

        context_block = ""
        if market_context:
            context_block = f"\nMarket Context:\n{market_context}\n"

        return f"""Analyze these {output.criteria.value} screening results from {output.universe}:

{results_block}

Total screened: {output.total_screened}
Failed to screen: {len(output.errors)} symbols
{context_block}
Provide your analysis in the following format:

SUMMARY: (2-3 sentences overview of the results)

TOP_PICKS: (list top 3 stocks with brief reasoning, one per line)
1. SYMBOL - reasoning
2. SYMBOL - reasoning
3. SYMBOL - reasoning

SECTOR_INSIGHTS: (1-2 sentences about sector concentration or patterns)

RISK_FACTORS: (1-2 sentences about common risks across these picks)

NEXT_STEPS: (1-2 sentences suggesting follow-up actions)"""

    def _parse_response(self, response: str) -> ScreeningAnalysis:
        """Parse LLM response into ScreeningAnalysis.

        Args:
            response: Raw LLM response

        Returns:
            Parsed ScreeningAnalysis
        """
        sections: dict[str, str | list] = {
            "summary": "",
            "top_picks": [],
            "sector_insights": "",
            "risk_factors": "",
            "next_steps": "",
        }

        section_headers = {
            "SUMMARY:": "summary",
            "TOP_PICKS:": "top_picks",
            "SECTOR_INSIGHTS:": "sector_insights",
            "RISK_FACTORS:": "risk_factors",
            "NEXT_STEPS:": "next_steps",
        }

        current_section = None
        for line in response.strip().split("\n"):
            line_upper = line.upper().strip()

            # Check for section header
            new_section = self._detect_section(line_upper, section_headers)
            if new_section:
                current_section = new_section
                content = self._extract_header_content(line)
                if content and new_section != "top_picks":
                    sections[new_section] = content
                continue

            # Process content for current section
            if current_section and line.strip():
                self._add_section_content(sections, current_section, line.strip())

        return self._build_analysis(sections)

    def _detect_section(self, line_upper: str, headers: dict[str, str]) -> str | None:
        """Detect if line starts a new section."""
        for header, section in headers.items():
            if line_upper.startswith(header):
                return section
        return None

    def _extract_header_content(self, line: str) -> str:
        """Extract content after colon in header line."""
        return line.split(":", 1)[1].strip() if ":" in line else ""

    def _add_section_content(self, sections: dict, section: str, content: str) -> None:
        """Add content to appropriate section."""
        if section == "top_picks":
            cleaned = content.lstrip("0123456789.)-: ") if content[0].isdigit() else content
            if cleaned:
                sections["top_picks"].append(cleaned)
        else:
            sections[section] += " " + content

    def _build_analysis(self, sections: dict) -> ScreeningAnalysis:
        """Build ScreeningAnalysis from parsed sections."""
        return ScreeningAnalysis(
            summary=sections["summary"].strip() or "Analysis of screening results.",
            top_picks=sections["top_picks"][:3] if sections["top_picks"] else ["No picks identified"],
            sector_insights=sections["sector_insights"].strip() or "Sector analysis unavailable.",
            risk_factors=sections["risk_factors"].strip() or "Risk analysis unavailable.",
            next_steps=sections["next_steps"].strip() or "Consider further research.",
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"ScreeningAnalyzer(llm={self._llm})"
