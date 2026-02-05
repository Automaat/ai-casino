"""LLM-powered analysis of screening results."""

from loguru import logger
from pydantic import BaseModel

from src.models.llm import LLMClient
from src.prompts import PromptLoader
from src.screening.screener import ScreeningOutput


class ScreeningAnalysis(BaseModel):
    """LLM analysis of screening results."""

    summary: str
    top_picks: list[str]
    sector_insights: str
    risk_factors: str
    next_steps: str


class ScreeningAnalyzer:
    """Analyze screening results using LLM."""

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize screening analyzer.

        Args:
            llm_client: LLM client for analysis
        """
        self._llm = llm_client
        self._prompts = PromptLoader("screening")
        logger.info("Initialized ScreeningAnalyzer")

    async def analyze(
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

        system_prompt = self._prompts.load("system")
        user_prompt = self._build_prompt(screening_output, market_context)
        response = await self._llm.acomplete(user_prompt, system=system_prompt, temperature=0.5)

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

        return self._prompts.load(
            "user",
            criteria=output.criteria.value,
            universe=output.universe,
            results_block=results_block,
            total_screened=output.total_screened,
            errors_count=len(output.errors),
            context_block=context_block,
        )

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
