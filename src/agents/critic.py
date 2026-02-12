"""Critic Agent for evaluating trading decision quality."""

from loguru import logger
from pydantic import BaseModel, Field

from src.agents.fundamental import FundamentalAnalysis
from src.agents.news import NewsAnalysis
from src.agents.risk import RiskAssessment
from src.agents.sentiment import SentimentAnalysis
from src.agents.technical import TechnicalAnalysis
from src.agents.trader import TradingDecision
from src.models.llm import LLMClient
from src.models.providers.base import StructuredOutputError
from src.prompts import PromptLoader


class DecisionEvaluationRequest(BaseModel):
    """Input for critique - decision + all analysis context."""

    symbol: str
    decision: TradingDecision
    technical: TechnicalAnalysis
    sentiment: SentimentAnalysis
    news: NewsAnalysis
    fundamental: FundamentalAnalysis | None
    risk: RiskAssessment
    game_plan_context: str | None
    portfolio_constraints: dict[str, float] | None = None
    recent_outcomes: list[str] | None = None


class CriticLLMResponse(BaseModel):
    """Structured LLM response for decision evaluation."""

    logical_consistency_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Score for logical consistency across all signals",
    )
    game_plan_alignment_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Score for alignment with game plan",
    )
    risk_constraint_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Score for compliance with risk constraints",
    )
    critical_issues: list[str] = Field(
        description="Critical issues that could lead to losses",
    )
    suggestions: list[str] = Field(
        description="Concrete suggestions for improvement",
    )


class CriticAnalysis(BaseModel):
    """Final critique result."""

    passed: bool
    overall_score: float
    logical_consistency_score: float
    game_plan_alignment_score: float
    risk_constraint_score: float
    critical_issues: list[str]
    suggestions: list[str]
    reflection: str


class CriticAgent:
    """Agent that evaluates trading decision quality.

    Evaluates decisions against three criteria:
    1. Logical consistency (50% weight) - signals align, no contradictions
    2. Game plan alignment (30% weight) - fits today's market strategy
    3. Risk constraint compliance (20% weight) - within limits
    """

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize critic agent.

        Args:
            llm_client: LLM client for evaluation
        """
        self.llm = llm_client
        self._prompts = PromptLoader("critic")
        logger.info("Initialized CriticAgent")

    async def evaluate(self, request: DecisionEvaluationRequest) -> CriticAnalysis:
        """Evaluate trading decision quality.

        Args:
            request: Evaluation request with decision and context

        Returns:
            CriticAnalysis with scores, issues, and reflection
        """
        logger.info(f"Evaluating decision for {request.symbol}")

        prompt = self._build_prompt(request)
        system = self._prompts.load("system")

        try:
            llm_response = await self.llm.astructured(
                prompt,
                CriticLLMResponse,
                system=system,
                temperature=0.3,
            )
        except StructuredOutputError as e:
            logger.opt(exception=True).warning(f"Structured output failed for critique, falling back: {e}")
            text_response = await self.llm.acomplete(prompt, system=system, temperature=0.3)
            return self._create_fallback_analysis(request, text_response)

        # Calculate overall score (weighted average)
        overall_score = (
            0.5 * llm_response.logical_consistency_score
            + 0.3 * llm_response.game_plan_alignment_score
            + 0.2 * llm_response.risk_constraint_score
        )

        # Decision passes if no critical issues AND overall score >= 0.6
        passed = len(llm_response.critical_issues) == 0 and overall_score >= 0.6

        # Generate reflection text for LLM context
        reflection = self._generate_reflection(
            request.symbol,
            passed,
            overall_score,
            llm_response,
        )

        return CriticAnalysis(
            passed=passed,
            overall_score=overall_score,
            logical_consistency_score=llm_response.logical_consistency_score,
            game_plan_alignment_score=llm_response.game_plan_alignment_score,
            risk_constraint_score=llm_response.risk_constraint_score,
            critical_issues=llm_response.critical_issues,
            suggestions=llm_response.suggestions,
            reflection=reflection,
        )

    def _build_prompt(self, request: DecisionEvaluationRequest) -> str:
        """Build evaluation prompt with all context.

        Args:
            request: Evaluation request

        Returns:
            Formatted prompt for LLM
        """
        # Format technical indicators
        tech_indicators = []
        if request.technical.rsi is not None:
            tech_indicators.append(f"RSI={request.technical.rsi:.1f}")
        if request.technical.macd_hist is not None:
            tech_indicators.append(f"MACD={request.technical.macd_hist:.3f}")
        tech_str = ", ".join(tech_indicators) if tech_indicators else "N/A"

        # Format sentiment
        sentiment_str = request.sentiment.overall_sentiment
        if hasattr(request.sentiment, "positive_ratio"):
            sentiment_str += f" ({request.sentiment.positive_ratio:.0%} positive)"

        # Format news themes
        news_themes = ", ".join(request.news.key_themes) if request.news.key_themes else "None"
        news_impact = request.news.impact_assessment

        # Format constraints
        constraints_str = "None"
        if request.portfolio_constraints:
            parts = []
            if "max_position_pct" in request.portfolio_constraints:
                parts.append(f"max_position={request.portfolio_constraints['max_position_pct']:.1%}")
            if "min_confidence" in request.portfolio_constraints:
                parts.append(f"min_confidence={request.portfolio_constraints['min_confidence']:.2f}")
            constraints_str = ", ".join(parts) if parts else "None"

        # Format recent outcomes
        outcomes_str = "None"
        if request.recent_outcomes:
            outcomes_str = "\n".join(f"- {outcome}" for outcome in request.recent_outcomes[-5:])

        # Format game plan
        game_plan = request.game_plan_context or "No game plan generated"

        # Build prompt from template
        return self._prompts.load(
            "user",
            symbol=request.symbol,
            action=request.decision.action.value,
            confidence=request.decision.confidence,
            reasoning=" ".join(request.decision.reasoning),
            technical_signal=request.technical.signal.value,
            technical_indicators=tech_str,
            sentiment=sentiment_str,
            news_themes=news_themes,
            news_impact=news_impact,
            risk_level=request.risk.validation.risk_level,
            game_plan_context=game_plan,
            portfolio_constraints=constraints_str,
            recent_outcomes=outcomes_str,
        )

    def _generate_reflection(
        self,
        symbol: str,
        passed: bool,
        overall_score: float,
        llm_response: CriticLLMResponse,
    ) -> str:
        """Generate structured reflection for LLM message history.

        Args:
            symbol: Stock symbol
            passed: Whether critique passed
            overall_score: Overall quality score
            llm_response: LLM evaluation response

        Returns:
            Formatted reflection text
        """
        status = "✅ PASSED" if passed else "⚠️ FAILED"

        reflection = f"**Critique for {symbol}: {status}**\n\n"
        reflection += f"**Overall Quality Score:** {overall_score:.0%}\n"
        reflection += f"- Logical Consistency: {llm_response.logical_consistency_score:.0%}\n"
        reflection += f"- Game Plan Alignment: {llm_response.game_plan_alignment_score:.0%}\n"
        reflection += f"- Risk Constraint Compliance: {llm_response.risk_constraint_score:.0%}\n\n"

        if llm_response.critical_issues:
            reflection += "**⚠️ Critical Issues:**\n"
            for issue in llm_response.critical_issues:
                reflection += f"- {issue}\n"
            reflection += "\n"

        if llm_response.suggestions:
            reflection += "**💡 Suggestions:**\n"
            for suggestion in llm_response.suggestions:
                reflection += f"- {suggestion}\n"
            reflection += "\n"

        if passed:
            reflection += "**Next Steps:** Decision quality validated. You may proceed with confidence."
        else:
            reflection += (
                "**Next Steps:** Consider re-analyzing with revised approach, "
                "reducing position size, or choosing HOLD given quality concerns."
            )

        return reflection

    def _create_fallback_analysis(
        self,
        request: DecisionEvaluationRequest,
        text_response: str,
    ) -> CriticAnalysis:
        """Create fallback analysis when structured output fails.

        Args:
            request: Evaluation request
            text_response: Text response from LLM

        Returns:
            CriticAnalysis with conservative scores
        """
        logger.warning(f"Using fallback analysis for {request.symbol}")

        # Conservative fallback - assume moderate quality
        return CriticAnalysis(
            passed=False,
            overall_score=0.5,
            logical_consistency_score=0.5,
            game_plan_alignment_score=0.5,
            risk_constraint_score=0.5,
            critical_issues=["Structured evaluation failed - manual review recommended"],
            suggestions=["Re-run analysis with stable LLM provider"],
            reflection=f"**Critique fallback for {request.symbol}:**\n\n{text_response}",
        )
