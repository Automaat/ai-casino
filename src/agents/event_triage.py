"""Event triage agent for real-time market signal watcher.

This agent uses LLM to assess event relevance, extract stock symbols, determine urgency,
and classify sentiment. Acts as a filter before triggering expensive trading analysis.
"""

from datetime import UTC, datetime
from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field

from src.daemon.events import BaseEvent, Sentiment, TriageResult, Urgency
from src.execution_tracking import track_agent
from src.models.llm import LLMClient
from src.models.providers.base import StructuredOutputError
from src.prompts import PromptLoader


class TriageLLMResponse(BaseModel):
    """LLM structured output for event triage."""

    relevance: float = Field(ge=0.0, le=1.0, description="Event relevance score 0-1")
    symbols: list[str] = Field(description="Extracted stock tickers (uppercase)")
    urgency: Literal["IMMEDIATE", "WATCHLIST", "IGNORE"] = Field(
        description="Urgency level: IMMEDIATE (analyze now), WATCHLIST (monitor), IGNORE (skip)"
    )
    sentiment: Literal["BULLISH", "BEARISH", "NEUTRAL"] = Field(description="Market sentiment direction")
    confidence: float = Field(ge=0.0, le=1.0, description="Triage confidence 0-1")
    reasoning: str = Field(description="Explanation for triage decision")


class EventTriageAgent:
    """Agent that triages real-time events using LLM.

    Assesses relevance, extracts symbols, determines urgency, and classifies sentiment.
    Uses structured output for consistency, falls back to low relevance on errors.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize event triage agent.

        Args:
            llm_client: LLM client for triage analysis
        """
        self.llm = llm_client
        self._prompts = PromptLoader("event_triage")
        logger.info("Initialized EventTriageAgent")

    @track_agent
    async def analyze(self, event: BaseEvent) -> TriageResult:
        """Triage an event to determine if analysis is warranted.

        Args:
            event: Event to triage (NewsEvent, SocialEvent, etc)

        Returns:
            TriageResult with relevance, symbols, urgency, sentiment
        """
        event_text = event.to_prompt_text()
        prompt = self._prompts.load("user", event_text=event_text)
        system = self._prompts.load("system")

        try:
            llm_response = await self.llm.astructured(
                prompt, TriageLLMResponse, system=system, temperature=0.3, max_tokens=512
            )

            result = TriageResult(
                event_id=event.event_id,
                event_type=event.event_type,
                relevance=llm_response.relevance,
                symbols=[s.upper() for s in llm_response.symbols],  # Normalize tickers
                urgency=Urgency(llm_response.urgency),
                sentiment=Sentiment(llm_response.sentiment),
                confidence=llm_response.confidence,
                reasoning=llm_response.reasoning,
                triaged_at=datetime.now(UTC),
            )

            logger.info(
                f"Triaged {event.event_type} event: relevance={result.relevance:.2f}, "
                f"urgency={result.urgency.value}, symbols={result.symbols}"
            )
            return result

        except StructuredOutputError as e:
            logger.opt(exception=True).warning(f"Triage structured output failed for {event.event_id}: {e}")
            # Fallback to low relevance (skip analysis)
            return TriageResult(
                event_id=event.event_id,
                event_type=event.event_type,
                relevance=0.3,
                symbols=[],
                urgency=Urgency.IGNORE,
                sentiment=Sentiment.NEUTRAL,
                confidence=0.0,
                reasoning=f"Triage failed: {str(e)[:100]}",
                triaged_at=datetime.now(UTC),
            )

    def __repr__(self) -> str:
        """String representation."""
        return f"EventTriageAgent(llm={self.llm})"
