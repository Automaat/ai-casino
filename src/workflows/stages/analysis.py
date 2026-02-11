"""Analysis stage implementation."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any, TypeVar, cast

from loguru import logger

if TYPE_CHECKING:
    from src.agents.comparative import ComparativeAnalyst
    from src.agents.fundamental import FundamentalAnalyst
    from src.agents.news import NewsAnalyst
    from src.agents.sentiment import SentimentAnalyst
    from src.agents.social import SocialSentimentAnalyst
    from src.agents.technical import TechnicalAnalyst
    from src.agents.thesis_researcher import ThesisResearcher
    from src.agents.trump import TrumpAnalyst
    from src.agents.web_researcher import WebResearchAgent
    from src.metrics.execution import ExecutionMetricsCollector

from src.agents.comparative import ComparativeAnalysis
from src.agents.fundamental import FundamentalAnalysis
from src.agents.news import NewsAnalysis
from src.agents.sentiment import SentimentAnalysis
from src.agents.social import SocialSentimentAnalysis
from src.agents.technical import TechnicalAnalysis
from src.agents.trump import TrumpAnalysis
from src.agents.web_researcher import WebResearchAnalysis
from src.workflows.models.analysis import AnalysisInput, AnalysisOutput
from src.workflows.stages.strategy_selection import _timed_agent_call

T = TypeVar("T")


def _unwrap_or_log(result: T | BaseException, label: str) -> T | None:
    """Return result if successful, log and return None if exception.

    Control-flow exceptions (e.g. cancellation, shutdown) are re-raised to
    allow proper propagation.
    """
    if isinstance(result, BaseException):
        # Re-raise control-flow exceptions so shutdown/cancellation propagates.
        if isinstance(result, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
            raise result
        # Log other exceptions with stack trace and return None.
        logger.opt(exception=result).error(f"{label} failed")
        return None
    return result


def _is_rate_limit_error(e: Exception) -> bool:
    """Check if exception is related to API rate limiting."""
    msg = str(e).lower()
    return any(
        pattern in msg
        for pattern in [
            "rate limit",
            "call frequency",
            "premium endpoint",
            "5 api calls per minute",
        ]
    )


def _handle_fundamental_result(
    result: object,
    warnings: list[str],
) -> FundamentalAnalysis | None:
    """Handle fundamental analysis result with rate-limit awareness.

    Args:
        result: Result from fundamental analysis
        warnings: List to append warnings to

    Returns:
        FundamentalAnalysis or None if rate-limited
    """
    if isinstance(result, Exception):
        if _is_rate_limit_error(result):
            warning = f"Fundamental analysis unavailable: {result}"
            logger.warning(warning)
            warnings.append(warning)
            return None
        raise result
    assert isinstance(result, FundamentalAnalysis)  # noqa: S101
    return result


def _handle_optional_result(
    result: T | Exception,
    name: str,
    warnings: list[str],
) -> T | None:
    """Handle optional analysis result, logging failures as warnings.

    Args:
        result: Result from analysis
        name: Name of analysis for logging
        warnings: List to append warnings to

    Returns:
        Analysis result or None if failed
    """
    if isinstance(result, Exception):
        warning = f"{name} analysis failed: {result}"
        logger.warning(warning)
        warnings.append(warning)
        return None
    return result


async def run_analyses(  # noqa: PLR0913
    input_data: AnalysisInput,
    technical_analyst: TechnicalAnalyst,
    sentiment_analyst: SentimentAnalyst,
    news_analyst: NewsAnalyst,
    fundamental_analyst: FundamentalAnalyst,
    comparative_analyst: ComparativeAnalyst,
    web_researcher: WebResearchAgent,
    social_analyst: SocialSentimentAnalyst,
    bullish_researcher: ThesisResearcher,
    bearish_researcher: ThesisResearcher,
    trump_mode: bool,
    trump_analyst: TrumpAnalyst | None,
    collector: ExecutionMetricsCollector | None = None,
) -> AnalysisOutput:
    """Run all analysis agents in parallel groups.

    Args:
        input_data: Analysis input with symbol, market data, news, trump posts
        technical_analyst: Technical analyst with selected strategy
        sentiment_analyst: Sentiment analyst
        news_analyst: News analyst
        fundamental_analyst: Fundamental analyst
        comparative_analyst: Comparative analyst
        web_researcher: Web research agent
        social_analyst: Social sentiment analyst
        bullish_researcher: Bullish researcher
        bearish_researcher: Bearish researcher
        trump_mode: Enable Trump analysis
        trump_analyst: Trump analyst (required if trump_mode=True)
        collector: Optional metrics collector

    Returns:
        AnalysisOutput with all analyses
    """
    warnings: list[str] = []

    # Validate market data
    if input_data.market_data is None:
        msg = "market_data is None, cannot run analyses"
        raise ValueError(msg)

    # Validate and extract data using helpers
    current_price = input_data.get_current_price()

    # Validate news_articles
    if input_data.news_articles is None:
        msg = "news_articles is None, cannot run sentiment/news analyses"
        raise ValueError(msg)

    # Handle empty news with warning
    if not input_data.news_articles:
        logger.warning(f"No news articles available for {input_data.symbol}, analyses may be degraded")
        warnings.append("No news articles available - sentiment and news analyses degraded")

    # Parallel Group 1: independent analyses (comparative, web_research, social, trump are optional)
    technical_task = _timed_agent_call(
        "technical",
        technical_analyst.analyze(
            input_data.symbol,
            input_data.market_data,
            enable_multi_timeframe=input_data.enable_multi_timeframe,
        ),
        collector,
    )
    sentiment_task = _timed_agent_call(
        "sentiment",
        sentiment_analyst.analyze(input_data.symbol, input_data.news_articles),
        collector,
    )
    news_task = _timed_agent_call(
        "news",
        news_analyst.analyze(input_data.symbol, input_data.news_articles),
        collector,
    )
    fundamental_task = _timed_agent_call(
        "fundamental",
        fundamental_analyst.analyze(input_data.symbol, current_price),
        collector,
    )
    comparative_task = _timed_agent_call(
        "comparative",
        comparative_analyst.analyze(input_data.symbol),
        collector,
    )
    web_research_task = _timed_agent_call(
        "web_research",
        web_researcher.research(input_data.symbol),
        collector,
    )
    social_task = _timed_agent_call(
        "social",
        social_analyst.analyze(input_data.symbol),
        collector,
    )

    # Wrap optional tasks to handle exceptions without canceling siblings
    async def safe_optional_task(coro: Coroutine) -> Any:  # noqa: ANN401
        try:
            return await coro
        except Exception as e:
            # Return exception as value - will be handled by _handle_optional_result
            return e

    # Run analyses in parallel using TaskGroup for structured concurrency
    async with asyncio.TaskGroup() as tg:
        # Core tasks (must succeed - let them raise)
        technical_result = tg.create_task(technical_task)
        sentiment_result = tg.create_task(sentiment_task)
        news_result = tg.create_task(news_task)

        # Optional tasks (wrap to return exceptions as values)
        fundamental_result = tg.create_task(safe_optional_task(fundamental_task))
        comparative_result = tg.create_task(safe_optional_task(comparative_task))
        web_research_result = tg.create_task(safe_optional_task(web_research_task))
        social_result = tg.create_task(safe_optional_task(social_task))

        trump_result = None
        if trump_mode and trump_analyst and input_data.trump_posts:
            trump_task = _timed_agent_call(
                "trump",
                trump_analyst.analyze(input_data.trump_posts),
                collector,
            )
            trump_result = tg.create_task(safe_optional_task(trump_task))

    # Extract results - core analyses (will raise if they failed)
    technical = technical_result.result()
    sentiment = sentiment_result.result()
    news = news_result.result()

    # Type narrowing
    assert isinstance(technical, TechnicalAnalysis)  # noqa: S101
    assert isinstance(sentiment, SentimentAnalysis)  # noqa: S101
    assert isinstance(news, NewsAnalysis)  # noqa: S101

    # Process optional analyses - wrapped tasks returned exceptions as values
    fundamental = _handle_fundamental_result(fundamental_result.result(), warnings)
    comparative = cast(
        "ComparativeAnalysis | None",
        _handle_optional_result(comparative_result.result(), "Comparative", warnings),
    )
    web_research = cast(
        "WebResearchAnalysis | None",
        _handle_optional_result(web_research_result.result(), "Web research", warnings),
    )
    social_sentiment = cast(
        "SocialSentimentAnalysis | None",
        _handle_optional_result(social_result.result(), "Social sentiment", warnings),
    )
    trump_analysis_processed = (
        cast(
            "TrumpAnalysis | None",
            _handle_optional_result(trump_result.result(), "Trump", warnings),
        )
        if trump_result
        else None
    )

    # Parallel Group 2: research (depends on Group 1)
    bullish_task = _timed_agent_call(
        "bullish_researcher",
        bullish_researcher.analyze(
            input_data.symbol,
            technical,
            sentiment,
            news,
            fundamental,
            comparative,
            trump_analysis_processed,
        ),
        collector,
    )
    bearish_task = _timed_agent_call(
        "bearish_researcher",
        bearish_researcher.analyze(
            input_data.symbol,
            technical,
            sentiment,
            news,
            fundamental,
            comparative,
            trump_analysis_processed,
        ),
        collector,
    )

    # Wrap research tasks to handle exceptions (these are optional)
    async def safe_research_task(coro: Coroutine) -> Any:  # noqa: ANN401
        try:
            return await coro
        except Exception as e:
            return e

    # Run research tasks in parallel using TaskGroup
    async with asyncio.TaskGroup() as tg:
        bullish_result = tg.create_task(safe_research_task(bullish_task))
        bearish_result = tg.create_task(safe_research_task(bearish_task))

    bullish = bullish_result.result()
    bearish = bearish_result.result()

    return AnalysisOutput(
        technical_analysis=technical,
        sentiment_analysis=sentiment,
        news_analysis=news,
        trump_analysis=trump_analysis_processed,
        fundamental_analysis=fundamental,
        comparative_analysis=comparative,
        web_research=web_research,
        social_sentiment_analysis=social_sentiment,
        bullish_research=_unwrap_or_log(bullish, "Bullish research"),
        bearish_research=_unwrap_or_log(bearish, "Bearish research"),
        warnings=warnings,
    )
