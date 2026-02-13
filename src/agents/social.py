"""Social Sentiment Analysis Agent."""

from datetime import UTC, datetime, timedelta

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from src.data.finnhub import FinnhubFetcher, NewsSentimentData, SocialSentimentData
from src.data.reddit import RedditFetcher, RedditSentimentData
from src.execution_tracking import track_agent
from src.models.llm import LLMClient
from src.models.providers.base import StructuredOutputError
from src.models.sentiment import FinBERTSentiment, SentimentScore, _analyze_batch_worker, get_finbert_executor
from src.prompts import PromptLoader


class SocialSentimentLLMResponse(BaseModel):
    """LLM structured output for social sentiment analysis."""

    interpretation: str = Field(description="Synthesis of social sentiment signals")
    sentiment_label: str = Field(description="BULLISH/BEARISH/NEUTRAL sentiment label")
    confidence_keywords: list[str] = Field(description="Keywords indicating confidence level")


class SocialSentimentAnalysis(BaseModel):
    """Social sentiment analysis result."""

    finnhub_sentiment: float | None  # -1 to 1
    reddit_sentiment: float | None  # -1 to 1
    overall_social_score: float  # -1 to 1 weighted average
    social_momentum: str  # rising/falling/stable
    wsb_mentions_24h: int
    sentiment_label: str  # BULLISH/BEARISH/NEUTRAL from LLM
    interpretation: str  # From LLM
    confidence: float  # 0.0-1.0 multi-factor


class SocialSentimentAnalyst:
    """Agent for analyzing social sentiment from Reddit and Finnhub."""

    def __init__(
        self,
        llm_client: LLMClient,
        finnhub_fetcher: FinnhubFetcher,
        reddit_fetcher: RedditFetcher,
        finbert: FinBERTSentiment,
    ) -> None:
        """Initialize social sentiment analyst.

        Args:
            llm_client: LLM client for interpretation
            finnhub_fetcher: Finnhub data fetcher
            reddit_fetcher: Reddit data fetcher
            finbert: FinBERT sentiment model
        """
        self.llm = llm_client
        self.finnhub = finnhub_fetcher
        self.reddit = reddit_fetcher
        self.finbert = finbert
        self._prompts = PromptLoader("social")
        logger.info("Initialized SocialSentimentAnalyst")

    @track_agent
    async def analyze(self, symbol: str) -> SocialSentimentAnalysis:
        """Analyze social sentiment from multiple sources.

        Args:
            symbol: Stock ticker symbol

        Returns:
            SocialSentimentAnalysis with unified social sentiment
        """
        logger.info(f"Analyzing social sentiment for {symbol}")

        # Fetch data from all sources
        finnhub_social, finnhub_news, reddit_data = await self._fetch_all_sources(symbol)

        # Compute individual sentiments
        finnhub_sentiment = self._compute_finnhub_sentiment(finnhub_social)
        reddit_sentiment = await self._compute_reddit_sentiment(reddit_data)

        # Compute overall score and momentum
        overall_score = self._compute_overall_social_score(finnhub_social, finnhub_news, reddit_sentiment)
        momentum = self._compute_social_momentum(finnhub_social)
        wsb_mentions = reddit_data.mention_count if reddit_data else 0

        # LLM interpretation
        interpretation, sentiment_label, confidence_keywords = await self._get_llm_interpretation(
            symbol, finnhub_social, finnhub_news, reddit_data, reddit_sentiment, overall_score, momentum
        )

        # Compute confidence
        confidence = self._compute_confidence(
            finnhub_social, finnhub_news, reddit_data, reddit_sentiment, confidence_keywords
        )

        logger.info(
            f"Social sentiment analysis complete for {symbol}: "
            f"score={overall_score:.2f}, label={sentiment_label}, confidence={confidence:.2f}"
        )

        return SocialSentimentAnalysis(
            finnhub_sentiment=finnhub_sentiment,
            reddit_sentiment=reddit_sentiment,
            overall_social_score=overall_score,
            social_momentum=momentum,
            wsb_mentions_24h=wsb_mentions,
            sentiment_label=sentiment_label,
            interpretation=interpretation,
            confidence=confidence,
        )

    def _process_fetch_results(
        self,
        results: tuple[
            BaseException | SocialSentimentData | None,
            BaseException | NewsSentimentData | None,
            BaseException | RedditSentimentData | None,
        ],
    ) -> tuple[SocialSentimentData | None, NewsSentimentData | None, RedditSentimentData | None]:
        """Process gather results, handling exceptions and logging errors.

        Args:
            results: Results from asyncio.gather with return_exceptions=True

        Returns:
            Tuple of (finnhub_social, finnhub_news, reddit_data)
        """
        finnhub_social: SocialSentimentData | None = None
        finnhub_news: NewsSentimentData | None = None
        reddit_data: RedditSentimentData | None = None

        if isinstance(results[0], BaseException):
            logger.error(f"Finnhub social fetch failed: {results[0]}")
        else:
            finnhub_social = results[0]

        if isinstance(results[1], BaseException):
            logger.error(f"Finnhub news fetch failed: {results[1]}")
        else:
            finnhub_news = results[1]

        if isinstance(results[2], BaseException):
            logger.error(f"Reddit fetch failed: {results[2]}")
        else:
            reddit_data = results[2]

        return finnhub_social, finnhub_news, reddit_data

    async def _fetch_all_sources(
        self, symbol: str
    ) -> tuple[SocialSentimentData | None, NewsSentimentData | None, RedditSentimentData | None]:
        """Fetch data from all sources in parallel.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Tuple of (finnhub_social, finnhub_news, reddit_data)
        """
        import asyncio

        async def fetch_finnhub_social() -> SocialSentimentData | None:
            try:
                # Last 7 days
                to_date = datetime.now(tz=UTC)
                from_date = to_date - timedelta(days=7)
                return await asyncio.to_thread(
                    self.finnhub.fetch_social_sentiment,
                    symbol,
                    from_date.strftime("%Y-%m-%d"),
                    to_date.strftime("%Y-%m-%d"),
                )
            except Exception as e:
                logger.opt(exception=True).warning(f"Finnhub social sentiment fetch failed: {e}")
                return None

        async def fetch_finnhub_news() -> NewsSentimentData | None:
            try:
                return await asyncio.to_thread(self.finnhub.fetch_sentiment_indicator, symbol)
            except Exception as e:
                logger.opt(exception=True).warning(f"Finnhub news sentiment fetch failed: {e}")
                return None

        async def fetch_reddit() -> RedditSentimentData | None:
            try:
                return await asyncio.to_thread(
                    self.reddit.fetch_mentions,
                    symbol,
                    subreddits=["wallstreetbets", "stocks", "investing"],
                    limit=25,
                    time_filter="day",
                )
            except Exception as e:
                logger.opt(exception=True).warning(f"Reddit fetch failed: {e}")
                return None

        # Run fetches in parallel using TaskGroup (exceptions handled within fetch functions)
        async with asyncio.TaskGroup() as tg:
            social_task = tg.create_task(fetch_finnhub_social())
            news_task = tg.create_task(fetch_finnhub_news())
            reddit_task = tg.create_task(fetch_reddit())

        # Extract results (TaskGroup propagates exceptions, fetch functions return None on failure)
        social_result = social_task.result()
        news_result = news_task.result()
        reddit_result = reddit_task.result()
        results = (social_result, news_result, reddit_result)

        return self._process_fetch_results(results)

    def _compute_finnhub_sentiment(self, data: SocialSentimentData | None) -> float | None:
        """Compute average Finnhub sentiment from Reddit and Twitter.

        Args:
            data: Finnhub social sentiment data

        Returns:
            Average sentiment -1 to 1, or None if no data
        """
        if not data or (not data.reddit and not data.twitter):
            return None

        scores = []
        if data.reddit:
            scores.extend(entry.score for entry in data.reddit)
        if data.twitter:
            scores.extend(entry.score for entry in data.twitter)

        return float(np.mean(scores)) if scores else None

    async def _compute_reddit_sentiment(self, data: RedditSentimentData | None) -> float | None:
        """Compute weighted Reddit sentiment using FinBERT.

        Args:
            data: Reddit sentiment data

        Returns:
            Weighted sentiment -1 to 1, or None if no data
        """
        if not data or not data.posts:
            return None

        # Analyze posts with FinBERT (async to avoid blocking)
        texts = [f"{post.title} {post.body}" for post in data.posts]

        import asyncio

        loop = asyncio.get_running_loop()
        # Use ProcessPoolExecutor for true parallelism (avoids GIL)
        executor = get_finbert_executor()
        score_dicts = await loop.run_in_executor(
            executor, _analyze_batch_worker, texts, self.finbert.device
        )
        sentiments: list[SentimentScore] = [SentimentScore(**s) for s in score_dicts]

        # Weight by upvote_ratio * score
        weighted_sum = 0.0
        total_weight = 0.0

        for post, sentiment in zip(data.posts, sentiments, strict=True):
            weight = post.upvote_ratio * max(1, post.score)
            weighted_sum += sentiment.score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _compute_overall_social_score(
        self,
        finnhub_social: SocialSentimentData | None,
        finnhub_news: NewsSentimentData | None,
        reddit_sentiment: float | None,
    ) -> float:
        """Compute weighted overall social score.

        Args:
            finnhub_social: Finnhub social data
            finnhub_news: Finnhub news sentiment data
            reddit_sentiment: Reddit FinBERT sentiment

        Returns:
            Weighted average -1 to 1
        """
        scores = []
        weights = []

        # Finnhub social: 40%
        if finnhub_social:
            finnhub_sent = self._compute_finnhub_sentiment(finnhub_social)
            if finnhub_sent is not None:
                scores.append(finnhub_sent)
                weights.append(0.4)

        # Reddit FinBERT: 40%
        if reddit_sentiment is not None:
            scores.append(reddit_sentiment)
            weights.append(0.4)

        # Finnhub news: 20%
        if finnhub_news:
            # Convert bullish/bearish percent to -1 to 1 scale
            news_score = (finnhub_news.sentiment.bullish_percent - 50.0) / 50.0
            scores.append(news_score)
            weights.append(0.2)

        if not scores:
            return 0.0

        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        return sum(score * weight for score, weight in zip(scores, normalized_weights, strict=True))

    def _compute_social_momentum(self, finnhub_social: SocialSentimentData | None) -> str:
        """Compute social momentum trend.

        Args:
            finnhub_social: Finnhub social sentiment data

        Returns:
            "rising", "falling", or "stable"
        """
        if not finnhub_social or (not finnhub_social.reddit and not finnhub_social.twitter):
            return "stable"

        # Combine reddit and twitter entries
        all_entries = list(finnhub_social.reddit) + list(finnhub_social.twitter)
        if len(all_entries) < 4:
            return "stable"

        # Sort by time
        sorted_entries = sorted(all_entries, key=lambda e: e.at_time)

        # Split into recent (0-3 days) and older (3-7 days)
        cutoff = datetime.now(tz=UTC) - timedelta(days=3)
        recent = [e for e in sorted_entries if e.at_time >= cutoff]
        older = [e for e in sorted_entries if e.at_time < cutoff]

        if not recent or not older:
            return "stable"

        recent_avg = np.mean([e.score for e in recent])
        older_avg = np.mean([e.score for e in older])

        diff = recent_avg - older_avg

        if diff > 0.1:
            return "rising"
        if diff < -0.1:
            return "falling"
        return "stable"

    async def _get_llm_interpretation(
        self,
        symbol: str,
        finnhub_social: SocialSentimentData | None,
        finnhub_news: NewsSentimentData | None,
        reddit_data: RedditSentimentData | None,
        reddit_sentiment: float | None,
        overall_score: float,
        momentum: str,
    ) -> tuple[str, str, list[str]]:
        """Get LLM interpretation of social sentiment.

        Args:
            symbol: Stock ticker
            finnhub_social: Finnhub social data
            finnhub_news: Finnhub news data
            reddit_data: Reddit data
            reddit_sentiment: Precomputed Reddit sentiment
            overall_score: Computed overall score
            momentum: Social momentum

        Returns:
            Tuple of (interpretation, sentiment_label, confidence_keywords)
        """
        finnhub_social_summary = self._format_finnhub_summary(finnhub_social)
        reddit_posts_text = self._format_reddit_posts(reddit_data)
        finnhub_news_summary = self._format_finnhub_news_summary(finnhub_news)

        prompt = self._prompts.load(
            "user",
            symbol=symbol,
            finnhub_social_summary=finnhub_social_summary,
            wsb_mentions=reddit_data.mention_count if reddit_data else 0,
            reddit_sentiment=reddit_sentiment,
            top_reddit_posts=reddit_posts_text,
            finnhub_news_summary=finnhub_news_summary,
            overall_score=overall_score,
            momentum=momentum,
        )
        system_prompt = self._prompts.load("system")

        try:
            llm_response = await self.llm.astructured(
                prompt, SocialSentimentLLMResponse, system=system_prompt, temperature=0.4
            )
            return (
                llm_response.interpretation,
                llm_response.sentiment_label,
                llm_response.confidence_keywords,
            )
        except StructuredOutputError as e:
            logger.opt(exception=True).warning(f"Structured output failed, falling back to text parsing: {e}")
            response = await self.llm.acomplete(prompt, system=system_prompt, temperature=0.4)

            # Extract sentiment label
            sentiment_label = "NEUTRAL"
            if "bullish" in response.lower():
                sentiment_label = "BULLISH"
            elif "bearish" in response.lower():
                sentiment_label = "BEARISH"

            # Extract confidence keywords
            confidence_keywords = []
            for keyword in ["strong", "clear", "high", "significant"]:
                if keyword in response.lower():
                    confidence_keywords.append(keyword)

            return response[:500], sentiment_label, confidence_keywords

    def _format_finnhub_summary(self, data: SocialSentimentData | None) -> str:
        """Format Finnhub social sentiment summary.

        Args:
            data: Finnhub social sentiment data

        Returns:
            Formatted summary string
        """
        if not data:
            return "No Finnhub social data available"

        reddit_count = len(data.reddit)
        twitter_count = len(data.twitter)
        reddit_avg = np.mean([e.score for e in data.reddit]) if data.reddit else 0.0
        twitter_avg = np.mean([e.score for e in data.twitter]) if data.twitter else 0.0

        return (
            f"Finnhub (7 days): Reddit {reddit_count} mentions (avg score: {reddit_avg:.2f}), "
            f"Twitter {twitter_count} mentions (avg score: {twitter_avg:.2f})"
        )

    def _format_reddit_posts(self, data: RedditSentimentData | None, limit: int = 5) -> str:
        """Format top Reddit posts.

        Args:
            data: Reddit sentiment data
            limit: Max posts to include

        Returns:
            Formatted posts string
        """
        if not data or not data.posts:
            return "No Reddit posts found"

        sorted_posts = sorted(data.posts, key=lambda p: p.score, reverse=True)[:limit]
        lines = []

        for i, post in enumerate(sorted_posts, 1):
            lines.append(
                f"{i}. r/{post.subreddit} (score: {post.score}, "
                f"upvote: {post.upvote_ratio:.2f}): {post.title[:100]}"
            )

        return "\n".join(lines)

    def _format_finnhub_news_summary(self, data: NewsSentimentData | None) -> str:
        """Format Finnhub news sentiment summary.

        Args:
            data: Finnhub news sentiment data

        Returns:
            Formatted summary string
        """
        if not data:
            return "No Finnhub news data available"

        return (
            f"News buzz: {data.buzz.articles_in_last_week} articles, "
            f"bullish: {data.sentiment.bullish_percent:.1f}%, "
            f"bearish: {data.sentiment.bearish_percent:.1f}%"
        )

    def _compute_confidence(
        self,
        finnhub_social: SocialSentimentData | None,
        finnhub_news: NewsSentimentData | None,
        reddit_data: RedditSentimentData | None,
        reddit_sentiment: float | None,
        llm_keywords: list[str],
    ) -> float:
        """Compute multi-factor confidence score.

        Args:
            finnhub_social: Finnhub social data
            finnhub_news: Finnhub news data
            reddit_data: Reddit data
            reddit_sentiment: Precomputed Reddit sentiment
            llm_keywords: Confidence keywords from LLM

        Returns:
            Confidence score 0.0-1.0
        """
        # Data availability factor (3 sources = 1.0)
        sources_available = sum(
            [
                finnhub_social is not None,
                finnhub_news is not None,
                reddit_data is not None and reddit_data.mention_count > 0,
            ]
        )
        availability_factor = sources_available / 3.0

        # Sample size factor (Reddit mentions)
        reddit_mentions = reddit_data.mention_count if reddit_data else 0
        if reddit_mentions >= 50:
            sample_factor = 1.0
        elif reddit_mentions >= 20:
            sample_factor = 0.8
        elif reddit_mentions >= 10:
            sample_factor = 0.6
        else:
            sample_factor = 0.4

        # Agreement factor (low std_dev = high agreement)
        scores = []
        if finnhub_social:
            finnhub_sent = self._compute_finnhub_sentiment(finnhub_social)
            if finnhub_sent is not None:
                scores.append(finnhub_sent)
        if reddit_sentiment is not None:
            scores.append(reddit_sentiment)
        if finnhub_news:
            news_score = (finnhub_news.sentiment.bullish_percent - 50.0) / 50.0
            scores.append(news_score)

        if len(scores) >= 2:
            std_dev = float(np.std(scores))
            agreement_factor = max(0.0, 1.0 - std_dev)
        else:
            agreement_factor = 0.5

        # LLM keyword boost
        keyword_boost = (
            0.1
            if any(keyword in llm_keywords for keyword in ["strong", "clear", "high", "significant"])
            else 0.0
        )

        # Combine factors
        base_confidence = availability_factor * 0.4 + sample_factor * 0.3 + agreement_factor * 0.3
        return min(1.0, base_confidence + keyword_boost)

    def __repr__(self) -> str:
        """String representation."""
        return f"SocialSentimentAnalyst(llm={self.llm.provider})"
