"""Comparative analysis worker."""

import asyncio
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from src.agents.comparative import ComparativeAnalysis, RelativeValuation
from src.data.comparative import ComparativeData, ComparativeDataFetcher
from src.models.llm import LLMClient
from src.models.providers.base import StructuredOutputError
from src.prompts import PromptLoader
from src.tools.models import ToolDefinition, ToolFunction, ToolParameter, ToolParametersSchema

# Valuation thresholds
PE_UNDERVALUED_THRESHOLD = 0.8  # P/E ratio < 0.8 of sector average indicates undervaluation
PE_OVERVALUED_THRESHOLD = 1.3  # P/E ratio > 1.3 of sector average indicates overvaluation
PERF_OUTPERFORM_THRESHOLD = 10  # >10% outperformance vs sector


class ComparativeLLMResponse(BaseModel):
    """LLM response for comparative analysis."""

    interpretation: str = Field(description="Comparative analysis interpretation")
    confidence_keywords: list[str] = Field(description="Confidence indicators")


class ComparativeWorker:
    """Worker for comparative analysis - stateless implementation."""

    def __init__(self, llm_client: LLMClient, fetcher: ComparativeDataFetcher) -> None:
        """Initialize comparative worker.

        Args:
            llm_client: LLM client for generating interpretations
            fetcher: Comparative data fetcher
        """
        self.llm = llm_client
        self.fetcher = fetcher
        self._prompts = PromptLoader("comparative")
        logger.info("Initialized ComparativeWorker")

    async def analyze(self, symbol: str) -> ComparativeAnalysis:
        """Perform comparative analysis on a stock.

        Args:
            symbol: Stock ticker symbol

        Returns:
            ComparativeAnalysis with relative valuation, metrics, and interpretation
        """
        logger.info(f"ComparativeWorker analyzing {symbol}")

        try:
            # Offload blocking fetcher call to thread
            data = await asyncio.to_thread(self.fetcher.fetch_comparative_data, symbol)
            metrics = self._calculate_relative_metrics(data)
            valuation = self._assess_relative_valuation(data, metrics)

            system = self._prompts.load("system")
            user_prompt = self._build_analysis_prompt(symbol, data, metrics, valuation)

            # Try structured output with fallback
            try:
                llm_response = await self.llm.astructured(
                    user_prompt, ComparativeLLMResponse, system=system, temperature=0.5
                )
                interpretation = llm_response.interpretation
                confidence = self._calculate_confidence_from_keywords(
                    data, metrics, llm_response.confidence_keywords
                )
            except StructuredOutputError as e:
                logger.opt(exception=True).warning(f"Structured output failed, falling back: {e}")
                interpretation = await self.llm.acomplete(user_prompt, system=system, temperature=0.5)
                confidence = self._calculate_confidence(data, metrics)

            logger.info(f"Comparative analysis complete: {valuation.value}, confidence={confidence:.2f}")

            return ComparativeAnalysis(
                relative_valuation=valuation,
                pe_vs_sector=metrics.get("pe_vs_sector"),
                pe_vs_market=metrics.get("pe_vs_market"),
                perf_vs_sector_ytd=metrics.get("perf_vs_sector_ytd"),
                perf_vs_sector_3m=metrics.get("perf_vs_sector_3m"),
                perf_vs_market_ytd=metrics.get("perf_vs_market_ytd"),
                perf_vs_market_3m=metrics.get("perf_vs_market_3m"),
                sector_etf=data.sector_etf,
                interpretation=interpretation,
                confidence=confidence,
            )

        except Exception as e:
            logger.opt(exception=True).error(f"Comparative analysis failed for {symbol}: {e}")
            raise

    def _calculate_relative_metrics(self, data: ComparativeData) -> dict[str, float | None]:
        """Calculate relative metrics vs sector and market.

        Args:
            data: Comparative data

        Returns:
            Dictionary of relative metrics
        """
        metrics: dict[str, float | None] = {}

        # P/E ratios relative to benchmarks
        stock_pe = data.stock_info.pe_ratio
        if stock_pe is not None and data.sector_pe is not None:
            metrics["pe_vs_sector"] = stock_pe / data.sector_pe
        else:
            metrics["pe_vs_sector"] = None

        if stock_pe is not None and data.market_pe is not None:
            metrics["pe_vs_market"] = stock_pe / data.market_pe
        else:
            metrics["pe_vs_market"] = None

        # Performance differences
        stock_perf = data.stock_performance
        sector_perf = data.sector_performance
        market_perf = data.market_performance

        metrics["perf_vs_sector_ytd"] = self._calc_diff(stock_perf.ytd_return, sector_perf.ytd_return)
        metrics["perf_vs_sector_3m"] = self._calc_diff(
            stock_perf.three_month_return, sector_perf.three_month_return
        )
        metrics["perf_vs_market_ytd"] = self._calc_diff(stock_perf.ytd_return, market_perf.ytd_return)
        metrics["perf_vs_market_3m"] = self._calc_diff(
            stock_perf.three_month_return, market_perf.three_month_return
        )

        return metrics

    def _calc_diff(self, a: float | None, b: float | None) -> float | None:
        """Calculate difference between two values."""
        if a is None or b is None:
            return None
        return a - b

    def _assess_relative_valuation(
        self, _data: ComparativeData, metrics: dict[str, float | None]
    ) -> RelativeValuation:
        """Assess relative valuation based on P/E and performance.

        Args:
            data: Comparative data
            metrics: Calculated relative metrics

        Returns:
            RelativeValuation enum
        """
        pe_vs_sector = metrics.get("pe_vs_sector")
        perf_vs_sector_3m = metrics.get("perf_vs_sector_3m")

        # P/E based valuation
        undervalued_pe = pe_vs_sector is not None and pe_vs_sector < PE_UNDERVALUED_THRESHOLD
        overvalued_pe = pe_vs_sector is not None and pe_vs_sector > PE_OVERVALUED_THRESHOLD

        # Performance + valuation combo: outperforming at discount
        outperforming_at_discount = (
            undervalued_pe and perf_vs_sector_3m is not None and perf_vs_sector_3m > PERF_OUTPERFORM_THRESHOLD
        )

        if outperforming_at_discount:
            return RelativeValuation.RELATIVELY_UNDERVALUED

        if undervalued_pe:
            return RelativeValuation.RELATIVELY_UNDERVALUED

        if overvalued_pe:
            return RelativeValuation.RELATIVELY_OVERVALUED

        return RelativeValuation.FAIRLY_VALUED

    def _build_analysis_prompt(
        self,
        symbol: str,
        data: ComparativeData,
        metrics: dict[str, float | None],
        valuation: RelativeValuation,
    ) -> str:
        """Build prompt for LLM interpretation.

        Args:
            symbol: Stock ticker
            data: Comparative data
            metrics: Calculated metrics
            valuation: Assessed valuation

        Returns:
            Formatted prompt string
        """
        valuation_parts = []
        if data.stock_info.pe_ratio is not None:
            valuation_parts.append(f"Stock P/E: {data.stock_info.pe_ratio:.2f}")
        if data.sector_pe is not None:
            valuation_parts.append(f"Sector P/E: {data.sector_pe:.2f}")
        if data.market_pe is not None:
            valuation_parts.append(f"Market P/E: {data.market_pe:.2f}")

        if metrics.get("pe_vs_sector") is not None:
            valuation_parts.append(f"P/E vs Sector: {metrics['pe_vs_sector']:.2f}x")
        if metrics.get("pe_vs_market") is not None:
            valuation_parts.append(f"P/E vs Market: {metrics['pe_vs_market']:.2f}x")

        performance_parts = []
        stock_perf = data.stock_performance
        if stock_perf.ytd_return is not None:
            performance_parts.append(f"Stock YTD: {stock_perf.ytd_return:.1f}%")
        if stock_perf.three_month_return is not None:
            performance_parts.append(f"Stock 3M: {stock_perf.three_month_return:.1f}%")

        if metrics.get("perf_vs_sector_ytd") is not None:
            sign = "+" if metrics["perf_vs_sector_ytd"] >= 0 else ""
            performance_parts.append(f"vs Sector YTD: {sign}{metrics['perf_vs_sector_ytd']:.1f}%")
        if metrics.get("perf_vs_market_ytd") is not None:
            sign = "+" if metrics["perf_vs_market_ytd"] >= 0 else ""
            performance_parts.append(f"vs Market YTD: {sign}{metrics['perf_vs_market_ytd']:.1f}%")

        return self._prompts.load(
            "user",
            symbol=symbol,
            sector_etf=data.sector_etf,
            valuation=valuation.value,
            valuation_metrics="\n".join(valuation_parts),
            performance_metrics="\n".join(performance_parts),
        )

    def _calculate_confidence(self, data: ComparativeData, metrics: dict[str, Any]) -> float:
        """Calculate confidence score based on data completeness.

        Args:
            data: Comparative data
            metrics: Calculated metrics

        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.5

        # Boost for data completeness
        if data.stock_info.pe_ratio is not None:
            confidence += 0.1
        if data.sector_pe is not None:
            confidence += 0.1
        if data.stock_performance.ytd_return is not None:
            confidence += 0.1
        if data.stock_performance.three_month_return is not None:
            confidence += 0.1

        # Boost for sector-specific ETF (not fallback to SPY)
        if data.sector_etf != "SPY":
            confidence += 0.05

        # Metrics completeness
        non_none_metrics = sum(1 for v in metrics.values() if v is not None)
        confidence += 0.05 * (non_none_metrics / len(metrics))

        return min(1.0, confidence)

    def _calculate_confidence_from_keywords(
        self, data: ComparativeData, metrics: dict[str, Any], keywords: list[str]
    ) -> float:
        """Calculate confidence using structured keywords.

        Args:
            data: Comparative data
            metrics: Calculated metrics
            keywords: Confidence keywords from LLM

        Returns:
            Confidence score between 0.0 and 1.0
        """
        base_confidence = self._calculate_confidence(data, metrics)

        # Boost/penalty from keywords
        keywords_lower = [k.lower() for k in keywords]
        if any(word in keywords_lower for word in ["strong", "clear", "high confidence"]):
            base_confidence += 0.1
        if any(word in keywords_lower for word in ["weak", "uncertain", "limited data"]):
            base_confidence -= 0.1

        return max(0.0, min(1.0, base_confidence))

    def get_tool_definition(self) -> ToolDefinition:
        """Get tool definition for supervisor integration.

        Returns:
            Tool definition
        """
        return ToolDefinition(
            type="function",
            function=ToolFunction(
                name="analyze_comparative",
                description="Compare stock to sector and market benchmarks",
                parameters=ToolParametersSchema(
                    type="object",
                    properties={
                        "symbol": ToolParameter(type="string", description="Stock ticker symbol"),
                    },
                    required=["symbol"],
                ),
            ),
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"ComparativeWorker(llm={self.llm}, fetcher={self.fetcher})"
