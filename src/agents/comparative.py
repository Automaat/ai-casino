"""Comparative analysis agent for relative valuation."""

from enum import StrEnum

from loguru import logger
from pydantic import BaseModel, Field

from src.data.comparative import ComparativeData, ComparativeDataFetcher
from src.models.llm import LLMClient


class RelativeValuation(StrEnum):
    """Relative valuation assessment."""

    RELATIVELY_UNDERVALUED = "RELATIVELY_UNDERVALUED"
    FAIRLY_VALUED = "FAIRLY_VALUED"
    RELATIVELY_OVERVALUED = "RELATIVELY_OVERVALUED"


class ComparativeAnalysis(BaseModel):
    """Comparative analysis result."""

    relative_valuation: RelativeValuation
    pe_vs_sector: float | None = Field(description="P/E ratio relative to sector (stock P/E / sector P/E)")
    pe_vs_market: float | None = Field(description="P/E ratio relative to market (stock P/E / market P/E)")
    perf_vs_sector_ytd: float | None = Field(
        description="YTD performance difference vs sector (stock - sector)"
    )
    perf_vs_sector_3m: float | None = Field(
        description="3M performance difference vs sector (stock - sector)"
    )
    perf_vs_market_ytd: float | None = Field(
        description="YTD performance difference vs market (stock - market)"
    )
    perf_vs_market_3m: float | None = Field(
        description="3M performance difference vs market (stock - market)"
    )
    sector_etf: str
    interpretation: str
    confidence: float = Field(ge=0.0, le=1.0)

    def __repr__(self) -> str:
        """String representation."""
        pe_str = f"{self.pe_vs_sector:.2f}" if self.pe_vs_sector else "N/A"
        return (
            f"ComparativeAnalysis(valuation={self.relative_valuation.value}, "
            f"pe_vs_sector={pe_str}, confidence={self.confidence:.2f})"
        )


class ComparativeAnalyst:
    """Analyzes stock relative to sector and market benchmarks."""

    def __init__(self, llm_client: LLMClient, fetcher: ComparativeDataFetcher) -> None:
        """Initialize comparative analyst.

        Args:
            llm_client: LLM client for generating interpretations
            fetcher: Comparative data fetcher
        """
        self.llm = llm_client
        self.fetcher = fetcher
        logger.info("Initialized ComparativeAnalyst")

    async def analyze(self, symbol: str) -> ComparativeAnalysis:
        """Perform comparative analysis on a stock.

        Args:
            symbol: Stock ticker symbol

        Returns:
            ComparativeAnalysis with relative valuation, metrics, and interpretation
        """
        logger.info(f"Analyzing {symbol} vs sector/market")

        try:
            data = self.fetcher.fetch_comparative_data(symbol)
            metrics = self._calculate_relative_metrics(data)
            valuation = self._assess_relative_valuation(data, metrics)

            prompt = self._build_analysis_prompt(symbol, data, metrics, valuation)
            system = (
                "You are a comparative analyst. Provide a concise interpretation "
                "of how this stock compares to its sector and the broader market in 2-3 sentences."
            )

            interpretation = await self.llm.acomplete(prompt, system=system, temperature=0.5)
            confidence = self._calculate_confidence(data, metrics)

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
            logger.error(f"Comparative analysis failed for {symbol}: {e}")
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
        if stock_pe and data.sector_pe:
            metrics["pe_vs_sector"] = stock_pe / data.sector_pe
        else:
            metrics["pe_vs_sector"] = None

        if stock_pe and data.market_pe:
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
        undervalued_pe = pe_vs_sector is not None and pe_vs_sector < 0.8
        overvalued_pe = pe_vs_sector is not None and pe_vs_sector > 1.3

        # Performance + valuation combo: outperforming at discount
        outperforming_at_discount = (
            undervalued_pe and perf_vs_sector_3m is not None and perf_vs_sector_3m > 10
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
        parts = [f"Analyze comparative data for {symbol} vs sector ({data.sector_etf}) and market (SPY):"]

        parts.append(f"Relative Valuation: {valuation.value}")

        if data.stock_info.pe_ratio is not None:
            parts.append(f"Stock P/E: {data.stock_info.pe_ratio:.2f}")
        if data.sector_pe is not None:
            parts.append(f"Sector P/E: {data.sector_pe:.2f}")
        if data.market_pe is not None:
            parts.append(f"Market P/E: {data.market_pe:.2f}")

        if metrics.get("pe_vs_sector") is not None:
            parts.append(f"P/E vs Sector: {metrics['pe_vs_sector']:.2f}x")
        if metrics.get("pe_vs_market") is not None:
            parts.append(f"P/E vs Market: {metrics['pe_vs_market']:.2f}x")

        # Performance
        stock_perf = data.stock_performance
        if stock_perf.ytd_return is not None:
            parts.append(f"Stock YTD: {stock_perf.ytd_return:.1f}%")
        if stock_perf.three_month_return is not None:
            parts.append(f"Stock 3M: {stock_perf.three_month_return:.1f}%")

        if metrics.get("perf_vs_sector_ytd") is not None:
            sign = "+" if metrics["perf_vs_sector_ytd"] >= 0 else ""
            parts.append(f"vs Sector YTD: {sign}{metrics['perf_vs_sector_ytd']:.1f}%")
        if metrics.get("perf_vs_market_ytd") is not None:
            sign = "+" if metrics["perf_vs_market_ytd"] >= 0 else ""
            parts.append(f"vs Market YTD: {sign}{metrics['perf_vs_market_ytd']:.1f}%")

        return "\n".join(parts)

    def _calculate_confidence(self, data: ComparativeData, metrics: dict[str, float | None]) -> float:
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

    def __repr__(self) -> str:
        """String representation."""
        return f"ComparativeAnalyst(llm={self.llm}, fetcher={self.fetcher})"
