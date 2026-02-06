"""Tools package for LLM function calling."""

from src.tools.analyze_stock import AnalyzeStockTool
from src.tools.backtest import RunBacktestTool
from src.tools.base import BaseTool
from src.tools.market_data import GetMarketDataTool
from src.tools.news import GetNewsTool
from src.tools.portfolio import OptimizePortfolioTool
from src.tools.registry import ToolRegistry
from src.tools.risk_metrics import GetRiskMetricsTool
from src.tools.screen_stocks import ScreenStocksTool
from src.tools.social_sentiment import GetSocialSentimentTool
from src.tools.tearsheet import GenerateTearsheetTool
from src.tools.trump_analysis import TrumpAnalysisTool
from src.tools.websearch import WebSearchTool

__all__ = [
    "AnalyzeStockTool",
    "BaseTool",
    "GenerateTearsheetTool",
    "GetMarketDataTool",
    "GetNewsTool",
    "GetRiskMetricsTool",
    "GetSocialSentimentTool",
    "OptimizePortfolioTool",
    "RunBacktestTool",
    "ScreenStocksTool",
    "ToolRegistry",
    "TrumpAnalysisTool",
    "WebSearchTool",
]
