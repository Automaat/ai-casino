"""Workers package - AI-based worker implementations."""

from src.workers.comparative import ComparativeWorker
from src.workers.fundamental import FundamentalWorker
from src.workers.news import NewsWorker
from src.workers.sentiment import SentimentWorker
from src.workers.social import SocialSentimentWorker
from src.workers.technical import TechnicalWorker
from src.workers.thesis_research import ThesisResearchWorker
from src.workers.trump import TrumpWorker
from src.workers.web_research import WebResearchWorker

__all__ = [
    "ComparativeWorker",
    "FundamentalWorker",
    "NewsWorker",
    "SentimentWorker",
    "SocialSentimentWorker",
    "TechnicalWorker",
    "ThesisResearchWorker",
    "TrumpWorker",
    "WebResearchWorker",
]
