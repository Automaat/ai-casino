"""Automated stock discovery system.

Provides multi-source discovery with intelligent scoring and portfolio-aware filtering.
"""

from src.discovery.engine import DiscoveryEngineConfig, StockDiscoveryEngine
from src.discovery.filters import PortfolioFilterConfig, PortfolioFilterEngine
from src.discovery.models import DiscoveryCandidate, DiscoveryResult, DiscoverySource
from src.discovery.scoring import MultiFactorScorer, ScoringWeights
from src.discovery.triggers import TriggerDetector

__all__ = [
    "DiscoveryCandidate",
    "DiscoveryEngineConfig",
    "DiscoveryResult",
    "DiscoverySource",
    "MultiFactorScorer",
    "PortfolioFilterConfig",
    "PortfolioFilterEngine",
    "ScoringWeights",
    "StockDiscoveryEngine",
    "TriggerDetector",
]
