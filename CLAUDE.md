# AI Casino

Multi-agent stock trading system using technical analysis, sentiment analysis, and news analysis to generate trading decisions. Built with LangGraph for agent orchestration and LiteLLM for flexible LLM provider switching.

**Tech Stack:** Python 3.12, LangGraph, LiteLLM, pandas-ta, transformers (FinBERT), yfinance

**Purpose:** Agentic AI system combining multiple analysis methods (technical indicators, sentiment, news) to make informed trading decisions (BUY/SELL/HOLD) with confidence scoring and risk assessment.

**Status:** MVP complete (44% - 11/25 features), functional paper trading foundation pending

---

## Project Structure

### Directory Layout

```
src/
├── agents/          # Trading agents (specialized analysts + final trader)
│   ├── technical.py   # Technical analysis (RSI/MACD via pandas-ta)
│   ├── sentiment.py   # Sentiment analysis (FinBERT on news)
│   ├── news.py        # News analysis (LLM-powered interpretation)
│   └── trader.py      # Final decision maker (synthesizes all inputs)
├── data/            # Data fetchers for market and news data
│   ├── market.py      # Alpha Vantage + yfinance
│   └── news.py        # Marketaux API
├── models/          # ML models and LLM wrappers
│   ├── llm.py         # LiteLLM client (Ollama/Claude/GPT abstraction)
│   └── sentiment.py   # FinBERT wrapper for sentiment analysis
├── strategies/      # Trading strategies
│   └── momentum.py    # RSI + MACD momentum strategy
├── workflows/       # Agent orchestration (LangGraph-style)
│   └── trading.py     # Sequential workflow: data → analysis → decision
└── main.py          # CLI entry point

tests/               # Full mirror of src structure
├── conftest.py      # Shared fixtures (sample_ohlcv_data, mock_llm_client)
├── test_agents/     # Agent tests
├── test_data/       # Data fetcher tests
├── test_models/     # Model tests
├── test_strategies/ # Strategy tests
└── test_workflows/  # Workflow tests
```

### Key Modules

- **agents/** - Specialized agents for analysis (TechnicalAnalyst, SentimentAnalyst, NewsAnalyst, TraderAgent)
- **workflows/trading.py** - Sequential pipeline: fetch data → technical → sentiment → news → final decision
- **models/llm.py** - LiteLLM abstraction for provider switching (Ollama dev → Claude/GPT prod)
- **strategies/momentum.py** - RSI + MACD momentum strategy using pandas-ta

---

## Development Workflow

### Before Coding

1. ASK clarifying questions (95% confident)
2. Research existing patterns (agents/, workflows/)
3. Create plan, get approval
4. Work incrementally

### Pre-Commit (MANDATORY)

```bash
mise check  # Must pass: format, lint, test
```

Never skip/disable on failure - fix properly, re-run until clean.

### Git

**Commits:** Conventional format (feat:, fix:, chore:, docs:, test:, refactor:) with `-s -S` flags
**Branches:** `feat/description`, `fix/description`, `chore/description`
**Hooks:** Strict ruff - adjust to pass, never work around

---

## Python Conventions

### Code Style

**Formatter/Linter:** ruff (45+ rule categories)
**Line length:** 110 | **Quotes:** Double | **Docstrings:** Google style | **Type hints:** Mandatory

**Linter errors:** Fix properly (research if needed), NEVER skip/disable (`# noqa`, `# type: ignore`). If stuck after research, ASK.

### Import Organization

**Order:** stdlib → third-party (alphabetical) → local (relative)

```python
"""Module docstring - Google style."""

# Standard library
import os
from datetime import datetime
from enum import Enum

# Third-party
import pandas as pd
from loguru import logger
from pydantic import BaseModel

# Local
from src.models.llm import LLMClient
from src.strategies.momentum import Signal
```

### Type Hints

**Required on all functions:**

```python
def __init__(self, llm_client: LLMClient, strategy: MomentumStrategy) -> None:
def fetch_daily(self, symbol: str, period_days: int = 90) -> MarketData:
def analyze(self, symbol: str, articles: list[NewsArticle]) -> SentimentAnalysis:
```

**Use Python 3.10+ syntax:** `list[str]`, `dict[str, int]`, `int | None` (not `Optional[int]`)

### Docstrings (Google Style)

```python
def analyze(self, symbol: str, market_data: pd.DataFrame) -> TechnicalAnalysis:
    """Perform technical analysis on market data.

    Args:
        symbol: Stock ticker symbol
        market_data: OHLCV dataframe with required columns

    Returns:
        TechnicalAnalysis with signal, indicators, and interpretation
    """
```

Class docstrings: 1-line sufficient

### Error Handling & Logging

**Pattern:** Try-except with logger.error + re-raise (no bare excepts)

```python
try:
    response = completion(model=self._model_id, messages=messages, temperature=temperature)
    return response.choices[0].message.content
except Exception as e:
    logger.error(f"LLM completion failed: {e}")
    raise
```

**Logging (loguru):** `logger.info/warning/error/debug()` - set level via `LOG_LEVEL` env var

### Pydantic Models & Enums

```python
class TechnicalAnalysis(BaseModel):
    """Technical analysis result."""
    signal: Signal
    rsi: float
    macd_hist: float
    interpretation: str
    confidence: float

    class Config:
        arbitrary_types_allowed = True  # When using DataFrame, etc.

class Signal(str, Enum):
    """Trading signal - str enum for JSON serialization."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
```

**All classes implement `__repr__`:** `return f"TechnicalAnalyst(strategy={self.strategy})"`

### Testing (pytest)

**Fixtures:** `sample_ohlcv_data`, `sample_news_articles`, `mock_llm_client`, `mock_finbert`
**Markers:** `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`

```python
def test_technical_analyst_analyze(mock_llm_client, sample_ohlcv_data):
    analyst = TechnicalAnalyst(mock_llm_client, MomentumStrategy())
    result = analyst.analyze("AAPL", sample_ohlcv_data)

    assert isinstance(result, TechnicalAnalysis)
    assert 0.0 <= result.confidence <= 1.0
    mock_llm_client.complete.assert_called_once()
```

**Rules:** Mock all external APIs, test ranges/types, no real API integration tests

---

## Simplicity Principles

### Anti-Patterns

❌ **NEVER:** TODOs, placeholders, incomplete error handling, obvious comments, over-engineering, premature abstractions, >100 line changes, print() (except main.py), bare excepts, commented code, backwards-compat hacks, provider-specific LLM (unless justified), globals, singletons

✅ **ALWAYS:** Simplest solution, reuse existing patterns, minimal changes, complete implementations

**Before implementing:** Can this be simpler? Abstractions needed NOW? Similar code exists? Minimal change?
**If unsure:** ASK for approval.

---

## Architecture Patterns

### Dependency Injection (MANDATORY)

**All classes accept dependencies via `__init__` - no singletons, no globals:**

```python
class TechnicalAnalyst:
    def __init__(self, llm_client: LLMClient, strategy: MomentumStrategy) -> None:
        self.llm = llm_client
        self.strategy = strategy
        logger.info("Initialized TechnicalAnalyst")
```

### LLM Abstraction (LiteLLM)

**Always use LiteLLM - case-by-case for provider-specific features:**

```python
class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None) -> None:
        self.provider = provider or os.getenv("LLM_PROVIDER", "ollama")
        self.model = model or os.getenv("LLM_MODEL", "qwen3:14b")
        self._model_id = f"{self.provider}/{self.model}"  # ollama/qwen3:14b

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.7) -> str:
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        return completion(model=self._model_id, messages=messages, temperature=temperature).choices[0].message.content
```

**Providers:** Dev: Ollama qwen3:14b, Prod: Claude sonnet-4, Alt: OpenAI gpt-4o

### Agent Pattern

**All agents return Pydantic models with confidence scores:**

```python
def analyze(self, symbol: str, market_data: pd.DataFrame) -> TechnicalAnalysis:
    logger.info(f"Analyzing {symbol} technicals")
    signal, indicators = self.strategy.generate_signal(market_data)

    prompt = f"Analyze indicators for {symbol}: RSI {indicators.rsi}, MACD {indicators.macd_hist}"
    response = self.llm.complete(prompt, system="You are a technical analyst.", temperature=0.3)

    return TechnicalAnalysis(
        signal=signal, rsi=indicators.rsi, macd_hist=indicators.macd_hist,
        interpretation=response, confidence=self._calculate_confidence(signal, indicators)
    )
```

### Workflow Pattern (Sequential Pipeline)

```python
def analyze(self, symbol: str, period_days: int = 90) -> TradingWorkflowResult:
    """Pipeline: fetch → technical → sentiment → news → decision"""
    state = self._fetch_data(symbol, period_days)
    state = self._run_technical_analysis(state)
    state = self._run_sentiment_analysis(state)
    state = self._run_news_analysis(state)
    state = self._make_decision(state)
    return TradingWorkflowResult(symbol=symbol, technical=state["technical_analysis"], ...)
```

---

## Domain-Specific Rules

### Trading Signals

**Always use Signal enum - never strings:**

```python
class Signal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

# Good
signal = Signal.BUY

# Bad
signal = "BUY"
```

### Confidence and Risk

**Confidence:** 0.0-1.0 float
**Risk:** LOW (≥0.75), MEDIUM (0.5-0.75), HIGH (<0.5)

```python
def _calculate_risk_level(self, confidence: float) -> str:
    return "LOW" if confidence >= 0.75 else "MEDIUM" if confidence >= 0.5 else "HIGH"
```

### LLM Temperature

- Technical analysis: `0.3` (deterministic)
- Trading decisions: `0.5` (balanced)
- General: `0.7` (default)

### Data Fetching

Prefer Alpha Vantage, fallback to yfinance, raise on empty data:

```python
def fetch_daily(self, symbol: str, period_days: int = 90) -> MarketData:
    try:
        data = self._fetch_alpha_vantage(symbol, period_days)
    except Exception as e:
        logger.warning(f"Alpha Vantage failed: {e}, falling back to yfinance")
        data = self._fetch_yfinance(symbol, period_days)
    if data.empty:
        raise ValueError(f"No market data available for {symbol}")
    return MarketData(symbol=symbol, data=data, last_updated=datetime.now())
```

### Technical Indicators (pandas-ta)

```python
def calculate_indicators(self, df: pd.DataFrame) -> IndicatorData:
    df.ta.rsi(length=14, append=True)  # RSI
    df.ta.macd(fast=12, slow=26, signal=9, append=True)  # MACD
    return IndicatorData(rsi=df["RSI_14"].iloc[-1], macd_hist=df["MACDh_12_26_9"].iloc[-1])
```

---

## Common Commands

### Development

```bash
# Install dependencies (using uv package manager)
uv sync --frozen --all-extras

# Run analysis (CLI entry point)
python -m src.main AAPL
python -m src.main TSLA --period 180

# Quality checks (run before every commit)
mise check              # All checks: format + lint + test

# Individual checks
mise format             # Format code with ruff
mise format:check       # Check formatting (CI mode)
mise lint               # Run ruff linter
mise test               # Run pytest
mise test:cov           # Run with coverage report

# Ollama management (for local LLM dev)
mise ollama:start       # Start Ollama server in background
mise ollama:stop        # Stop Ollama server
mise ollama:status      # Check if Ollama running

# Activate virtual environment (if needed)
source .venv/bin/activate
```

### Configuration (.env)

Required: `ALPHA_VANTAGE_API_KEY`
Optional: `MARKETAUX_API_KEY`, `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`
LLM: `LLM_PROVIDER` (ollama|anthropic|openai), `LLM_MODEL`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`
Logging: `LOG_LEVEL` (DEBUG|INFO|WARNING|ERROR)

---

## Project-Specific Context

### Domain Knowledge

**Agents:** Technical (RSI/MACD), Sentiment (FinBERT), News (LLM), Trader (synthesizer)

**Indicators:**
- RSI: 0-100, oversold <30, overbought >70
- MACD: Histogram >0 bullish, <0 bearish

**Workflow:** fetch data → technical → sentiment → news → decision

**State:** TypedDict with symbol, market_data, news_articles, *_analysis fields

### Gotchas

- Alpha Vantage: 5 req/min free tier (cache in data/cache/)
- FinBERT: 440MB download first run
- Ollama: Must run locally for dev (`mise ollama:start`)
- Empty news: Handle with warning, not error
- MACD: Needs ~35 data points minimum

### Integration Points

- **Alpha Vantage:** Market data (ALPHA_VANTAGE_API_KEY required, 5 req/min free)
- **Marketaux:** News (MARKETAUX_API_KEY optional, 100 req/day)
- **Ollama:** Local LLM (http://localhost:11434, qwen3:14b recommended, `mise ollama:start`)
- **LiteLLM:** Unified API (Ollama/Anthropic/OpenAI via env vars)

---

## Additional Resources

- **Implementation Plan:** ./implem-plan.md (25 features, MVP roadmap)
- **Research:** ./agentic-stock-trading-system-research.md (architecture deep dive)
- **Dependencies:** pyproject.toml (pinned versions via renovate)
- **Linting Config:** ruff.toml (45+ rule categories, complexity limits)
- **CI Workflows:** .github/workflows/ci.yml (lint, test, yamllint, actionlint)
