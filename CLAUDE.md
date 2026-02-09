# AI Casino

Multi-agent stock trading system using technical analysis, sentiment analysis, and news analysis to generate trading decisions. Built with LangGraph for agent orchestration and custom provider abstraction for flexible LLM switching.

**Tech Stack:** Python 3.12, LangGraph, pandas-ta, transformers (FinBERT), yfinance, Anthropic SDK, OpenAI SDK

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
│   ├── llm.py         # LLM client facade (provider abstraction)
│   ├── providers/     # Provider implementations (Anthropic, OpenAI, Ollama)
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
- **models/llm.py** - LLM client facade with custom provider abstraction (Ollama dev → Claude/GPT prod)
- **models/providers/** - Provider implementations using native SDKs (AnthropicProvider, OpenAIProvider, OllamaProvider)
- **strategies/momentum.py** - RSI + MACD momentum strategy using pandas-ta

---

## Development Workflow

### Before Coding

1. ASK clarifying questions (95% confident)
2. Research existing patterns (agents/, workflows/)
3. Create plan, get approval
4. Work incrementally

### Configuration Changes

**When adding new daemon config:**
1. Add config model to `src/daemon/config.py`
2. Add field to `DaemonConfig` with `Field(default_factory=...)`
3. Update `DaemonConfig.from_yaml()` to extract and pass the data
4. **MANDATORY: Update `docs/daemon.yaml.example`** with comprehensive documentation
   - Add section with all fields commented out
   - Include inline comments explaining each field
   - Document valid ranges, defaults, and examples
   - Keep example file comprehensive - users rely on it for discovery

### Pre-Commit (MANDATORY)

```bash
mise check  # Must pass: format, lint, test
mise audit  # Check for CVEs (optional locally, enforced in CI)
```

Never skip/disable on failure - fix properly, re-run until clean.

### Git

**Commits:** Conventional format (feat:, fix:, chore:, docs:, test:, refactor:) with `-s -S` flags
**Branches:** `feat/description`, `fix/description`, `chore/description`
**Hooks:** Strict ruff - adjust to pass, never work around

### Pull Requests

**Format:**
```markdown
## Motivation
<!-- Why are we doing this change -->

## Implementation information
<!-- Explain how this was done and potentially alternatives considered and discarded -->

## Supporting documentation
Fixes #<issue-number>
<!-- Include MADR or related PRs if applicable -->
```

**ALWAYS link an issue** - create one if needed before PR

---

## Python Conventions

### Code Style

**Formatter/Linter:** ruff (45+ rule categories)
**Line length:** 110 | **Quotes:** Double | **Docstrings:** Google style | **Type hints:** Mandatory

**Linter errors:** Fix properly (research if needed), NEVER skip/disable (`# noqa`, `# type: ignore`). If stuck after research, ASK.

### Import Organization

**Order:** stdlib → third-party (alphabetical) → local (relative)

**TYPE_CHECKING blocks:** Generally avoid for clarity. **Exception:** Use for deferring expensive imports (torch, transformers, large models) when only type hints needed. Import actual classes for runtime usage.

Example:
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.workflows.types import TradingWorkflowResult  # Heavy type

def process(result: "TradingWorkflowResult") -> None:  # String annotation
```

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

❌ **NEVER:** TODOs, placeholders, incomplete error handling, obvious comments, over-engineering, premature abstractions, >100 line changes, print() (except main.py), bare excepts, commented code, backwards-compat hacks, provider-specific LLM (unless justified), globals, singletons, dicts/kwargs for structured data

✅ **ALWAYS:** Simplest solution, reuse existing patterns, minimal changes, complete implementations, typed classes over dicts

**Before implementing:** Can this be simpler? Abstractions needed NOW? Similar code exists? Minimal change?
**If unsure:** ASK for approval.

### Types vs Dicts

**ALWAYS create typed classes (Pydantic/dataclasses) instead of dicts/kwargs for structured data:**

```python
# ❌ BAD - dict and kwargs
def analyze(self, **kwargs: Any) -> dict[str, Any]:
    symbol = kwargs.get("symbol")
    data = kwargs.get("data")
    return {"signal": "BUY", "confidence": 0.8}

# ✅ GOOD - typed classes
class AnalysisRequest(BaseModel):
    symbol: str
    data: pd.DataFrame

class AnalysisResult(BaseModel):
    signal: Signal
    confidence: float

def analyze(self, request: AnalysisRequest) -> AnalysisResult:
    return AnalysisResult(signal=Signal.BUY, confidence=0.8)
```

**Why:** Type safety, IDE autocomplete, validation, self-documenting code, catches errors at definition time

**Exceptions:** Only use dicts for truly dynamic key-value stores (e.g., JSON from external API that you immediately parse into types)

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

### LLM Abstraction (Custom Provider Pattern)

**Architecture:** Custom provider abstraction using native SDKs (Anthropic, OpenAI) and direct HTTP for Ollama.

**Provider implementations:**
- `BaseLLMProvider`: Abstract interface defining `acomplete()`, `astream()`, `astructured()`, `acomplete_with_tools()`
- `AnthropicProvider`: Uses `AsyncAnthropic` from `anthropic` package
- `OpenAIProvider`: Uses OpenAI SDK with native client
- `OllamaProvider`: Direct HTTP client for local inference

**LLMClient facade:**

```python
class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None) -> None:
        self.provider = provider or os.getenv("LLM_PROVIDER", "ollama")
        self.model = model or os.getenv("LLM_MODEL", "qwen3:14b")
        self._provider: BaseLLMProvider = self._create_provider()  # Factory pattern

    def _create_provider(self) -> BaseLLMProvider:
        if self.provider == "ollama":
            return OllamaProvider(model=self.model, base_url=self.base_url)
        if self.provider == "anthropic":
            return AnthropicProvider(model=self.model, api_key=self._api_key)
        if self.provider == "openai":
            return OpenAIProvider(model=self.model, api_key=self._api_key)

    async def acomplete(self, prompt: str, system: str | None = None, temperature: float = 0.7) -> str:
        messages = self._build_messages(prompt, system)
        async with _get_semaphore():  # Concurrency control
            return await self._provider.acomplete(messages, temperature)
```

**Providers:** Dev: Ollama qwen3:14b, Prod: Claude sonnet-4, Alt: OpenAI gpt-4o
**Features:** Structured output (Pydantic), tool calling, streaming, retry logic, concurrency limits

### Agent Pattern

**All agents use structured output with `*LLMResponse` models:**

```python
# 1. Define LLM response model (what LLM returns)
class TechnicalLLMResponse(BaseModel):
    """LLM response for technical analysis."""
    interpretation: str = Field(description="Technical analysis interpretation")
    confidence_keywords: list[str] = Field(description="Confidence indicators")

# 2. Define agent output model (final result)
class TechnicalAnalysis(BaseModel):
    """Technical analysis result."""
    signal: Signal
    rsi: float | None
    interpretation: str
    confidence: float

# 3. Use structured output with fallback
async def analyze(self, symbol: str, market_data: pd.DataFrame) -> TechnicalAnalysis:
    prompt = self._prompts.load("user", symbol=symbol, ...)
    system = self._prompts.load("system")

    try:
        llm_response = await self.llm.astructured(prompt, TechnicalLLMResponse, system=system)
        interpretation = llm_response.interpretation
    except StructuredOutputError as e:
        logger.warning(f"Structured output failed, falling back: {e}")
        interpretation = await self.llm.acomplete(prompt, system=system)

    return TechnicalAnalysis(signal=signal, interpretation=interpretation, ...)
```

**Naming:** `{Agent}LLMResponse` (e.g., `NewsLLMResponse`, `TraderLLMResponse`)

### Prompt Management

**All prompts externalized to `src/prompts/{agent_name}/`:**

```
src/prompts/
├── technical/
│   ├── system_momentum.txt
│   └── user_momentum.txt
├── news/
│   ├── system.txt
│   └── user.txt
└── trader/
    ├── system.txt
    └── user_base.txt
```

**Load via PromptLoader:**

```python
from src.prompts import PromptLoader

class NewsAnalyst:
    def __init__(self, llm_client: LLMClient) -> None:
        self._prompts = PromptLoader("news")

    async def analyze(self, symbol: str, articles: list[NewsArticle]) -> NewsAnalysis:
        prompt = self._prompts.load("user", symbol=symbol, headlines_text=text)
        system = self._prompts.load("system")
```

**Never hardcode prompts in agent code** - always use PromptLoader

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

### Trading Sessions

**Session types (StrEnum):**
- `REGULAR`: 9:30 AM - 4:00 PM ET (standard market hours)
- `PRE_MARKET`: 4:00 AM - 9:30 AM ET (optional, config-enabled)

**Daemon config:**
```yaml
daemon:
  schedule:
    enable_pre_market: true  # Default: false
```

**Behavior:**
- Same analysis pipeline both sessions
- No automatic confidence adjustment (data quality varies naturally)
- Session flagged in `TradingWorkflowResult.trading_session` and state
- UI shows `(PRE-MARKET)` badge in daemon logs
- Same interval as regular hours

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
mise audit              # Check dependencies for known CVEs (pip-audit)

# Ollama management (for local LLM dev)
mise ollama:start       # Start Ollama server in background
mise ollama:stop        # Stop Ollama server
mise ollama:status      # Check if Ollama running

# Activate virtual environment (if needed)
source .venv/bin/activate
```

### Configuration

**Daemon config:** `docs/daemon.yaml.example` - comprehensive example with all config sections
**Env vars (.env):**
- Required: `ALPHA_VANTAGE_API_KEY`
- Optional: `MARKETAUX_API_KEY`, `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`
- LLM: `LLM_PROVIDER` (ollama|anthropic|openai), `LLM_MODEL`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `LLM_MAX_CONCURRENT` (1-20, default 5)
- Logging: `LOG_LEVEL` (DEBUG|INFO|WARNING|ERROR)

**Note:** Daemon config values take priority over env vars. See `docs/daemon.yaml.example` for all options.

### TUI Logs

TUI worker logs: `~/.ai-casino/worker.log` (debug level, includes LLM errors)
Chat history: `~/.ai-casino/chat-history.json` (last 100 messages)

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
- **Transformers logging:** Must suppress BEFORE import cascade. Set env vars at CLI entry point (`src/cli/app.py`) and call `hf_logging.set_verbosity_error()` in modules that import transformers. Env vars alone don't catch all output.
- **OpenAI structured output:** Requires `additionalProperties: false` recursively in JSON schema for strict mode

### Integration Points

- **Alpha Vantage:** Market data (ALPHA_VANTAGE_API_KEY required, 5 req/min free)
- **Marketaux:** News (MARKETAUX_API_KEY optional, 100 req/day)
- **Ollama:** Local LLM (http://localhost:11434, qwen3:14b recommended, `mise ollama:start`)
- **Anthropic/OpenAI SDKs:** Cloud LLM providers via native SDK clients (ANTHROPIC_API_KEY/OPENAI_API_KEY)

---

## Additional Resources

- **Implementation Plan:** ./implem-plan.md (25 features, MVP roadmap)
- **Research:** ./agentic-stock-trading-system-research.md (architecture deep dive)
- **Dependencies:** pyproject.toml (pinned versions via renovate)
- **Linting Config:** ruff.toml (45+ rule categories, complexity limits)
- **CI Workflows:** .github/workflows/ci.yml (lint, test, yamllint, actionlint)
