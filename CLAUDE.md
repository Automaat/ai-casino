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

## UI/UX Patterns

### Empty States Must Distinguish Feature Status

**Pattern:** Empty UI sections must show why they're empty - distinguish "disabled" vs "no data yet".

**Bad:**

```svelte
{:else}
  <div>No data available</div>
{/if}
```

**Good:**

```svelte
{:else if !response?.enabled}
  <div>
    <div class="font-medium">Feature disabled</div>
    <div class="text-sm">Enable in config: <code>feature.enabled: true</code></div>
  </div>
{:else}
  <div>
    <div class="font-medium">No data yet</div>
    <div class="text-sm">Waiting for first run</div>
  </div>
{/if}
```

**Implementation:**

1. Add status fields to API response model (`enabled: bool`, `database_enabled: bool`, `has_data: bool`)
2. Update endpoint to populate status from config
3. Update frontend types to match
4. Update UI to show contextual messages

**Example:** Portfolio snapshots/rebalancing endpoints

---

## Development Workflow

### Before Coding

1. ASK clarifying questions (95% confident)
2. Research existing patterns (agents/, workflows/)
3. Create plan, get approval
4. Work incrementally

### Configuration Changes

**CRITICAL: YAML-only configuration**

- **NEVER** use environment variables for config in this project
- **ALWAYS** configure via `~/.ai-casino/daemon-production.yaml`
- Environment variables only allowed as fallbacks in DI provider functions (`resolve_config_or_env`)
- docker-compose.yml must NOT set config via environment variables

**When adding new daemon config:**

1. Add config model to `src/daemon/config/{module}.py`
2. Add field to `DaemonConfig` with `Field(default_factory=...)`
3. Update `DaemonConfig.from_yaml()` to extract and pass the data
4. **MANDATORY: Update `docs/daemon.yaml.example`** with comprehensive documentation
   - Add section with all fields commented out
   - Include inline comments explaining each field
   - Document valid ranges, defaults, and examples
   - Keep example file comprehensive - users rely on it for discovery
5. Update DI provider in `src/di/providers/` to read from `daemon_config.{section}`

### Pre-Commit (MANDATORY)

```bash
mise check  # Must pass: format, lint, typecheck, test
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

**Formatter:** ruff (fast formatter)
**Linter:** ruff (45+ rule categories)
**Type Checker:** pyrefly (high-performance type checker, faster than mypy/pyright)
**Line length:** 110 | **Quotes:** Double | **Docstrings:** Google style | **Type hints:** Mandatory
**File length:** Max 400 lines per file - split into logical modules if exceeded
**Method length:** Max 60 lines per method/function - extract helper methods if exceeded

**Linter/type errors:** Fix properly (research if needed), NEVER skip/disable (`# noqa`, `# type: ignore`). If stuck after research, ASK.

**When `# type: ignore` is acceptable:**
- Third-party library missing type stubs (use `# type: ignore[import-untyped]`)
- Complex generic patterns pyrefly can't infer (add comment explaining why)
- Interfacing with untyped external APIs (prefer typed wrapper when possible)

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

**Syntax:** Python 3.10+ - `list[str]`, `dict[str, int]`, `int | None` (not `Optional[int]`)

**Best Practices:**
- Type all function parameters and return values (no `Any` unless truly dynamic)
- Use `TypedDict` for structured dicts with known keys
- Prefer concrete types over `Any`: `object` for truly unknown, protocol types for duck-typed interfaces
- Use `collections.abc` types for parameters (`Sequence`, `Mapping`) for broader compatibility
- Annotate class attributes in `__init__` or at class level
- Use `Final` for constants: `TIMEOUT: Final[int] = 30`
- Use string annotations for forward references: `def process(self, result: "TradingWorkflowResult") -> None:`

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

**Critical vs Non-Critical:**
- **Critical** (always propagate): data fetchers, LLM calls, broker API, database writes, user-facing operations
- **Non-Critical** (may swallow): batch processing (screening 500+ stocks, optimization 100+ trials), cache, metrics

**Swallowing Exceptions (Non-Critical Only):**
```python
# ALWAYS use logger.opt(exception=True) when swallowing for traceback
except ValueError as e:
    logger.opt(exception=True).warning(f"Invalid data, skipping: {e}")
    return None

# NOT this (missing context):
except Exception as e:
    logger.warning(f"Failed: {e}")  # ❌ No traceback
    return None
```

**Specific Exceptions First:**
```python
# Hierarchical exception handling (specific → general)
except HTTPStatusError as e:
    logger.error(f"HTTP {e.response.status_code}: {url}")
    raise
except HTTPError as e:
    logger.error(f"Network error: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise
```

**Never:**
- Bare `except Exception: return None` without logging
- `except Exception` in critical paths (use specific exceptions)
- Warning-level logs without `logger.opt(exception=True)` when swallowing exception

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

### Async & Concurrency

**Async-first API:** All I/O-bound methods must be `async`. Sync wrappers (`asyncio.run()`) only at CLI entry points — never inside async context.

**Blocking I/O offloading (MANDATORY):**
- Network/disk/ML inference → `await asyncio.to_thread(blocking_fn, *args)`
- CPU-heavy (FinBERT) → `await loop.run_in_executor(None, fn, *args)`
- Never call blocking functions directly in async code (freezes event loop)

```python
# ✅ GOOD
daily = await asyncio.to_thread(self._market_fetcher.fetch_daily, symbol, 30)
scores = await loop.run_in_executor(None, self.finbert.analyze_batch, texts)

# ❌ BAD - blocks event loop
daily = self._market_fetcher.fetch_daily(symbol, 30)
```

**Concurrency control:**
- `asyncio.Semaphore` for rate limiting (LLM calls, API requests)
- `asyncio.Lock` for shared async state
- `threading.Lock` only for thread-shared state (cache, model access)
- Never use `threading.Lock` in async code — use `asyncio.Lock`

**Parallel execution:** `asyncio.gather(*tasks, return_exceptions=True)` — always handle exceptions:

```python
results = await asyncio.gather(*tasks, return_exceptions=True)
for result in results:
    if isinstance(result, (asyncio.CancelledError, KeyboardInterrupt)):
        raise result
    if isinstance(result, Exception):
        logger.error(f"Task failed: {result}")
```

**HTTP clients:**
- Reuse clients for connection pooling (don't create per-request)
- `async with httpx.AsyncClient()` for short-lived scopes
- Store as instance attribute for long-lived services

**Anti-patterns:**
- ❌ `nest_asyncio` — archived, breaks cancellation/exceptions
- ❌ `asyncio.run()` inside async functions — RuntimeError
- ❌ Blocking calls (`requests.get`, `time.sleep`, `open().read()`) in async
- ❌ Fire-and-forget `asyncio.create_task()` without error handling
- ❌ Unbounded `asyncio.gather()` — use semaphore for backpressure

---

## Simplicity Principles

### Anti-Patterns

❌ **NEVER:** TODOs, placeholders, incomplete error handling, obvious comments, over-engineering, premature abstractions, >100 line changes, >400 line files, >60 line methods, print() (except main.py), bare excepts, commented code, backwards-compat hacks, provider-specific LLM (unless justified), globals, singletons, dicts/kwargs for structured data, inheritance hierarchies (prefer composition)

✅ **ALWAYS:** Simplest solution, reuse existing patterns, minimal changes, complete implementations, typed classes over dicts, split files >400 lines into logical modules, extract helper methods for >60 line functions, composition over inheritance, extract proper encapsulated abstractions

**Before implementing:** Can this be simpler? Abstractions needed NOW? Similar code exists? Minimal change? File too large (>400 lines)? Method too long (>60 lines)?
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

### Dict vs Typed Class Decision Matrix

**ALWAYS create typed classes (Pydantic/dataclasses) instead of dicts/kwargs for structured data.**

#### ✅ When Dict/Kwargs is Acceptable

1. **Framework Requirements**
   - LLM tool execution: `execute(**kwargs)` - Required for function calling interface
   - Decorators: `wrapper(*args, **kwargs)` with ParamSpec - Preserves signatures
   - UI frameworks: `super().__init__(**kwargs)` - Framework inheritance requirement

2. **Dynamic Key-Value Storage**
   - Caches: `_cache: dict[str, tuple[datetime, Any]]` - Truly dynamic keys
   - Registries: `_tools: dict[str, BaseTool]` - Runtime registration
   - Lookups: `STRATEGY_MAP: dict[MarketRegime, str]` - Enum mappings

3. **Template Interpolation**
   - Prompt loading: `load("user", **variables)` - f-string format() requires kwargs
   - Must be typed: `**kwargs: str | int | float | list | dict` (not `Any`)

4. **External API Intermediate**
   - Alpha Vantage raw response: Parse immediately into typed model
   - Keep dict only during validation/parsing step

#### ❌ When to Use Typed Class

1. **Function Return Values**
   ```python
   # ❌ Bad
   def get_earnings_flags(...) -> dict:
       return {"upcoming_earnings": True, "days_until_earnings": 5}

   # ✅ Good
   class EarningsFlags(BaseModel):
       upcoming_earnings: bool
       days_until_earnings: int | None

   def get_earnings_flags(...) -> EarningsFlags:
       return EarningsFlags(upcoming_earnings=True, days_until_earnings=5)
   ```

2. **Function Parameters**
   ```python
   # ❌ Bad
   def analyze(self, metrics: dict[str, float | None]) -> Analysis:

   # ✅ Good
   class FundamentalMetrics(BaseModel):
       pe_ratio: float | None = None
       eps: float | None = None

   def analyze(self, metrics: FundamentalMetrics) -> Analysis:
   ```

3. **State Objects**
   ```python
   # ❌ Bad
   state = {**state, "final_decision": decision}

   # ✅ Good
   class DecisionState(BaseModel):
       final_decision: TradingDecision

   state = DecisionState(final_decision=decision)
   ```

4. **Configuration**
   ```python
   # ❌ Bad
   kwargs = {"universe": "sp500", "top_n": 10}
   screen(**kwargs)

   # ✅ Good
   class ScreeningArgs(BaseModel):
       universe: str
       top_n: int

   args = ScreeningArgs(universe="sp500", top_n=10)
   screen(args)
   ```

5. **API Schemas**
   ```python
   # ❌ Bad
   def get_tool_definition(self) -> dict:
       return {"type": "function", "function": {...}}

   # ✅ Good
   class ToolDefinition(BaseModel):
       type: str = "function"
       function: ToolFunction

   def get_tool_definition(self) -> ToolDefinition:
       return ToolDefinition(function=ToolFunction(...))
   ```

#### Model Creation Checklist

When creating a new Pydantic model:

1. ✅ Use descriptive name: `{Component}{Purpose}` (e.g., `EarningsFlags`, `FundamentalMetrics`)
2. ✅ Add docstring: One-line description
3. ✅ Use `Field()` for validation: `Field(ge=0.0, le=1.0, description="...")`
4. ✅ Use `Field(default_factory=list)` for mutable defaults
5. ✅ Add `class Config: arbitrary_types_allowed = True` if using DataFrame/datetime
6. ✅ Use `| None` for optional fields (not `Optional[T]`)
7. ✅ Use StrEnum for fixed string values
8. ✅ Add `@property` for computed fields (not extra model fields)
9. ✅ Implement `__repr__()` for debugging

Example:
```python
class TechnicalMetrics(BaseModel):
    """Technical analysis metrics."""

    rsi: float = Field(ge=0.0, le=100.0, description="RSI indicator")
    macd_hist: float
    interpretation: str
    confidence: float = Field(ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    @property
    def is_oversold(self) -> bool:
        """Check if RSI indicates oversold."""
        return self.rsi < 30.0

    def __repr__(self) -> str:
        return f"TechnicalMetrics(rsi={self.rsi:.1f}, confidence={self.confidence:.2f})"
```

#### Migration Pattern

When refactoring dict → typed class:

1. Create model in appropriate location (`src/{module}/models.py`)
2. Update function signature: `-> dict` → `-> ModelName`
3. Replace dict construction: `return {...}` → `return ModelName(...)`
4. Update consumers: Access via attributes not keys
5. Add `.model_dump()` at API boundaries if needed temporarily
6. Run `mise typecheck` to verify

---

## Architecture Patterns

### Dependency Injection (MANDATORY)

**All classes accept dependencies via `__init__` - no singletons, no globals. ALWAYS use the DI container (`src/di/container.py`) for dependency resolution.**

**Basic DI pattern:**
```python
class TechnicalAnalyst:
    def __init__(self, llm_client: LLMClient, strategy: MomentumStrategy) -> None:
        self.llm = llm_client
        self.strategy = strategy
        logger.info("Initialized TechnicalAnalyst")
```

**DI Container Usage:**

The project uses `dependency-injector` for centralized dependency management. The container is defined in `src/di/container.py` and provides:
- **Singleton** providers for stateful services (cache, database, API clients)
- **Factory** providers for per-request instances (workflows, agents)

**Creating dependencies from container:**
```python
from src.di.container import create_container

# Create container (optionally with config path)
container = create_container(config_path="~/.ai-casino/daemon-production.yaml")

# Get singleton instances (shared across app)
llm_client = container.llm_client()
market_fetcher = container.market_fetcher()
finnhub_fetcher = container.finnhub_fetcher()

# Create workflow instances (new instance each time)
workflow = container.workflow_meta(
    broker=broker,
    metrics_tracker=tracker,
    container=container,  # IMPORTANT: Explicitly pass container to factories
)
```

**CRITICAL: Factory providers and `providers.Self()`**

`providers.Self()` does NOT work reliably with Factory providers. It evaluates to `None` instead of the container instance.

```python
# ❌ BAD - providers.Self() doesn't work with Factory
workflow_meta = providers.Factory(
    create_workflow_meta,
    container=providers.Self(),  # This will be None!
)

# ✅ GOOD - pass container explicitly when calling factory
workflow_meta = providers.Factory(
    create_workflow_meta,
    # Don't include container in factory definition
)

# Then in usage:
workflow = container.workflow_meta(
    broker=broker,
    container=container,  # Explicitly pass here
)
```

**Adding new providers:**

When adding new services to the container:

1. **Singleton for stateful services:**
```python
# In src/di/container.py
new_service = providers.Singleton(
    create_new_service,
    dependency1=other_provider,
    daemon_config=daemon_config,
)
```

2. **Factory for per-request instances:**
```python
# In src/di/container.py
new_workflow = providers.Factory(
    create_new_workflow,
    llm_client=llm_client,
    # Don't include container=providers.Self() - won't work!
)
```

3. **Create provider function in `src/di/providers/`:**
```python
# In src/di/providers/data.py (or appropriate module)
def create_new_service(daemon_config: DaemonConfig) -> NewService:
    """Create NewService with resolved config."""
    api_key = resolve_config_or_env(
        daemon_config.api_keys.new_service_api_key,
        "NEW_SERVICE_API_KEY",
    )
    return NewService(api_key=api_key)
```

4. **Pass container explicitly when needed:**
```python
# In code that uses the factory
instance = container.new_workflow(
    param1=value1,
    container=container,  # Explicitly pass container
)
```

**Best practices:**
- NEVER create service instances directly (e.g., `FinnhubFetcher()`) - always use container
- NEVER use `providers.Self()` with Factory providers
- Always pass `container` parameter explicitly when calling factories
- For optional dependencies, check container first: `service = container.service() if container else None`
- Add fallback only as last resort: `service = param or (container.service() if container else None) or Service()`

### Composition over Inheritance (MANDATORY)

**ALWAYS prefer composition over inheritance. Extract proper encapsulated abstractions and compose them together.**

**Core principles:**
- Favor "has-a" relationships over "is-a" relationships
- Extract single-responsibility components that can be composed
- Each abstraction should be independently testable and reusable
- Compose abstractions via dependency injection

**Why composition:**
- **Flexibility:** Change behavior at runtime by swapping components
- **Testability:** Mock individual components independently
- **Maintainability:** Changes to one component don't cascade through inheritance hierarchy
- **Clarity:** Explicit dependencies make code relationships obvious
- **Reusability:** Components can be used in different contexts without inheritance constraints

**Pattern:**

```python
# ❌ BAD - inheritance hierarchy
class BaseAnalyst:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client

    def _format_result(self, data: dict) -> str:
        return json.dumps(data, indent=2)

class TechnicalAnalyst(BaseAnalyst):
    def analyze(self, symbol: str, market_data: pd.DataFrame) -> TechnicalAnalysis:
        result = self._run_analysis(market_data)
        formatted = self._format_result(result)  # Inherited method
        return TechnicalAnalysis(...)

class SentimentAnalyst(BaseAnalyst):
    def analyze(self, articles: list[NewsArticle]) -> SentimentAnalysis:
        result = self._run_sentiment(articles)
        formatted = self._format_result(result)  # Inherited method
        return SentimentAnalysis(...)

# ✅ GOOD - composition with extracted abstractions
class ResultFormatter:
    """Encapsulated formatting abstraction."""
    def format(self, data: dict) -> str:
        return json.dumps(data, indent=2)

class TechnicalAnalyst:
    def __init__(self, llm_client: LLMClient, formatter: ResultFormatter) -> None:
        self.llm = llm_client
        self.formatter = formatter  # Composed dependency

    def analyze(self, symbol: str, market_data: pd.DataFrame) -> TechnicalAnalysis:
        result = self._run_analysis(market_data)
        formatted = self.formatter.format(result)  # Composed behavior
        return TechnicalAnalysis(...)

class SentimentAnalyst:
    def __init__(self, llm_client: LLMClient, formatter: ResultFormatter) -> None:
        self.llm = llm_client
        self.formatter = formatter  # Composed dependency

    def analyze(self, articles: list[NewsArticle]) -> SentimentAnalysis:
        result = self._run_sentiment(articles)
        formatted = self.formatter.format(result)  # Composed behavior
        return SentimentAnalysis(...)
```

**When to extract abstractions:**

1. **Shared behavior across multiple classes** → Extract to composable component
2. **Complex logic that can be isolated** → Extract to single-responsibility class
3. **Behavior that might change independently** → Extract to swappable component
4. **Logic with its own dependencies** → Extract to injected component

**Example: Extract validation logic**

```python
# ❌ BAD - validation mixed in class
class OrderExecutor:
    def execute(self, order: Order) -> ExecutionResult:
        # Validation logic embedded
        if order.quantity <= 0:
            raise ValueError("Invalid quantity")
        if order.price <= 0:
            raise ValueError("Invalid price")
        if not order.symbol:
            raise ValueError("Missing symbol")

        # Execution logic
        return self._submit_order(order)

# ✅ GOOD - extracted validation abstraction
class OrderValidator:
    """Encapsulated validation abstraction."""
    def validate(self, order: Order) -> None:
        if order.quantity <= 0:
            raise ValueError("Invalid quantity")
        if order.price <= 0:
            raise ValueError("Invalid price")
        if not order.symbol:
            raise ValueError("Missing symbol")

class OrderExecutor:
    def __init__(self, validator: OrderValidator, broker: Broker) -> None:
        self.validator = validator  # Composed validation
        self.broker = broker

    def execute(self, order: Order) -> ExecutionResult:
        self.validator.validate(order)  # Delegated validation
        return self._submit_order(order)  # Focused execution
```

**Benefits in testing:**

```python
# Easy to test with composition
def test_order_executor_with_valid_order():
    mock_validator = Mock(spec=OrderValidator)
    mock_broker = Mock(spec=Broker)
    executor = OrderExecutor(mock_validator, mock_broker)

    result = executor.execute(order)

    mock_validator.validate.assert_called_once_with(order)
    assert result.success

# Validator is independently testable
def test_order_validator_rejects_negative_quantity():
    validator = OrderValidator()
    invalid_order = Order(quantity=-10, price=100, symbol="AAPL")

    with pytest.raises(ValueError, match="Invalid quantity"):
        validator.validate(invalid_order)
```

**Exceptions:**

Inheritance is acceptable ONLY for:
- Protocol/ABC definitions (interfaces)
- Pydantic BaseModel subclasses (data models)
- Framework-required inheritance (pytest fixtures, Django models)
- Enum subclasses

**Never use inheritance for code reuse** - always extract and compose instead.

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
mise check              # All checks: format + lint + typecheck + test

# Individual checks
mise format             # Format code with ruff
mise format:check       # Check formatting (CI mode)
mise lint               # Run ruff linter
mise typecheck          # Run pyrefly type checker
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

**Primary config:** `~/.ai-casino/daemon-production.yaml` - YAML-only configuration (see `docs/daemon.yaml.example`)

**Configuration hierarchy:**

1. **daemon-production.yaml** (only source) - all config must be here
2. DI container resolves config
3. Environment variables as fallback ONLY in DI providers (via `resolve_config_or_env`)

**Environment variables (fallback only, not primary config):**

- API keys: `ALPHA_VANTAGE_API_KEY`, `MARKETAUX_API_KEY`, `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`
- Runtime: `LOG_LEVEL` (DEBUG|INFO|WARNING|ERROR)

**NEVER configure via:**

- ❌ docker-compose.yml environment section (except `LOG_LEVEL`)
- ❌ `.env` files for daemon config
- ❌ Command-line arguments for config

**ALWAYS configure via:**

- ✅ `~/.ai-casino/daemon-production.yaml`
- ✅ DI container reads from `daemon_config`
- ✅ `resolve_config_or_env()` for API keys with env fallback

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
