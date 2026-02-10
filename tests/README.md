# Testing Guide

Comprehensive guide for writing and maintaining tests in the AI Casino project using the DI container pattern.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Container Fundamentals](#container-fundamentals)
4. [Common Patterns](#common-patterns)
5. [Override Patterns](#override-patterns)
6. [Test Isolation](#test-isolation)
7. [Migration Guide](#migration-guide)
8. [Async Testing](#async-testing)
9. [Troubleshooting](#troubleshooting)
10. [Quick Reference](#quick-reference)

---

## Overview

### What Changed

The test suite migrated from fixture-based mocking to DI container pattern (issue #315). This aligns with the production codebase migration from `nest_asyncio` to proper dependency injection.

**Before:**
```python
def test_agent(mock_llm_client, mock_finbert):
    agent = NewsAnalyst(mock_llm_client)
    result = agent.analyze("AAPL", articles)
```

**After:**
```python
def test_agent(test_container):
    agent = test_container.news_analyst()
    result = agent.analyze("AAPL", articles)
```

### Why

- **Single source of truth**: Container wiring matches production
- **Better isolation**: Container per test prevents state leakage
- **Cleaner architecture**: Eliminates ~500 lines of fixture code
- **Production parity**: Tests use same DI wiring as production code

### Scope

- ✅ **Migrated**: Agent tests (262), Tool tests (117), Workflow tests (22) = 401 tests
- ✅ **No migration needed**: Daemon tests (447) - use local fixtures
- ❌ **Kept as-is**: Data fetcher tests (~100) - patch-based appropriate for API mocking

---

## Quick Start

### Basic Test Structure

```python
def test_agent_basic(test_container):
    """Test with minimal container overrides."""
    agent = test_container.news_analyst()
    result = agent.analyze("AAPL", articles)

    assert result.key_themes
    assert 0.0 <= result.sentiment_score <= 1.0
```

### Test with Custom Override

```python
def test_agent_custom_llm(test_container):
    """Test with custom LLM mock."""
    from unittest.mock import AsyncMock

    mock_llm = AsyncMock()
    mock_llm.acomplete.return_value = "Custom response"
    test_container.llm_client.override(
        providers.Factory(lambda: mock_llm)
    )

    agent = test_container.news_analyst()
    result = agent.analyze("AAPL", articles)

    assert "Custom response" in result.interpretation
```

### Available Fixtures

```python
# Minimal overrides - for agent/component tests
test_container(tmp_path)
# Overrides: llm_client, finbert_sentiment
# Real: all fetchers

# Full overrides - for integration tests
test_container_full(tmp_path)
# Overrides: llm_client, finbert_sentiment, all fetchers, all tools

# Agent-focused - for agent integration tests
test_container_agents(tmp_path)
# Overrides: llm_client, finbert_sentiment
# Real: fetchers (for integration testing)
```

---

## Container Fundamentals

### Provider Types

The container uses two provider patterns:

#### 1. Singleton Pattern

**Used for:** Stateful components that should be reused (cache, fetchers, analyzers)

```python
# In container
finbert_sentiment = providers.Singleton(
    model_providers.create_finbert_sentiment
)

# Override
mock_finbert = create_mock_finbert()
container.finbert_sentiment.override(mock_finbert)

# Reset
container.finbert_sentiment.reset_override()
```

#### 2. Factory Pattern

**Used for:** Stateless components that need fresh instances (LLM clients, agents)

```python
# In container
llm_client = providers.Factory(
    model_providers.create_llm_client,
    provider=daemon_config.provided.api_keys.llm_provider,
    model=daemon_config.provided.api_keys.llm_model,
)

# Override (REQUIRES providers.Factory wrapper)
from dependency_injector import providers

mock_llm = create_mock_llm_client()
container.llm_client.override(
    providers.Factory(lambda: mock_llm)
)

# Reset
container.llm_client.reset_override()
```

### Why Factory Needs providers.Factory

**Problem:**
```python
# ❌ WRONG - Returns function, not instance
container.llm_client.override(lambda: mock_llm)
agent = container.news_analyst()  # agent.llm is a function!
```

**Solution:**
```python
# ✅ CORRECT - Wraps lambda in Factory provider
container.llm_client.override(providers.Factory(lambda: mock_llm))
agent = container.news_analyst()  # agent.llm is mock_llm instance
```

---

## Common Patterns

### Agent Tests

```python
async def test_news_analyst(test_container, sample_news_articles):
    """Basic agent test."""
    analyst = test_container.news_analyst()
    result = await analyst.analyze("AAPL", sample_news_articles)

    assert isinstance(result, NewsAnalysis)
    assert result.key_themes
```

### Agent with Factory Parameter

Some agents use factory pattern requiring parameters:

```python
async def test_technical_analyst(test_container, sample_ohlcv_data):
    """Technical analyst needs strategy parameter."""
    from src.strategies.momentum import MomentumStrategy

    # technical_analyst() returns factory function
    analyst = test_container.technical_analyst()(MomentumStrategy())
    result = await analyst.analyze("AAPL", sample_ohlcv_data)

    assert isinstance(result, TechnicalAnalysis)
```

### Tool Tests

```python
def test_analyze_stock_tool(test_container_full):
    """Tool test with full mocks."""
    tool = AnalyzeStockTool(container=test_container_full)
    result = tool.execute(symbol="AAPL", period_days=90)

    assert "AAPL" in result
    assert "BUY" in result or "SELL" in result or "HOLD" in result
```

### Workflow Tests

```python
async def test_workflow_analyze(test_container):
    """Basic workflow test."""
    workflow = test_container.workflow_momentum()
    result = await workflow.analyze("AAPL", period_days=90)

    assert isinstance(result, TradingWorkflowResult)
    assert result.symbol == "AAPL"
    assert result.decision
    assert result.risk
```

### Workflow with Custom Config

```python
async def test_workflow_with_broker(test_container):
    """Workflow with broker and custom config."""
    from unittest.mock import MagicMock

    mock_broker = MagicMock()
    mock_broker.get_account_info.return_value = BrokerAccountInfo(...)

    workflow = TradingWorkflow(
        llm_client=test_container.llm_client(),
        market_fetcher=test_container.market_fetcher(),
        news_fetcher=test_container.news_fetcher(),
        finbert=test_container.finbert_sentiment(),
        fundamental_fetcher=test_container.fundamental_fetcher(),
        broker=mock_broker,
        use_meta_agent=False,
    )

    result = await workflow.analyze("AAPL")
    assert result.risk.validation.approved
```

### Custom Override Mid-Test

```python
async def test_with_custom_override(test_container, sample_news_articles):
    """Override fetcher for specific test behavior."""
    from unittest.mock import MagicMock

    # Override news fetcher to return sample articles
    mock_news_fetcher = MagicMock()
    mock_news_fetcher.fetch_company_news.return_value = sample_news_articles
    test_container.news_fetcher.override(mock_news_fetcher)

    workflow = test_container.workflow_momentum()
    state = await workflow._fetch_data("AAPL", 90)

    assert len(state["news_articles"]) > 0
```

---

## Override Patterns

### Singleton Overrides

```python
# Config
test_config = DaemonConfig(api_keys=ApiKeysConfig(...))
container.daemon_config.override(test_config)

# Cache
test_cache = HistoricalCache(db_path=str(tmp_path / "test.db"))
container.historical_cache.override(test_cache)

# Fetchers (all Singleton)
mock_market = create_mock_market_fetcher()
container.market_fetcher.override(mock_market)

mock_news = create_mock_news_fetcher()
container.news_fetcher.override(mock_news)

# FinBERT
mock_finbert = create_mock_finbert()
container.finbert_sentiment.override(mock_finbert)
```

### Factory Overrides

```python
from dependency_injector import providers

# LLM client (MUST wrap in Factory)
mock_llm = create_mock_llm_client()
container.llm_client.override(providers.Factory(lambda: mock_llm))

# Workflows (Factory pattern)
mock_workflow = MagicMock(spec=TradingWorkflow)
mock_workflow.analyze = AsyncMock(return_value=mock_result)
container.workflow_momentum.override(providers.Factory(lambda: mock_workflow))
```

### Reset Between Tests

The test fixtures handle reset automatically:

```python
@pytest.fixture
def test_container(tmp_path):
    """Container with automatic cleanup."""
    container = create_test_container(...)
    yield container
    reset_test_container(container)  # Automatic reset
```

Manual reset if needed:

```python
def test_something(test_container):
    # Custom override
    test_container.news_fetcher.override(mock)

    # Do test work...

    # Manual reset specific providers
    test_container.news_fetcher.reset_override()
```

---

## Test Isolation

### Best Practices

**DO:**
- ✅ Use container fixtures (`test_container`, `test_container_full`)
- ✅ Override only what you need for the specific test
- ✅ Let fixtures handle cleanup automatically
- ✅ Use `tmp_path` for cache/state files
- ✅ Create fresh mocks per test if mocking manually

**DON'T:**
- ❌ Share mocks between tests (state leakage)
- ❌ Forget `providers.Factory()` wrapper for Factory overrides
- ❌ Modify global state without cleanup
- ❌ Use real API credentials in tests
- ❌ Rely on test execution order

### Avoiding State Leakage

**Problem: Shared mock state**
```python
# ❌ BAD - Mock shared between tests
mock_llm = AsyncMock()  # Module level

def test_one(test_container):
    container.llm_client.override(providers.Factory(lambda: mock_llm))
    # ...

def test_two(test_container):
    # Uses same mock_llm with state from test_one!
```

**Solution: Fresh mocks**
```python
# ✅ GOOD - Fresh mock per test
def test_one(test_container):
    mock_llm = AsyncMock()
    container.llm_client.override(providers.Factory(lambda: mock_llm))
    # ...

def test_two(test_container):
    mock_llm = AsyncMock()  # Fresh instance
    container.llm_client.override(providers.Factory(lambda: mock_llm))
    # ...
```

### FinBERT Singleton Cleanup

FinBERT uses a module-level singleton that must be cleared between tests:

```python
# Automatic cleanup via conftest.py autouse fixture
@pytest.fixture(autouse=True)
def _clear_finbert_between_tests():
    """Clear FinBERT singleton between tests."""
    yield
    clear_finbert_sentiment()
```

---

## Migration Guide

### Step-by-Step: Migrating Agent Tests

**Before:**
```python
def test_news_analyst(mock_llm_client, sample_news_articles):
    analyst = NewsAnalyst(mock_llm_client)
    result = await analyst.analyze("AAPL", sample_news_articles)

    assert result.key_themes
    mock_llm_client.acomplete.assert_called_once()
```

**After:**
```python
def test_news_analyst(test_container, sample_news_articles):
    analyst = test_container.news_analyst()
    result = await analyst.analyze("AAPL", sample_news_articles)

    assert result.key_themes
    # Remove mock assertions - focus on behavior
```

**Changes:**
1. Replace `mock_llm_client` with `test_container`
2. Use `test_container.news_analyst()` instead of `NewsAnalyst(mock)`
3. Remove mock assertions (focus on output behavior)
4. Keep sample data fixtures (`sample_news_articles`)

### Step-by-Step: Migrating Tool Tests

**Before:**
```python
def test_tool(mock_llm_client):
    with patch("src.di.container.create_container") as mock_create:
        mock_container = MagicMock()
        mock_container.llm_client.return_value = mock_llm_client
        mock_create.return_value = mock_container

        tool = AnalyzeStockTool()
        result = tool.execute(symbol="AAPL")
```

**After:**
```python
def test_tool(test_container_full):
    tool = AnalyzeStockTool(container=test_container_full)
    result = tool.execute(symbol="AAPL")

    assert "AAPL" in result
```

**Changes:**
1. Remove complex patch setup
2. Pass `container=test_container_full` to tool
3. Remove mock container creation
4. Focus on tool behavior

### Step-by-Step: Migrating Workflow Tests

**Before:**
```python
@pytest.fixture
def mock_workflow_dependencies(mock_llm, mock_finbert, ...):
    market = MagicMock()
    market.fetch_daily.return_value = MarketData(...)
    return market, news, llm, finbert, fund

def test_workflow(mock_workflow_dependencies):
    market, news, llm, finbert, fund = mock_workflow_dependencies
    workflow = TradingWorkflow(llm, market, news, finbert, fund)
    result = await workflow.analyze("AAPL")
```

**After:**
```python
def test_workflow(test_container):
    workflow = test_container.workflow_momentum()
    result = await workflow.analyze("AAPL")

    assert isinstance(result, TradingWorkflowResult)
    assert result.symbol == "AAPL"
```

**Changes:**
1. Remove composite fixture
2. Use `test_container.workflow_momentum()`
3. Remove manual dependency unpacking
4. Override only when custom behavior needed

---

## Async Testing

### AsyncMock Patterns

**Basic AsyncMock:**
```python
from unittest.mock import AsyncMock

async def test_async_method(test_container):
    mock_llm = AsyncMock()
    mock_llm.acomplete.return_value = "Mock response"
    test_container.llm_client.override(providers.Factory(lambda: mock_llm))

    agent = test_container.news_analyst()
    result = await agent.analyze("AAPL", articles)

    mock_llm.acomplete.assert_called_once()
```

### Structured Output Mock

**Problem: side_effect conflicts with return_value**
```python
# ❌ WRONG - side_effect already set in create_mock_llm_client()
agent = test_container.event_triage_agent()
agent.llm.astructured.return_value = llm_response  # Ignored!
```

**Solution: Replace entire AsyncMock**
```python
# ✅ CORRECT - Clear side_effect by replacing method
from unittest.mock import AsyncMock

agent = test_container.event_triage_agent()
agent.llm.astructured = AsyncMock(return_value=llm_response)

result = await agent.analyze(event)
assert result.relevance == 0.85
```

### Dependency Wiring After Creation

Some agents need dependencies wired after creation:

```python
async def test_meta_agent(test_container):
    """Meta agent needs market_fetcher wired."""
    mock_market = create_mock_market_fetcher()

    agent = test_container.meta_agent()
    agent.market_fetcher = mock_market  # Wire after creation

    result = await agent.analyze("AAPL")
```

---

## Troubleshooting

### Common Errors

#### 1. AttributeError: 'function' object has no attribute 'X'

**Symptom:**
```python
AttributeError: 'function' object has no attribute 'acomplete'
```

**Cause:** Factory override missing `providers.Factory()` wrapper

**Fix:**
```python
# ❌ WRONG
container.llm_client.override(lambda: mock_llm)

# ✅ CORRECT
from dependency_injector import providers
container.llm_client.override(providers.Factory(lambda: mock_llm))
```

#### 2. AsyncMock return_value Ignored

**Symptom:**
```python
mock.astructured.return_value = X  # Has no effect
```

**Cause:** `side_effect` already set (takes precedence over `return_value`)

**Fix:**
```python
# ✅ Replace entire AsyncMock
from unittest.mock import AsyncMock
agent.llm.astructured = AsyncMock(return_value=llm_response)
```

#### 3. AttributeError: 'DynamicContainer' object has no attribute 'X'

**Symptom:**
```python
AttributeError: 'DynamicContainer' object has no attribute 'workflow_ensemble'
```

**Cause:** Provider doesn't exist in container

**Fix:** Check available providers:
```python
# Available: workflow_momentum, workflow_meta, workflow_trump, workflow_full
# NOT available: workflow_ensemble (create manually)

workflow = TradingWorkflow(
    llm_client=test_container.llm_client(),
    ...,
    use_ensemble=True,
)
```

#### 4. Empty News/Data in Tests

**Symptom:**
```python
assert len(state["news_articles"]) > 0
# AssertionError: assert 0 > 0
```

**Cause:** `test_container` has `override_fetchers=False`, hits real API with test keys

**Fix:** Override fetcher or use `test_container_full`:
```python
# Option 1: Override
mock_news = MagicMock()
mock_news.fetch_company_news.return_value = sample_news_articles
test_container.news_fetcher.override(mock_news)

# Option 2: Use test_container_full
def test_something(test_container_full, sample_news_articles):
    # test_container_full has all fetchers mocked
```

#### 5. State Leakage Between Tests

**Symptom:** Tests pass individually but fail when run together

**Cause:** Shared mock state or missing reset

**Fix:**
```python
# Ensure fixture handles reset
@pytest.fixture
def test_container(tmp_path):
    container = create_test_container(...)
    yield container
    reset_test_container(container)  # Must reset

# Or create fresh mocks per test
def test_one(test_container):
    mock_llm = AsyncMock()  # Fresh mock
    # ...
```

### Debug Checklist

When tests fail:

1. ✅ Check provider type (Singleton vs Factory)
2. ✅ Verify `providers.Factory()` wrapper for Factory overrides
3. ✅ Confirm test fixture (`test_container` vs `test_container_full`)
4. ✅ Check for shared mock state
5. ✅ Verify AsyncMock setup (replace vs return_value)
6. ✅ Ensure sample data fixtures provided
7. ✅ Check container reset in fixture cleanup

---

## Quick Reference

### Provider Types Cheat Sheet

| Provider | Type | Override Pattern | Reset |
|----------|------|------------------|-------|
| `daemon_config` | Singleton | `container.daemon_config.override(config)` | `reset_override()` |
| `historical_cache` | Singleton | `container.historical_cache.override(cache)` | `reset_override()` |
| `llm_client` | **Factory** | `container.llm_client.override(providers.Factory(lambda: mock))` | `reset_override()` |
| `finbert_sentiment` | Singleton | `container.finbert_sentiment.override(mock)` | `reset_override()` |
| `market_fetcher` | Singleton | `container.market_fetcher.override(mock)` | `reset_override()` |
| `news_fetcher` | Singleton | `container.news_fetcher.override(mock)` | `reset_override()` |
| `fundamental_fetcher` | Singleton | `container.fundamental_fetcher.override(mock)` | `reset_override()` |
| `finnhub_fetcher` | Singleton | `container.finnhub_fetcher.override(mock)` | `reset_override()` |
| `comparative_fetcher` | Singleton | `container.comparative_fetcher.override(mock)` | `reset_override()` |
| `alpaca_broker` | Singleton | `container.alpaca_broker.override(mock)` | `reset_override()` |
| `backtest_runner` | Factory | `container.backtest_runner.override(providers.Factory(lambda: mock))` | `reset_override()` |
| `optuna_optimizer` | Factory | `container.optuna_optimizer.override(providers.Factory(lambda: mock))` | `reset_override()` |
| `metrics_tracker` | Singleton | `container.metrics_tracker.override(mock)` | `reset_override()` |
| `quantstats_reporter` | Singleton | `container.quantstats_reporter.override(mock)` | `reset_override()` |
| `stock_screener` | Singleton | `container.stock_screener.override(mock)` | `reset_override()` |
| `workflow_momentum` | Factory | `container.workflow_momentum.override(providers.Factory(lambda: mock))` | `reset_override()` |
| `news_analyst` | Factory | (via factory call) | N/A |
| `technical_analyst` | Factory | (via factory call) | N/A |

### Mock Creation Functions

Located in `tests/di/container_test.py`:

```python
create_mock_llm_client() -> MagicMock
create_mock_finbert() -> MagicMock
create_mock_market_fetcher() -> MagicMock
create_mock_news_fetcher() -> MagicMock
create_mock_fundamental_fetcher() -> MagicMock
create_mock_finnhub_fetcher() -> MagicMock
create_mock_reddit_fetcher() -> MagicMock
create_mock_truth_social_fetcher() -> MagicMock
create_mock_web_search_fetcher() -> MagicMock
create_mock_earnings_fetcher() -> MagicMock
create_mock_comparative_fetcher() -> MagicMock
create_mock_alpaca_broker() -> MagicMock
create_mock_backtest_runner() -> MagicMock
create_mock_optuna_optimizer() -> MagicMock
create_mock_metrics_tracker() -> MagicMock
create_mock_quantstats_reporter() -> MagicMock
create_mock_stock_screener() -> MagicMock
```

### Test Fixtures

```python
# In conftest.py

test_container(tmp_path) -> AppContainer
# Minimal overrides: llm_client, finbert_sentiment
# Use for: Agent unit tests, component tests

test_container_full(tmp_path) -> AppContainer
# Full overrides: All mocks enabled
# Use for: Tool tests, integration tests

test_container_agents(tmp_path) -> AppContainer
# Agent-focused: llm_client, finbert_sentiment
# Real fetchers for integration testing
# Use for: Agent integration tests

# Sample data fixtures (unchanged)
sample_ohlcv_data() -> pd.DataFrame
sample_news_articles() -> list[NewsArticle]
sample_bullish_research() -> BullishResearchAnalysis
sample_bearish_research() -> BearishResearchAnalysis
```

### Common Test Patterns Summary

```python
# Agent test
def test_agent(test_container):
    agent = test_container.news_analyst()
    result = await agent.analyze("AAPL", data)
    assert result

# Tool test
def test_tool(test_container_full):
    tool = AnalyzeStockTool(container=test_container_full)
    result = tool.execute(symbol="AAPL")
    assert result

# Workflow test
async def test_workflow(test_container):
    workflow = test_container.workflow_momentum()
    result = await workflow.analyze("AAPL")
    assert result

# Custom override
def test_custom(test_container):
    mock = create_mock_news_fetcher()
    test_container.news_fetcher.override(mock)
    # ... test code
```

---

## Additional Resources

- **Container implementation**: `src/di/container.py`
- **Test container utils**: `tests/di/container_test.py`
- **Provider factories**: `src/di/providers/`
- **Migration PR**: #315
- **Related issues**: #7 (nest_asyncio removal)

---

*Last updated: 2026-02-10 | Migration completed in 4 phases*
