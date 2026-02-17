# Pydantic AI Migration Plan - Supervisor Pattern

**Status:** Planning
**Target:** Migrate from LangGraph to Pydantic AI with supervisor pattern
**Estimated Effort:** 4-6 weeks

---

## Migration Rationale

### Why Pydantic AI

**Advantages:**

- Type-safe, maintainable agents (FastAPI-like experience)
- Smaller package size (~70MB vs ~300MB)
- Python-native control flow vs graph abstractions
- Better IDE autocomplete, validation
- Simpler state management (Pydantic models vs TypedDict)
- Aligns with existing codebase standards (already using Pydantic everywhere)

**Trade-offs:**

- Less sophisticated checkpointing than LangGraph
- Smaller ecosystem
- Newer framework (less battle-tested)

### Why Supervisor Pattern

**Benefits for AI Casino:**

- Dynamic analysis selection (skip unnecessary analyses)
- Multi-strategy support (momentum, mean-reversion, breakout)
- Intelligent routing based on market conditions
- Foundation for portfolio-level coordination
- Better scalability

**Costs:**

- +1-3 LLM calls per analysis (routing overhead)
- More complex testing
- Initial latency increase

---

## Current Architecture

### Sequential Pipeline (Current)

```
TradingWorkflow
  ├── fetch_data(symbol)
  ├── run_technical_analysis() → TechnicalAnalysis
  ├── run_sentiment_analysis() → SentimentAnalysis
  ├── run_news_analysis() → NewsAnalysis
  └── make_decision() → TradingDecision
```

**Characteristics:**

- Fixed sequence, all analyses always run
- Simple, predictable, easy to debug
- No dynamic routing
- Workers implemented as agents (TechnicalAnalyst, etc.)

---

## Target Architecture

### Supervisor Pattern (Pydantic AI)

```
TradingSupervisor (Pydantic AI Agent)
  ├── Receives request → determines needed analyses
  ├── Routes to workers (via tools)
  │   ├── @technical_worker.tool
  │   ├── @sentiment_worker.tool
  │   ├── @news_worker.tool
  │   ├── @fundamental_worker.tool (new)
  │   └── @risk_validator.tool (new)
  ├── Aggregates results
  └── Synthesizes final decision
```

**Characteristics:**

- Dynamic routing based on context
- Workers as Pydantic AI tools
- Parallel execution when possible
- Conditional analyses based on data quality/market conditions

---

## Migration Phases

### Phase 1: Foundation (Week 1-2)

**Add Pydantic AI dependency**

- Install pydantic-ai package
- Configure in pyproject.toml
- Update DI container for dual framework support

**Create base supervisor**

- Implement TradingSupervisor as Pydantic AI Agent
- Define SupervisorState (Pydantic model)
- Create basic task decomposition logic
- Add supervisor prompts

**Convert first worker**

- Convert TechnicalAnalyst to TechnicalWorker
- Implement as @agent.tool
- Test standalone execution
- Preserve existing functionality

### Phase 2: Worker Migration (Week 2-3)

**Convert remaining workers**

- SentimentWorker (@agent.tool)
- NewsWorker (@agent.tool)
- Unified BaseWorker interface (Pydantic models)

**Add new workers**

- FundamentalWorker (P/E, earnings analysis)
- RiskValidator (decision validation)

**Parallel execution**

- Implement parallelization pattern
- Use asyncio.gather for independent workers
- Handle errors gracefully

### Phase 3: Integration (Week 3-4)

**Update TradingWorkflow**

- Replace sequential pipeline with supervisor
- Integrate with existing data fetchers
- Maintain backward compatibility (feature flag)

**DI container updates**

- Update providers for Pydantic AI agents
- Configure supervisor with all workers
- Manage dependency injection

**Testing**

- Unit tests for supervisor logic
- Integration tests for workflow
- A/B testing framework (sequential vs supervisor)

### Phase 4: Optimization (Week 4-5)

**Routing intelligence**

- Context-aware routing (pre-market, earnings, volatile)
- Skip unnecessary analyses
- Dynamic confidence thresholds

**Performance tuning**

- Measure latency overhead
- Optimize LLM calls
- Cache routing decisions

**Error handling**

- Graceful degradation
- Fallback strategies
- Retry logic

### Phase 5: Production Readiness (Week 5-6)

**Monitoring & observability**

- Add supervisor metrics
- Track routing decisions
- Log worker execution times

**Documentation**

- Update CLAUDE.md with Pydantic AI patterns
- Document supervisor architecture
- Create migration guide

**Cleanup**

- Remove LangGraph dependency (if fully migrated)
- Clean up unused code
- Update tests

---

## Implementation Details

### Supervisor Implementation

**File:** `src/agents/supervisor.py`

```python
from pydantic_ai import Agent
from pydantic import BaseModel

class SupervisorState(BaseModel):
    """Supervisor execution state."""
    symbol: str
    market_data: pd.DataFrame | None = None
    needed_analyses: list[str] = []
    completed_analyses: dict[str, Any] = {}
    final_decision: TradingDecision | None = None

class TradingSupervisor:
    """Coordinates trading analysis workflow."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.agent = Agent(
            model=llm_client,
            result_type=SupervisorDecision,
            system_prompt=self._load_system_prompt(),
        )

    async def coordinate(
        self,
        symbol: str,
        context: AnalysisContext,
    ) -> TradingDecision:
        """Execute supervisor workflow."""
        state = SupervisorState(symbol=symbol)

        # 1. Determine needed analyses
        plan = await self._create_plan(state, context)
        state.needed_analyses = plan.analyses

        # 2. Execute workers (parallel if possible)
        results = await self._execute_workers(state)
        state.completed_analyses = results

        # 3. Synthesize decision
        decision = await self._synthesize(state)
        return decision
```

### Worker Implementation

**File:** `src/workers/technical.py`

```python
from pydantic_ai import Agent

class TechnicalWorker:
    """Technical analysis worker."""

    def __init__(self, llm_client: LLMClient, strategy: MomentumStrategy) -> None:
        self.agent = Agent(
            model=llm_client,
            result_type=TechnicalAnalysis,
            system_prompt=self._load_system_prompt(),
        )
        self.strategy = strategy

    @property
    def tool(self):
        """Return tool definition for supervisor."""
        return {
            "name": "technical_analysis",
            "description": "Perform RSI/MACD momentum analysis",
            "function": self.analyze,
        }

    async def analyze(
        self,
        symbol: str,
        market_data: pd.DataFrame,
    ) -> TechnicalAnalysis:
        """Execute technical analysis."""
        indicators = self.strategy.calculate_indicators(market_data)

        result = await self.agent.run(
            f"Analyze {symbol} with RSI={indicators.rsi}, MACD={indicators.macd}"
        )

        return TechnicalAnalysis(
            signal=self._determine_signal(indicators),
            rsi=indicators.rsi,
            macd_hist=indicators.macd_hist,
            interpretation=result.data.interpretation,
            confidence=result.data.confidence,
        )
```

### DI Container Updates

**File:** `src/di/container.py`

```python
# Workers
technical_worker = providers.Singleton(
    create_technical_worker,
    llm_client=llm_client,
    strategy=momentum_strategy,
)

sentiment_worker = providers.Singleton(
    create_sentiment_worker,
    llm_client=llm_client,
    finbert=finbert_model,
)

# Supervisor
supervisor = providers.Singleton(
    create_supervisor,
    llm_client=llm_client,
    workers={
        "technical": technical_worker,
        "sentiment": sentiment_worker,
        "news": news_worker,
        "fundamental": fundamental_worker,
        "risk": risk_validator,
    },
)

# Workflow
workflow = providers.Factory(
    TradingWorkflow,
    supervisor=supervisor,
    market_fetcher=market_fetcher,
)
```

---

## Testing Strategy

### Unit Tests

**Supervisor tests:**

- Task decomposition logic
- Routing decisions
- State management
- Error handling

**Worker tests:**

- Tool definitions
- Standalone execution
- Result validation

### Integration Tests

**Workflow tests:**

- End-to-end supervisor execution
- Parallel worker execution
- Error propagation
- Backward compatibility

### A/B Testing

**Comparison metrics:**

- Latency (sequential vs supervisor)
- Accuracy (signal correctness)
- Cost (LLM calls)
- Decision quality

---

## Migration Checklist

### Phase 1: Foundation

- [ ] Add pydantic-ai to pyproject.toml
- [ ] Create src/agents/supervisor.py
- [ ] Create src/workers/ directory
- [ ] Implement SupervisorState (Pydantic model)
- [ ] Create supervisor prompts (src/prompts/supervisor/)
- [ ] Convert TechnicalAnalyst → TechnicalWorker
- [ ] Unit tests for supervisor + technical worker

### Phase 2: Worker Migration

- [ ] Convert SentimentAnalyst → SentimentWorker
- [ ] Convert NewsAnalyst → NewsWorker
- [ ] Create FundamentalWorker (new)
- [ ] Create RiskValidator (new)
- [ ] Implement parallel execution pattern
- [ ] Error handling for worker failures
- [ ] Unit tests for all workers

### Phase 3: Integration

- [ ] Update TradingWorkflow for supervisor
- [ ] Add feature flag (ENABLE_SUPERVISOR)
- [ ] Update DI container providers
- [ ] Integration tests (sequential + supervisor)
- [ ] A/B testing framework
- [ ] Validate backward compatibility

### Phase 4: Optimization

- [ ] Context-aware routing logic
- [ ] Performance benchmarks
- [ ] Cache routing decisions
- [ ] Optimize LLM calls
- [ ] Graceful degradation
- [ ] Retry logic

### Phase 5: Production

- [ ] Add supervisor metrics
- [ ] Update daemon config schema
- [ ] Monitoring dashboards
- [ ] Update CLAUDE.md
- [ ] Migration guide doc
- [ ] Remove LangGraph dependency
- [ ] Clean up unused code

---

## Rollback Plan

### Feature Flag Strategy

**Config:** `~/.ai-casino/daemon-production.yaml`

```yaml
workflow:
  pattern: "supervisor"  # "sequential" | "supervisor"
```

**Rollback steps:**

1. Change config to `pattern: "sequential"`
2. Restart daemon
3. Monitor for issues
4. Keep both implementations until supervisor proven stable

### Gradual Rollout

**Week 1-2:** Internal testing only (not in daemon)
**Week 3-4:** Daemon with feature flag (sequential default)
**Week 5:** Daemon with supervisor enabled (monitor closely)
**Week 6:** Remove sequential code if supervisor stable

---

## Success Criteria

### Performance

- Latency: <2x sequential pipeline
- Cost: <50% increase in LLM calls
- Accuracy: ≥95% of sequential baseline

### Functionality

- All existing analyses supported
- New analyses (fundamental, risk) working
- Parallel execution functional
- Error handling robust

### Code Quality

- Type safety maintained
- Tests passing (>90% coverage)
- Documentation complete
- No regressions

---

## Risks & Mitigation

### Risk: Higher Latency

**Mitigation:**

- Optimize routing prompts
- Cache common decisions
- Parallel worker execution
- Performance monitoring

### Risk: Increased Complexity

**Mitigation:**

- Comprehensive testing
- Feature flag for rollback
- Gradual rollout
- Clear documentation

### Risk: LLM Call Cost

**Mitigation:**

- Smart routing (skip unnecessary analyses)
- Cache routing decisions
- Use cheaper model for routing
- Monitor costs closely

### Risk: New Framework Bugs

**Mitigation:**

- Extensive testing
- A/B comparison with sequential
- Keep sequential code until proven
- Monitor error rates

---

## Resources

### Documentation

- [Pydantic AI Docs](https://ai.pydantic.dev/)
- [Multi-Agent Patterns](https://ai.pydantic.dev/multi-agent-applications/)
- [Supervisor Pattern](https://medium.com/aitech/the-supervisor-pattern-for-gen-ai-agent-systems-d1920c0bdbbb)

### Examples

- [AWS Pydantic AI Multi-Agent](https://github.com/aws-samples/sample-pydantic-ai-streaming-rag-multiagent)
- [Parallelization Patterns](https://dylancastillo.co/til/parallelization-orchestrator-workers-pydantic-ai.html)

### Internal Docs

- `knowledge/agentic-architectures.md` - Research compilation
- `CLAUDE.md` - Project conventions
- `docs/analysis-pipeline.md` - Current architecture
