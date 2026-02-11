# Daemon Architecture

System-level architecture of the AI Casino daemon mode: component relationships, lifecycles, concurrency, and market session handling.

## Component Dependency Graph

```mermaid
graph TD
    subgraph DI Container
        Container[AppContainer]
    end

    subgraph Daemon Layer
        DR[DaemonRunner]
        DCO[DaemonCycleOrchestrator]
        TW[TrumpWatcher]
        MS[MarketScheduler]
        DS[DaemonState]
        DC[DaemonConfig]
        CB[DaemonContextBuilder]
    end

    subgraph Data Layer
        Cache[HistoricalCache]
        DB[Database]
        MF[MarketDataFetcher]
        NF[NewsFetcher]
        FF[FinnhubFetcher]
        RF[RedditFetcher]
        TSF[TruthSocialFetcher]
        FDF[FundamentalDataFetcher]
    end

    subgraph Orchestration
        TradingWorkflow
        MetaAgent
    end

    subgraph Analysis Agents
        TA[TechnicalAnalyst]
        SA[SentimentAnalyst]
        NA[NewsAnalyst]
        FA[FundamentalAnalyst]
        CA[ComparativeAnalyst]
        WR[WebResearchAgent]
        SSA[SocialSentimentAnalyst]
        TRA[TrumpAnalyst]
        BR[BullishResearcher]
        BER[BearishResearcher]
    end

    subgraph Decision & Risk
        Trader[TraderAgent]
        RM[RiskManagementAgent]
    end

    subgraph Execution & Tracking
        AB[AlpacaBroker]
        MT[MetricsTracker]
        PSR[PortfolioSnapshotRepository]
        AR[AnalysisRepository]
        PR[PositionRepository]
    end

    Container --> DR
    Container --> TradingWorkflow
    Container --> Cache
    Container --> DB

    DR --> DC
    DR --> MS
    DR --> DS
    DR --> DCO
    DR --> CB

    DCO --> TradingWorkflow
    TW --> TradingWorkflow
    TW --> TRA

    MS --> DC

    TradingWorkflow --> Cache
    TradingWorkflow --> MF
    TradingWorkflow --> NF
    TradingWorkflow --> TSF
    TradingWorkflow --> MetaAgent
    TradingWorkflow --> TA
    TradingWorkflow --> SA
    TradingWorkflow --> NA
    TradingWorkflow --> FA
    TradingWorkflow --> CA
    TradingWorkflow --> WR
    TradingWorkflow --> SSA
    TradingWorkflow --> TRA
    TradingWorkflow --> BR
    TradingWorkflow --> BER
    TradingWorkflow --> Trader
    TradingWorkflow --> RM
    TradingWorkflow --> AB
    TradingWorkflow --> MT
    TradingWorkflow --> PSR
    TradingWorkflow --> AR

    MF --> Cache
    NF --> Cache
    SSA --> FF
    SSA --> RF
    FA --> FDF
    CA --> FDF

    PSR --> DB
    AR --> DB
    PR --> DB
```

## DaemonRunner Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Init: from_config_file()

    Init --> WaitingForMarket: market_hours_only=true
    Init --> RunningCycle: market_hours_only=false

    WaitingForMarket --> RunningCycle: is_market_open()=true
    WaitingForMarket --> WaitingForMarket: sleep(min(time_until_open, 60s))

    RunningCycle --> Analyzing: _analyze_watchlist()
    Analyzing --> LogResults: _log_results()
    LogResults --> SaveState: state.save()
    SaveState --> Sleeping: sleep(interval_minutes * 60)
    Sleeping --> WaitingForMarket: market_hours_only=true
    Sleeping --> RunningCycle: market_hours_only=false

    RunningCycle --> ErrorRecovery: exception
    ErrorRecovery --> Sleeping: sleep(60s)

    WaitingForMarket --> Shutdown: SIGINT/SIGTERM
    RunningCycle --> Shutdown: SIGINT/SIGTERM
    Analyzing --> Shutdown: CancelledError
    Sleeping --> Shutdown: SIGINT/SIGTERM

    Shutdown --> [*]: state.save()
```

**Key transitions:**

- **Init**: Loads `DaemonConfig` from TOML, creates `MarketScheduler` and `DaemonState`
- **WaitingForMarket**: Polls `is_market_open()` every `min(time_until_open, 60)` seconds
- **Analyzing**: Runs all watchlist symbols concurrently via `asyncio.Semaphore`
- **Sleeping**: Waits `interval_minutes` (default: 30 min) between cycles
- **Shutdown**: Graceful — saves state, sets `running=False`

## TrumpWatcher Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Init: TrumpWatcher(poll_interval, max_analyses)

    Init --> Polling: run()

    Polling --> CheckingPosts: _check_new_posts()
    CheckingPosts --> Polling: no new posts, sleep(poll_interval)

    CheckingPosts --> IdentifyingStocks: new posts found
    IdentifyingStocks --> IdentifyingStocks: keyword match (SECTOR_STOCKS)
    IdentifyingStocks --> IdentifyingStocks: LLM extraction (top 5 posts)

    IdentifyingStocks --> Polling: no stocks identified
    IdentifyingStocks --> AnalyzingStocks: symbols found

    AnalyzingStocks --> EmittingSignal: _analyze_stocks()
    EmittingSignal --> Polling: _emit_signal(), sleep(poll_interval)

    Polling --> ErrorRecovery: exception
    ErrorRecovery --> Polling: sleep(60s)

    Polling --> Shutdown: SIGINT/SIGTERM
    AnalyzingStocks --> Shutdown: CancelledError

    Shutdown --> [*]
```

**Key details:**

- **Polling**: Checks Truth Social every `poll_interval` (default: 60s)
- **First run**: Looks back 1 hour for recent posts
- **Stock identification**: Two-pass — keyword matching against `SECTOR_STOCKS` map, then LLM extraction
- **SECTOR_STOCKS sectors**: tariff, china, crypto, oil, bank, tech, defense, pharma (2 stocks per sector match)
- **Analysis cap**: Max `max_analyses` (default: 5) stocks per cycle
- **Content sanitization**: Posts truncated to 200 chars for LLM processing

## Concurrency Model

```mermaid
graph LR
    subgraph DaemonRunner
        DS[Daemon Semaphore<br/>max=3]
        DS --> S1[Symbol 1<br/>analyze]
        DS --> S2[Symbol 2<br/>analyze]
        DS --> S3[Symbol 3<br/>analyze]
        DS -.-> S4[Symbol 4<br/>waiting]
    end

    subgraph TrumpWatcher
        TS[Trump Semaphore<br/>max=2]
        TS --> T1[Stock 1<br/>analyze]
        TS --> T2[Stock 2<br/>analyze]
        TS -.-> T3[Stock 3<br/>waiting]
    end

    subgraph LLM Layer
        LS[LLM Semaphore<br/>max=LLM_MAX_CONCURRENT]
        LS --> L1[LLM Call 1]
        LS --> L2[LLM Call 2]
        LS --> L3[LLM Call 3]
        LS --> L4[LLM Call 4]
        LS --> L5[LLM Call 5]
    end

    subgraph Data Layer
        Cache[HistoricalCache<br/>threading.Lock]
        DBPool[Database Pool<br/>async sessions]
    end

    S1 --> LS
    S2 --> LS
    S3 --> LS
    T1 --> LS
    T2 --> LS

    S1 --> Cache
    S2 --> Cache
    S3 --> Cache
    T1 --> Cache
    T2 --> Cache

    S1 --> DBPool
    S2 --> DBPool
    S3 --> DBPool
```

| Semaphore/Lock | Default Limit | Configurable | Source |
|---|---|---|---|
| Daemon concurrent analyses | 3 | `max_concurrent_analyses` in config | `runner.py` |
| Trump concurrent analyses | 2 | No (hardcoded) | `trump_watcher.py` |
| LLM concurrent calls | 5 | `LLM_MAX_CONCURRENT` env var (1-20) | `models/llm.py` |
| HistoricalCache | threading.Lock (shared singleton) | No | `data/cache.py` |
| Database sessions | async pool (per-request sessions) | No | `database/engine.py` |

All semaphores use `asyncio.Semaphore`; concurrency is async, with blocking market/news (and optional Truth Social) fetches offloaded to a thread pool via `asyncio.to_thread(...)`. HistoricalCache uses `threading.Lock` for thread-safe access. Database uses async sessions from connection pool.

## Trading Sessions Timeline

```mermaid
gantt
    title Trading Sessions (Eastern Time)
    dateFormat HH:mm
    axisFormat %H:%M

    section Market Closed
    Overnight           :done, 00:00, 04:00

    section Pre-Market
    Pre-Market (optional) :active, 04:00, 09:30

    section Regular Hours
    Regular Session      :crit, 09:30, 16:00

    section Market Closed
    After Hours          :done, 16:00, 23:59
```

| Session | Hours (ET) | Config Flag | Behavior |
|---|---|---|---|
| Pre-Market | 04:00 — 09:30 | `enable_pre_market: true` | Same pipeline, flagged as `PRE_MARKET` in results |
| Regular | 09:30 — 16:00 | Always enabled | Standard analysis cycle |
| Closed | 16:00 — 04:00 | — | Daemon waits (if `market_hours_only=true`) |

- Weekend handling: Friday close → Monday open (scheduler skips Saturday/Sunday)
- No automatic confidence adjustment between sessions — data quality varies naturally
- Session type recorded in `AnalysisRecord.trading_session` and `TradingWorkflowResult`

---

**See also:** [Analysis Pipeline](analysis-pipeline.md) | [Daemon Operations](daemon-operations.md)
