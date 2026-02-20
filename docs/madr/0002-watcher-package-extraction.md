# MADR 0002 — Watcher Package Extraction

**Status:** Accepted
**Date:** 2026-02-19

---

## Context

Watchers lived in `src/daemon/watchers/` and inherited from `EventWatcher`, which mixed:
- Loop infrastructure (`run()`, sleep, graceful shutdown)
- LLM event triage (`EventTriageAgent`)
- Direct `TradingWorkflow` execution (bypassed `MarketEventQueue`)

Problems:
1. Inheritance coupled unrelated concerns; watchers could not be tested in isolation
2. `EventWatcher._analyze_stocks()` called `TradingWorkflow` directly — bypassed the established queue-based flow
3. Standalone watchers (`EconomicCalendarWatcher`, `OptionsFlowWatcher`, `SocialSentimentWatcher`) duplicated their own `run()` loop instead of reusing `EventWatcher`

---

## Decisions

### 1. Extract `src/v1/watchers/` package

**Decision:** Move all 8 watcher files from `src/daemon/watchers/` to `src/v1/watchers/`. Delete `src/daemon/watchers/` and `src/daemon/event_watcher.py`.

**Rationale:** Watchers are not daemon-internal concerns — they are reusable components that poll external sources. A dedicated package gives them a stable import path and makes the dependency direction explicit (`v1/watchers` → `daemon/events`, `v1/watchers` → `event_queue`; `daemon` → `v1/watchers`).

### 2. Replace `EventWatcher` inheritance with `EventTriagePipeline` composition

**Decision:** Introduce `Watcher` ABC + `PeriodicWatcher` base in `src/v1/watchers/base.py`. Extract triage + enqueue logic into `EventTriagePipeline` service in `src/v1/watchers/pipeline.py`. Inject pipeline via constructor.

**Alternatives considered:**
- Keep `EventWatcher` base, move it to `src/watchers/` — rejected: mixing loop + triage in one class remains an SRP violation
- Make `EventTriagePipeline` a mixin — rejected: composition is simpler and more testable

**Rationale:** Each abstraction now has a single responsibility. `PeriodicWatcher` owns the 1s-granularity sleep loop. `EventTriagePipeline` owns LLM triage and queue routing. Watchers own source-specific `_fetch_events()` logic. Dependencies are constructor-injected and easily mocked in tests.

### 3. Watchers emit to `MarketEventQueue`; coordinator consumes via `EventQueueConsumer`

**Decision:** `EventTriagePipeline.process()` routes IMMEDIATE events to `MarketEventQueue.enqueue()`. WATCHLIST events go to `DaemonState` discovery candidates. Watchers no longer call `TradingWorkflow` directly.

**Alternatives considered:**
- Keep direct `TradingWorkflow` execution in watchers — rejected: bypasses TTL, deduplication, `process_after` scheduling that `MarketEventQueue` provides
- Run `EventQueueConsumer` separately from the daemon — already the case; no change needed

**Rationale:** `MarketEventQueue` provides TTL, deduplication, delayed processing, and observability. Routing all events through the queue makes the full event lifecycle visible and testable.

---

## Consequences

- All watcher imports change from `src.daemon.watchers.*` to `src.v1.watchers.*`
- Event-driven watchers (5): inject `EventTriagePipeline` via constructor; IMMEDIATE events now enqueue instead of triggering `TradingWorkflow` directly
- Standalone watchers (3): remove own `run()` loop; inherit from `PeriodicWatcher`; rename `_fetch_and_assess[_all]()` → `_tick()`
- `EventQueueConsumer` must be running for IMMEDIATE events to be processed (it is, in the production daemon)
- When `MarketEventQueue` is unavailable (DB disabled), IMMEDIATE events are logged and dropped — `EventTriagePipeline` accepts `queue: MarketEventQueue | None`
