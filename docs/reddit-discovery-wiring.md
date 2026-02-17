# Reddit → Discovery Pipeline: Wiring Gaps

Current state as of 2026-02-17. Documents what works, what's broken, and what needs to be connected.

---

## Intended Flow

```
PeriodicRedditScrapingTask (Playwright)
  → reddit_posts / reddit_ticker_mentions (DB)
    → SocialWatcher (polls DB every 15min)
      → SocialEvent (volume spike / viral post)
        → EventTriageAgent (LLM urgency scoring)
          → IMMEDIATE → discovery_callback → StockDiscoveryEngine
          → WATCHLIST → state.discovery.add_event_candidates()
            → StockDiscoveryEngine._fetch_reddit_candidates()
              → TradingWorkflow (full analysis)
```

---

## What Actually Works

- `PeriodicRedditScrapingTask` scrapes posts/comments via Playwright, extracts tickers via LLM, writes to `reddit_posts`, `reddit_comments`, `reddit_ticker_mentions`, `reddit_ticker_sentiment` tables
- `SocialWatcher._fetch_events()` reads those DB tables (DB-first path, falls back to PRAW API)
- Volume spike and viral post detection logic works correctly
- `WATCHLIST` urgency events reach `state.discovery.add_event_candidates()` — this path is functional

---

## Gaps To Fix

### 1. `SocialWatcher` missing `discovery_mode` and `discovery_callback`

**File:** `src/di/providers/watchers.py` → `create_social_watcher()`

`SocialWatcher` is instantiated without `discovery_mode=True` or `discovery_callback`. Result: `IMMEDIATE` urgency events bypass discovery entirely and fire a direct `TradingWorkflow` instead of queuing a `DiscoveryCandidate`.

**Fix:** Pass `discovery_mode=True` and wire a `discovery_callback` that calls the discovery engine's candidate intake. The `EventWatcherIntegrationConfig.social_watcher_use_discovery` flag already exists in config — read it.

```python
# src/di/providers/watchers.py
return SocialWatcher(
    ...
    discovery_mode=config.event_watchers.social_watcher_use_discovery,
    discovery_callback=discovery_engine.add_candidates,  # needs discovery_engine injected
)
```

---

### 2. Dead config: `EventWatcherIntegrationConfig.social_watcher_use_discovery`

**File:** `src/daemon/config/events.py`

The flag `social_watcher_use_discovery: bool = True` is declared but never read anywhere. No code path in factory, providers, or lifecycle consumes it.

**Fix:** Read it in `create_social_watcher()` (see gap 1).

---

### 3. `NewsTrendingWatcher` has `discovery_mode=True` but no `discovery_callback`

**File:** `src/daemon/watchers/news_trending_watcher.py` (hardcodes `discovery_mode=True` in `__init__`)
**File:** `src/di/providers/watchers.py` → `create_news_trending_watcher()`

`_route_to_discovery()` in `EventWatcher` base class guards on `if all_candidates and self._discovery_callback:` — since `discovery_callback=None`, all IMMEDIATE candidates are silently dropped.

**Fix:** Wire the same `discovery_callback` to `NewsTrendingWatcher` as to `SocialWatcher` (see gap 1). Also applies to any other watcher with `discovery_mode=True`.

---

### 4. `StockDiscoveryEngine._fetch_reddit_candidates()` is a stub

**File:** `src/daemon/` (StockDiscoveryEngine, exact file TBD)

Contains `# TODO: Implement Reddit trending integration` and always returns `[]`. The `enable_reddit_trending` config flag enables dead code.

**Fix:** Implement using `RedditTickerMentionRepository.get_mentions_in_window()` and `RedditTickerSentimentRepository` aggregates — the data is already in DB, just needs to be queried and converted to `DiscoveryCandidate` objects with `source=DiscoverySource.REDDIT`.

---

### 5. `StockDiscoveryEngine` created with `reddit_fetcher=None`

**File:** `src/daemon/factory.py` → `_create_discovery_engine()`

`OptionalServices(reddit_fetcher=None, ...)` — Reddit fetcher is explicitly excluded from discovery services even though the container has a `reddit_fetcher` singleton.

**Fix:** Pass `reddit_fetcher=container.reddit_fetcher()` once gap 4 is implemented.

---

### 6. `PeriodicRedditScrapingTask` has no state persistence

**File:** `src/daemon/tasks/data_tasks.py`

`get_last_run()` always returns `None`. `record_success()` has `# TODO` comments. The interval-based dedup uses an in-memory class variable (`_last_run_time`) that resets on daemon restart.

**Fix:** Persist last run timestamp to daemon state (same pattern as `EarningsCalendarFetch` uses `state.set_last_earnings_fetch()`). Requires adding `get_last_reddit_scraping()` / `set_last_reddit_scraping()` to the state facade.

---

## Dependency Order for Fixes

```
6 (state persistence) — independent
4 + 5 (discovery engine stub) — independent, needs DB repos
1 + 2 + 3 (watcher wiring) — depends on 4 being implemented first
```

Gaps 4 and 6 can be done in parallel. Gap 1/2/3 should be done after gap 4 so the discovery_callback has something real to call.
