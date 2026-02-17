# Adding a New Data Source to Constant Discovery

Guide for wiring any new data source (social media, news, alternative data) into the continuous discovery pipeline.

---

## How Discovery Works

The discovery pipeline continuously surfaces stock candidates from multiple sources and funnels them into the trading workflow:

```
Data Source (scraper / API / watcher)
  → DB storage (raw data)
    → EventWatcher (polls, detects signals)
      → SocialEvent / NewsEvent / ...
        → EventTriageAgent (LLM urgency scoring)
          → IMMEDIATE → discovery_callback → StockDiscoveryEngine
          → WATCHLIST → state.discovery.add_event_candidates()
            → StockDiscoveryEngine (merges + deduplicates candidates)
              → TradingWorkflow (full analysis → BUY/SELL/HOLD)
```

There are two entry points into `StockDiscoveryEngine`:
- **Event-driven** (via `EventWatcher`): signals detected in real time (volume spikes, viral posts, breaking news)
- **Pull-based** (via `_fetch_*_candidates()` methods): periodic polling of DB aggregates or external APIs

---

## Checklist: Adding a New Data Source

### 1. Data Collection Layer

Create a scraper or fetcher in `src/data/`:
- Use `async` for all I/O
- Return typed dataclasses (`RedditPost`, `NewsArticle`, etc.) — not dicts
- Include a `PeriodicXxxTask` in `src/daemon/tasks/` for scheduled execution (follow `EarningsCalendarFetch` as a pattern)
- Persist raw data to DB via repository classes in `src/database/repositories/`

**State persistence requirement:** `get_last_run()` must return a real timestamp from daemon state, not `None`. `record_success()` must persist. Without this, dedup resets on every daemon restart.

```python
# Pattern from EarningsCalendarFetch
async def get_last_run(self) -> datetime | None:
    return await self.components.state.get_last_xxx_fetch()

async def record_success(self, duration: float) -> None:
    await self.components.state.set_last_xxx_fetch(datetime.now(UTC))
```

---

### 2. Signal Detection Layer (`EventWatcher`)

Create a watcher in `src/daemon/watchers/` that extends `EventWatcher`:

```python
class XxxWatcher(EventWatcher):
    async def _fetch_events(self) -> list[BaseEvent]:
        # Query DB or API, detect signals, return events
        ...
```

Signal types to detect (examples from `SocialWatcher`):
- **Volume spike**: mention count grew ≥ threshold since last poll
- **Viral post**: high score + high upvote ratio + recent age
- **Trending**: sudden appearance in top N

**Window sizing:** use separate windows for different detection types. Volume spikes use the poll interval; viral/trending checks should use the full age window (e.g., 60 min to match `_check_viral_post`'s `age_seconds > 3600` guard).

Register the watcher in:
- `src/di/providers/watchers.py` — `create_xxx_watcher()` function
- `src/daemon/factory.py` → `_create_event_watchers()`
- `src/daemon/lifecycle.py` → `_start_watchers()`

---

### 3. Wiring to Discovery (critical — easy to miss)

When creating the watcher in `src/di/providers/watchers.py`, **always pass both**:

```python
return XxxWatcher(
    ...
    discovery_mode=config.event_watchers.xxx_use_discovery,   # read from config
    discovery_callback=discovery_engine.add_candidates,        # must not be None
)
```

**Without `discovery_mode=True`:** IMMEDIATE urgency events skip discovery and fire a direct `TradingWorkflow` — bypassing candidate deduplication and rate limiting.

**Without `discovery_callback`:** `_route_to_discovery()` in the base class silently no-ops. No error, no log — candidates are dropped.

Add the config flag to `EventWatcherIntegrationConfig` in `src/daemon/config/events.py`:

```python
xxx_use_discovery: bool = True
```

And expose it in `docs/daemon.yaml.example`.

---

### 4. Pull-Based Discovery (optional, for DB aggregates)

If the data source populates DB tables with aggregated signals (e.g., `reddit_ticker_sentiment`), implement a fetch method in `StockDiscoveryEngine`:

```python
async def _fetch_xxx_candidates(self) -> list[DiscoveryCandidate]:
    mentions = await self._xxx_repo.get_mentions_in_window(window_minutes=60)
    return [
        DiscoveryCandidate(
            symbol=symbol,
            source=DiscoverySource.XXX,
            confidence=self._compute_confidence(count),
            discovery_timestamp=datetime.now(UTC),
        )
        for symbol, count in mentions
        if count >= self._config.min_mentions
    ]
```

Wire the repository into `StockDiscoveryEngine` via `OptionalServices` in `src/daemon/factory.py` → `_create_discovery_engine()`.

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| `discovery_callback=None` | IMMEDIATE candidates silently dropped, no error | Pass `discovery_engine.add_candidates` |
| `discovery_mode=False` (default) | IMMEDIATE events trigger direct TradingWorkflow, bypassing dedup | Pass `discovery_mode=True` |
| Config flag declared but not read | Changing config has no effect | Read flag in provider function, not just declare it |
| `get_last_run()` returns `None` | Interval dedup resets on restart, scraper runs too frequently | Persist to daemon state |
| Viral window = poll interval | Posts 16–60 min old never checked for viral status | Use `viral_window = 60` separate from `poll_window` |
| `on_conflict_do_nothing` on scores | Stale scores in DB, viral detection misses posts that gained score | Use `on_conflict_do_update` for mutable fields (score, upvote_ratio) |

---

## Reference Implementations

| Component | Good example |
|-----------|-------------|
| Periodic scraping task | `EarningsCalendarFetch` in `src/daemon/tasks/data_tasks.py` |
| EventWatcher with signal detection | `SocialWatcher` in `src/daemon/watchers/social_watcher.py` |
| DB repository with upsert | `RedditPostRepository.bulk_insert()` in `src/database/repositories/reddit.py` |
| Pull-based discovery fetch | `StockDiscoveryEngine._fetch_news_candidates()` |
| DI wiring for watcher | `create_news_trending_watcher()` in `src/di/providers/watchers.py` |
