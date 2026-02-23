# v1 Watchers

All watchers extend `PeriodicWatcher` (polls on fixed interval with 1s-granularity sleep for responsive shutdown).

Events flow through `EventTriagePipeline` → LLM triage → `MarketEventQueue` (IMMEDIATE) or discovery candidates (WATCHLIST).

---

## Watchers

### NewsWatcher

Polls financial news sources for breaking news. Filters by recency (default: 15min window) and keyword detection (earnings, FDA, merger, lawsuit, etc.). Deduplicates by URL, preferring higher-weighted sources (marketaux > finnhub > newsdata > duckduckgo).

- **Poll interval:** 300s (5min)
- **Sources:** Marketaux (fallback), Finnhub, NewsData, DuckDuckGo
- **Produces events:** yes → `NewsEvent` (type: `news`)

---

### TrumpWatcher

Monitors Trump's Truth Social feed for new posts. Deduplicates via rolling post ID window (500 entries). On first run fetches last 1h; subsequent runs fetch since last check.

- **Poll interval:** 300s (5min)
- **Source:** TruthSocialFetcher
- **Produces events:** yes → `TrumpEvent` (type: `trump`)

---

### SocialWatcher

Monitors Reddit communities (default: wallstreetbets, stocks) for two signal types:
- **Volume spikes:** ≥50% increase in symbol mentions between polls
- **Viral posts:** <1h old, score >1000, upvote ratio >80%

Primary: queries Reddit DB. Fallback: Reddit API via RedditFetcher.

- **Poll interval:** 900s (15min)
- **Source:** Reddit DB / RedditFetcher API
- **Produces events:** yes → `SocialEvent` (type: `social`)

---

### AnomalyWatcher

Detects market data anomalies across a configured watchlist. Uses round-robin rotation (default: 5 symbols/cycle) to stay within Alpha Vantage rate limits. Detects per symbol:
- **Volume spike:** current volume ≥ 2x 20-day average
- **Price move:** intraday change ≥ 5% from open
- **Gap:** open vs previous close ≥ 3%

- **Poll interval:** 900s (15min)
- **Source:** Alpha Vantage intraday + daily via MarketDataFetcher
- **Produces events:** yes → `AnomalyEvent` (type: `anomaly`, subtypes: `volume_spike`, `price_move`, `gap`)

---

### NewsTrendingWatcher

Discovers new trading candidates via web search. Runs configurable queries (default: "trending stocks today", "hot stocks right now", "stock market movers"), extracts ticker mentions using regex, tracks rolling mention frequency, and fires when a symbol exceeds `min_mention_threshold` (default: 3) in the trailing window.

- **Poll interval:** 600s (10min)
- **Source:** WebSearchFetcher
- **Produces events:** yes → `NewsTrendingEvent` (type: `news_trending`)

---

### EconomicCalendarWatcher

Polls FRED economic calendar. Filters to US HIGH/MEDIUM impact events in a 24h lookahead window. Computes a risk signal with trading recommendation:
- HIGH impact within 2h → `AVOID_NEW_POSITIONS`
- HIGH impact within 24h → `REDUCE_SIZE`
- MEDIUM impact within 4h → `REDUCE_SIZE`
- Otherwise → `TRADE_NORMALLY`

Signal is exposed via `current_signal` property (sync) — consumed by coordinator tools, not pushed to event queue.

- **Poll interval:** 3600s (60min)
- **Source:** EconomicCalendarFetcher (FRED)
- **Produces events:** no — updates `_current_signal` in-memory only

---

### OptionsFlowWatcher

Polls yfinance options chains for a configured symbol list. Computes per-symbol signals:
- Put/call ratio
- Volume spike vs 5-session rolling average
- Block trade detection (premium ≥ $100k)
- Net premium direction (BULLISH/BEARISH/NEUTRAL)
- Composite significance score (40% vol spike, 30% block trades, 30% PCR)

Signal is exposed via `get_signal(symbol)` (sync) — consumed by coordinator tools, not pushed to event queue.

- **Poll interval:** 900s (15min)
- **Source:** OptionsFlowFetcher (yfinance)
- **Produces events:** no — updates `_signals` dict in-memory only

---

### SocialSentimentWatcher

Aggregates retail sentiment from two sources per symbol: ApeWisdom (trending rank + mention delta vs 24h ago) and Reddit DB (hourly sentiment aggregates). Computes composite signal: direction (BULLISH/BEARISH/NEUTRAL), buzz score, significance score, and trending status.

Signal is exposed via `get_signal(symbol)` (sync) — consumed by coordinator tools, not pushed to event queue.

- **Poll interval:** 1800s (30min)
- **Sources:** ApeWisdomFetcher + Reddit DB (`RedditTickerSentimentORM`)
- **Produces events:** no — updates `_signals` dict in-memory only

---

## Summary

| Watcher | Poll | Produces Events | Event Type |
|---|---|---|---|
| NewsWatcher | 5min | yes | `NewsEvent` |
| TrumpWatcher | 5min | yes | `TrumpEvent` |
| SocialWatcher | 15min | yes | `SocialEvent` |
| AnomalyWatcher | 15min | yes | `AnomalyEvent` |
| NewsTrendingWatcher | 10min | yes | `NewsTrendingEvent` |
| EconomicCalendarWatcher | 60min | no | — (in-memory signal) |
| OptionsFlowWatcher | 15min | no | — (in-memory signal) |
| SocialSentimentWatcher | 30min | no | — (in-memory signal) |
