# NewsWatcher E2E Manual Test Guide

End-to-end verification of NewsWatcher breaking news detection, triage, and analysis pipeline.

---

## Prerequisites

- **API Keys:**
  - `ALPHA_VANTAGE_API_KEY` (optional, but recommended)
  - `MARKETAUX_API_KEY` (for premium news source)
  - `FINNHUB_API_KEY` (alternative source)
  - `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` (for LLM triage)

- **Paper Trading Account:**
  - Alpaca paper trading API key/secret (for workflow analysis)

- **System:**
  - Python 3.14+
  - All dependencies installed (`mise install`)

---

## Test Configuration

Create test config at `~/.ai-casino/daemon-test.yaml`:

```yaml
daemon:
  watchlist:
    - AAPL
    - TSLA
    - NVDA
  interval_minutes: 60
  market_hours_only: false
  auto_trade: false

  news_watcher:
    enabled: true
    poll_interval_minutes: 2  # 2 minutes (aggressive for testing)
    relevance_threshold: 0.7
    cooldown_minutes: 30
    breaking_threshold_minutes: 15
    max_concurrent_analyses: 2

  api_keys:
    alpha_vantage_api_key: "YOUR_KEY"
    marketaux_api_key: "YOUR_KEY"
    finnhub_api_key: "YOUR_KEY"
    anthropic_api_key: "YOUR_KEY"

  database:
    enable_persistence: false

  api:
    enabled: false
```

**Config notes:**

- `poll_interval_minutes: 2` - Check every 2 minutes (vs default 5 min)
- `breaking_threshold_minutes: 15` - Only news <15min old
- `market_hours_only: false` - Run 24/7 for testing
- `auto_trade: false` - Paper analysis only, no actual trades

---

## Execution Steps

### 1. Start Daemon

```bash
# Clean logs
rm -f ~/.ai-casino/worker.log

# Start daemon (Ctrl+C to stop)
python -m src.cli.app daemon --config ~/.ai-casino/daemon-test.yaml
```

**Expected output:**

```
Daemon started
Watchlist: AAPL, TSLA, NVDA
Interval: 60 minutes
...
Event watchers: 1 active
NewsWatcher Started
Poll interval: 120s
```

### 2. Monitor Logs

```bash
# In another terminal, tail worker logs
tail -f ~/.ai-casino/worker.log | grep NewsWatcher

# Or filter for key events
tail -f ~/.ai-casino/worker.log | grep -E "Breaking|EVENT SIGNAL|NewsWatcher"
```

### 3. Wait for Breaking News

NewsWatcher polls every 2 minutes. Watch for:

```log
2026-02-12 10:05:23 | INFO | Breaking from marketaux: Apple announces new iPhone... (8.3m)
2026-02-12 10:05:24 | INFO | Found 1 new event(s)
2026-02-12 10:05:25 | INFO | Found 1 high-relevance event(s)
2026-02-12 10:05:26 | INFO | Analyzing 1 symbols: ['AAPL']
```

### 4. Verify Event Signal

If breaking news is relevant (relevance ≥0.7, urgency=IMMEDIATE):

```log
═══ EVENT SIGNAL DETECTED ═══
Event Type: news
Source: marketaux
Relevance: 0.92
Urgency: IMMEDIATE
Sentiment: BULLISH
Reasoning: Major product launch announcement...

Analyzed Stocks (1):
  AAPL: BUY (confidence: 0.78)
═══════════════════════════════
```

---

## Verification Checklist

### ✅ 1. Breaking News Detection

- [ ] NewsWatcher starts polling every 2 minutes
- [ ] Breaking keywords detected (e.g., "breaking", "announces", "earnings")
- [ ] Only recent articles (<15 min) processed
- [ ] Old articles filtered out

**Verify:**

```bash
grep -c "Breaking from" ~/.ai-casino/worker.log  # Should be >0
grep -c "No new events" ~/.ai-casino/worker.log  # Many cycles skip
```

### ✅ 2. Multi-Source Deduplication

- [ ] Articles fetched from multiple sources (Marketaux, Finnhub, etc.)
- [ ] Duplicate URLs deduplicated by highest weight source
- [ ] _seen_urls prevents re-processing same article

**Verify:**

```bash
grep "Dedup: prefer" ~/.ai-casino/worker.log  # Should show source preference
```

### ✅ 3. LLM Triage

- [ ] EventTriageAgent analyzes breaking news
- [ ] Extracts symbols, relevance, urgency, sentiment
- [ ] Low-relevance events skipped (relevance <0.7)

**Verify:**

```bash
grep "high-relevance event" ~/.ai-casino/worker.log
```

### ✅ 4. Cooldown

- [ ] After analyzing AAPL, it enters 30-minute cooldown
- [ ] Subsequent AAPL news skipped during cooldown
- [ ] Other symbols (TSLA, NVDA) still analyzed

**Verify:**

```bash
grep "in cooldown" ~/.ai-casino/worker.log
grep "skipped (in cooldown)" ~/.ai-casino/worker.log
```

### ✅ 5. Concurrent Analysis

- [ ] Max 2 symbols analyzed concurrently (`max_concurrent_analyses`)
- [ ] Semaphore prevents analysis storm

**Verify:**

```bash
grep "Analyzing.*symbols" ~/.ai-casino/worker.log
# Should never see >2 symbols in one batch
```

### ✅ 6. Graceful Shutdown

- [ ] Ctrl+C triggers graceful shutdown
- [ ] Watchers stopped cleanly
- [ ] No exceptions during shutdown

**Verify:**

```bash
# Press Ctrl+C
# Check logs:
grep "Stopping event watchers" ~/.ai-casino/worker.log
grep "NewsWatcher Stopped" ~/.ai-casino/worker.log
grep "Daemon shutdown complete" ~/.ai-casino/worker.log
```

---

## Success Criteria

**All verification points pass:**

- ✅ Breaking news detected and logged
- ✅ Deduplication works (no duplicate analysis)
- ✅ Triage extracts correct symbols
- ✅ Cooldown prevents re-analysis
- ✅ Graceful shutdown (no errors)

**Example successful run:**

- 30-minute test window
- 15 poll cycles (every 2 min)
- 1-3 breaking news events detected
- 1-2 EVENT SIGNALS emitted
- 0 rate limit errors
- Clean shutdown

---

## Troubleshooting

### No Breaking News Detected

- **Check news sources:** Verify API keys in config
- **Lower threshold:** Set `breaking_threshold_minutes: 30` (more lenient)
- **Check keywords:** Look at fetched articles in logs, verify titles match `BREAKING_KEYWORDS`

### Rate Limit Errors (429)

```bash
grep "429" ~/.ai-casino/worker.log
```

**Fix:** Increase `poll_interval` to 300s (5 min) or use higher API tier

### Triage Failures

```bash
grep "Event triage failed" ~/.ai-casino/worker.log
```

**Fix:** Check LLM provider status, verify API key

### High Memory Usage

```bash
ps aux | grep daemon  # Check RSS memory
```

**Fix:** Reduce `max_concurrent_analyses` to 1

---

## Cleanup

```bash
# Stop daemon (Ctrl+C)

# Archive test logs
mv ~/.ai-casino/worker.log ~/.ai-casino/worker-test-$(date +%Y%m%d-%H%M%S).log

# Reset config
# (Keep daemon-production.yaml for production use)
```

---

## Next Steps

After successful E2E test:

1. Review [Load Test Guide](./news-watcher-load-test.md) for stability testing
2. Configure production settings in `daemon-production.yaml`
3. Enable auto-trading if desired (`auto_trade: true`)
