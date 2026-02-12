# NewsWatcher Load Test Guide

1-hour stability test for memory leaks, rate limit handling, and error recovery.

---

## Objective

Verify NewsWatcher stability under aggressive configuration:

- **Memory growth** <50MB over 1 hour
- **Rate limit handling** graceful degradation (no crashes)
- **Error recovery** continues after transient failures
- **No crashes** for entire test duration

---

## Setup

### Test Configuration

Create `~/.ai-casino/daemon-loadtest.yaml`:

```yaml
daemon:
  watchlist:
    - AAPL
    - TSLA
    - GOOGL
    - MSFT
    - AMZN  # 5 symbols
  interval_minutes: 60
  market_hours_only: false
  auto_trade: false

  news_watcher:
    enabled: true
    poll_interval_minutes: 1  # 1 minute (very aggressive)
    relevance_threshold: 0.5  # Lower threshold (more events)
    cooldown_minutes: 10  # Short cooldown
    breaking_threshold_minutes: 30  # Longer window
    max_concurrent_analyses: 3

  api_keys:
    alpha_vantage_api_key: "YOUR_KEY"
    marketaux_api_key: "YOUR_KEY"  # Premium tier recommended
    finnhub_api_key: "YOUR_KEY"
    newsdata_api_key: "YOUR_KEY"
    anthropic_api_key: "YOUR_KEY"

  database:
    enable_persistence: false

  api:
    enabled: false
```

**Load test parameters:**

- `poll_interval_minutes: 1` - Poll every minute (60 fetches/hour)
- `5 symbols` - Larger watchlist for more analysis
- `relevance_threshold: 0.5` - More events analyzed
- All 4 news sources enabled

---

## Execution

### 1. Baseline Memory

```bash
# Start daemon
python -m src.cli.app daemon --config ~/.ai-casino/daemon-loadtest.yaml

# In another terminal, capture baseline
ps aux | grep "daemon" | grep -v grep | awk '{print "Initial RSS:", $6/1024, "MB"}'
```

**Record:** Initial memory (typically 150-200 MB)

### 2. Run for 1 Hour

Let daemon run unattended for 60 minutes.

### 3. Periodic Memory Checks

```bash
# At 15 min
ps aux | grep "daemon" | grep -v grep | awk '{print "15min RSS:", $6/1024, "MB"}'

# At 30 min
ps aux | grep "daemon" | grep -v grep | awk '{print "30min RSS:", $6/1024, "MB"}'

# At 45 min
ps aux | grep "daemon" | grep -v grep | awk '{print "45min RSS:", $6/1024, "MB"}'

# At 60 min
ps aux | grep "daemon" | grep -v grep | awk '{print "60min RSS:", $6/1024, "MB"}'
```

### 4. Monitor Logs

```bash
# Tail logs in real-time
tail -f ~/.ai-casino/worker.log | grep -E "ERROR|WARNING|Breaking|EVENT SIGNAL"
```

---

## Verification Metrics

### ✅ 1. Memory Growth

**Pass criteria:** Memory growth <50 MB over 1 hour

**Verify:**

```bash
# Calculate delta
# Expected: 150MB initial → <200MB final
```

**If memory grows >50MB:**

- Check for memory leaks in _seen_urls dict
- Verify garbage collection of old Event objects
- Review LLM client connection pooling

### ✅ 2. Rate Limit Handling

**Pass criteria:** Rate limit errors logged but daemon continues

**Verify:**

```bash
grep -c "429" ~/.ai-casino/worker.log  # Count rate limit hits
grep -c "fetch failed" ~/.ai-casino/worker.log  # Graceful degradation
```

**Expected:**

- 0-5 rate limit errors (depends on API tier)
- Errors logged with `logger.warning` (not `error`)
- Fetcher continues polling after 429

**If excessive 429 errors:**

- Increase `poll_interval` to 120s
- Upgrade Marketaux/Finnhub API tier

### ✅ 3. Error Recovery

**Pass criteria:** Transient errors don't crash daemon

**Verify:**

```bash
grep "Event triage failed" ~/.ai-casino/worker.log
grep "Analysis task failed" ~/.ai-casino/worker.log
```

**Expected:**

- Triage failures logged, cycle continues
- Partial analysis results processed
- No daemon crash

### ✅ 4. No Crashes

**Pass criteria:** Daemon runs for full 60 minutes without exit

**Verify:**

```bash
# Check if process still running
ps aux | grep daemon

# Verify continuous polling
grep -c "No new events" ~/.ai-casino/worker.log  # Should be ~30-40 cycles
```

### ✅ 5. Event Processing

**Pass criteria:** Events processed correctly throughout test

**Verify:**

```bash
# Count breaking news detected
grep -c "Breaking from" ~/.ai-casino/worker.log

# Count signals emitted
grep -c "EVENT SIGNAL DETECTED" ~/.ai-casino/worker.log

# Verify variety
grep -E "AAPL|TSLA|GOOGL|MSFT|AMZN" ~/.ai-casino/worker.log | wc -l
```

**Expected:**

- 5-20 breaking news events (varies by market activity)
- 1-5 EVENT SIGNALS emitted
- Multiple symbols analyzed (not just one stuck symbol)

---

## Success Criteria

**All checks pass:**

- ✅ Memory growth <50 MB
- ✅ Rate limits handled gracefully
- ✅ Errors recovered, no crashes
- ✅ Daemon runs full 60 minutes
- ✅ Events processed continuously

**Example successful run:**

```
Initial RSS: 185 MB
60min RSS: 220 MB  (35 MB growth - PASS)

Rate limit errors: 2 (Marketaux 429 x2)
Analysis failures: 0
Daemon crashes: 0

Breaking news detected: 12
EVENT SIGNALS emitted: 3
Symbols analyzed: AAPL (2x), TSLA (1x), GOOGL (1x)
```

---

## Troubleshooting

### Memory Growth >50 MB

**Diagnosis:**

```bash
# Check _seen_urls dict size (should not grow unbounded)
grep "Dedup" ~/.ai-casino/worker.log | tail -20

# Check for leaked references
# (Would require Python profiler like memory_profiler)
```

**Fix:**

- Implement _seen_urls LRU eviction (e.g., keep last 1000 URLs)
- Add periodic cleanup task

### Excessive Rate Limits

**Diagnosis:**

```bash
grep "429" ~/.ai-casino/worker.log | head
```

**Fix:**

- Reduce poll frequency: `poll_interval: 120` (2 min)
- Disable lower-tier sources: Remove newsdata/duckduckgo
- Upgrade API tier (Marketaux Pro)

### Daemon Crash

**Diagnosis:**

```bash
# Check last error before crash
tail -100 ~/.ai-casino/worker.log | grep ERROR

# Check for uncaught exceptions
grep "Traceback" ~/.ai-casino/worker.log
```

**Fix:**

- File bug report with full traceback
- Add try/except around identified crash point
- Reduce `max_concurrent_analyses` to 1

### No Events Processed

**Diagnosis:**

```bash
grep "Breaking from" ~/.ai-casino/worker.log  # Empty?
grep "fetch failed" ~/.ai-casino/worker.log  # All sources failing?
```

**Fix:**

- Verify API keys: Check config
- Verify API keys in daemon config YAML
- Lower `breaking_threshold_minutes: 60`

---

## Cleanup

```bash
# Stop daemon (Ctrl+C)

# Archive load test logs
mkdir -p ~/.ai-casino/loadtest-logs
mv ~/.ai-casino/worker.log ~/.ai-casino/loadtest-logs/loadtest-$(date +%Y%m%d-%H%M%S).log

# Generate summary report
echo "Load Test Summary" > ~/.ai-casino/loadtest-logs/summary.txt
grep -c "Breaking from" ~/.ai-casino/loadtest-logs/loadtest-*.log >> ~/.ai-casino/loadtest-logs/summary.txt
grep -c "429" ~/.ai-casino/loadtest-logs/loadtest-*.log >> ~/.ai-casino/loadtest-logs/summary.txt
```

---

## Production Recommendations

After successful load test:

1. **Reduce poll frequency:** `poll_interval: 300` (5 min) for production
2. **Raise relevance threshold:** `relevance_threshold: 0.7` (filter noise)
3. **Enable persistence:** `database.enable_persistence: true`
4. **Monitor metrics:** Set up Prometheus/Grafana for RSS tracking
5. **Alert on 429:** Configure alerting for sustained rate limit errors

---

## Next Steps

- Deploy to production with conservative config
- Monitor first 24 hours closely
- Tune thresholds based on event volume
- Schedule periodic load tests (monthly)
