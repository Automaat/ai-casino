# Scripts

## profile_daemon.py

Profile daemon cycle to identify performance bottlenecks.

### Usage

```bash
# Profile with default stocks (AAPL, TSLA, GOOGL, MSFT, NVDA)
python scripts/profile_daemon.py

# Profile custom stocks
python scripts/profile_daemon.py --stocks "AAPL,MSFT,AMZN"

# Adjust concurrency
python scripts/profile_daemon.py --max-concurrent 5

# Skip creating GitHub issues
python scripts/profile_daemon.py --no-issues
```

### Prerequisites

- API keys configured (Alpha Vantage, LLM provider)
- LLM provider configured (Ollama/Anthropic/OpenAI)
- `gh` CLI installed (for issue creation)

### Output

1. **Console output**: Rich tables showing top 10 bottlenecks and category breakdown
2. **JSON report**: `~/.ai-casino/profiles/benchmark/bottleneck_report_*.json`
3. **Profile data**: `~/.ai-casino/profiles/benchmark/YYYY-MM-DD/cycle_*.pstats`
4. **GitHub issues**: Created for top 3 bottlenecks (unless `--no-issues`)

### Categories

- `llm_api`: LLM API calls (Anthropic, OpenAI, Ollama)
- `finbert`: FinBERT sentiment analysis
- `market_data`: Market data fetching (Alpha Vantage, yfinance)
- `database`: Database operations (SQLite)
- `technical_indicators`: pandas-ta calculations
- `news_data`: News fetching
- `other`: Uncategorized functions

### Example Report

```
Top 10 Bottlenecks
┌──────┬───────────────────┬────────────────┬────────────┬───────┬──────────┬─────────────┐
│ Rank │ Category          │ Cumulative Time│ % of Total │ Calls │ Per Call │ Function    │
├──────┼───────────────────┼────────────────┼────────────┼───────┼──────────┼─────────────┤
│ 1    │ llm_api           │ 45.234s        │ 62.3%      │ 15    │ 3.0156s  │ anthropic...│
│ 2    │ finbert           │ 12.456s        │ 17.1%      │ 120   │ 0.1038s  │ torch...    │
│ 3    │ market_data       │ 8.123s         │ 11.2%      │ 25    │ 0.3249s  │ yfinance... │
└──────┴───────────────────┴────────────────┴────────────┴───────┴──────────┴─────────────┘
```

### Related

- Issue #520: Profile daemon to identify bottlenecks
- `src/daemon/profiling/`: Profiling infrastructure
