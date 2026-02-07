# AI Casino - Agentic Stock Trading System

Multi-agent stock trading system with technical analysis, sentiment analysis, and news analysis.

## Architecture

```
Technical Analyst →
Sentiment Analyst →  Trader Agent → Trading Decision (BUY/SELL/HOLD)
News Analyst     →
```

- **Technical Analyst**: RSI + MACD momentum indicators
- **Sentiment Analyst**: FinBERT sentiment on news
- **News Analyst**: LLM-powered news interpretation
- **Trader Agent**: Synthesizes all inputs for final decision

## Features

- 📊 Technical analysis (RSI, MACD via pandas-ta)
- 🧠 Sentiment analysis (FinBERT for financial text)
- 📰 News analysis (Marketaux API)
- 🤖 LLM-powered agents (Ollama/Claude/GPT via LiteLLM)
- ✅ Strict linting (ruff with comprehensive rules)
- 🧪 Full test coverage (pytest)
- 🔧 mise for dependency management

## Setup

### Prerequisites

- Python 3.12+
- [mise](https://mise.jdx.dev/) installed

### Installation

```bash
# Install tools (Python, uv, ruff, Ollama)
mise install

# Install Python dependencies
uv sync

# Copy environment template
cp .env.example .env

# Edit .env with your API keys
# Required: ALPHA_VANTAGE_API_KEY
# Optional: MARKETAUX_API_KEY, ALPACA_API_KEY
```

### LLM Setup

#### Option 1: Local Development (Ollama)

```bash
# Start Ollama server in background
mise ollama:start

# Check Ollama status
mise ollama:status

# Pull LLM model
ollama pull qwen3:14b

# In .env:
LLM_PROVIDER=ollama
LLM_MODEL=qwen3:14b

# Stop Ollama when done
mise ollama:stop
```

#### Option 2: Production (Claude)

```bash
# In .env:
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-20250514
ANTHROPIC_API_KEY=sk-ant-...
```

#### Option 3: Production (OpenAI)

```bash
# In .env:
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
OPENAI_API_KEY=sk-...
```

## Usage

### Analyze a Stock

```bash
python -m src.main AAPL
```

### Custom Period

```bash
python -m src.main TSLA --period 180
```

### View Execution Metrics

```bash
python -m src.main AAPL --metrics
```

### Output Example

```
Trading Analysis for AAPL

Technical Analysis:
Signal: BUY
RSI: 35.42
MACD Histogram: 0.2341
Confidence: 0.82

Sentiment Analysis:
Overall: positive
Score: 0.65
Articles: 10

Final Decision: BUY
Confidence: 0.85
Risk Level: LOW
```

## Trading Modes

### Paper Trading (Default)

Paper trading mode allows safe strategy testing without risking real capital:

```bash
# Set paper trading credentials (optional, falls back to live if not set)
export ALPACA_PAPER_API_KEY=<key>
export ALPACA_PAPER_SECRET_KEY=<secret>

# In daemon.toml
[daemon]
trading_mode = "paper"  # Default
auto_trade = true
```

### Live Trading Promotion

**IMPORTANT:** Live trading requires passing paper trading validation first. The system enforces minimum criteria before allowing live trades:

1. **Validate Readiness**
   ```bash
   ai-casino validate-paper-trading --config daemon.toml
   ```

   Checks:
   - Min 30 days paper trading
   - Min 20 executed trades
   - Sharpe ratio ≥ 0.5
   - Max drawdown ≤ 15%
   - Win rate ≥ 45%

2. **Configure Live Mode**
   ```bash
   # Set live credentials
   export ALPACA_API_KEY=<key>
   export ALPACA_SECRET_KEY=<secret>

   # In daemon.toml
   [daemon]
   trading_mode = "live"
   ```

3. **Start Daemon**
   ```bash
   # Requires validation to pass
   ai-casino daemon --config daemon.toml

   # Or bypass validation (advanced users only)
   ai-casino daemon --config daemon.toml --force-live
   ```

### Validation Criteria

Configure paper trading requirements in `daemon.toml`:

```toml
[daemon.paper_trading]
min_duration_days = 30        # Minimum paper trading period
min_trades = 20               # Minimum executed trades
min_sharpe = 0.5              # Minimum Sharpe ratio
max_drawdown_percent = 15.0   # Maximum drawdown allowed
min_win_rate = 0.45           # Minimum win rate (45%)
```

### Automated Notifications

When paper trading criteria are met, you'll receive a Telegram notification (if configured):

```
🎉 Paper Trading Validation Complete

Duration: 32 days
Trades: 25
Sharpe: 0.68
Max DD: 12.3%

✅ Ready for live trading promotion
Run: ai-casino validate-paper-trading to review
```

## Development

### Run Tests

```bash
mise test
```

### Run Tests with Coverage

```bash
mise test:cov
```

### Lint Code

```bash
mise lint
```

### Format Code

```bash
mise format
```

### Run All Checks

```bash
mise check
```

### Ollama Management

```bash
# Start Ollama server
mise ollama:start

# Check status
mise ollama:status

# Stop Ollama server
mise ollama:stop
```

## Project Structure

```
src/
├── agents/          # Trading agents
│   ├── technical.py
│   ├── sentiment.py
│   ├── news.py
│   └── trader.py
├── data/            # Data fetchers
│   ├── market.py
│   └── news.py
├── models/          # ML models
│   ├── llm.py
│   └── sentiment.py
├── strategies/      # Trading strategies
│   └── momentum.py
├── workflows/       # Agent orchestration
│   └── trading.py
└── main.py          # CLI entry point

tests/               # Full test suite
```

## API Keys

### Required

- **Alpha Vantage**: Free tier (500 calls/day) - [Get key](https://www.alphavantage.co/support/#api-key)

### Optional

- **Marketaux**: Free tier - [Get key](https://www.marketaux.com/)
- **Alpaca**: Paper trading - [Get key](https://alpaca.markets/)
- **Anthropic**: Claude API - [Get key](https://console.anthropic.com/)
- **OpenAI**: GPT API - [Get key](https://platform.openai.com/)

## Technology Stack

- **LLM**: LiteLLM (Ollama/Claude/GPT)
- **Sentiment**: FinBERT (HuggingFace)
- **Technical**: pandas-ta (150+ indicators)
- **Agents**: LangGraph
- **Market Data**: yfinance, Alpha Vantage
- **News**: Marketaux API
- **Testing**: pytest
- **Linting**: ruff (strict mode)
- **Tools**: mise, uv

## License

MIT
