# 🤖 Agentic Stock Trading System - Research Guide

**Date:** 2026-01-05
**Tags:** #research #trading #ai #agents #python
**Focus:** Open source/cheap components for technical analysis + news sentiment trading bot

---

## 📊 Summary

Building an agentic trading system requires:
1. **Multi-agent architecture** - Specialized agents for different analysis types
2. **Technical analysis engine** - Price/volume indicators (RSI, MACD, etc.)
3. **Sentiment analysis** - News/social media NLP processing
4. **Paper trading** - Simulated execution for testing
5. **LLM backbone** - Decision synthesis (local or cheap API)

---

## 🏗️ Architecture Overview

### Recommended Multi-Agent Pattern

Based on [TradingAgents framework](https://github.com/TauricResearch/TradingAgents):

```
┌─────────────────────────────────────────────────────────┐
│                    ANALYST TEAM                         │
├─────────────┬─────────────┬─────────────┬──────────────┤
│ Technical   │ Sentiment   │ News        │ Fundamental  │
│ Analyst     │ Analyst     │ Analyst     │ Analyst      │
│ (RSI,MACD)  │ (FinBERT)   │ (Headlines) │ (Financials) │
└──────┬──────┴──────┬──────┴──────┬──────┴───────┬──────┘
       │             │             │              │
       ▼             ▼             ▼              ▼
┌─────────────────────────────────────────────────────────┐
│                   RESEARCHER TEAM                        │
├────────────────────────┬────────────────────────────────┤
│    Bullish Researcher  │    Bearish Researcher          │
│    (Optimistic view)   │    (Risk-focused view)         │
└───────────────┬────────┴──────────────┬─────────────────┘
                │                       │
                ▼                       ▼
┌─────────────────────────────────────────────────────────┐
│                    TRADER AGENT                          │
│         Synthesizes debates → Trading decision           │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│               RISK MANAGEMENT AGENT                      │
│         Position sizing, stop-loss, approval             │
└─────────────────────────────────────────────────────────┘
```

[Source: TradingAgents GitHub - "Comprises both bullish and bearish researchers who critically assess the insights provided by the Analyst Team"]

---

## 🔧 Component Breakdown

### 1. 📈 Technical Analysis Engine

| Library | Indicators | Notes |
|---------|-----------|-------|
| **[pandas-ta](https://pypi.org/project/pandas-ta/)** | 150+ | "contains more than 150 indicators and utilities" ⭐ Recommended |
| [ta](https://github.com/bukosabino/ta) | 40+ | Simpler, pandas-native |
| [TA-Lib](https://ta-lib.org/) | 200+ | C library, requires compilation |

**Key indicators to implement:**
- 📊 **Trend:** SMA, EMA, MACD, ADX
- 📈 **Momentum:** RSI, Stochastic, Williams %R
- 📉 **Volatility:** Bollinger Bands, ATR
- 📦 **Volume:** OBV, Volume SMA

**Example Strategy Definition:**
```python
import pandas_ta as ta

CustomStrategy = ta.Strategy(
    name="Momo and Volatility",
    ta=[
        {"kind": "sma", "length": 50},
        {"kind": "sma", "length": 200},
        {"kind": "bbands", "length": 20},
        {"kind": "rsi"},
        {"kind": "macd", "fast": 8, "slow": 21},
    ]
)
df.ta.strategy(CustomStrategy)
```

[Source: pandas-ta PyPI documentation]

---

#### 🔀 Multi-Strategy & Real-Time Optimization

**Yes, you can run multiple strategies in parallel and tune them live!**

##### Pattern 1: Strategy Ensemble

[Source: arXiv - Deep RL Ensemble](https://arxiv.org/html/2511.12120v1)

> "Ensemble strategy integrates best features of multiple algorithms, robustly adjusting to different market situations"

```python
from dataclasses import dataclass
from typing import Callable
import optuna

@dataclass
class Strategy:
    name: str
    params: dict
    signal_fn: Callable  # Returns BUY/SELL/HOLD
    weight: float = 1.0

# Run multiple strategies in parallel
strategies = [
    Strategy("momentum", {"rsi_period": 14, "threshold": 30}, momentum_signal),
    Strategy("mean_reversion", {"bb_period": 20, "std": 2}, mean_reversion_signal),
    Strategy("trend_following", {"sma_fast": 50, "sma_slow": 200}, trend_signal),
]

def ensemble_decision(strategies, market_data):
    votes = {}
    for s in strategies:
        signal = s.signal_fn(market_data, s.params)
        votes[signal] = votes.get(signal, 0) + s.weight
    return max(votes, key=votes.get)  # Weighted majority
```

##### Pattern 2: Dynamic Parameter Tuning with Optuna

[Source: Optuna Docs](https://optuna.org/)

> "Efficiently search large spaces and prune unpromising trials"

```python
import optuna

def optimize_strategy(trial):
    # Define search space
    rsi_period = trial.suggest_int("rsi_period", 7, 21)
    rsi_oversold = trial.suggest_int("rsi_oversold", 20, 40)
    rsi_overbought = trial.suggest_int("rsi_overbought", 60, 80)

    # Backtest with these params
    result = backtest(rsi_period, rsi_oversold, rsi_overbought)
    return result.sharpe_ratio

# Find optimal params
study = optuna.create_study(direction="maximize")
study.optimize(optimize_strategy, n_trials=100)

best_params = study.best_params  # Use in live trading
```

##### Pattern 3: Live A/B Testing (Paper Trading)

```python
class StrategyABTest:
    def __init__(self, strategy_a, strategy_b, allocation=0.5):
        self.strategies = {"A": strategy_a, "B": strategy_b}
        self.allocation = allocation  # % of capital to A
        self.performance = {"A": [], "B": []}

    def execute(self, market_data):
        for name, strategy in self.strategies.items():
            signal = strategy.decide(market_data)
            # Paper trade with allocated capital
            result = paper_trade(signal, self.get_allocation(name))
            self.performance[name].append(result)

    def get_winner(self):
        sharpe_a = calculate_sharpe(self.performance["A"])
        sharpe_b = calculate_sharpe(self.performance["B"])
        return "A" if sharpe_a > sharpe_b else "B"
```

##### Pattern 4: Adaptive Strategy Selection (RL-based)

[Source: Springer - Dynamic Stock Decision](https://link.springer.com/article/10.1007/s10489-022-03606-0)

> "Dynamically select agents according to current market situation"

```python
class AdaptiveStrategySelector:
    """Switch strategies based on market regime"""

    def __init__(self, strategies: dict):
        self.strategies = strategies  # {"trending": ..., "ranging": ...}
        self.regime_detector = MarketRegimeDetector()

    def select_strategy(self, market_data):
        regime = self.regime_detector.detect(market_data)
        # High volatility → use mean reversion
        # Trending market → use momentum
        if regime == "high_volatility":
            return self.strategies["mean_reversion"]
        elif regime == "trending":
            return self.strategies["trend_following"]
        return self.strategies["default"]
```

##### Frameworks Supporting Multi-Strategy

| Framework | Multi-Strategy | Live Optimization | Notes |
|-----------|---------------|-------------------|-------|
| **[Jesse](https://jesse.trade/)** | ✅ | ✅ Optuna | "Batch backtests, compare across strategies" |
| **[Freqtrade](https://www.freqtrade.io/)** | ✅ | ✅ ML-based | "Strategy optimization by machine learning" |
| **[Backtrader](https://www.backtrader.com/)** | ✅ | Manual | Multiple strategies, data feeds |
| **[QSTrader](https://github.com/mhallsmoore/qstrader)** | ✅ | Manual | Portfolio-level strategies |

**Recommendation:** Start with static ensemble → Add Optuna for param tuning → Graduate to adaptive selection

---

#### 🎯 Pattern 5: Meta-Agent Strategy Selection (Manager Agent)

**Yes! Separate evaluation agents + one "manager" agent is a proven architecture.**

[Source: MetaTrader Paper](https://arxiv.org/pdf/2210.01774)

> "MetaTrader learns a meta-policy to select suitable policies to execute from the learned trading policy set, conditioned on current market condition"

##### Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    STRATEGY AGENTS (parallel)               │
├─────────────────┬─────────────────┬─────────────────────────┤
│  Momentum Agent │ Mean Rev Agent  │  Trend Agent            │
│  - RSI/MACD     │ - Bollinger     │  - SMA cross            │
│  - Output:      │ - Output:       │  - Output:              │
│    signal +     │   signal +      │    signal +             │
│    confidence   │   confidence    │    confidence           │
│    + reasoning  │   + reasoning   │    + reasoning          │
└────────┬────────┴────────┬────────┴────────────┬────────────┘
         │                 │                     │
         ▼                 ▼                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    META-AGENT (Manager)                      │
│  Inputs:                                                     │
│  - All strategy signals + confidence + reasoning             │
│  - Recent performance metrics per strategy                   │
│  - Current market regime                                     │
│                                                              │
│  Decision criteria:                                          │
│  - Which strategy fits current market?                       │
│  - Historical accuracy of each strategy                      │
│  - Risk-adjusted performance                                 │
│                                                              │
│  Output: Final trading decision + allocation                 │
└─────────────────────────────────────────────────────────────┘
```

##### What Meta-Agent Can Evaluate

| Criterion | Description | Source |
|-----------|-------------|--------|
| **Sharpe Ratio** | Risk-adjusted return per strategy | Rolling window |
| **Win Rate** | % profitable trades recently | Trade history |
| **Drawdown** | Max loss from peak | Portfolio tracker |
| **Confidence Score** | Strategy's self-reported certainty | Agent output |
| **Market Regime Match** | Does strategy fit current conditions? | Regime detector |
| **Correlation** | Avoid strategies with correlated signals | Statistical |

[Source: Macrosynergy](https://macrosynergy.com/research/measuring-value-added-of-algorithmic-trading-strategies/)

> "Criteria include statistical metrics like forecast accuracy, as well as trading performance like moving average of trade PNL"

##### Implementation

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class StrategyOutput:
    name: str
    signal: Literal["BUY", "SELL", "HOLD"]
    confidence: float  # 0.0 - 1.0
    reasoning: str
    predicted_return: float

@dataclass
class StrategyPerformance:
    name: str
    sharpe_30d: float
    win_rate_30d: float
    max_drawdown: float
    recent_accuracy: float  # How often signal was correct

class MetaAgent:
    """LLM-powered manager that picks best strategy"""

    def __init__(self, llm, strategy_agents: list):
        self.llm = llm
        self.strategy_agents = strategy_agents
        self.performance_tracker = PerformanceTracker()

    def decide(self, market_data: dict) -> dict:
        # 1. Get signals from all strategy agents (parallel)
        outputs: list[StrategyOutput] = []
        for agent in self.strategy_agents:
            output = agent.analyze(market_data)
            outputs.append(output)

        # 2. Get historical performance
        performances = self.performance_tracker.get_metrics()

        # 3. Detect market regime
        regime = self.detect_regime(market_data)

        # 4. Ask LLM meta-agent to pick best
        prompt = f"""
You are a portfolio manager selecting the best trading strategy.

MARKET REGIME: {regime}

STRATEGY OUTPUTS:
{self._format_outputs(outputs)}

RECENT PERFORMANCE (30 days):
{self._format_performance(performances)}

SELECTION CRITERIA:
1. Which strategy historically performs best in {regime} markets?
2. Which has highest confidence with best recent accuracy?
3. Risk: avoid strategies in drawdown

OUTPUT FORMAT:
- selected_strategy: name
- allocation: 0.0-1.0 (how much capital)
- reasoning: why this choice
"""
        return self.llm.decide(prompt)

    def detect_regime(self, market_data) -> str:
        volatility = market_data["atr_14"] / market_data["close"]
        trend_strength = abs(market_data["adx"])

        if volatility > 0.03:
            return "high_volatility"
        elif trend_strength > 25:
            return "trending"
        else:
            return "ranging"
```

##### Debate Pattern (Alternative)

[Source: TradingAgents](https://tradingagents-ai.github.io/)

> "Agents engage in structured debates, exchanging rationales to resolve disagreements and synthesize robust decisions"

```python
class DebateOrchestrator:
    """Multiple rounds of argument between strategies"""

    def run_debate(self, outputs: list[StrategyOutput], rounds: int = 2):
        # Round 1: Each strategy presents case
        arguments = [o.reasoning for o in outputs]

        # Round 2: Counter-arguments
        for i, output in enumerate(outputs):
            others = [o for j, o in enumerate(outputs) if j != i]
            counter = self.llm.generate_counter(output, others)
            arguments.append(counter)

        # Final: Facilitator picks winner
        return self.llm.select_winner(arguments)
```

##### When to Use Meta-Agent vs Simple Ensemble

| Approach | Best For | Complexity |
|----------|----------|------------|
| **Weighted Voting** | Stable markets, similar strategies | ⭐ Simple |
| **Meta-Agent (LLM)** | Regime-dependent selection, reasoning needed | ⭐⭐⭐ |
| **RL Meta-Policy** | High-frequency, learned patterns | ⭐⭐⭐⭐ |

**Recommendation:** Start with weighted voting → Add LLM meta-agent when you need reasoning about *why* a strategy was chosen

---

### 2. 🧠 Sentiment Analysis (NLP)

#### FinBERT - Financial Sentiment ⭐

[Source: ProsusAI/finBERT GitHub](https://github.com/ProsusAI/finBERT)

- **Model:** Pre-trained BERT fine-tuned on financial corpus
- **Output:** Positive/Negative/Neutral + confidence score
- **Accuracy:** ~79-80% on financial sentiment tasks
- **Free:** ✅ Open source, runs locally

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def get_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return {
        "positive": probs[0][0].item(),
        "negative": probs[0][1].item(),
        "neutral": probs[0][2].item()
    }
```

#### Alternative: LLM-based Sentiment

[Source: ScienceDirect paper](https://www.sciencedirect.com/science/article/pii/S1544612324002575)
- "GPT-3-based OPT model significantly outperforms the others, predicting stock market returns with an accuracy of 74.4%"

---

### 3. 📰 News Data Sources

| Source | Free Tier | Features |
|--------|-----------|----------|
| **[Marketaux](https://www.marketaux.com)** | ✅ 100% free | "80+ markets and 5,000+ sources" |
| **[Finnhub](https://finnhub.io/)** | ✅ Free tier | Real-time news, company news, market news |
| [Alpha Vantage](https://www.alphavantage.co/) | 500 calls/day | News + sentiment scores built-in |
| [Benzinga](https://www.benzinga.com/apis/) | ✅ Free | "Stock News feed is completely free" |
| [EODHD](https://eodhd.com/) | 20 calls/day | Financial news + sentiment |

**Recommendation:** Start with Marketaux (most generous free tier)

---

### 4. 📊 Market Data Sources

| Source | Price | Best For |
|--------|-------|----------|
| **[yfinance](https://pypi.org/project/yfinance/)** | Free | Historical data, quick prototyping ⚠️ |
| **[Alpha Vantage](https://www.alphavantage.co/)** | Free 500/day | Reliable, fundamentals, news |
| **[Finnhub](https://finnhub.io/)** | Free tier | Real-time quotes, fundamentals |
| [Financial Modeling Prep](https://financialmodelingprep.com/) | Free tier | Financial statements |

⚠️ **yfinance warning:** "aggressive rate limiting, frequent IP bans, and inconsistent data availability" [Source: Medium comparison article]

**Recommendation:** Alpha Vantage for reliability, yfinance for quick testing

---

### 5. 🧪 Paper Trading / Backtesting

#### Paper Trading APIs

| Platform | Free | Initial Balance | Notes |
|----------|------|-----------------|-------|
| **[Alpaca](https://docs.alpaca.markets/docs/paper-trading)** | ✅ | $100,000 | "Free IEX real-time market data" ⭐ |
| [QuantConnect](https://www.quantconnect.com/) | ✅ | Configurable | Cloud-based, good for complex strategies |
| [TradeStation](https://www.tradestation.com/) | ✅ | Simulated | Stocks, options, futures |

**Alpaca limitations:**
- ❌ No dividend simulation
- ❌ No market impact/slippage simulation
- ❌ No queue position simulation
[Source: Alpaca documentation]

#### Backtesting Frameworks

| Framework | Speed | Live Trading | Notes |
|-----------|-------|--------------|-------|
| **[Backtesting.py](https://kernc.github.io/backtesting.py/)** | Fast | ❌ | Simple, great visualizations ⭐ |
| **[Backtrader](https://www.backtrader.com/)** | Medium | ✅ IB, Oanda | "feature-rich Python framework" |
| [vectorbt](https://vectorbt.dev/) | Very fast | ❌ | "testing many thousands of strategies in seconds" |
| [Zipline](https://github.com/quantopian/zipline) | Medium | ❌ | Used by Quantopian |

**Recommendation:** Backtesting.py for development → Alpaca for paper trading

---

### 6. 🤖 LLM Options (Agent Brain)

#### 🔄 LiteLLM - Unified Provider Abstraction ⭐⭐⭐

[Source: LiteLLM GitHub](https://github.com/BerriAI/litellm)

**Use [LiteLLM](https://docs.litellm.ai/) to swap between providers with ONE line change:**

> "Python SDK, Proxy Server (AI Gateway) to call 100+ LLM APIs in OpenAI format"

```python
from litellm import completion
import os

# Configure providers
os.environ["ANTHROPIC_API_KEY"] = "sk-..."
os.environ["OPENAI_API_KEY"] = "sk-..."

# 🧪 Development: Ollama (free, local)
response = completion(
    model="ollama/qwen3:14b",
    messages=[{"role": "user", "content": "Analyze AAPL technicals"}]
)

# 🚀 Production: Claude (switch ONE param)
response = completion(
    model="anthropic/claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Analyze AAPL technicals"}]
)

# 💰 Or OpenAI
response = completion(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Analyze AAPL technicals"}]
)
```

**Benefits:**

- ✅ Same code for dev (Ollama) and prod (Claude/GPT)
- ✅ OpenAI-compatible exceptions across all providers
- ✅ Built-in cost tracking, rate limiting, logging
- ✅ Easy A/B testing between models

**Installation:**

```bash
pip install litellm
```

---

#### Local LLMs for Development (Free)

[Source: LocalLLM guide](https://localllm.in/blog/complete-guide-ollama-alternatives)

| Tool | Ease | Performance | Notes |
|------|------|-------------|-------|
| **[Ollama](https://ollama.ai/)** | ⭐⭐⭐ | Good | "go-to solution for running LLMs locally" |
| [LM Studio](https://lmstudio.ai/) | ⭐⭐⭐ | Good | Desktop GUI |
| [vLLM](https://github.com/vllm-project/vllm) | ⭐⭐ | Best | Production-grade |

**Recommended local models for trading agents:**

- **Qwen3 14B/32B** - Good reasoning, tool calling
- **Llama 3.3 70B** - Strong general capability
- **DeepSeek R1** - Excellent reasoning

[Source: HuggingFace blog - "new models deliver powerful reasoning... agentic coding capabilities, and built-in tool-calling"]

---

#### Production APIs

| Provider | Model | Price | Notes |
|----------|-------|-------|-------|
| **Anthropic** | Claude Sonnet | ~$3/M tokens | Best reasoning |
| **OpenAI** | GPT-4o | ~$2.50/M tokens | Fast, reliable |
| **Groq** | Llama/Mixtral | Very cheap | Fastest inference |
| **Together AI** | Open models | ~$0.20/M tokens | Cheap |
| **DeepSeek** | DeepSeek V3 | $0.14/M tokens | Cheapest quality |

**Recommendation:** Ollama + Qwen3 for dev → Claude Sonnet for prod via LiteLLM

---

### 7. 🔗 Agent Framework

**[LangGraph](https://github.com/langchain-ai/langgraph)** ⭐ Recommended

- Used by TradingAgents: "built with LangGraph to ensure flexibility and modularity"
- Supports complex multi-agent workflows
- State management for agent collaboration
- Works with any LLM (local or API)

```python
from langgraph.graph import StateGraph
from typing import TypedDict

class TradingState(TypedDict):
    symbol: str
    technical_analysis: dict
    sentiment_analysis: dict
    news_analysis: dict
    decision: str

# Build graph
workflow = StateGraph(TradingState)
workflow.add_node("technical_analyst", technical_analyst)
workflow.add_node("sentiment_analyst", sentiment_analyst)
workflow.add_node("trader", trader_agent)
# ... add edges
```

---

## 💰 Cost Breakdown (Budget Build)

| Component | Tool | Cost |
|-----------|------|------|
| Technical Analysis | pandas-ta | Free |
| Sentiment Analysis | FinBERT (local) | Free |
| News Data | Marketaux | Free |
| Market Data | Alpha Vantage | Free (500/day) |
| Paper Trading | Alpaca | Free |
| Backtesting | Backtesting.py | Free |
| LLM | Ollama + Qwen3 | Free |
| LLM Abstraction | LiteLLM | Free |
| Agent Framework | LangGraph | Free |
| Strategy Optimization | Optuna | Free |
| **Total** | | **$0** 🎉 |

---

## 🏠 Homelab Deployment

**Yes! Everything can run self-hosted on modest hardware.**

### Hardware Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **CPU** | 4 cores | 8+ cores | More = faster backtesting |
| **RAM** | 16 GB | 32 GB | LLM + FinBERT + data |
| **GPU** | None | 8GB VRAM | For faster LLM inference |
| **Storage** | 50 GB | 200 GB+ | Models + historical data |
| **Network** | Stable | Low latency | For real-time data feeds |

#### Component Memory Breakdown

| Service | RAM Usage | GPU VRAM |
|---------|-----------|----------|
| **Ollama (Qwen3 14B)** | 10-14 GB | 10 GB (or CPU) |
| **FinBERT** | 500 MB | 200 MB |
| **PostgreSQL** | 500 MB | - |
| **Redis** | 100 MB | - |
| **Trading App** | 1-2 GB | - |
| **Total** | ~16 GB | ~10 GB |

[Source: HuggingFace FinBERT](https://huggingface.co/ProsusAI/finbert/discussions/20) - "Float16: 208.82 MB VRAM for inference"

### Docker Compose Stack

```yaml
# docker-compose.yml
version: '3.8'

services:
  # 🤖 Local LLM
  ollama:
    image: ollama/ollama:latest
    container_name: trading-ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  # 🧠 Sentiment Analysis API
  finbert:
    build: ./services/finbert
    container_name: trading-finbert
    ports:
      - "8001:8000"
    environment:
      - DEVICE=cuda  # or cpu
    restart: unless-stopped

  # 📊 Trading Bot Core
  trading-bot:
    build: ./services/trading-bot
    container_name: trading-bot
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - FINBERT_HOST=http://finbert:8000
      - REDIS_HOST=redis
      - POSTGRES_HOST=postgres
      - ALPACA_API_KEY=${ALPACA_API_KEY}
      - ALPACA_SECRET_KEY=${ALPACA_SECRET_KEY}
      - ALPHA_VANTAGE_KEY=${ALPHA_VANTAGE_KEY}
    depends_on:
      - ollama
      - finbert
      - redis
      - postgres
    restart: unless-stopped

  # 💾 State & Caching
  redis:
    image: redis:alpine
    container_name: trading-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  # 📈 Trade History & Metrics
  postgres:
    image: postgres:16-alpine
    container_name: trading-postgres
    environment:
      - POSTGRES_USER=trading
      - POSTGRES_PASSWORD=trading
      - POSTGRES_DB=trades
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  # 📊 Monitoring Dashboard
  grafana:
    image: grafana/grafana:latest
    container_name: trading-grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
    restart: unless-stopped

volumes:
  ollama_data:
  redis_data:
  postgres_data:
  grafana_data:
```

### FinBERT Service Dockerfile

```dockerfile
# services/finbert/Dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN pip install transformers torch fastapi uvicorn

COPY app.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```python
# services/finbert/app.py
from fastapi import FastAPI
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

@app.post("/sentiment")
async def analyze(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    labels = ["positive", "negative", "neutral"]
    return {labels[i]: probs[0][i].item() for i in range(3)}
```

### Directory Structure

```text
trading-system/
├── docker-compose.yml
├── .env                          # API keys
├── services/
│   ├── finbert/
│   │   ├── Dockerfile
│   │   └── app.py
│   └── trading-bot/
│       ├── Dockerfile
│       ├── requirements.txt
│       └── src/
│           ├── agents/
│           │   ├── technical.py
│           │   ├── sentiment.py
│           │   ├── news.py
│           │   └── meta.py
│           ├── strategies/
│           │   ├── momentum.py
│           │   ├── mean_reversion.py
│           │   └── trend.py
│           ├── data/
│           │   ├── market.py      # Alpha Vantage
│           │   └── news.py        # Marketaux
│           └── main.py
├── data/
│   └── historical/               # Cached market data
└── dashboards/
    └── grafana/                  # Monitoring configs
```

### Startup Commands

```bash
# 1. Clone and setup
git clone <your-repo>
cd trading-system

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 3. Pull and start
docker compose pull
docker compose up -d

# 4. Download LLM model
docker exec -it trading-ollama ollama pull qwen3:14b

# 5. Verify services
docker compose ps
curl http://localhost:11434/api/tags  # Ollama
curl http://localhost:8001/docs       # FinBERT
curl http://localhost:8000/health     # Trading bot
```

### Alternative: Lightweight No-GPU Setup

If no GPU available, use smaller models:

```yaml
# .env
LLM_MODEL=qwen3:8b        # Smaller, runs on CPU
FINBERT_DEVICE=cpu        # CPU inference
```

| Setup | RAM Needed | Speed |
|-------|------------|-------|
| **GPU (RTX 3080+)** | 16 GB | ⚡ Fast |
| **CPU (8 cores)** | 24 GB | 🐢 Slower but works |
| **Apple M1/M2/M3** | 16 GB | ⚡ Fast (Metal) |

### Existing Frameworks with Docker Support

| Framework | Docker | Self-Hosted | Notes |
|-----------|--------|-------------|-------|
| **[Freqtrade](https://www.freqtrade.io/)** | ✅ Official | ✅ | Crypto focus, ML optimization |
| **[Jesse](https://jesse.trade/)** | ✅ | ✅ | "Installation with Docker is extremely positive" |
| **[LEAN](https://www.lean.io/)** | ✅ | ✅ | "Replicate full QuantConnect experience on-premise" |
| **[OpenAlgo](https://openalgo.in/)** | ✅ | ✅ | "Self-hosted algo trading platform" |

[Source: Freqtrade Docker Quickstart](https://www.freqtrade.io/en/stable/docker_quickstart/)

### Monitoring & Observability

```yaml
# Add to docker-compose.yml for full observability

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  # Log aggregation
  loki:
    image: grafana/loki
    ports:
      - "3100:3100"
```

**Grafana dashboards to create:**

- 📈 Strategy performance (Sharpe, returns, drawdown)
- 🤖 Agent decisions log
- 💻 System resources (CPU, RAM, GPU)
- 📊 Trade history

---

## 🇵🇱 Polish GPW (Warsaw Stock Exchange) Setup

**Yes, but requires some adaptations.**

### Key Differences from US Markets

| Aspect | US Markets | GPW (Poland) |
|--------|------------|--------------|
| **Data APIs** | Many free options | Limited, Stooq best free |
| **Broker APIs** | Alpaca, IBKR | XTB, IBKR |
| **Paper Trading** | Easy (Alpaca) | Manual simulation |
| **Sentiment** | FinBERT (English) | HerBERT (Polish) |
| **News Sources** | Many APIs | Polish sites, RSS |
| **Trading Hours** | 9:30-16:00 EST | 9:00-17:00 CET |

### 📊 GPW Data Sources

#### Free Options

| Source | Data Type | Access | Notes |
|--------|-----------|--------|-------|
| **[Stooq](https://stooq.com/)** | Historical OHLCV | Free | Best free source for GPW |
| **[EODHD](https://eodhd.com/)** | Historical + Fundamentals | 20 calls/day free | Has GPW.WAR data |
| **[GPW Official](https://www.gpw.pl/market-data)** | Real-time | Paid license | "Market Data Gateway" |

[Source: QuantStart](https://www.quantstart.com/articles/an-introduction-to-stooq-pricing-data/) - Stooq provides free historical data

#### Stooq Python Integration

```python
# Option 1: pystooq package
from pystooq import StooqDataFetcher
from datetime import date

fetcher = StooqDataFetcher()
data = fetcher.get_data(
    tickers=["PKO", "CDR", "PKN", "PZU"],  # Polish stocks
    start=date(2023, 1, 1),
    end=date(2024, 12, 31)
)

# Option 2: pandas-datareader (may have CAPTCHA issues)
from pandas_datareader import data as pdr

df = pdr.DataReader("PKO", "stooq", start="2023-01-01")
```

[Source: pystooq PyPI](https://pypi.org/project/pystooq/)

⚠️ **Note:** "Stooq disabled automatic downloads and requires CAPTCHA" - may need manual bulk download from [stooq.com/db/h/](https://stooq.com/db/h/)

### 🏦 Polish Broker APIs

| Broker | API | Paper Trading | Notes |
|--------|-----|---------------|-------|
| **[XTB](https://www.xtb.com/)** | REST API | ❌ | "REST API reliable for real-time data" |
| **[Interactive Brokers](https://www.interactivebrokers.com/)** | Full API | ✅ | Best option, supports GPW |
| **mBank eMakler** | ❌ No API | ❌ | Manual only |
| **DM BOŚ** | Limited | ❌ | Check current offerings |

[Source: Medium - XTB Python](https://medium.com/@slisowski/build-your-own-trading-bot-in-python-step-by-step-part-i-f711b25f7ef3)

**Recommendation:** Use Interactive Brokers for API trading on GPW

#### XTB API Example (if available)

```python
# XTB xStation5 API (CFD trading)
import xapi

async def connect_xtb():
    async with await xapi.connect(
        accountId="YOUR_ACCOUNT",
        password="YOUR_PASSWORD",
        host="xapi.xtb.com",
        type="real",  # or "demo"
        safe=True
    ) as x:
        response = await x.socket.getSymbol(symbol="PKO.PL")
        print(response)
```

### 🧠 Polish Sentiment Analysis

**FinBERT won't work well for Polish news - use HerBERT instead.**

#### HerBERT Sentiment Model

[Source: Hugging Face Voicelab](https://huggingface.co/Voicelab/herbert-base-cased-sentiment)

> "HerBERT is a BERT-based Language Model trained on Polish Corpora"

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Polish sentiment model
tokenizer = AutoTokenizer.from_pretrained("Voicelab/herbert-base-cased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained(
    "Voicelab/herbert-base-cased-sentiment"
)

def analyze_polish_sentiment(text: str) -> dict:
    """Analyze Polish text sentiment (negative/neutral/positive)"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    labels = ["negative", "neutral", "positive"]
    return {labels[i]: probs[0][i].item() for i in range(3)}

# Example
sentiment = analyze_polish_sentiment(
    "PKN Orlen notuje rekordowe zyski w trzecim kwartale"
)
# {'negative': 0.02, 'neutral': 0.15, 'positive': 0.83}
```

#### Alternative Polish Models

| Model | Use Case | Accuracy |
|-------|----------|----------|
| **[Voicelab/herbert-base-cased-sentiment](https://huggingface.co/Voicelab/herbert-base-cased-sentiment)** | General Polish | ⭐ Best |
| [eevvgg/bert-polish-sentiment-politics](https://huggingface.co/eevvgg/bert-polish-sentiment-politics) | Political tweets | Good |
| [nlptown/bert-base-multilingual-uncased-sentiment](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment) | Multilingual | OK for Polish |

### 📰 Polish Financial News Sources

| Source | Type | Access |
|--------|------|--------|
| **[Bankier.pl](https://www.bankier.pl/)** | News portal | RSS available |
| **[Money.pl](https://www.money.pl/)** | Finance portal | Scraping |
| **[Parkiet.com](https://www.parkiet.com/)** | Stock news | RSS |
| **[StockWatch.pl](https://www.stockwatch.pl/)** | Analysis | Scraping |
| **[PAP Biznes](https://biznes.pap.pl/)** | News agency | API (paid) |

#### RSS Feed Integration

```python
import feedparser
from datetime import datetime

POLISH_NEWS_FEEDS = [
    "https://www.bankier.pl/rss/wiadomosci.xml",
    "https://www.parkiet.com/rss/parkiet.xml",
]

def fetch_polish_news():
    articles = []
    for feed_url in POLISH_NEWS_FEEDS:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:10]:
            articles.append({
                "title": entry.title,
                "summary": entry.get("summary", ""),
                "published": entry.get("published", ""),
                "link": entry.link,
            })
    return articles

# Then analyze with HerBERT
for article in fetch_polish_news():
    sentiment = analyze_polish_sentiment(article["title"])
    print(f"{article['title'][:50]}... → {max(sentiment, key=sentiment.get)}")
```

### 🔄 Adapted Docker Stack for GPW

```yaml
# docker-compose.gpw.yml - GPW-specific additions

services:
  # Replace FinBERT with HerBERT
  herbert-sentiment:
    build: ./services/herbert
    container_name: trading-herbert
    ports:
      - "8001:8000"
    environment:
      - MODEL_NAME=Voicelab/herbert-base-cased-sentiment
    restart: unless-stopped

  # GPW data fetcher
  gpw-data:
    build: ./services/gpw-data
    container_name: trading-gpw-data
    environment:
      - STOOQ_CACHE_DIR=/data/stooq
    volumes:
      - ./data/stooq:/data/stooq
    restart: unless-stopped
```

### ⚠️ GPW-Specific Challenges

| Challenge | Workaround |
|-----------|------------|
| **No free paper trading** | Build local simulator with historical data |
| **Limited real-time data** | Use delayed quotes or pay for GPW feed |
| **Polish news parsing** | HerBERT + RSS feeds |
| **Broker API limitations** | Interactive Brokers or XTB demo |
| **Lower liquidity** | Stick to WIG20 / mWIG40 stocks |

### 📋 GPW Implementation Checklist

- [ ] Set up Stooq data fetcher (bulk download + cache)
- [ ] Replace FinBERT service with HerBERT
- [ ] Configure Polish news RSS feeds
- [ ] Open Interactive Brokers account (or XTB demo)
- [ ] Build local paper trading simulator
- [ ] Focus on liquid stocks (WIG20: PKO, CDR, PKN, PZU, etc.)

### 🎯 Recommended GPW Tickers for Testing

| Ticker | Company | Sector | Liquidity |
|--------|---------|--------|-----------|
| **PKO** | PKO Bank Polski | Banking | ⭐⭐⭐ High |
| **CDR** | CD Projekt | Gaming | ⭐⭐⭐ High |
| **PKN** | PKN Orlen | Energy | ⭐⭐⭐ High |
| **PZU** | PZU SA | Insurance | ⭐⭐⭐ High |
| **KGH** | KGHM | Mining | ⭐⭐⭐ High |
| **PEO** | Bank Pekao | Banking | ⭐⭐ Medium |
| **DNP** | Dino Polska | Retail | ⭐⭐ Medium |
| **ALE** | Allegro | E-commerce | ⭐⭐ Medium |

---

## 🏁 Implementation Roadmap

### Phase 1: Foundation 🔨
- [ ] Set up Python environment with dependencies
- [ ] Implement data fetchers (Alpha Vantage, Marketaux)
- [ ] Build technical indicator calculator (pandas-ta)
- [ ] Test FinBERT sentiment analysis

### Phase 2: Agents 🤖
- [ ] Design agent communication protocol
- [ ] Implement Technical Analyst agent
- [ ] Implement Sentiment Analyst agent
- [ ] Implement News Analyst agent
- [ ] Build Trader agent with decision logic

### Phase 3: Risk & Execution ⚖️
- [ ] Add Risk Management agent
- [ ] Implement position sizing logic
- [ ] Connect to Alpaca paper trading
- [ ] Add trade logging/monitoring

### Phase 4: Testing & Refinement 🧪
- [ ] Backtest on historical data
- [ ] Paper trade for 30+ days
- [ ] Analyze performance metrics
- [ ] Tune agent prompts/logic

---

## 📚 Key Resources

### Official Docs
- [TradingAgents GitHub](https://github.com/TauricResearch/TradingAgents) - Complete multi-agent implementation
- [pandas-ta Documentation](https://pypi.org/project/pandas-ta/) - Technical indicators
- [Alpaca API Docs](https://docs.alpaca.markets/docs/paper-trading) - Paper trading
- [FinBERT on HuggingFace](https://huggingface.co/ProsusAI/finbert) - Sentiment model

### Learning
- [PyQuant News - Technical Analysis](https://www.pyquantnews.com/free-python-resources/building-and-backtesting-trading-strategies-with-python)
- [Backtesting.py Tutorial](https://kernc.github.io/backtesting.py/)
- [LangGraph Documentation](https://github.com/langchain-ai/langgraph)

---

## ❓ What Sources DON'T Cover

- **Specific strategy performance** - No concrete backtesting results for multi-agent systems
- **Optimal agent prompts** - TradingAgents uses proprietary prompts
- **Real-world slippage impact** - Paper trading doesn't simulate accurately
- **Regulatory considerations** - Not covered (important for real trading)
- **Tax implications** - Not researched

---

## 🔗 All Sources

1. [TradingAgents GitHub](https://github.com/TauricResearch/TradingAgents)
2. [Alpaca Paper Trading Docs](https://docs.alpaca.markets/docs/paper-trading)
3. [FinBERT GitHub](https://github.com/ProsusAI/finBERT)
4. [pandas-ta PyPI](https://pypi.org/project/pandas-ta/)
5. [Backtesting.py](https://kernc.github.io/backtesting.py/)
6. [Marketaux](https://www.marketaux.com)
7. [Finnhub](https://finnhub.io/)
8. [Alpha Vantage](https://www.alphavantage.co/)
9. [LiteLLM GitHub](https://github.com/BerriAI/litellm)
10. [Ollama Alternatives Guide](https://localllm.in/blog/complete-guide-ollama-alternatives)
11. [Open Source LLMs 2025](https://huggingface.co/blog/daya-shankar/open-source-llms)
12. [Sentiment Trading with LLMs (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S1544612324002575)
13. [Financial Data APIs Comparison](https://medium.com/coinmonks/the-7-best-financial-apis-for-investors-and-developers-in-2025-in-depth-analysis-and-comparison-adbc22024f68)
14. [Optuna - Hyperparameter Optimization](https://optuna.org/)
15. [Deep RL Ensemble Strategy (arXiv)](https://arxiv.org/html/2511.12120v1)
16. [Dynamic Stock Decision Ensemble (Springer)](https://link.springer.com/article/10.1007/s10489-022-03606-0)
17. [Jesse Trading Framework](https://jesse.trade/)
18. [Freqtrade](https://www.freqtrade.io/)
19. [MetaTrader RL Paper (arXiv)](https://arxiv.org/pdf/2210.01774)
20. [Macrosynergy - Measuring Strategy Value](https://macrosynergy.com/research/measuring-value-added-of-algorithmic-trading-strategies/)
21. [Local LLM Stack (Docker)](https://github.com/dalekurt/local-llm-stack)
22. [OpenAlgo - Self-hosted Trading](https://openalgo.in/)
23. [LEAN Engine](https://www.lean.io/)
24. [Stooq - GPW Data (QuantStart)](https://www.quantstart.com/articles/an-introduction-to-stooq-pricing-data/)
25. [pystooq PyPI](https://pypi.org/project/pystooq/)
26. [HerBERT Polish Sentiment (Voicelab)](https://huggingface.co/Voicelab/herbert-base-cased-sentiment)
27. [XTB Python Trading Bot](https://medium.com/@slisowski/build-your-own-trading-bot-in-python-step-by-step-part-i-f711b25f7ef3)

---

## 🎯 Quick Start Suggestion

**Minimal Viable Agent (simplest approach):**

```python
# 1. Fetch data
import yfinance as yf
import pandas_ta as ta

# 2. Technical signals
df = yf.download("AAPL", period="3mo")
df.ta.rsi(append=True)
df.ta.macd(append=True)

# 3. News sentiment
from transformers import pipeline
sentiment = pipeline("sentiment-analysis", model="ProsusAI/finbert")
news_sentiment = sentiment("Apple reports record iPhone sales")

# 4. Simple decision logic (replace with LLM agent)
def decide(rsi, macd_hist, sentiment_score):
    if rsi < 30 and sentiment_score > 0.5:
        return "BUY"
    elif rsi > 70 and sentiment_score < -0.3:
        return "SELL"
    return "HOLD"

# 5. Paper trade via Alpaca
# ... implement execution
```

Then gradually add more sophisticated agents with LangGraph!

---

**Suggested Obsidian location:** 3_Resources/Trading/
**Potential MOCs:** [[AI Projects MOC]], [[Trading MOC]], [[Python Projects MOC]]
**Tags:** #ai #trading #agents #python #research
