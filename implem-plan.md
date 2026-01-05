# AI Casino - Implementation Plan

**Project:** Agentic Stock Trading System
**Status:** MVP Complete (44% overall)
**Last Updated:** 2026-01-05

---

## ✅ Completed Features (11/25)

### Core Architecture
- [x] Multi-agent architecture (Technical, Sentiment, News, Trader)
- [x] LangGraph workflow orchestration
- [x] LiteLLM integration (Ollama/Claude/GPT)
- [x] CLI interface with rich output

### Analysis Agents
- [x] Technical Analyst (RSI + MACD via pandas-ta)
- [x] FinBERT sentiment analysis integration
- [x] News Analyst (Marketaux API)

### Data & Infrastructure
- [x] Market data fetching (yfinance, Alpha Vantage)
- [x] Full test suite with pytest
- [x] Strict linting with ruff
- [x] mise tooling (Python, uv, ruff, Ollama)

---

## 📋 Pending Features (14/25)

### 🔴 High Priority - Core Trading

#### 1. Risk Management Agent
**Priority:** HIGH
**Effort:** Medium
**Dependencies:** None
**Description:**
- Position sizing based on account balance
- Stop-loss calculation
- Risk validation before trades
- Max exposure limits

#### 2. Alpaca Paper Trading Integration
**Priority:** HIGH
**Effort:** Medium
**Dependencies:** Risk Management Agent
**Description:**
- Execute simulated trades via Alpaca API
- Track paper trading portfolio
- Order placement and management
- Real-time position tracking

#### 3. Backtesting Framework
**Priority:** HIGH
**Effort:** Large
**Dependencies:** None
**Description:**
- Integrate backtesting.py
- Historical data replay
- Strategy performance metrics
- Compare strategies on past data

#### 4. Performance Metrics Tracking
**Priority:** HIGH
**Effort:** Medium
**Dependencies:** Paper trading or backtesting
**Description:**
- Sharpe ratio calculation
- Maximum drawdown tracking
- Win rate statistics
- Risk-adjusted returns

---

### 🟡 Medium Priority - Advanced Agents

#### 5. Fundamental Analyst Agent
**Priority:** MEDIUM
**Effort:** Large
**Dependencies:** Fundamental data API
**Description:**
- P/E ratio analysis
- EPS, revenue growth
- Balance sheet metrics
- Financial statement parsing

#### 6. Bullish Researcher Agent
**Priority:** MEDIUM
**Effort:** Medium
**Dependencies:** All analyst agents
**Description:**
- Optimistic case synthesis
- Bull thesis generation
- Debate with bearish researcher
- Confidence scoring

#### 7. Bearish Researcher Agent
**Priority:** MEDIUM
**Effort:** Medium
**Dependencies:** All analyst agents
**Description:**
- Risk-focused analysis
- Bear thesis generation
- Counter-arguments
- Downside scenario modeling

---

### 🟢 Advanced Features - Multi-Strategy

#### 8. Mean Reversion Strategy
**Priority:** LOW
**Effort:** Small
**Dependencies:** None
**Description:**
- Bollinger Bands implementation
- Overbought/oversold detection
- Statistical arbitrage signals

#### 9. Trend Following Strategy
**Priority:** LOW
**Effort:** Small
**Dependencies:** None
**Description:**
- SMA crossover (50/200)
- ADX trend strength
- Momentum confirmation

#### 10. Multi-Strategy Ensemble System
**Priority:** LOW
**Effort:** Large
**Dependencies:** Multiple strategies
**Description:**
- Run strategies in parallel
- Weighted voting system
- Signal aggregation
- Conflict resolution

#### 11. Meta-Agent for Strategy Selection
**Priority:** LOW
**Effort:** Large
**Dependencies:** Multi-strategy ensemble
**Description:**
- Dynamic strategy selection
- Market regime detection
- Performance-based weighting
- Adaptive allocation

#### 12. Optuna Strategy Optimization
**Priority:** LOW
**Effort:** Medium
**Dependencies:** Backtesting framework
**Description:**
- Parameter tuning automation
- Multi-objective optimization
- Hyperparameter search
- Cross-validation

---

### 🔵 Infrastructure

#### 13. Trade History Database
**Priority:** LOW
**Effort:** Medium
**Dependencies:** Paper trading
**Description:**
- PostgreSQL setup
- Trade logging
- Portfolio snapshots
- Historical analysis queries

#### 14. Monitoring Dashboard
**Priority:** LOW
**Effort:** Large
**Dependencies:** Database, metrics tracking
**Description:**
- Grafana integration
- Real-time performance charts
- Strategy comparison views
- Alert configuration

---

## 🎯 Recommended Implementation Order

### Phase 1: Trading Foundation (MVP+)
1. Risk Management Agent
2. Alpaca Paper Trading Integration
3. Performance Metrics Tracking
4. Trade History Database (basic)

**Goal:** Functional paper trading system with risk controls

### Phase 2: Advanced Analysis
5. Fundamental Analyst Agent
6. Bullish Researcher Agent
7. Bearish Researcher Agent

**Goal:** Complete research team with debate functionality

### Phase 3: Strategy Evolution
8. Backtesting Framework
9. Mean Reversion Strategy
10. Trend Following Strategy
11. Multi-Strategy Ensemble System

**Goal:** Multiple strategies with historical validation

### Phase 4: Optimization & Monitoring
12. Meta-Agent for Strategy Selection
13. Optuna Strategy Optimization
14. Monitoring Dashboard (Grafana)

**Goal:** Self-optimizing system with full observability

---

## 📊 Progress Tracking

**Overall Progress:** 11/25 (44%)

### By Category
- **Architecture & Core:** 4/4 (100%) ✅
- **Analysis Agents:** 3/6 (50%)
- **Trading & Execution:** 0/4 (0%)
- **Strategies:** 0/3 (0%)
- **Infrastructure:** 1/4 (25%)
- **Optimization:** 0/2 (0%)
- **DevOps:** 3/3 (100%) ✅

---

## 🔗 References

- [Research Document](./agentic-stock-trading-system-research.md)
- [TradingAgents Framework](https://github.com/TauricResearch/TradingAgents)
- [Alpaca Paper Trading Docs](https://docs.alpaca.markets/docs/paper-trading)
- [Backtesting.py](https://kernc.github.io/backtesting.py/)
- [Optuna](https://optuna.org/)

---

## 📝 Notes

- Current MVP supports single-stock analysis with basic momentum strategy
- All completed features have full test coverage
- Strict linting enforced via ruff
- Hybrid LLM setup (Ollama dev → Claude/GPT prod)
