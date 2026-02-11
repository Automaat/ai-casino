# Analysis Pipeline

Per-symbol analysis flow from raw market data to a BUY/SELL/HOLD decision with risk gating.

## Complete Pipeline

```mermaid
graph TD
    Start([Symbol Input]) --> FetchData

    subgraph Stage 1: Data Fetch
        FetchData[Fetch Market Data<br/>90 days OHLCV]
        FetchNews[Fetch News Articles]
        FetchAccount[Fetch Account Info]
    end

    FetchData --> StrategySelect
    FetchNews --> StrategySelect
    FetchAccount --> StrategySelect

    subgraph Stage 2: Strategy Selection
        StrategySelect[MetaAgent<br/>Regime Detection]
        StrategySelect --> |regime confidence ≥ 0.5| SingleStrategy[Single Strategy]
        StrategySelect --> |regime confidence < 0.5| EnsembleStrat[Ensemble Strategy]
    end

    SingleStrategy --> Group1
    EnsembleStrat --> Group1

    subgraph Stage 3: Parallel Group 1
        direction LR
        Technical[TechnicalAnalyst]
        Sentiment[SentimentAnalyst]
        News[NewsAnalyst]
        Fundamental[FundamentalAnalyst]
        Comparative[ComparativeAnalyst]
        WebResearch[WebResearchAgent]
        Social[SocialSentimentAnalyst]
        Trump[TrumpAnalyst]
    end

    Group1 --> Group2

    subgraph Stage 4: Parallel Group 2
        direction LR
        Bullish[BullishResearcher]
        Bearish[BearishResearcher]
    end

    Group2 --> Decision

    subgraph Stage 5: Decision
        Decision[TraderAgent<br/>Synthesize all analyses]
    end

    Decision --> Risk

    subgraph Stage 6: Risk Gate
        Risk[RiskManagementAgent]
        Risk --> |approved| Execute[Execute Trade]
        Risk --> |rejected| NoTrade[No Trade]
    end

    Execute --> Done([Result])
    NoTrade --> Done
```

**Stage details:**

| Stage | Async | Agents | Depends On |
|---|---|---|---|
| 1. Data Fetch | Yes | — | Symbol input |
| 2. Strategy Selection | Yes | MetaAgent | Market data |
| 3. Group 1 | Parallel | 7 core (optional Trump when enabled) | Data + strategy |
| 4. Group 2 | Parallel | 2 agents | Group 1 results |
| 5. Decision | Sequential | TraderAgent | All analyses |
| 6. Risk Gate | Sequential | RiskManagementAgent | Decision + account |

## Strategy Selection

```mermaid
graph TD
    MD[Market Data] --> RD[MarketRegimeDetector<br/>detect_regime]
    RD --> Regime{Regime Confidence?}

    Regime --> |≥ 0.5| Map{Detected Regime}
    Map --> |TRENDING_BULLISH| TF[TrendFollowingStrategy<br/>SMA 20/50]
    Map --> |TRENDING_BEARISH| TF
    Map --> |RANGING| MR[MeanReversionStrategy<br/>BB 20/2σ]
    Map --> |HIGH_VOLATILITY| MOM[MomentumStrategy<br/>RSI 14, MACD 12/26/9]

    Regime --> |< 0.5| Ensemble[EnsembleStrategy]
    Ensemble --> Weights[Calculate Weights]
    Weights --> |Base| BW["momentum: 0.33<br/>mean_reversion: 0.33<br/>trend_following: 0.34"]
    BW --> Boost["+ Regime boost: up to 0.2<br/>+ Performance boost: recent_score × 0.1 (when metrics available)"]
    Boost --> Normalize[Normalize to Σ=1.0]
```

**Regime-to-strategy mapping (`STRATEGY_REGIME_MAP`):**

| Regime | Strategy | Key Parameters |
|---|---|---|
| `TRENDING_BULLISH` | TrendFollowing | SMA fast=20, slow=50 |
| `TRENDING_BEARISH` | TrendFollowing | SMA fast=20, slow=50 |
| `RANGING` | MeanReversion | BB period=20, std=2.0 |
| `HIGH_VOLATILITY` | Momentum | RSI 14, MACD 12/26/9 |

**Ensemble weight constants:**

| Constant | Value |
|---|---|
| `LOW_CONFIDENCE_THRESHOLD` | 0.5 |
| `REGIME_WEIGHT_BOOST` | 0.2 |
| `PERFORMANCE_WEIGHT_BOOST` | 0.1 |
| `WIN_RATE_THRESHOLD` | 0.5 |

## Strategy Signal Logic

### Momentum (RSI + MACD)

```mermaid
graph LR
    RSI["RSI(14)"] --> RSI_Check{RSI Value?}
    RSI_Check --> |< 30| Oversold[Oversold]
    RSI_Check --> |> 70| Overbought[Overbought]
    RSI_Check --> |30-70| Neutral[Neutral]

    MACD["MACD(12,26,9)"] --> MACD_Check{MACD vs Signal?}
    MACD_Check --> |MACD > Signal| Bullish[Bullish]
    MACD_Check --> |MACD < Signal| Bearish[Bearish]

    Oversold --> AND_BUY{AND}
    Bullish --> AND_BUY
    AND_BUY --> BUY([BUY])

    Overbought --> AND_SELL{AND}
    Bearish --> AND_SELL
    AND_SELL --> SELL([SELL])

    Neutral --> HOLD([HOLD])
```

### Mean Reversion (Bollinger Bands)

```mermaid
graph LR
    Price[Close Price] --> BB{"vs Bollinger Bands<br/>(20, 2σ)"}
    BB --> |Close < Lower Band| BUY([BUY])
    BB --> |Close > Upper Band| SELL([SELL])
    BB --> |Within Bands| HOLD([HOLD])
```

### Trend Following (SMA + ADX)

```mermaid
graph LR
    SMA["SMA Cross<br/>(fast/slow)"] --> Cross{Crossover?}
    Cross --> |Golden Cross + Bullish DI| BUY_CROSS([BUY])
    Cross --> |Death Cross + Bearish DI| SELL_CROSS([SELL])

    ADX["ADX(14)"] --> Trend{ADX ≥ 25?}
    Trend --> |Yes + DI+ > DI- + Close > Fast SMA| BUY_TREND([BUY])
    Trend --> |Yes + DI- > DI+ + Close < Fast SMA| SELL_TREND([SELL])

    Cross --> |No crossover| Check_ADX{Check ADX}
    Check_ADX --> Trend
    Trend --> |No strong trend| HOLD([HOLD])
```

### Ensemble Aggregation

| Method | Logic | Conflict Resolution |
|---|---|---|
| `WEIGHTED_VOTING` | Sum weights per signal, highest wins | If BUY-SELL margin < 10% → HOLD |
| `MAJORITY_VOTE` | 1 vote per strategy, majority wins | If tie → HOLD |
| `UNANIMOUS` | All strategies must agree | Any disagreement → HOLD |

**Ensemble confidence formula:**
```
confidence = (agreement_ratio × 0.4) + (weighted_score × 0.4) + (signal_strength × 0.2)
```

**Default ensemble weights:**

| Strategy | Weight |
|---|---|
| Momentum | 0.40 |
| Mean Reversion | 0.25 |
| Trend Following | 0.35 |

**Conflict margin threshold:** 0.10 (10%)

## Risk Assessment Flow

```mermaid
graph TD
    Input["Decision<br/>(symbol, action, price,<br/>confidence, account)"] --> HoldCheck{Action = HOLD?}
    HoldCheck --> |Yes| PassThrough[Return HOLD assessment<br/>No risk calc needed]

    HoldCheck --> |No| StopLoss[Calculate Stop-Loss]

    StopLoss --> ATR{ATR available?}
    ATR --> |Yes| ATR_SL["ATR-based stop<br/>stop = price ± ATR × 2.0"]
    ATR --> |No| Fixed_SL["Fixed % stop<br/>stop = price ± 2.0%"]

    ATR_SL --> Trailing{BUY + trailing enabled?}
    Fixed_SL --> Trailing
    Trailing --> |Yes| Trail["Add trailing stop<br/>trail: 3.0%, activation: 5.0%"]
    Trailing --> |No| PosSize

    Trail --> PosSize[Calculate Position Size]
    PosSize --> MaxRisk["max_risk = balance × 2.0%"]
    MaxRisk --> Shares["shares = max_risk / risk_per_share"]
    Shares --> CapCash["Cap by available cash"]
    CapCash --> CapPosition["Cap by max single position (20%)"]

    CapPosition --> Validate[Validate Constraints]

    Validate --> V1{"Exposure < 80%?"}
    Validate --> V2{"Sufficient cash?"}
    Validate --> V3{"Owns stock? (for SELL)"}
    Validate --> V4{"Confidence ≥ 0.6?"}

    V1 --> Score[Calculate Risk Score]
    V2 --> Score
    V3 --> Score
    V4 --> Score

    Score --> Level{Risk Score}
    Level --> |≥ 0.75| LOW[LOW Risk<br/>Approved]
    Level --> |0.5 — 0.75| MEDIUM[MEDIUM Risk<br/>Approved with warnings]
    Level --> |< 0.5| HIGH[HIGH Risk<br/>Rejected]

    LOW --> Audit[Write audit log]
    MEDIUM --> Audit
    HIGH --> Audit
```

## Agent Reference

| Agent | Input | Output Model | Core/Optional | Uses LLM |
|---|---|---|---|---|
| TechnicalAnalyst | Market data (OHLCV) | `TechnicalAnalysis` | Core | Yes |
| SentimentAnalyst | News articles | `SentimentAnalysis` | Core | No (FinBERT) |
| NewsAnalyst | News articles | `NewsAnalysis` | Core | Yes |
| FundamentalAnalyst | Symbol | `FundamentalAnalysis` | Optional | Yes |
| ComparativeAnalyst | Symbol + market data | `ComparativeAnalysis` | Optional | Yes |
| WebResearchAgent | Symbol | `WebResearchAnalysis` | Optional | Yes |
| SocialSentimentAnalyst | Symbol | `SocialSentimentAnalysis` | Optional | Yes |
| TrumpAnalyst | Truth Social posts | `TrumpAnalysis` | Optional | Yes |
| BullishResearcher | Group 1 results | `BullishResearchAnalysis` | Core | Yes |
| BearishResearcher | Group 1 results | `BearishResearchAnalysis` | Core | Yes |
| TraderAgent | All analyses | `TradingDecision` | Core | Yes |
| RiskManagementAgent | Decision + account | `RiskAssessment` | Core | No |
| MetaAgent | Market data | `StrategySelection` | Core | No |

## Indicator Thresholds

| Indicator | Parameter | Value | Source |
|---|---|---|---|
| RSI period | `rsi_period` | 14 | `momentum.py` |
| RSI oversold | `rsi_oversold` | 30.0 | `momentum.py` |
| RSI overbought | `rsi_overbought` | 70.0 | `momentum.py` |
| MACD fast | `macd_fast` | 12 | `momentum.py` |
| MACD slow | `macd_slow` | 26 | `momentum.py` |
| MACD signal | `macd_signal` | 9 | `momentum.py` |
| BB period | `bb_period` | 20 | `mean_reversion.py` |
| BB std dev | `bb_std` | 2.0 | `mean_reversion.py` |
| SMA fast | `sma_fast` | 50 (default) / 20 (meta) | `trend_following.py` |
| SMA slow | `sma_slow` | 200 (default) / 50 (meta) | `trend_following.py` |
| ADX period | `adx_period` | 14 | `trend_following.py` |
| ADX threshold | `adx_threshold` | 25.0 | `trend_following.py` |
| ATR period | — | 14 | `risk.py` |
| ATR multiplier | `ATR_MULTIPLIER` | 2.0 | `risk.py` |

## Risk Constants

| Constant | Value | Description |
|---|---|---|
| `MAX_POSITION_RISK_PERCENT` | 2.0% | Max risk per trade |
| `MAX_TOTAL_EXPOSURE_PERCENT` | 80.0% | Max total portfolio exposure |
| `MAX_SINGLE_POSITION_PERCENT` | 20.0% | Max single position size |
| `DEFAULT_STOP_LOSS_PERCENT` | 2.0% | Fallback stop-loss |
| `ATR_MULTIPLIER` | 2.0 | ATR-based stop-loss multiplier |
| `TRAILING_STOP_PERCENT` | 3.0% | Trailing stop distance |
| `TRAILING_ACTIVATION_PERCENT` | 5.0% | Trailing stop activation |
| `MIN_DECISION_CONFIDENCE` | 0.6 | Minimum confidence to approve |
| `RISK_LEVEL_LOW_THRESHOLD` | 0.75 | Risk score for LOW level |
| `RISK_LEVEL_MEDIUM_THRESHOLD` | 0.5 | Risk score for MEDIUM level |
| `REJECTED_CONFIDENCE_PENALTY` | 0.3 | Confidence penalty if rejected |
| `RISK_SCORE_WEIGHT` | 0.6 | Risk score weight in final confidence |
| `DECISION_CONFIDENCE_WEIGHT` | 0.4 | Decision confidence weight |

**Risk score formula:**
```
risk_score = (risk_component × 0.3) + (exposure_component × 0.3) + (confidence × 0.4)
```

---

**See also:** [Daemon Architecture](daemon-architecture.md) | [Daemon Operations](daemon-operations.md)
