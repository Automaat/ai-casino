# Position Sizing Configuration Guide

## Overview

The position sizing system combines multiple methods (risk-based, portfolio optimization, confidence weighting, Monte Carlo) into a unified, configurable approach. You control the behavior with three high-level settings:

1. **Primary Goal** - What are you optimizing for?
2. **Risk Tolerance** - How aggressive are constraints?
3. **Complexity** - How sophisticated is the logic?

## Quick Start

Add to `daemon.yaml`:

```yaml
daemon:
  position_sizing:
    primary_goal: "balanced"        # maximize_returns | minimize_risk | balanced
    risk_tolerance: "moderate"      # conservative | moderate | aggressive
    complexity: "simple"            # simple | advanced
```

## Configuration Reference

### Primary Goal

**Determines the base position sizing method:**

| Value | Behavior | Best For |
|-------|----------|----------|
| `maximize_returns` | Use portfolio optimization weights primarily | Growth, rebalancing-driven strategies |
| `minimize_risk` | Use risk-based sizing (stop-loss distance) | Capital preservation, defensive trading |
| `balanced` | Blend both methods 50/50 | Moderate growth + risk management |

### Risk Tolerance

**Controls how strictly constraints are applied:**

| Value | Behavior | Max Risk | Max Position | Total Exposure |
|-------|----------|----------|--------------|----------------|
| `conservative` | Minimum of all methods, strict limits | 1% | 10-15% | 60-70% |
| `moderate` | Layered constraints | 2% | 20% | 80% |
| `aggressive` | Looser limits, confidence-weighted | 3-5% | 25-30% | 90% |

### Complexity

**Determines advanced features:**

| Value | Features | Use When |
|-------|----------|----------|
| `simple` | Basic layered constraints | Starting out, prefer transparency |
| `advanced` | + Confidence scaling<br>+ Monte Carlo adjustment | Have historical data, want adaptive sizing |

## How Position Sizing Works

### Method Selection Logic

```
if primary_goal == "maximize_returns":
    base_shares = portfolio_weight * portfolio_value / price
elif primary_goal == "minimize_risk":
    base_shares = (balance * risk_pct) / (price - stop_loss_price)
else:  # balanced
    opt_shares = portfolio_weight * portfolio_value / price
    risk_shares = (balance * risk_pct) / (price - stop_loss_price)
    base_shares = blend_weight_optimization * opt_shares + blend_weight_risk_based * risk_shares
```

### Constraint Layers (Always Applied)

```
1. Available Cash: shares = min(shares, available_cash / price)
2. Max Single Position: shares = min(shares, balance * max_single_position_pct / price)
3. Max Risk Per Trade: shares = min(shares, balance * max_risk_per_trade_pct / risk_per_share)
```

### Advanced Features (complexity="advanced")

**Confidence Scaling:**
```
if confidence < confidence_low_threshold:
    shares *= confidence_low_reduction_factor  # Default: 0.5
elif confidence < confidence_high_threshold:
    # Linear interpolation
    factor = interpolate(confidence, low_threshold, high_threshold)
    shares *= factor
```

**Monte Carlo Adjustment:**
```
if monte_carlo_stress_test_failed:
    shares *= monte_carlo_risk_multiplier  # Default: 0.7
```

## Configuration Examples

### 1. Conservative Growth (Safe)

```yaml
position_sizing:
  primary_goal: "maximize_returns"
  risk_tolerance: "conservative"
  complexity: "simple"
  max_risk_per_trade_pct: 1.0
  max_single_position_pct: 10.0
  max_total_exposure_pct: 70.0
```

**Result:** Small, diversified positions following optimizer, never exceeding 1% risk.

### 2. Aggressive Returns (Growth Focus)

```yaml
position_sizing:
  primary_goal: "maximize_returns"
  risk_tolerance: "aggressive"
  complexity: "advanced"
  max_risk_per_trade_pct: 3.0
  max_single_position_pct: 25.0
  confidence_scaling_enabled: true
  use_monte_carlo_adjustment: true
```

**Result:** Larger positions following optimizer, reduced on low confidence or high tail risk.

### 3. Safety First (Capital Preservation)

```yaml
position_sizing:
  primary_goal: "minimize_risk"
  risk_tolerance: "conservative"
  complexity: "simple"
  max_risk_per_trade_pct: 1.0
  max_single_position_pct: 15.0
  max_total_exposure_pct: 60.0
```

**Result:** All sizing based on stop-loss distance, very small risk per trade.

### 4. Balanced (Default, Recommended)

```yaml
position_sizing:
  primary_goal: "balanced"
  risk_tolerance: "moderate"
  complexity: "simple"
  max_risk_per_trade_pct: 2.0
  max_single_position_pct: 20.0
  max_total_exposure_pct: 80.0
  blend_weight_optimization: 0.5
  blend_weight_risk_based: 0.5
```

**Result:** 50/50 blend of optimizer + risk-based, moderate constraints.

### 5. Adaptive Advanced (Sophisticated)

```yaml
position_sizing:
  primary_goal: "balanced"
  risk_tolerance: "moderate"
  complexity: "advanced"
  blend_weight_optimization: 0.6
  blend_weight_risk_based: 0.4
  confidence_scaling_enabled: true
  confidence_high_threshold: 0.8
  confidence_low_threshold: 0.6
  confidence_low_reduction_factor: 0.5
  use_monte_carlo_adjustment: true
  monte_carlo_risk_multiplier: 0.7
```

**Result:** Optimizer-weighted blend, scales down on low confidence or stress test failures.

## Fine-Tuning Parameters

### Risk Limits (Always Applied)

```yaml
position_sizing:
  # Maximum risk if stop-loss hits (% of account balance)
  max_risk_per_trade_pct: 2.0  # Range: 0.1-10.0

  # Maximum single position size (% of account balance)
  max_single_position_pct: 20.0  # Range: 1.0-50.0

  # Maximum total portfolio exposure (% of account balance)
  max_total_exposure_pct: 80.0  # Range: 10.0-100.0
```

### Blend Weights (primary_goal="balanced")

```yaml
position_sizing:
  # Must sum to 1.0
  blend_weight_optimization: 0.5  # Range: 0.0-1.0
  blend_weight_risk_based: 0.5    # Range: 0.0-1.0
```

**Examples:**
- `0.7/0.3` - Trust optimizer more
- `0.3/0.7` - Trust risk management more
- `1.0/0.0` - Equivalent to `primary_goal: "maximize_returns"`

### Confidence Scaling (complexity="advanced")

```yaml
position_sizing:
  confidence_scaling_enabled: true

  # No reduction above this threshold
  confidence_high_threshold: 0.8  # Range: 0.5-1.0

  # Reduction applied below this threshold
  confidence_low_threshold: 0.6   # Range: 0.3-0.9

  # Multiplier for low confidence (<low_threshold)
  confidence_low_reduction_factor: 0.5  # Range: 0.1-0.9
```

**Behavior:**
- Confidence ≥ 0.8: Full position size
- Confidence 0.6-0.8: Linear interpolation
- Confidence < 0.6: 50% position size

### Monte Carlo Adjustment (complexity="advanced")

```yaml
position_sizing:
  # Requires monte_carlo.enabled=true
  use_monte_carlo_adjustment: false

  # Multiplier when stress test shows high tail risk
  monte_carlo_risk_multiplier: 0.7  # Range: 0.1-1.0
```

## Integration with Other Features

### With Portfolio Rebalancing

```yaml
position_sizing:
  primary_goal: "maximize_returns"  # Use optimizer weights
  risk_tolerance: "moderate"

rebalancing:
  enabled: true
  method: "max_sharpe"
  rebalance_threshold: 0.01
```

**Result:** Position sizes follow optimal portfolio weights, constrained by risk limits.

### With Monte Carlo Stress Testing

```yaml
position_sizing:
  complexity: "advanced"
  use_monte_carlo_adjustment: true
  monte_carlo_risk_multiplier: 0.7

monte_carlo:
  enabled: true
  adjust_position_sizing: true  # Must be enabled
  loss_threshold: 0.10
  max_acceptable_prob: 0.15
```

**Result:** Positions reduced by 30% when stress test fails.

### With Pre-Trade Backtesting

```yaml
position_sizing:
  complexity: "advanced"
  confidence_scaling_enabled: true

pre_trade_backtesting:
  enabled: true
  confidence_penalty_multiplier: 0.7
```

**Result:** Failed backtest reduces confidence → smaller position via confidence scaling.

## Migration from Legacy

### Environment Variables (Old)

```bash
MAX_POSITION_RISK=2.0
MAX_SINGLE_POSITION=20.0
MAX_EXPOSURE=80.0
```

### YAML Config (New, Recommended)

```yaml
position_sizing:
  max_risk_per_trade_pct: 2.0
  max_single_position_pct: 20.0
  max_total_exposure_pct: 80.0
```

**Note:** Config takes priority over env vars when both are set.

## Validation and Limits

### Automatic Validation

```yaml
# Blend weights must sum to 1.0
blend_weight_optimization: 0.5
blend_weight_risk_based: 0.5
# ✓ Valid

blend_weight_optimization: 0.6
blend_weight_risk_based: 0.3
# ✗ Invalid: sum=0.9 (must be 1.0)

# Confidence thresholds must be ordered
confidence_high_threshold: 0.8
confidence_low_threshold: 0.6
# ✓ Valid

confidence_high_threshold: 0.6
confidence_low_threshold: 0.8
# ✗ Invalid: low >= high
```

### Range Constraints

All parameters have validated ranges. See `src/daemon/config.py:PositionSizingConfig` for bounds.

## Troubleshooting

### Positions Too Small

**Symptom:** All positions < 5% despite 20% max

**Solutions:**
1. Check `risk_tolerance: "conservative"` → try `"moderate"`
2. Increase `max_risk_per_trade_pct` (e.g., 2.0 → 3.0)
3. Widen stop-losses (affects risk-based sizing)
4. If using `complexity: "advanced"`, check confidence thresholds

### Positions Too Large

**Symptom:** Positions exceed comfortable size

**Solutions:**
1. Lower `max_single_position_pct` (e.g., 20 → 15)
2. Enable confidence scaling to reduce on lower-confidence signals
3. Use `primary_goal: "minimize_risk"`
4. Set `risk_tolerance: "conservative"`

### Unexpected Sizing Behavior

**Debug:**
1. Check `primary_goal` - controls base method
2. If `"balanced"`, verify `blend_weight_*` sum to 1.0
3. Check risk limits - may be constraining too much
4. If `complexity: "advanced"`, review confidence/Monte Carlo settings

## FAQ

**Q: What's the default behavior?**
A: `balanced` goal, `moderate` tolerance, `simple` complexity. 50/50 blend of optimizer + risk-based, 2% max risk, 20% max position.

**Q: Can I use only portfolio optimization?**
A: Yes. Set `primary_goal: "maximize_returns"` and `blend_weight_optimization: 1.0`.

**Q: Can I use only risk-based?**
A: Yes. Set `primary_goal: "minimize_risk"` and `blend_weight_risk_based: 1.0`.

**Q: Do I need rebalancing enabled for this to work?**
A: No. Weight-based sizing only applies when rebalancing provides target weights. Otherwise falls back to risk-based.

**Q: What if I don't set position_sizing in YAML?**
A: Defaults are used (see above). System is backwards-compatible.

**Q: Can I override per symbol?**
A: Not currently. Position sizing is portfolio-wide. Consider using confidence adjustments instead.

## Next Steps

1. Start with defaults (balanced/moderate/simple)
2. Run in paper trading for 30+ days
3. Analyze position sizes in tearsheets
4. Adjust based on risk tolerance and performance
5. Graduate to `complexity: "advanced"` after sufficient historical data

For implementation details, see:
- Config model: `src/daemon/config.py:PositionSizingConfig`
- Risk agent: `src/agents/risk.py:RiskManagementAgent`
- Example config: `docs/daemon.yaml.example`
