# Multi-Regime Strategy Benchmark

This report covers **5,000 seeded synthetic sessions**. It is not historical or live-market evidence.

| Strategy | Mean net P&L | Session P&L Sharpe | Mean peak inventory | Mean adverse loss | Mean fills |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed_spread | -30.62 | -0.044 | 21.01 | 68.47 | 161.68 |
| inventory_aware | 19.98 | 0.120 | 4.77 | 41.64 | 117.73 |
| avellaneda_stoikov | 16.55 | 0.084 | 7.59 | 45.33 | 124.92 |
| bayesian_toxicity | 40.11 | 0.328 | 4.22 | 26.78 | 93.78 |

## Method

- Every strategy quotes before the next aggressive order is revealed.
- Strategies share the same event stream and per-event fill uniform for paired comparisons.
- The regime grid varies volatility, informed-flow intensity, and transaction costs.
- Net P&L includes per-fill transaction costs and terminal inventory liquidation.
- Paired bootstrap intervals are stored in the JSON summary.
