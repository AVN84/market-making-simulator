# Interview Guide

## Thirty-second version

I built a C++20 L2 limit order book and a separate Python research harness for
market-making strategy experiments. The C++ book enforces price-time priority
and supports partial fills plus direct cancellation. The Python side evaluates
four policies on the same seeded event streams, with transaction costs,
inventory liquidation, and a bounded queue-fill model. Across 5,000 synthetic
sessions, the Bayesian toxicity policy raised session P&L Sharpe from -0.044 for
fixed spread to 0.328. It reduced mean peak inventory by about 80% and negative
one-step markout loss by about 61%. I also trained a leakage-free PPO policy for
200,000 steps. On 1,000 disjoint holdout sessions, it improved mean net P&L by
59% and reduced peak inventory by 24% versus the Avellaneda-Stoikov
approximation. The Bayesian policy still performed best overall.

## What is actually implemented

- C++20 order book with integer ticks and FIFO price levels.
- Direct order-ID locator for cancellation.
- Deterministic matching tests compiled with warnings as errors.
- Four synthetic regimes covering volatility and transaction costs.
- Shared event streams and shared per-event fill uniforms.
- Fixed-spread and inventory-aware policies.
- A discrete Avellaneda-Stoikov approximation.
- A Glosten-Milgrom-inspired Bayesian toxicity policy.
- A Gymnasium environment with 27 bounded quote actions.
- Stable-Baselines3 PPO with disjoint training and evaluation seeds.
- Row-level CSV, aggregate JSON, readable reports, and paired bootstrap
  intervals.

## Result table

Rule-based benchmark: 5,000 synthetic sessions of 500 events.

| Policy | Mean net P&L | Session P&L Sharpe | Mean peak inventory | Mean adverse loss |
| --- | ---: | ---: | ---: | ---: |
| Fixed spread | -30.62 | -0.044 | 21.01 | 68.47 |
| Inventory aware | 19.98 | 0.120 | 4.77 | 41.64 |
| Avellaneda-Stoikov | 16.55 | 0.084 | 7.59 | 45.33 |
| Bayesian toxicity | 40.11 | 0.328 | 4.22 | 26.78 |

PPO holdout: 1,000 disjoint synthetic sessions of 300 events.

| Policy | Mean net P&L | Session P&L Sharpe | Mean peak inventory | Mean adverse loss |
| --- | ---: | ---: | ---: | ---: |
| Avellaneda-Stoikov | 6.22 | 0.043 | 6.61 | 27.27 |
| PPO | 9.91 | 0.091 | 5.05 | 21.13 |
| Bayesian toxicity | 22.55 | 0.269 | 3.88 | 16.10 |

## Why there is no look-ahead

The policy acts before the current aggressive order is revealed. It sees only
inventory and rolling public history from earlier events. The order side,
urgency, informed flag, and next price move arrive after the quote is fixed.
The first observation is identical across seeds within a regime, and a unit
test protects that timing rule.

## Strategy intuition

### Fixed spread

Always quotes the same distance around the public mid. It earns more fills in
calm sessions but accumulates inventory and gets picked off in toxic flow.

### Inventory aware

Moves its reservation price against current inventory and widens when exposure
grows. It gives up fills in exchange for much lower position risk.

### Avellaneda-Stoikov

Uses the reservation price

```text
r = s - q * gamma * sigma^2 * time_remaining
```

Long inventory moves both quotes down, while short inventory moves them up.
Recent variance and time remaining also affect the spread.

### Bayesian toxicity

Persistent buy flow moves its public fair-value estimate upward, while
persistent sell flow moves it downward. If signed post-trade price responses
suggest informed flow, the policy widens. It is inspired by Glosten-Milgrom but
is not a structural calibration of that model.

### PPO

The observation uses six public pre-trade features. The action selects spread
width, inventory-skew strength, and a small public-flow tilt. Reward is marked
wealth change less a quadratic inventory penalty, with terminal liquidation.

## Questions to expect

### Why did the Bayesian policy beat PPO?

The synthetic generator has a simple persistent-flow structure that the
Bayesian rule matches directly. PPO improved over Avellaneda-Stoikov but did not
beat the strongest specialized baseline. That is why strong rule-based
benchmarks matter.

### Why use common random numbers?

Every policy sees the same events and fill uniforms. Differences are less
likely to come from one strategy receiving an easier random session.

### What does the Sharpe number mean?

It is mean net P&L divided by population P&L standard deviation across synthetic
sessions. It is not annualized and should never be presented as a live-trading
Sharpe ratio.

### Why keep C++ and Python separate?

The C++ component isolates matching invariants. Python makes the research
assumptions and experiments easy to inspect. The separation is testable, but it
leaves integration through bindings as future work.

### What would you do next?

Bind the C++ book into Python, add latency and exchange-specific fees, and
evaluate on carefully licensed replay data. Only then would it make sense to
discuss calibration or external validity.

## Do not claim

- No historical feed or live trading.
- No calibrated queue position.
- No exact structural Glosten-Milgrom implementation.
- No annualized Sharpe.
- No claim that PPO is the best policy.
- No claim that the Python simulator currently executes through the C++ book.
