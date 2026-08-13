# Experiment Design

## Question

How do fixed, inventory-aware, stochastic-control, Bayesian, and learned quoting
policies trade off synthetic P&L against inventory and adverse selection when
they receive the same market events?

The project is a controlled software experiment. It does not estimate how any
policy would perform on a real exchange.

## Timing and leakage guard

At event `t`, a policy observes only:

- inventory after event `t - 1`;
- trailing signed order flow;
- trailing mid-price volatility;
- trailing signed price response;
- the previous mid-price change; and
- time remaining.

The policy then submits a quote. Only after that action is fixed does the
simulator reveal the next aggressive order and the subsequent mid-price move.
The first observation is identical across seeds within a regime because no
order has arrived yet. A regression test enforces this property.

## Common random numbers

Each session is generated once and replayed against every rule-based strategy.
An event includes one uniform random draw for the queue-fill decision. A policy
with fill probability `p` fills when that shared draw is below `p`. This keeps
execution randomness aligned even when policies quote at different prices.

## Regimes

The benchmark crosses two volatility levels with two transaction-cost levels.
Volatile regimes also have more informed flow and larger informed price impact.
Sessions rotate through the four settings so each policy receives the same
number of observations from every regime.

## Strategies

### Fixed spread

Quotes a constant distance around the public mid and ignores inventory.

### Inventory aware

Moves the quote center against current inventory and widens after inventory
crosses fixed thresholds.

### Avellaneda-Stoikov approximation

The reservation price is

```text
r = s - q * gamma * sigma^2 * tau
```

where `s` is the public mid, `q` is inventory, `gamma` is risk aversion,
`sigma^2` is trailing variance, and `tau` is the fraction of the session left.
The spread uses the standard risk and arrival-decay terms, then rounds outward
to integer ticks.

### Bayesian toxicity

This policy is inspired by Glosten-Milgrom rather than presented as a structural
implementation. It uses public signed-flow persistence to shift its estimate of
fair value. An EWMA of signed post-trade price response widens the spread when
recent flow appears informed.

### PPO

PPO chooses one of 27 combinations of spread offset, inventory-skew strength,
and public-flow tilt. Reward is the one-step change in marked wealth less a
quadratic inventory penalty. Terminal reward also pays the same liquidation
cost used by the rule-based evaluator.

## Evaluation

The rule-based benchmark runs 5,000 sessions of 500 events. The PPO model trains
on randomized seeds and regimes, then evaluates on 1,000 sessions from a
disjoint seed range. The summary reports:

- net P&L after fees and liquidation;
- mean P&L divided by population P&L standard deviation across sessions;
- gross spread capture;
- negative one-step markout loss;
- maximum and mean absolute inventory; and
- fills plus fill ratio.

Paired bootstrap intervals resample per-session strategy differences. They are
deterministic because the bootstrap seed is fixed.

## Limitations

The market process is synthetic. Queue fills are a bounded probability model.
The experiment has no historical calibration, hidden liquidity, network
latency, exchange rebates, order amendments, or cross-asset interactions. The
C++ order book and Python research harness remain separate components.
