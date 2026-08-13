# Resume Metric Verification

All results below come from seeded **synthetic** sessions. They are not
historical, calibrated, annualized, or live-market performance.

## Reproduction scope

- C++20 matching-engine tests: run separately with warnings treated as errors.
- Python tests: cover deterministic generation, accounting, all four policies,
  paired benchmarking, PPO action controls, and the pre-trade leakage guard.
- Rule-based benchmark: **5,000 sessions**, four regimes,
  and four strategies.
- PPO training: **200,000 timesteps** across randomized
  regimes.
- PPO holdout: **1,000 disjoint sessions** over
  seeds 100001-101000.

## Rule-based result

The strongest session P&L Sharpe came from `bayesian_toxicity`:

- session P&L Sharpe: **0.328**, versus
  **-0.044** for fixed spread;
- mean net P&L: **40.11** synthetic ticks, versus
  **-30.62**;
- mean peak inventory reduction: **79.9%**;
- mean negative one-step markout loss reduction: **60.9%**;
- paired bootstrap 95% interval for the net P&L difference: **[51.99, 89.42]**.

## PPO result

Against Avellaneda-Stoikov on the disjoint holdout:

- mean net P&L changed **+59.3%**;
- mean peak inventory changed **-23.5%**;
- PPO session P&L Sharpe was **0.091**, versus
  **0.043**.

PPO should be presented as a leakage-free research policy and an empirical
risk-return tradeoff. Do not claim it beat the strongest rule-based strategy
unless the recorded holdout values actually show that.

## Resume-safe wording

> Implemented a C++20 L2 limit order book with price-time priority, FIFO price
> levels, partial fills, and locator-based cancellation, with deterministic
> invariant tests compiled under warnings-as-errors.

> Benchmarked four quoting policies over 5,000 common-seed
> synthetic sessions across volatility and cost regimes; `bayesian_toxicity` reduced
> mean peak inventory 80% and negative one-step markout loss
> 61% versus fixed-spread quoting.

> Built a leakage-free Gymnasium environment and trained PPO across randomized
> regimes, then evaluated it against four rule-based policies on
> 1,000 disjoint synthetic holdout sessions.
