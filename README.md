# Market-Making Simulator

An interview-oriented engineering project that separates a **C++20 limit order
book** from a **deterministic Python market-making simulator**. The goal is to
make the assumptions inspectable: matching logic lives in C++; synthetic replay,
strategy comparison, and an optional PPO experiment live in Python.

> All prices, P&L figures, fills, and experiments in this repository are
> synthetic tick-based simulations. This is educational software, not a live
> trading system, investment advice, or a claim about real-market performance.

## What is implemented

### C++20 matching engine

- Integer-tick price levels with price-time priority.
- FIFO resting orders at every price level.
- Limit-order matching, market-order matching, cancellation, and locator-based
  order lookup.
- Tests for FIFO behavior, cancellation, and non-resting unfilled market flow.

### Synthetic execution and strategy comparison

- Seeded aggressive order-flow generator with a bounded directional-pressure
  proxy after sufficiently aggressive orders.
- Fixed-spread reference strategy and inventory-aware reservation-price strategy.
- A **bounded queue-aware fill proxy**: a marketable order is not guaranteed to
  reach our quote. At the touch it fills with a configured base probability;
  every tick it crosses through the quote adds a capped probability increment.
- One-step post-fill markout tracking. Negative values mean the following
  synthetic mid moved against the fill; this is a diagnostic for the explicit
  adverse-selection proxy, not a calibrated impact estimate.
- Multi-seed experiment runner that emits row-level CSV, aggregate JSON, and a
  readable report under `artifacts/`.

### Optional RL milestone

- A Gymnasium environment where a policy selects one of five bounded inventory
  skew strengths per event.
- Stable-Baselines3 PPO training/evaluation script. It is intentionally a small
  smoke-test workflow, not evidence of a production-quality RL strategy.

## Architecture

```text
C++20 order book
  order ID -> resting-order locator
  bids (descending map) / asks (ascending map)
  price level -> FIFO order list

Python synthetic simulator
  seeded aggressive order flow
        -> fixed or inventory-aware quote
        -> bounded queue-aware fill decision
        -> cash, inventory, mark-to-market, post-fill markout
        -> multi-seed artifacts

Optional Gymnasium/PPO layer
  observation: mid displacement, inventory, order-flow side/urgency, time
  action: one of five inventory-skew multipliers
  reward: one-step marked-wealth change less a small inventory penalty
```

The Python simulator currently does not bind to the C++ book. Keeping the two
layers separate makes the matching invariants and replay assumptions easier to
review. A future integration could replay events through C++ bindings.

## Quick start

### C++ tests

Requires a C++20 compiler. If CMake is installed:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

On machines without CMake, compile the test directly:

```bash
clang++ -std=c++20 -Wall -Wextra -Wpedantic -Icpp/include \
  cpp/src/order_book.cpp cpp/tests/order_book_tests.cpp -o build/order_book_tests
./build/order_book_tests
```

### Python simulator and tests

Requires Python 3.10+.

```bash
PYTHONPATH=python python3 -m market_making.simulator
PYTHONPATH=python python3 -m unittest discover -s python/tests -v
```

### Reproduce the strategy experiment

The following compares both strategies across the same 100 deterministic seeds
and writes actual results to `artifacts/`:

```bash
PYTHONPATH=python python3 -m market_making.experiments --seeds 100 --steps 500
```

It produces:

- `artifacts/multi_seed_results.csv`: one row per strategy/seed.
- `artifacts/multi_seed_summary.json`: aggregate mean, population standard
  deviation, min, and max.
- `artifacts/multi_seed_report.md`: concise interpretation of those exact runs.

### Optional PPO smoke test

Gymnasium and PyTorch are required. Install Stable-Baselines3 in your local
Python environment if needed:

```bash
python3 -m pip install --user stable-baselines3==2.3.2
PYTHONPATH=python python3 -m market_making.train_ppo --timesteps 4096 --eval-seeds 30
```

The training script writes `artifacts/ppo_evaluation.csv` and
`artifacts/ppo_training_summary.json`. It evaluates an agent against the
fixed-skew action on common synthetic seeds. A small PPO run is a dependency and
integration check only; it must not be described as an optimized trading model.

## How to read the metrics

- **Mark-to-market P&L**: `cash + inventory * final_mid`. It is in synthetic
  ticks and is scenario-specific.
- **Post-fill markout**: one-step value of a fill at the next synthetic mid;
  negative values flag adverse movement after a fill.
- **Max absolute inventory**: the largest directional inventory exposure seen.
- **Filled quantity**: quantity that reached the quote under the queue proxy.

For a fair comparison, both strategies receive the identical seeded replay. The
inventory-aware strategy may lower inventory exposure by accepting fewer fills;
that trade-off is the result to discuss, not a claim that one strategy wins in
all markets.

## Interview walkthrough

1. Start with the C++ book: explain integer ticks, price-time priority, FIFO
   lists, and why order IDs need a locator for cancellation.
2. Explain why a backtest cannot assume every marketable order fills. Point to
   `QueueAwareFillModel` and name its limitation: it is a capped probability
   proxy for unknown queue position, not a matching-engine queue simulation.
3. Contrast the two quote policies. The fixed baseline ignores inventory; the
   inventory-aware rule shifts its reservation price and widens at larger
   inventory.
4. Show `artifacts/multi_seed_report.md`, emphasizing common seeds and
   synthetic-only scope.
5. Describe the Gym environment as a narrow research interface: PPO chooses a
   bounded skew multiplier; no claim is made beyond the recorded synthetic
   evaluation.

## Limitations and next steps

- Synthetic data only: no historical feeds, calibration, live orders, or real
  money.
- No exchange-specific queue, hidden liquidity, fees, latency, partial
  cancellations, or realistic cross-asset effects.
- Directional pressure and queue-aware fills are explicit bounded assumptions,
  not estimates fit to data.
- PPO uses a small discrete action space and lightweight run; it is not tuned or
  validated out of sample.
- Useful next engineering steps: bind the Python replay to the C++ book, add
  fees/latency, validate on carefully licensed historical data, and evaluate
  policies on held-out regimes with risk-adjusted metrics.
