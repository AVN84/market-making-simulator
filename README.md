# Market-Making Simulator

An interview-oriented market microstructure project with two testable layers:

- a C++20 price-time-priority limit order book; and
- a Python research harness for reproducible synthetic strategy experiments.

All prices, fills, P&L values, and comparisons in this repository come from
seeded synthetic simulations. This is educational software, not a live trading
system, investment advice, or evidence of real-market performance.

## What is implemented

### C++20 matching engine

- Integer-tick prices with ordered bid and ask maps.
- FIFO time priority within each price level.
- Limit and market matching, partial fills, cancellation, and direct order-ID
  lookup through stable list iterators.
- Deterministic tests for matching order, cancellation, and non-resting market
  order remainders.

### Synthetic research harness

- Four regimes spanning calm and volatile flow with low and high transaction
  costs.
- Persistent public order flow plus a latent informed-flow process that creates
  controlled adverse selection.
- A bounded queue-aware fill model. Strategies share the same event stream and
  per-event fill uniform, which supports paired comparisons.
- Explicit per-fill transaction costs and terminal inventory liquidation.
- Rolling public features computed only from orders that have already arrived.
  No strategy sees the next order's side, urgency, informed flag, or subsequent
  price move before quoting.

### Rule-based strategies

1. `fixed_spread`: constant two-sided reference quote.
2. `inventory_aware`: reservation-price skew and inventory-dependent widening.
3. `avellaneda_stoikov`: discrete-tick approximation using inventory, recent
   variance, risk aversion, and time remaining.
4. `bayesian_toxicity`: Glosten-Milgrom-inspired public-flow belief that shifts
   fair value and widens quotes when recent signed price responses look toxic.

The fourth policy is intentionally labeled "inspired." It is not a structural
calibration of the original Glosten-Milgrom model.

### PPO research policy

The Gymnasium environment exposes six pre-trade public features:

- normalized inventory;
- trailing flow imbalance;
- trailing volatility;
- an EWMA toxicity estimate;
- the previous mid-price change; and
- fraction of the session elapsed.

PPO chooses among 27 bounded quote actions covering spread width, inventory
skew strength, and a small public-flow tilt. Training randomizes both replay
seed and regime. Evaluation uses a disjoint seed range and compares PPO against
all four rule-based policies.

## Architecture

```text
C++20 matching engine
  ordered price levels -> FIFO orders -> trades and cancellation

Python research harness
  seeded latent market process
    -> public state before the next order
    -> strategy quote
    -> shared queue-fill uniform
    -> cash, inventory, costs, markouts, liquidation
    -> paired CSV and JSON artifacts

Gymnasium and PPO
  public-history observation -> bounded quote action
  -> marked-wealth reward less inventory risk and liquidation cost
  -> disjoint multi-regime holdout evaluation
```

The Python simulator does not call the C++ book. The C++ layer validates
matching invariants, while Python keeps research assumptions easy to inspect.
Binding them is a future integration step, not a completed claim.

## Quick start

### C++ tests

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Without CMake:

```bash
mkdir -p build
clang++ -std=c++20 -Wall -Wextra -Wpedantic -Werror -Icpp/include \
  cpp/src/order_book.cpp cpp/tests/order_book_tests.cpp \
  -o build/order_book_tests
./build/order_book_tests
```

### Python tests

```bash
PYTHONPATH=python python3 -m unittest discover -s python/tests -v
```

### Four-strategy benchmark

```bash
PYTHONPATH=python python3 -m market_making.benchmark \
  --sessions 5000 \
  --steps 500 \
  --output-dir artifacts/research/benchmark
```

Outputs:

- `strategy_results.csv`: one row per strategy and session.
- `strategy_summary.json`: aggregate and per-regime metrics, plus paired
  bootstrap confidence intervals against fixed spread.
- `strategy_report.md`: readable summary of the exact run.

### PPO training and holdout evaluation

Install the optional dependencies if needed:

```bash
python3 -m pip install -e '.[rl]'
```

Train and evaluate:

```bash
PYTHONPATH=python python3 -m market_making.train_ppo \
  --timesteps 200000 \
  --episode-steps 300 \
  --eval-sessions 1000 \
  --eval-seed-start 100001 \
  --output-dir artifacts/research/ppo
```

The output directory contains the saved model, row-level evaluation CSV, and a
JSON summary for PPO and all rule-based baselines.

## Metrics

- **Net P&L**: marked cash and inventory, minus transaction costs already paid
  and terminal liquidation cost.
- **Session P&L Sharpe**: mean net P&L across sessions divided by its population
  standard deviation. It is a synthetic cross-session score, not an annualized
  live-trading Sharpe ratio.
- **Gross spread capture**: execution price relative to the public mid at the
  time of the fill, before fees and later price movement.
- **Post-fill markout**: fill value at the next synthetic mid.
- **Negative markout loss**: magnitude of one-step markouts below zero.
- **Peak inventory**: maximum absolute position within a session.
- **Fill ratio**: filled quantity divided by quantity that could reach the
  quote under the crossing rule.

## Reproducibility and limitations

- Randomness is seed-controlled and paired across strategies.
- Benchmark confidence intervals use deterministic paired bootstrap resampling.
- Training and evaluation seed ranges are disjoint.
- The market process is synthetic and deliberately simple. It is not calibrated
  to an exchange, historical feed, or asset.
- The queue model is a bounded probability proxy, not actual queue position.
- The project omits hidden liquidity, latency races, order amendments, exchange
  rebates, and cross-asset effects.
- PPO is a research prototype. Its results should be reported even when a
  simpler policy performs better.

## Interview walkthrough

1. Start with the C++ book and explain price-time priority plus direct
   cancellation through the order locator.
2. State the synthetic scope and explain why strategies share event streams and
   fill uniforms.
3. Contrast inventory-aware, Avellaneda-Stoikov, and Bayesian toxicity quoting.
4. Explain the look-ahead safeguard: the quote is chosen before the next order
   is revealed.
5. Show the multi-regime benchmark and discuss the P&L, fill, and inventory
   tradeoff rather than claiming one universal winner.
6. Treat PPO as an experiment, then name C++ bindings and licensed replay data
   as the next serious engineering steps.
