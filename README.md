# Market-Making Simulator

A compact, interview-ready baseline for learning how a matching engine and an
inventory-aware quoting strategy fit together. It has two deliberately separate
components:

- A **C++20 limit order book** with integer tick prices, FIFO price-time
  priority, limit-order matching, market-order matching, and cancellation.
- A **Python synthetic replay/backtest** using an inventory-aware, reservation
  price quoting baseline inspired by Avellaneda-Stoikov-style inventory skew.

The Python replay is intentionally synthetic and deterministic. It makes the
project runnable without external exchange data, API keys, or large downloads.

## Architecture

```text
C++20 matching engine
  OrderId -> resting-order locator
  price level -> FIFO list of orders
  bids: descending price map | asks: ascending price map

Python backtest
  seeded synthetic aggressive orders
        -> inventory-aware bid/ask quote
        -> simulated fills
        -> cash, inventory, mark-to-market P&L metrics
```

The C++ engine is a standalone core with explicit tests. The Python baseline is
also standalone so its assumptions are easy to inspect during an interview.

## Quick Start

### C++ limit order book

Requires CMake 3.20+ and a C++20 compiler.

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

### Python synthetic replay

Requires Python 3.10+. There are no third-party Python dependencies.

```bash
PYTHONPATH=python python3 -m market_making.simulator
PYTHONPATH=python python3 -m unittest discover -s python/tests -v
```

The command prints deterministic JSON for the default seed. Change
`BacktestConfig(seed=..., steps=...)` in a script or REPL to explore another
reproducible scenario.

## Metrics Reported

The backtest reports:

- total fills, buy fills, and sell fills
- final cash and inventory
- final synthetic mid-price
- mark-to-market P&L: `cash + inventory * final_mid_price`
- maximum absolute inventory observed

These values describe only the generated scenario. They are not trading
performance claims and should not be extrapolated to live markets.

## Tests

The C++ tests cover price-time priority, cancellation, and the invariant that
unfilled market-order quantity does not rest on the book. The Python tests cover
deterministic replay generation and a deterministic end-to-end backtest.

## Limitations

- The components are intentionally separate; the Python baseline does not call
  the C++ order book through bindings yet.
- The replay is synthetic, not historical market data, and has no calibration.
- There are no fees, latency, queue-position uncertainty, adverse selection,
  partial quote cancellations, or exchange-specific rules.
- The quote model is a small inventory-aware heuristic, not a fitted
  Avellaneda-Stoikov implementation and not PPO/RL.
- This is educational software, not investment or execution advice.
