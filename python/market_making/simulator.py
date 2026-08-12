"""A small, reproducible synthetic replay for testing market-making logic.

This module intentionally uses only Python's standard library. It is a learning
and interview project, not a production trading system or a calibrated model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import random
from typing import Literal


AggressorSide = Literal["buy", "sell"]


@dataclass(frozen=True)
class MarketEvent:
    timestamp: int
    mid_price: int
    aggressor_side: AggressorSide
    limit_price: int
    quantity: int


@dataclass(frozen=True)
class BacktestConfig:
    seed: int = 7
    steps: int = 500
    initial_mid_price: int = 10_000
    base_half_spread_ticks: int = 2
    inventory_skew_ticks: float = 0.35
    max_quote_size: int = 2


@dataclass(frozen=True)
class Quote:
    bid: int
    ask: int
    size: int


@dataclass(frozen=True)
class BacktestResult:
    steps: int
    fills: int
    buy_fills: int
    sell_fills: int
    final_inventory: int
    final_cash: int
    final_mid_price: int
    mark_to_market_pnl: int
    max_abs_inventory: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


class InventoryAwareQuoter:
    """Simple reservation-price quoter inspired by inventory-skew models."""

    def __init__(self, config: BacktestConfig) -> None:
        self._config = config

    def quote(self, mid_price: int, inventory: int) -> Quote:
        reservation = mid_price - self._config.inventory_skew_ticks * inventory
        half_spread = self._config.base_half_spread_ticks + min(abs(inventory), 3) // 2
        bid = int(round(reservation - half_spread))
        ask = int(round(reservation + half_spread))
        if bid >= ask:
            ask = bid + 1
        return Quote(bid=bid, ask=ask, size=self._config.max_quote_size)


def generate_synthetic_replay(config: BacktestConfig) -> list[MarketEvent]:
    """Generate a deterministic sequence of simple aggressive limit orders."""
    if config.steps <= 0:
        raise ValueError("steps must be positive")
    if config.max_quote_size <= 0:
        raise ValueError("max_quote_size must be positive")

    rng = random.Random(config.seed)
    mid = config.initial_mid_price
    events: list[MarketEvent] = []
    for timestamp in range(config.steps):
        mid += rng.choice((-1, 0, 0, 1))
        side: AggressorSide = "buy" if rng.random() < 0.5 else "sell"
        aggression = rng.choice((0, 1, 2, 3, 4))
        limit = mid + aggression if side == "buy" else mid - aggression
        events.append(MarketEvent(timestamp, mid, side, limit, rng.choice((1, 1, 2))))
    return events


def run_backtest(config: BacktestConfig = BacktestConfig()) -> BacktestResult:
    """Replay synthetic flow against one inventory-aware two-sided quote."""
    inventory = 0
    cash = 0
    fills = buy_fills = sell_fills = max_abs_inventory = 0
    replay = generate_synthetic_replay(config)
    quoter = InventoryAwareQuoter(config)

    for event in replay:
        quote = quoter.quote(event.mid_price, inventory)
        quantity = min(event.quantity, quote.size)
        if event.aggressor_side == "buy" and event.limit_price >= quote.ask:
            inventory -= quantity
            cash += quote.ask * quantity
            fills += quantity
            sell_fills += quantity
        elif event.aggressor_side == "sell" and event.limit_price <= quote.bid:
            inventory += quantity
            cash -= quote.bid * quantity
            fills += quantity
            buy_fills += quantity
        max_abs_inventory = max(max_abs_inventory, abs(inventory))

    final_mid = replay[-1].mid_price
    return BacktestResult(
        steps=config.steps,
        fills=fills,
        buy_fills=buy_fills,
        sell_fills=sell_fills,
        final_inventory=inventory,
        final_cash=cash,
        final_mid_price=final_mid,
        mark_to_market_pnl=cash + inventory * final_mid,
        max_abs_inventory=max_abs_inventory,
    )


def main() -> None:
    print(run_backtest().to_json())


if __name__ == "__main__":
    main()
