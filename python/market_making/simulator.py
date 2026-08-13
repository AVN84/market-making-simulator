"""Deterministic synthetic replay and bounded execution assumptions.

This module is deliberately a learning/backtesting harness, not an exchange
simulator. Prices are integer ticks and every random choice is seed-controlled
so strategy comparisons are reproducible.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from typing import Literal

AggressorSide = Literal["buy", "sell"]
StrategyName = Literal["fixed_spread", "inventory_aware"]


@dataclass(frozen=True)
class MarketEvent:
    timestamp: int
    mid_price: int
    aggressor_side: AggressorSide
    limit_price: int
    quantity: int
    aggression_ticks: int


@dataclass(frozen=True)
class BacktestConfig:
    seed: int = 7
    steps: int = 500
    initial_mid_price: int = 10_000
    base_half_spread_ticks: int = 2
    inventory_skew_ticks: float = 0.35
    max_quote_size: int = 2
    queue_base_fill_probability: float = 0.35
    queue_fill_probability_per_crossed_tick: float = 0.15
    adverse_selection_probability: float = 0.55
    adverse_selection_min_aggression_ticks: int = 2
    adverse_selection_impact_ticks: int = 2


@dataclass(frozen=True)
class Quote:
    bid: int
    ask: int
    size: int


@dataclass(frozen=True)
class BacktestResult:
    strategy: StrategyName
    steps: int
    quote_opportunities: int
    fill_opportunity_quantity: int
    fills: int
    buy_fills: int
    sell_fills: int
    final_inventory: int
    final_cash: int
    final_mid_price: int
    mark_to_market_pnl: int
    max_abs_inventory: int
    gross_spread_capture_pnl: int
    post_fill_markout_pnl: int
    negative_markout_quantity: int
    negative_markout_loss: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


class FixedSpreadQuoter:
    """Two-sided reference strategy with no inventory response."""

    def __init__(self, config: BacktestConfig) -> None:
        self._config = config

    def quote(self, mid_price: int, inventory: int) -> Quote:
        del inventory
        return Quote(
            bid=mid_price - self._config.base_half_spread_ticks,
            ask=mid_price + self._config.base_half_spread_ticks,
            size=self._config.max_quote_size,
        )


class InventoryAwareQuoter:
    """Simple reservation-price quoter with inventory-dependent skew."""

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


class QueueAwareFillModel:
    """A bounded approximation of queue priority for marketable synthetic flow.

    A marketable event is not guaranteed to reach our resting quote: at the
    touch it fills with ``queue_base_fill_probability`` and each tick the order
    crosses through our quote adds a bounded incremental chance. This models
    unknown queue position and competing displayed liquidity. It does *not*
    model a real exchange queue, hidden liquidity, or latency.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self._config = config

    def fill_probability(self, event: MarketEvent, quote: Quote) -> float:
        crossed_ticks = (
            event.limit_price - quote.ask
            if event.aggressor_side == "buy"
            else quote.bid - event.limit_price
        )
        if crossed_ticks < 0:
            return 0.0
        probability = (
            self._config.queue_base_fill_probability
            + crossed_ticks * self._config.queue_fill_probability_per_crossed_tick
        )
        return min(1.0, max(0.0, probability))

    def should_fill(self, event: MarketEvent, quote: Quote, rng: random.Random) -> bool:
        return rng.random() < self.fill_probability(event, quote)


def _validate_config(config: BacktestConfig) -> None:
    if config.steps <= 0:
        raise ValueError("steps must be positive")
    if config.max_quote_size <= 0:
        raise ValueError("max_quote_size must be positive")
    for name, value in (
        ("queue_base_fill_probability", config.queue_base_fill_probability),
        ("queue_fill_probability_per_crossed_tick", config.queue_fill_probability_per_crossed_tick),
        ("adverse_selection_probability", config.adverse_selection_probability),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between zero and one")
    if config.adverse_selection_min_aggression_ticks < 0:
        raise ValueError("adverse_selection_min_aggression_ticks must be non-negative")
    if config.adverse_selection_impact_ticks <= 0:
        raise ValueError("adverse_selection_impact_ticks must be positive")


def generate_synthetic_replay(config: BacktestConfig) -> list[MarketEvent]:
    """Generate deterministic aggressive flow with bounded directional pressure.

    When an aggressive order has sufficient urgency, the next synthetic mid can
    move one tick in the aggressor's direction. This is a bounded adverse-
    selection proxy: it creates one-step post-fill markouts, not a calibrated
    price-impact model.
    """
    _validate_config(config)
    rng = random.Random(config.seed)
    mid = config.initial_mid_price
    events: list[MarketEvent] = []
    for timestamp in range(config.steps):
        side: AggressorSide = "buy" if rng.random() < 0.5 else "sell"
        aggression = rng.choice((0, 1, 2, 3, 4))
        limit = mid + aggression if side == "buy" else mid - aggression
        events.append(MarketEvent(timestamp, mid, side, limit, rng.choice((1, 1, 2)), aggression))

        noise = rng.choice((-1, 0, 0, 1))
        direction = 1 if side == "buy" else -1
        impact = 0
        if (
            aggression >= config.adverse_selection_min_aggression_ticks
            and rng.random() < config.adverse_selection_probability
        ):
            impact = direction * config.adverse_selection_impact_ticks
        mid += noise + impact
    return events


def _make_quoter(config: BacktestConfig, strategy: StrategyName):
    if strategy == "fixed_spread":
        return FixedSpreadQuoter(config)
    if strategy == "inventory_aware":
        return InventoryAwareQuoter(config)
    raise ValueError(f"unknown strategy: {strategy}")


def run_backtest(
    config: BacktestConfig | None = None,
    strategy: StrategyName = "inventory_aware",
) -> BacktestResult:
    """Replay synthetic flow against a selected two-sided quoting strategy."""
    config = config or BacktestConfig()
    _validate_config(config)
    inventory = 0
    cash = 0
    fills = buy_fills = sell_fills = max_abs_inventory = 0
    quote_opportunities = fill_opportunity_quantity = 0
    gross_spread_capture_pnl = post_fill_markout_pnl = 0
    negative_markout_quantity = negative_markout_loss = 0
    replay = generate_synthetic_replay(config)
    quoter = _make_quoter(config, strategy)
    fill_model = QueueAwareFillModel(config)
    execution_rng = random.Random(config.seed ^ 0x5F3759DF)

    for index, event in enumerate(replay):
        quote = quoter.quote(event.mid_price, inventory)
        fill_probability = fill_model.fill_probability(event, quote)
        quantity = min(event.quantity, quote.size)
        if fill_probability == 0.0:
            continue

        quote_opportunities += 1
        fill_opportunity_quantity += quantity
        if not fill_model.should_fill(event, quote, execution_rng):
            continue

        next_mid = replay[index + 1].mid_price if index + 1 < len(replay) else event.mid_price
        if event.aggressor_side == "buy":
            inventory -= quantity
            cash += quote.ask * quantity
            sell_fills += quantity
            gross_spread_capture = (quote.ask - event.mid_price) * quantity
            markout = (quote.ask - next_mid) * quantity
        else:
            inventory += quantity
            cash -= quote.bid * quantity
            buy_fills += quantity
            gross_spread_capture = (event.mid_price - quote.bid) * quantity
            markout = (next_mid - quote.bid) * quantity
        fills += quantity
        gross_spread_capture_pnl += gross_spread_capture
        post_fill_markout_pnl += markout
        if markout < 0:
            negative_markout_quantity += quantity
            negative_markout_loss += -markout
        max_abs_inventory = max(max_abs_inventory, abs(inventory))

    final_mid = replay[-1].mid_price
    return BacktestResult(
        strategy=strategy,
        steps=config.steps,
        quote_opportunities=quote_opportunities,
        fill_opportunity_quantity=fill_opportunity_quantity,
        fills=fills,
        buy_fills=buy_fills,
        sell_fills=sell_fills,
        final_inventory=inventory,
        final_cash=cash,
        final_mid_price=final_mid,
        mark_to_market_pnl=cash + inventory * final_mid,
        max_abs_inventory=max_abs_inventory,
        gross_spread_capture_pnl=gross_spread_capture_pnl,
        post_fill_markout_pnl=post_fill_markout_pnl,
        negative_markout_quantity=negative_markout_quantity,
        negative_markout_loss=negative_markout_loss,
    )


def main() -> None:
    print(run_backtest().to_json())


if __name__ == "__main__":
    main()
