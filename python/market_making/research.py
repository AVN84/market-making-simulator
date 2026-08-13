"""Research harness for synthetic market-making strategy comparisons.

The module is deliberately explicit about its assumptions. Strategies quote
before the next aggressive order is revealed, every strategy receives the same
seeded event stream and fill uniforms, and all P&L values are synthetic ticks.
"""

from __future__ import annotations

import math
import random
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from statistics import mean, pstdev
from typing import Literal, Protocol

AggressorSide = Literal["buy", "sell"]
StrategyName = Literal[
    "fixed_spread",
    "inventory_aware",
    "avellaneda_stoikov",
    "bayesian_toxicity",
]


@dataclass(frozen=True)
class MarketRegime:
    name: str
    volatility_ticks: float
    informed_trade_probability: float
    informed_impact_ticks: int
    flow_persistence: float
    transaction_cost_ticks: float


REGIMES: tuple[MarketRegime, ...] = (
    MarketRegime("calm_low_cost", 0.65, 0.12, 2, 0.55, 0.02),
    MarketRegime("calm_high_cost", 0.65, 0.12, 2, 0.55, 0.15),
    MarketRegime("volatile_low_cost", 1.35, 0.32, 3, 0.72, 0.02),
    MarketRegime("volatile_high_cost", 1.35, 0.32, 3, 0.72, 0.15),
)


@dataclass(frozen=True)
class SessionConfig:
    seed: int = 7
    steps: int = 500
    initial_mid_price: int = 10_000
    base_half_spread_ticks: int = 2
    max_quote_size: int = 2
    inventory_skew_ticks: float = 0.32
    inventory_widen_threshold: int = 4
    queue_base_fill_probability: float = 0.35
    queue_fill_probability_per_crossed_tick: float = 0.15
    history_window: int = 20
    as_risk_aversion: float = 0.08
    as_arrival_decay: float = 0.75
    toxicity_ewma_alpha: float = 0.15
    toxicity_center_ticks: float = 1.25
    toxicity_spread_ticks: float = 1.5
    liquidation_half_spread_ticks: float = 2.0
    regime: MarketRegime = REGIMES[0]


@dataclass(frozen=True)
class ResearchEvent:
    timestamp: int
    mid_price: int
    next_mid_price: int
    aggressor_side: AggressorSide
    limit_price: int
    quantity: int
    aggression_ticks: int
    informed: bool
    fill_uniform: float


@dataclass(frozen=True)
class Quote:
    bid: int
    ask: int
    size: int


@dataclass(frozen=True)
class PublicState:
    mid_price: int
    inventory: int
    step: int
    horizon: int
    recent_volatility: float
    flow_imbalance: float
    toxicity_estimate: float
    recent_return: float


@dataclass(frozen=True)
class SessionResult:
    seed: int
    regime: str
    strategy: str
    steps: int
    quote_opportunities: int
    fill_opportunity_quantity: int
    fills: int
    buy_fills: int
    sell_fills: int
    final_inventory: int
    max_abs_inventory: int
    mean_abs_inventory: float
    gross_spread_capture_pnl: float
    transaction_cost_pnl: float
    post_fill_markout_pnl: float
    negative_markout_quantity: int
    negative_markout_loss: float
    liquidation_cost_pnl: float
    mark_to_market_pnl: float
    net_pnl: float

    @property
    def fill_ratio(self) -> float:
        if self.fill_opportunity_quantity == 0:
            return 0.0
        return self.fills / self.fill_opportunity_quantity

    def as_row(self) -> dict[str, str | int | float]:
        row = asdict(self)
        row["fill_ratio"] = round(self.fill_ratio, 8)
        return row


class Quoter(Protocol):
    def quote(self, state: PublicState) -> Quote: ...


def _validate_config(config: SessionConfig) -> None:
    if config.steps <= 1:
        raise ValueError("steps must be greater than one")
    if config.base_half_spread_ticks <= 0 or config.max_quote_size <= 0:
        raise ValueError("spread and quote size must be positive")
    if config.history_window <= 1:
        raise ValueError("history_window must be greater than one")
    for name, value in (
        ("queue_base_fill_probability", config.queue_base_fill_probability),
        ("queue_fill_probability_per_crossed_tick", config.queue_fill_probability_per_crossed_tick),
        ("informed_trade_probability", config.regime.informed_trade_probability),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between zero and one")


def generate_session(config: SessionConfig) -> list[ResearchEvent]:
    """Generate a deterministic session with latent informed flow.

    The current order is never exposed to a strategy before it quotes. Informed
    flow is aligned with a subsequent synthetic mid move, which creates a
    controlled adverse-selection channel without claiming market calibration.
    """

    _validate_config(config)
    rng = random.Random(config.seed)
    mid = config.initial_mid_price
    flow_signal = 0.0
    events: list[ResearchEvent] = []

    for timestamp in range(config.steps):
        flow_signal = (
            config.regime.flow_persistence * flow_signal
            + rng.gauss(0.0, 1.0)
        )
        informed = rng.random() < config.regime.informed_trade_probability
        if informed:
            direction = 1 if flow_signal >= 0.0 else -1
        else:
            buy_probability = 1.0 / (1.0 + math.exp(-0.55 * flow_signal))
            direction = 1 if rng.random() < buy_probability else -1

        noise_move = int(round(rng.gauss(0.0, config.regime.volatility_ticks)))
        impact_move = direction * config.regime.informed_impact_ticks if informed else 0
        next_mid = max(1, mid + noise_move + impact_move)
        aggression_pool = (2, 3, 3, 4) if informed else (0, 1, 1, 2, 3)
        aggression = rng.choice(aggression_pool)
        side: AggressorSide = "buy" if direction > 0 else "sell"
        limit_price = mid + aggression if side == "buy" else mid - aggression
        events.append(
            ResearchEvent(
                timestamp=timestamp,
                mid_price=mid,
                next_mid_price=next_mid,
                aggressor_side=side,
                limit_price=limit_price,
                quantity=rng.choice((1, 1, 2)),
                aggression_ticks=aggression,
                informed=informed,
                fill_uniform=rng.random(),
            )
        )
        mid = next_mid

    return events


def _safe_quote(center: float, half_spread: float, size: int) -> Quote:
    half_spread = max(1.0, half_spread)
    bid = int(math.floor(center - half_spread))
    ask = int(math.ceil(center + half_spread))
    if bid >= ask:
        ask = bid + 1
    return Quote(bid=bid, ask=ask, size=size)


class FixedSpreadQuoter:
    def __init__(self, config: SessionConfig) -> None:
        self.config = config

    def quote(self, state: PublicState) -> Quote:
        return _safe_quote(
            state.mid_price,
            self.config.base_half_spread_ticks,
            self.config.max_quote_size,
        )


class InventoryAwareQuoter:
    def __init__(self, config: SessionConfig) -> None:
        self.config = config

    def quote(self, state: PublicState) -> Quote:
        center = state.mid_price - self.config.inventory_skew_ticks * state.inventory
        widen = abs(state.inventory) // self.config.inventory_widen_threshold
        return _safe_quote(
            center,
            self.config.base_half_spread_ticks + widen,
            self.config.max_quote_size,
        )


class AvellanedaStoikovQuoter:
    """Discrete-tick approximation of Avellaneda-Stoikov reservation pricing."""

    def __init__(self, config: SessionConfig) -> None:
        self.config = config

    def quote(self, state: PublicState) -> Quote:
        remaining = max(0.0, 1.0 - state.step / state.horizon)
        variance = max(0.25, state.recent_volatility**2)
        gamma = self.config.as_risk_aversion
        kappa = self.config.as_arrival_decay
        reservation = state.mid_price - state.inventory * gamma * variance * remaining
        theoretical_half_spread = (
            0.5 * gamma * variance * remaining
            + math.log1p(gamma / kappa) / gamma
        )
        half_spread = max(self.config.base_half_spread_ticks, theoretical_half_spread)
        return _safe_quote(reservation, half_spread, self.config.max_quote_size)


class BayesianToxicityQuoter:
    """Glosten-Milgrom-inspired belief update over public order-flow history.

    This is not a full structural Glosten-Milgrom calibration. It uses an EWMA
    toxicity belief to shift fair value toward persistent public flow and widen
    quotes when recent signed price responses suggest informed trading.
    """

    def __init__(self, config: SessionConfig) -> None:
        self.config = config

    def quote(self, state: PublicState) -> Quote:
        flow_shift = self.config.toxicity_center_ticks * state.flow_imbalance
        inventory_shift = self.config.inventory_skew_ticks * state.inventory
        center = state.mid_price + flow_shift - inventory_shift
        half_spread = (
            self.config.base_half_spread_ticks
            + self.config.toxicity_spread_ticks * state.toxicity_estimate
        )
        return _safe_quote(center, half_spread, self.config.max_quote_size)


def make_quoter(config: SessionConfig, strategy: StrategyName) -> Quoter:
    if strategy == "fixed_spread":
        return FixedSpreadQuoter(config)
    if strategy == "inventory_aware":
        return InventoryAwareQuoter(config)
    if strategy == "avellaneda_stoikov":
        return AvellanedaStoikovQuoter(config)
    if strategy == "bayesian_toxicity":
        return BayesianToxicityQuoter(config)
    raise ValueError(f"unknown strategy: {strategy}")


def fill_probability(config: SessionConfig, event: ResearchEvent, quote: Quote) -> float:
    crossed_ticks = (
        event.limit_price - quote.ask
        if event.aggressor_side == "buy"
        else quote.bid - event.limit_price
    )
    if crossed_ticks < 0:
        return 0.0
    probability = (
        config.queue_base_fill_probability
        + crossed_ticks * config.queue_fill_probability_per_crossed_tick
    )
    return min(1.0, max(0.0, probability))


class PublicHistory:
    """Rolling public features updated only after each order arrives."""

    def __init__(self, config: SessionConfig) -> None:
        self._returns: deque[float] = deque(maxlen=config.history_window)
        self._sides: deque[int] = deque(maxlen=config.history_window)
        self._toxicity = 0.0
        self._alpha = config.toxicity_ewma_alpha

    def state(self, mid: int, inventory: int, step: int, horizon: int) -> PublicState:
        volatility = pstdev(self._returns) if len(self._returns) > 1 else 0.0
        flow_imbalance = mean(self._sides) if self._sides else 0.0
        recent_return = self._returns[-1] if self._returns else 0.0
        return PublicState(
            mid_price=mid,
            inventory=inventory,
            step=step,
            horizon=horizon,
            recent_volatility=volatility,
            flow_imbalance=flow_imbalance,
            toxicity_estimate=self._toxicity,
            recent_return=recent_return,
        )

    def update(self, event: ResearchEvent) -> None:
        side = 1 if event.aggressor_side == "buy" else -1
        price_return = event.next_mid_price - event.mid_price
        adverse_response = min(1.0, max(0.0, side * price_return / 4.0))
        self._toxicity = (
            (1.0 - self._alpha) * self._toxicity
            + self._alpha * adverse_response
        )
        self._returns.append(float(price_return))
        self._sides.append(side)


def run_session(
    config: SessionConfig,
    strategy: StrategyName,
    events: Sequence[ResearchEvent] | None = None,
) -> SessionResult:
    """Evaluate one strategy on one session with pre-trade information only."""

    _validate_config(config)
    replay = list(events) if events is not None else generate_session(config)
    if len(replay) != config.steps:
        raise ValueError("event count must equal config.steps")
    quoter = make_quoter(config, strategy)
    history = PublicHistory(config)
    inventory = 0
    cash = 0.0
    fills = buy_fills = sell_fills = 0
    quote_opportunities = fill_opportunity_quantity = 0
    max_abs_inventory = inventory_time_sum = 0
    gross_spread_capture = transaction_cost = post_fill_markout = 0.0
    negative_markout_quantity = 0
    negative_markout_loss = 0.0

    for event in replay:
        state = history.state(event.mid_price, inventory, event.timestamp, config.steps)
        quote = quoter.quote(state)
        probability = fill_probability(config, event, quote)
        quantity = min(event.quantity, quote.size)
        if probability > 0.0:
            quote_opportunities += 1
            fill_opportunity_quantity += quantity

        if probability > 0.0 and event.fill_uniform < probability:
            fee = config.regime.transaction_cost_ticks * quantity
            if event.aggressor_side == "buy":
                inventory -= quantity
                cash += quote.ask * quantity - fee
                sell_fills += quantity
                spread_capture = (quote.ask - event.mid_price) * quantity
                markout = (quote.ask - event.next_mid_price) * quantity
            else:
                inventory += quantity
                cash -= quote.bid * quantity + fee
                buy_fills += quantity
                spread_capture = (event.mid_price - quote.bid) * quantity
                markout = (event.next_mid_price - quote.bid) * quantity
            fills += quantity
            gross_spread_capture += spread_capture
            transaction_cost += fee
            post_fill_markout += markout
            if markout < 0.0:
                negative_markout_quantity += quantity
                negative_markout_loss += -markout

        max_abs_inventory = max(max_abs_inventory, abs(inventory))
        inventory_time_sum += abs(inventory)
        history.update(event)

    final_mid = replay[-1].next_mid_price
    mark_to_market = cash + inventory * final_mid
    liquidation_cost = abs(inventory) * (
        config.liquidation_half_spread_ticks + config.regime.transaction_cost_ticks
    )
    return SessionResult(
        seed=config.seed,
        regime=config.regime.name,
        strategy=strategy,
        steps=config.steps,
        quote_opportunities=quote_opportunities,
        fill_opportunity_quantity=fill_opportunity_quantity,
        fills=fills,
        buy_fills=buy_fills,
        sell_fills=sell_fills,
        final_inventory=inventory,
        max_abs_inventory=max_abs_inventory,
        mean_abs_inventory=inventory_time_sum / config.steps,
        gross_spread_capture_pnl=gross_spread_capture,
        transaction_cost_pnl=transaction_cost,
        post_fill_markout_pnl=post_fill_markout,
        negative_markout_quantity=negative_markout_quantity,
        negative_markout_loss=negative_markout_loss,
        liquidation_cost_pnl=liquidation_cost,
        mark_to_market_pnl=mark_to_market,
        net_pnl=mark_to_market - liquidation_cost,
    )


def session_configs(
    count: int,
    *,
    steps: int = 500,
    seed_start: int = 1,
    regimes: Iterable[MarketRegime] = REGIMES,
) -> list[SessionConfig]:
    if count <= 0:
        raise ValueError("count must be positive")
    regime_list = tuple(regimes)
    if not regime_list:
        raise ValueError("at least one regime is required")
    return [
        SessionConfig(
            seed=seed_start + index,
            steps=steps,
            regime=regime_list[index % len(regime_list)],
        )
        for index in range(count)
    ]


def with_regime(config: SessionConfig, regime: MarketRegime) -> SessionConfig:
    return replace(config, regime=regime)
