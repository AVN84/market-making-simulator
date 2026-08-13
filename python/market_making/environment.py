"""Leakage-free Gymnasium environment for the synthetic research harness."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from market_making.research import (
    REGIMES,
    PublicHistory,
    PublicState,
    Quote,
    SessionConfig,
    fill_probability,
    generate_session,
)


@dataclass(frozen=True)
class ActionSpec:
    spread_offset_ticks: int
    inventory_skew_multiplier: float
    flow_tilt_ticks: int


ACTION_SPECS: tuple[ActionSpec, ...] = tuple(
    ActionSpec(spread_offset, inventory_multiplier, flow_tilt)
    for spread_offset in (0, 1, 2)
    for inventory_multiplier in (0.5, 1.0, 1.5)
    for flow_tilt in (-1, 0, 1)
)


class MarketMakingEnv(gym.Env[np.ndarray, int]):
    """Choose spread, inventory skew, and public-flow tilt before each order.

    Observations contain inventory and rolling public history only. The current
    aggressive order's side, urgency, informed flag, and future price move are
    intentionally hidden until after the action is selected.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: SessionConfig | None = None,
        *,
        randomize_reset_seed: bool = False,
        randomize_regime: bool = False,
        inventory_penalty: float = 0.01,
    ) -> None:
        super().__init__()
        config = config or SessionConfig()
        self.base_config = config
        self.config = config
        self._randomize_reset_seed = randomize_reset_seed
        self._randomize_regime = randomize_regime
        self._inventory_penalty = inventory_penalty
        self._reset_rng = random.Random(config.seed)
        self.action_space = spaces.Discrete(len(ACTION_SPECS))
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0, 0.0, -1.0, 0.0], dtype=np.float32),
            high=np.ones(6, dtype=np.float32),
            dtype=np.float32,
        )
        self._events = []
        self._history = PublicHistory(config)
        self._index = 0
        self._inventory = 0
        self._cash = 0.0
        self._fills = 0
        self._max_abs_inventory = 0
        self._inventory_time_sum = 0
        self._gross_spread_capture = 0.0
        self._transaction_cost = 0.0
        self._post_fill_markout = 0.0
        self._negative_markout_quantity = 0
        self._negative_markout_loss = 0.0

    def _public_state(self) -> PublicState:
        if not self._events:
            mid = self.config.initial_mid_price
        elif self._index < len(self._events):
            mid = self._events[self._index].mid_price
        else:
            mid = self._events[-1].next_mid_price
        return self._history.state(mid, self._inventory, self._index, self.config.steps)

    def _observation(self) -> np.ndarray:
        if self._index >= len(self._events):
            return np.zeros(6, dtype=np.float32)
        state = self._public_state()
        max_inventory = self.config.steps * self.config.max_quote_size
        volatility_scale = max(1.0, 4.0 * self.config.regime.volatility_ticks)
        return np.array(
            [
                np.clip(state.inventory / max_inventory, -1.0, 1.0),
                np.clip(state.flow_imbalance, -1.0, 1.0),
                np.clip(state.recent_volatility / volatility_scale, 0.0, 1.0),
                np.clip(state.toxicity_estimate, 0.0, 1.0),
                np.clip(state.recent_return / 6.0, -1.0, 1.0),
                np.clip(state.step / state.horizon, 0.0, 1.0),
            ],
            dtype=np.float32,
        )

    def quote_for_action(self, action: int, state: PublicState | None = None) -> Quote:
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action}")
        current = self._public_state() if state is None else state
        spec = ACTION_SPECS[action]
        center = (
            current.mid_price
            - self.config.inventory_skew_ticks
            * spec.inventory_skew_multiplier
            * current.inventory
            + spec.flow_tilt_ticks * current.flow_imbalance
        )
        half_spread = self.config.base_half_spread_ticks + spec.spread_offset_ticks
        bid = int(math.floor(center - half_spread))
        ask = max(bid + 1, int(math.ceil(center + half_spread)))
        return Quote(bid=bid, ask=ask, size=self.config.max_quote_size)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._reset_rng.seed(seed)
        replay_seed = (
            self._reset_rng.randrange(1, 2**31)
            if self._randomize_reset_seed
            else self.base_config.seed if seed is None else seed
        )
        if options and "regime_index" in options:
            regime = REGIMES[int(options["regime_index"]) % len(REGIMES)]
        elif self._randomize_regime:
            regime = REGIMES[self._reset_rng.randrange(len(REGIMES))]
        else:
            regime = self.base_config.regime
        self.config = replace(self.base_config, seed=replay_seed, regime=regime)
        self._events = generate_session(self.config)
        self._history = PublicHistory(self.config)
        self._index = 0
        self._inventory = 0
        self._cash = 0.0
        self._fills = 0
        self._max_abs_inventory = 0
        self._inventory_time_sum = 0
        self._gross_spread_capture = 0.0
        self._transaction_cost = 0.0
        self._post_fill_markout = 0.0
        self._negative_markout_quantity = 0
        self._negative_markout_loss = 0.0
        return self._observation(), {"seed": replay_seed, "regime": regime.name}

    def step(self, action: int):
        if self._index >= len(self._events):
            raise RuntimeError("episode has terminated; call reset before step")
        event = self._events[self._index]
        quote = self.quote_for_action(action)
        previous_wealth = self._cash + self._inventory * event.mid_price
        probability = fill_probability(self.config, event, quote)
        quantity = min(event.quantity, quote.size)
        filled = probability > 0.0 and event.fill_uniform < probability

        if filled:
            fee = self.config.regime.transaction_cost_ticks * quantity
            if event.aggressor_side == "buy":
                self._inventory -= quantity
                self._cash += quote.ask * quantity - fee
                spread_capture = (quote.ask - event.mid_price) * quantity
                markout = (quote.ask - event.next_mid_price) * quantity
            else:
                self._inventory += quantity
                self._cash -= quote.bid * quantity + fee
                spread_capture = (event.mid_price - quote.bid) * quantity
                markout = (event.next_mid_price - quote.bid) * quantity
            self._fills += quantity
            self._gross_spread_capture += spread_capture
            self._transaction_cost += fee
            self._post_fill_markout += markout
            if markout < 0.0:
                self._negative_markout_quantity += quantity
                self._negative_markout_loss += -markout

        marked_wealth = self._cash + self._inventory * event.next_mid_price
        reward = marked_wealth - previous_wealth - self._inventory_penalty * self._inventory**2
        self._max_abs_inventory = max(self._max_abs_inventory, abs(self._inventory))
        self._inventory_time_sum += abs(self._inventory)
        self._history.update(event)
        self._index += 1
        terminated = self._index >= len(self._events)
        liquidation_cost = 0.0
        if terminated:
            liquidation_cost = abs(self._inventory) * (
                self.config.liquidation_half_spread_ticks
                + self.config.regime.transaction_cost_ticks
            )
            reward -= liquidation_cost
        net_pnl = marked_wealth - liquidation_cost
        info = {
            "action": int(action),
            "filled_quantity": quantity if filled else 0,
            "inventory": self._inventory,
            "cash": self._cash,
            "mark_to_market_pnl": marked_wealth,
            "net_pnl": net_pnl,
            "max_abs_inventory": self._max_abs_inventory,
            "mean_abs_inventory": self._inventory_time_sum / self._index,
            "gross_spread_capture_pnl": self._gross_spread_capture,
            "transaction_cost_pnl": self._transaction_cost,
            "post_fill_markout_pnl": self._post_fill_markout,
            "negative_markout_quantity": self._negative_markout_quantity,
            "negative_markout_loss": self._negative_markout_loss,
            "liquidation_cost_pnl": liquidation_cost,
            "fill_probability": probability,
            "regime": self.config.regime.name,
        }
        return self._observation(), float(reward), terminated, False, info
