"""A small Gymnasium environment over the deterministic synthetic replay.

The environment is intentionally compact. It lets a policy choose the strength
of inventory skew at each event; it is not connected to the C++ matching engine
and it does not model real market microstructure.
"""

from __future__ import annotations

from dataclasses import replace
import random
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from market_making.simulator import (
    BacktestConfig,
    QueueAwareFillModel,
    Quote,
    generate_synthetic_replay,
)


class MarketMakingEnv(gym.Env[np.ndarray, int]):
    """Choose a bounded inventory-skew strength against synthetic order flow.

    Actions 0--4 correspond to multipliers ``(0.0, 0.5, 1.0, 1.5, 2.0)`` on
    the configured reservation-price inventory skew. Reward is one-step change
    in marked wealth, less a small inventory holding penalty. All figures are
    synthetic tick values, not financial performance.
    """

    metadata = {"render_modes": []}
    _SKEW_MULTIPLIERS = (0.0, 0.5, 1.0, 1.5, 2.0)

    def __init__(self, config: BacktestConfig = BacktestConfig()) -> None:
        super().__init__()
        self.config = config
        self.action_space = spaces.Discrete(len(self._SKEW_MULTIPLIERS))
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.ones(5, dtype=np.float32),
            dtype=np.float32,
        )
        self._replay = []
        self._execution_rng = random.Random()
        self._fill_model = QueueAwareFillModel(config)
        self._index = 0
        self._inventory = 0
        self._cash = 0

    def _observation(self) -> np.ndarray:
        if self._index >= len(self._replay):
            return np.zeros(5, dtype=np.float32)
        event = self._replay[self._index]
        max_mid_displacement = self.config.steps * (1 + self.config.adverse_selection_impact_ticks)
        max_inventory = self.config.steps * self.config.max_quote_size
        return np.array(
            [
                (event.mid_price - self.config.initial_mid_price) / max_mid_displacement,
                self._inventory / max_inventory,
                1.0 if event.aggressor_side == "buy" else -1.0,
                event.aggression_ticks / 4.0,
                self._index / self.config.steps,
            ],
            dtype=np.float32,
        )

    def _quote(self, mid_price: int, action: int) -> Quote:
        multiplier = self._SKEW_MULTIPLIERS[action]
        reservation = mid_price - self.config.inventory_skew_ticks * multiplier * self._inventory
        half_spread = self.config.base_half_spread_ticks + min(abs(self._inventory), 3) // 2
        bid = int(round(reservation - half_spread))
        ask = max(bid + 1, int(round(reservation + half_spread)))
        return Quote(bid=bid, ask=ask, size=self.config.max_quote_size)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        del options
        replay_seed = self.config.seed if seed is None else seed
        replay_config = replace(self.config, seed=replay_seed)
        self._replay = generate_synthetic_replay(replay_config)
        self._execution_rng = random.Random(replay_seed ^ 0x5F3759DF)
        self._fill_model = QueueAwareFillModel(replay_config)
        self._index = 0
        self._inventory = 0
        self._cash = 0
        return self._observation(), {"seed": replay_seed}

    def step(self, action: int):
        if self._index >= len(self._replay):
            raise RuntimeError("episode has terminated; call reset before step")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action}")

        event = self._replay[self._index]
        quote = self._quote(event.mid_price, action)
        previous_wealth = self._cash + self._inventory * event.mid_price
        quantity = min(event.quantity, quote.size)
        fill_probability = self._fill_model.fill_probability(event, quote)
        filled = fill_probability > 0.0 and self._fill_model.should_fill(event, quote, self._execution_rng)

        if filled and event.aggressor_side == "buy":
            self._inventory -= quantity
            self._cash += quote.ask * quantity
        elif filled:
            self._inventory += quantity
            self._cash -= quote.bid * quantity

        next_mid = (
            self._replay[self._index + 1].mid_price
            if self._index + 1 < len(self._replay)
            else event.mid_price
        )
        marked_wealth = self._cash + self._inventory * next_mid
        reward = float(marked_wealth - previous_wealth - 0.02 * abs(self._inventory))
        self._index += 1
        terminated = self._index >= len(self._replay)
        info = {
            "action": int(action),
            "filled_quantity": quantity if filled else 0,
            "inventory": self._inventory,
            "cash": self._cash,
            "mark_to_market_pnl": marked_wealth,
            "fill_probability": fill_probability,
        }
        return self._observation(), reward, terminated, False, info
