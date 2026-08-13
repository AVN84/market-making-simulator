from __future__ import annotations

import unittest

import numpy as np
from market_making.environment import ACTION_SPECS, MarketMakingEnv
from market_making.ppo import evaluate_actions
from market_making.research import PublicState, SessionConfig


class MarketMakingEnvironmentTests(unittest.TestCase):
    def test_reset_with_same_seed_is_reproducible(self) -> None:
        env = MarketMakingEnv(SessionConfig(steps=20))
        first, first_info = env.reset(seed=11)
        second, second_info = env.reset(seed=11)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_info, second_info)

    def test_first_observation_does_not_reveal_current_order(self) -> None:
        first_env = MarketMakingEnv(SessionConfig(seed=1, steps=20))
        second_env = MarketMakingEnv(SessionConfig(seed=2, steps=20))
        first, _ = first_env.reset(seed=1)
        second, _ = second_env.reset(seed=2)
        np.testing.assert_array_equal(first, second)

    def test_randomized_training_resets_are_reproducible_and_distinct(self) -> None:
        first_env = MarketMakingEnv(
            SessionConfig(steps=20),
            randomize_reset_seed=True,
            randomize_regime=True,
        )
        second_env = MarketMakingEnv(
            SessionConfig(steps=20),
            randomize_reset_seed=True,
            randomize_regime=True,
        )
        first_sequence = [first_env.reset()[1] for _ in range(4)]
        second_sequence = [second_env.reset()[1] for _ in range(4)]
        self.assertEqual(first_sequence, second_sequence)
        self.assertEqual(len({item["seed"] for item in first_sequence}), 4)

    def test_fixed_action_episode_is_deterministic(self) -> None:
        configs = [SessionConfig(seed=13, steps=30)]
        first = evaluate_actions(configs, lambda observation: 13)
        second = evaluate_actions(configs, lambda observation: 13)
        self.assertEqual(first, second)

    def test_action_space_controls_spread_and_inventory_response(self) -> None:
        env = MarketMakingEnv(SessionConfig(steps=20))
        state = PublicState(10_000, 10, 5, 20, 1.0, 0.5, 0.2, 1.0)
        narrow = env.quote_for_action(3, state)
        wide = env.quote_for_action(21, state)
        self.assertLess(narrow.ask - narrow.bid, wide.ask - wide.bid)
        self.assertEqual(len(ACTION_SPECS), 27)

    def test_step_reports_valid_transition(self) -> None:
        env = MarketMakingEnv(SessionConfig(steps=10))
        observation, _ = env.reset(seed=7)
        self.assertTrue(env.observation_space.contains(observation))
        observation, reward, terminated, truncated, info = env.step(13)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertIsInstance(reward, float)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIn("net_pnl", info)
        self.assertIn("negative_markout_loss", info)


if __name__ == "__main__":
    unittest.main()
