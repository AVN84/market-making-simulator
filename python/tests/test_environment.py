import unittest

import numpy as np

from market_making.environment import MarketMakingEnv
from market_making.ppo import evaluate_actions
from market_making.simulator import BacktestConfig


class MarketMakingEnvironmentTests(unittest.TestCase):
    def test_reset_with_same_seed_is_reproducible(self) -> None:
        env = MarketMakingEnv(BacktestConfig(steps=20))
        first, first_info = env.reset(seed=41)
        second, second_info = env.reset(seed=41)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_info, second_info)

    def test_fixed_action_episode_is_deterministic(self) -> None:
        config = BacktestConfig(steps=30)
        first = evaluate_actions(config, [13], lambda observation: 2)
        second = evaluate_actions(config, [13], lambda observation: 2)

        self.assertEqual(first, second)
        self.assertGreaterEqual(first[0]["fills"], 0)

    def test_step_reports_valid_transition(self) -> None:
        env = MarketMakingEnv(BacktestConfig(steps=3))
        observation, _ = env.reset(seed=9)
        self.assertTrue(env.observation_space.contains(observation))
        next_observation, reward, terminated, truncated, info = env.step(2)

        self.assertTrue(env.observation_space.contains(next_observation))
        self.assertIsInstance(reward, float)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIn("mark_to_market_pnl", info)


if __name__ == "__main__":
    unittest.main()
