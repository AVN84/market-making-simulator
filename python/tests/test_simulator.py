import unittest

from market_making.simulator import BacktestConfig, generate_synthetic_replay, run_backtest


class SyntheticReplayTests(unittest.TestCase):
    def test_replay_is_deterministic(self) -> None:
        config = BacktestConfig(seed=19, steps=20)
        self.assertEqual(generate_synthetic_replay(config), generate_synthetic_replay(config))

    def test_end_to_end_backtest_is_deterministic(self) -> None:
        config = BacktestConfig(seed=19, steps=100)
        first = run_backtest(config)
        second = run_backtest(config)
        self.assertEqual(first, second)
        self.assertEqual(first.steps, 100)
        self.assertGreater(first.fills, 0)
        self.assertEqual(first.buy_fills + first.sell_fills, first.fills)
        self.assertGreaterEqual(first.max_abs_inventory, abs(first.final_inventory))


if __name__ == "__main__":
    unittest.main()
