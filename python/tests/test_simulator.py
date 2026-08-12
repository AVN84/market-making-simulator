import unittest

from market_making.simulator import (
    BacktestConfig,
    MarketEvent,
    QueueAwareFillModel,
    Quote,
    generate_synthetic_replay,
    run_backtest,
)


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

    def test_queue_proxy_increases_fill_chance_for_deeper_crossing(self) -> None:
        model = QueueAwareFillModel(BacktestConfig())
        quote = Quote(bid=99, ask=101, size=2)
        at_touch = MarketEvent(0, 100, "buy", 101, 1, 1)
        deep_cross = MarketEvent(0, 100, "buy", 104, 1, 4)
        not_marketable = MarketEvent(0, 100, "buy", 100, 1, 0)

        self.assertEqual(model.fill_probability(not_marketable, quote), 0.0)
        self.assertLess(model.fill_probability(at_touch, quote), model.fill_probability(deep_cross, quote))
        self.assertLessEqual(model.fill_probability(deep_cross, quote), 1.0)

    def test_directional_pressure_can_create_negative_post_fill_markout(self) -> None:
        config = BacktestConfig(
            seed=8,
            steps=100,
            base_half_spread_ticks=1,
            queue_base_fill_probability=1.0,
            adverse_selection_probability=1.0,
            adverse_selection_min_aggression_ticks=0,
            adverse_selection_impact_ticks=3,
        )
        result = run_backtest(config, "fixed_spread")

        self.assertGreater(result.fills, 0)
        self.assertGreater(result.negative_markout_quantity, 0)


if __name__ == "__main__":
    unittest.main()
