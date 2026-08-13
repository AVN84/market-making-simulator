from __future__ import annotations

import unittest
from dataclasses import replace

from market_making.benchmark import STRATEGIES, benchmark_rows, summarize_rows
from market_making.research import (
    AvellanedaStoikovQuoter,
    BayesianToxicityQuoter,
    PublicState,
    SessionConfig,
    generate_session,
    run_session,
)


class ResearchHarnessTests(unittest.TestCase):
    def test_session_generation_is_deterministic(self) -> None:
        config = SessionConfig(seed=19, steps=40)
        self.assertEqual(generate_session(config), generate_session(config))

    def test_all_strategies_share_replay_and_produce_valid_accounting(self) -> None:
        config = SessionConfig(seed=23, steps=80)
        events = generate_session(config)
        for strategy in STRATEGIES:
            result = run_session(config, strategy, events)
            self.assertEqual(result.buy_fills + result.sell_fills, result.fills)
            self.assertGreaterEqual(result.max_abs_inventory, abs(result.final_inventory))
            self.assertGreaterEqual(result.transaction_cost_pnl, 0.0)
            self.assertAlmostEqual(
                result.net_pnl,
                result.mark_to_market_pnl - result.liquidation_cost_pnl,
            )

    def test_avellaneda_stoikov_reservation_price_moves_against_inventory(self) -> None:
        config = SessionConfig(steps=50)
        quoter = AvellanedaStoikovQuoter(config)
        flat = PublicState(10_000, 0, 10, 50, 2.0, 0.0, 0.0, 0.0)
        long = replace(flat, inventory=10)
        self.assertLess(quoter.quote(long).bid, quoter.quote(flat).bid)
        self.assertLess(quoter.quote(long).ask, quoter.quote(flat).ask)

    def test_bayesian_policy_widens_when_toxicity_rises(self) -> None:
        config = SessionConfig(steps=50)
        quoter = BayesianToxicityQuoter(config)
        quiet = PublicState(10_000, 0, 10, 50, 1.0, 0.0, 0.0, 0.0)
        toxic = replace(quiet, toxicity_estimate=1.0)
        quiet_quote = quoter.quote(quiet)
        toxic_quote = quoter.quote(toxic)
        self.assertGreater(toxic_quote.ask - toxic_quote.bid, quiet_quote.ask - quiet_quote.bid)

    def test_small_benchmark_has_paired_rows_and_intervals(self) -> None:
        rows = benchmark_rows(8, 30, 1)
        self.assertEqual(len(rows), 8 * len(STRATEGIES))
        summary = summarize_rows(rows)
        self.assertEqual(summary["session_count"], 8)
        paired = summary["paired_vs_fixed_spread"]
        self.assertIn("avellaneda_stoikov", paired)
        self.assertIn("bootstrap_95pct_lower", paired["avellaneda_stoikov"]["net_pnl"])


if __name__ == "__main__":
    unittest.main()
