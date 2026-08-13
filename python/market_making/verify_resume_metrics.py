"""Generate resume-safe claims from checked-in benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _change(reference: float, candidate: float) -> float:
    if reference == 0.0:
        return 0.0
    return 100.0 * (candidate - reference) / abs(reference)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-summary",
        type=Path,
        default=Path("artifacts/research/benchmark/strategy_summary.json"),
    )
    parser.add_argument(
        "--ppo-summary",
        type=Path,
        default=Path("artifacts/research/ppo/ppo_training_summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/research/resume_metrics_verification.md"),
    )
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark_summary.read_text(encoding="utf-8"))
    ppo = json.loads(args.ppo_summary.read_text(encoding="utf-8"))
    strategies = benchmark["strategies"]
    best_name = max(strategies, key=lambda name: strategies[name]["session_pnl_sharpe"])
    fixed = strategies["fixed_spread"]
    best = strategies[best_name]
    paired = benchmark["paired_vs_fixed_spread"].get(best_name, {})
    ppo_metrics = ppo["policies"]["ppo"]
    as_metrics = ppo["policies"]["avellaneda_stoikov"]

    peak_reduction = -_change(
        fixed["max_abs_inventory"]["mean"],
        best["max_abs_inventory"]["mean"],
    )
    adverse_reduction = -_change(
        fixed["negative_markout_loss"]["mean"],
        best["negative_markout_loss"]["mean"],
    )
    ppo_peak_change = _change(
        as_metrics["max_abs_inventory"]["mean"],
        ppo_metrics["max_abs_inventory"]["mean"],
    )
    ppo_pnl_change = _change(
        as_metrics["net_pnl"]["mean"],
        ppo_metrics["net_pnl"]["mean"],
    )

    net_interval = paired.get("net_pnl", {})
    interval_text = (
        f"[{net_interval['bootstrap_95pct_lower']:.2f}, "
        f"{net_interval['bootstrap_95pct_upper']:.2f}]"
        if net_interval
        else "not applicable"
    )
    report = f"""# Resume Metric Verification

All results below come from seeded **synthetic** sessions. They are not
historical, calibrated, annualized, or live-market performance.

## Reproduction scope

- C++20 matching-engine tests: run separately with warnings treated as errors.
- Python tests: cover deterministic generation, accounting, all four policies,
  paired benchmarking, PPO action controls, and the pre-trade leakage guard.
- Rule-based benchmark: **{benchmark['session_count']:,} sessions**, four regimes,
  and four strategies.
- PPO training: **{ppo['training']['timesteps']:,} timesteps** across randomized
  regimes.
- PPO holdout: **{ppo['evaluation']['session_count']:,} disjoint sessions** over
  seeds {ppo['evaluation']['seed_range'][0]}-{ppo['evaluation']['seed_range'][1]}.

## Rule-based result

The strongest session P&L Sharpe came from `{best_name}`:

- session P&L Sharpe: **{best['session_pnl_sharpe']:.3f}**, versus
  **{fixed['session_pnl_sharpe']:.3f}** for fixed spread;
- mean net P&L: **{best['net_pnl']['mean']:.2f}** synthetic ticks, versus
  **{fixed['net_pnl']['mean']:.2f}**;
- mean peak inventory reduction: **{peak_reduction:.1f}%**;
- mean negative one-step markout loss reduction: **{adverse_reduction:.1f}%**;
- paired bootstrap 95% interval for the net P&L difference: **{interval_text}**.

## PPO result

Against Avellaneda-Stoikov on the disjoint holdout:

- mean net P&L changed **{ppo_pnl_change:+.1f}%**;
- mean peak inventory changed **{ppo_peak_change:+.1f}%**;
- PPO session P&L Sharpe was **{ppo_metrics['session_pnl_sharpe']:.3f}**, versus
  **{as_metrics['session_pnl_sharpe']:.3f}**.

PPO should be presented as a leakage-free research policy and an empirical
risk-return tradeoff. Do not claim it beat the strongest rule-based strategy
unless the recorded holdout values actually show that.

## Resume-safe wording

> Implemented a C++20 L2 limit order book with price-time priority, FIFO price
> levels, partial fills, and locator-based cancellation, with deterministic
> invariant tests compiled under warnings-as-errors.

> Benchmarked four quoting policies over {benchmark['session_count']:,} common-seed
> synthetic sessions across volatility and cost regimes; `{best_name}` reduced
> mean peak inventory {peak_reduction:.0f}% and negative one-step markout loss
> {adverse_reduction:.0f}% versus fixed-spread quoting.

> Built a leakage-free Gymnasium environment and trained PPO across randomized
> regimes, then evaluated it against four rule-based policies on
> {ppo['evaluation']['session_count']:,} disjoint synthetic holdout sessions.
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
