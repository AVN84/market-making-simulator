"""Run reproducible multi-regime market-making benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import mean, pstdev

from market_making.research import (
    StrategyName,
    generate_session,
    run_session,
    session_configs,
)

STRATEGIES: tuple[StrategyName, ...] = (
    "fixed_spread",
    "inventory_aware",
    "avellaneda_stoikov",
    "bayesian_toxicity",
)
METRICS = (
    "net_pnl",
    "mark_to_market_pnl",
    "gross_spread_capture_pnl",
    "transaction_cost_pnl",
    "post_fill_markout_pnl",
    "negative_markout_loss",
    "fills",
    "fill_ratio",
    "max_abs_inventory",
    "mean_abs_inventory",
)


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(mean(values), 6),
        "population_stddev": round(pstdev(values), 6),
        "minimum": round(min(values), 6),
        "maximum": round(max(values), 6),
    }


def _paired_interval(
    reference: list[float],
    candidate: list[float],
    *,
    samples: int = 2_000,
    seed: int = 20260813,
) -> dict[str, float]:
    if len(reference) != len(candidate) or not reference:
        raise ValueError("paired samples must be non-empty and equal length")
    differences = [
        candidate_value - reference_value
        for reference_value, candidate_value in zip(reference, candidate, strict=True)
    ]
    rng = random.Random(seed)
    bootstrapped = []
    for _ in range(samples):
        bootstrapped.append(mean(differences[rng.randrange(len(differences))] for _ in differences))
    bootstrapped.sort()
    lower = bootstrapped[int(0.025 * samples)]
    upper = bootstrapped[int(0.975 * samples)]
    reference_mean = mean(reference)
    percent_change = (
        100.0 * mean(differences) / abs(reference_mean)
        if reference_mean != 0.0
        else 0.0
    )
    return {
        "mean_difference": round(mean(differences), 6),
        "percent_change_vs_reference": round(percent_change, 4),
        "bootstrap_95pct_lower": round(lower, 6),
        "bootstrap_95pct_upper": round(upper, 6),
    }


def benchmark_rows(
    session_count: int,
    steps: int,
    seed_start: int,
) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    for config in session_configs(session_count, steps=steps, seed_start=seed_start):
        events = generate_session(config)
        for strategy in STRATEGIES:
            rows.append(run_session(config, strategy, events).as_row())
    return rows


def summarize_rows(rows: list[dict[str, str | int | float]]) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot summarize empty rows")
    session_keys = sorted({(int(row["seed"]), str(row["regime"])) for row in rows})
    strategies: dict[str, object] = {}
    by_regime: dict[str, object] = {}

    for strategy in STRATEGIES:
        selected = [row for row in rows if row["strategy"] == strategy]
        metric_summary = {
            metric: _summary([float(row[metric]) for row in selected])
            for metric in METRICS
        }
        pnl = [float(row["net_pnl"]) for row in selected]
        pnl_stddev = pstdev(pnl)
        metric_summary["session_pnl_sharpe"] = (
            round(mean(pnl) / pnl_stddev, 6) if pnl_stddev else 0.0
        )
        strategies[strategy] = metric_summary

    for regime in sorted({str(row["regime"]) for row in rows}):
        by_regime[regime] = {
            strategy: {
                metric: _summary(
                    [
                        float(row[metric])
                        for row in rows
                        if row["regime"] == regime and row["strategy"] == strategy
                    ]
                )
                for metric in ("net_pnl", "negative_markout_loss", "max_abs_inventory", "fills")
            }
            for strategy in STRATEGIES
        }

    fixed_by_key = {
        (int(row["seed"]), str(row["regime"])): row
        for row in rows
        if row["strategy"] == "fixed_spread"
    }
    paired: dict[str, object] = {}
    for strategy in STRATEGIES[1:]:
        candidate_by_key = {
            (int(row["seed"]), str(row["regime"])): row
            for row in rows
            if row["strategy"] == strategy
        }
        paired[strategy] = {
            metric: _paired_interval(
                [float(fixed_by_key[key][metric]) for key in session_keys],
                [float(candidate_by_key[key][metric]) for key in session_keys],
            )
            for metric in (
                "net_pnl",
                "negative_markout_loss",
                "max_abs_inventory",
                "gross_spread_capture_pnl",
            )
        }

    return {
        "scope": "seeded synthetic simulations only; no historical or live-market performance",
        "session_count": len(session_keys),
        "row_count": len(rows),
        "strategies": strategies,
        "by_regime": by_regime,
        "paired_vs_fixed_spread": paired,
    }


def _write_report(summary: dict[str, object], output: Path) -> None:
    strategies = summary["strategies"]
    assert isinstance(strategies, dict)
    lines = [
        "# Multi-Regime Strategy Benchmark",
        "",
        f"This report covers **{summary['session_count']:,} seeded synthetic sessions**. "
        "It is not historical or live-market evidence.",
        "",
        "| Strategy | Mean net P&L | Session P&L Sharpe | Mean peak inventory | "
        "Mean adverse loss | Mean fills |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy in STRATEGIES:
        metrics = strategies[strategy]
        assert isinstance(metrics, dict)
        lines.append(
            f"| {strategy} | {metrics['net_pnl']['mean']:.2f} | "
            f"{metrics['session_pnl_sharpe']:.3f} | "
            f"{metrics['max_abs_inventory']['mean']:.2f} | "
            f"{metrics['negative_markout_loss']['mean']:.2f} | "
            f"{metrics['fills']['mean']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Method",
            "",
            "- Every strategy quotes before the next aggressive order is revealed.",
            "- Strategies share the same event stream and per-event fill uniform for "
            "paired comparisons.",
            "- The regime grid varies volatility, informed-flow intensity, and transaction costs.",
            "- Net P&L includes per-fill transaction costs and terminal inventory liquidation.",
            "- Paired bootstrap intervals are stored in the JSON summary.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(rows: list[dict[str, str | int | float]], output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "strategy_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize_rows(rows)
    (output_dir / "strategy_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(summary, output_dir / "strategy_report.md")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=5_000)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/research/benchmark"))
    args = parser.parse_args()
    if args.sessions <= 0 or args.steps <= 1:
        parser.error("sessions must be positive and steps must exceed one")
    rows = benchmark_rows(args.sessions, args.steps, args.seed_start)
    write_outputs(rows, args.output_dir)
    print(args.output_dir / "strategy_report.md")


if __name__ == "__main__":
    main()
