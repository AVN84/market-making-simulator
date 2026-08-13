"""Train PPO without order-flow look-ahead and evaluate on disjoint regimes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from market_making.environment import MarketMakingEnv
from market_making.ppo import evaluate_actions, summarize_episodes
from market_making.research import (
    REGIMES,
    StrategyName,
    generate_session,
    run_session,
    session_configs,
)

BASELINES: tuple[StrategyName, ...] = (
    "fixed_spread",
    "inventory_aware",
    "avellaneda_stoikov",
    "bayesian_toxicity",
)


def _baseline_rows(configs):
    rows = []
    for config in configs:
        events = generate_session(config)
        for strategy in BASELINES:
            result = run_session(config, strategy, events)
            row = result.as_row()
            row["policy"] = strategy
            row["total_reward"] = row["net_pnl"]
            row["action_counts"] = ""
            rows.append(row)
    return rows


def _percent_change(reference: float, candidate: float) -> float:
    if reference == 0.0:
        return 0.0
    return 100.0 * (candidate - reference) / abs(reference)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--episode-steps", type=int, default=300)
    parser.add_argument("--eval-sessions", type=int, default=1_000)
    parser.add_argument("--eval-seed-start", type=int, default=100_001)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/research/ppo"))
    args = parser.parse_args()
    if args.timesteps <= 0 or args.eval_sessions <= 0 or args.episode_steps <= 1:
        parser.error("timesteps and eval sessions must be positive; episode steps must exceed one")

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.utils import set_random_seed
    except ImportError as exc:
        raise SystemExit(
            "PPO requires stable-baselines3, Gymnasium, and PyTorch. "
            "Install the optional rl dependencies first."
        ) from exc

    training_seed = 20260813
    set_random_seed(training_seed)
    training_env = MarketMakingEnv(
        session_configs(1, steps=args.episode_steps, seed_start=training_seed)[0],
        randomize_reset_seed=True,
        randomize_regime=True,
    )
    model = PPO(
        "MlpPolicy",
        training_env,
        seed=training_seed,
        learning_rate=3e-4,
        n_steps=1_024,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [128, 128]},
        verbose=0,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "ppo_market_maker")
    evaluation_configs = session_configs(
        args.eval_sessions,
        steps=args.episode_steps,
        seed_start=args.eval_seed_start,
        regimes=REGIMES,
    )
    ppo_rows = evaluate_actions(
        evaluation_configs,
        lambda observation: int(model.predict(observation, deterministic=True)[0]),
    )
    baseline_rows = _baseline_rows(evaluation_configs)
    rows = baseline_rows + ppo_rows
    csv_path = output_dir / "ppo_evaluation.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    policies = tuple(BASELINES) + ("ppo",)
    policy_summaries = {
        policy: summarize_episodes([row for row in rows if row["policy"] == policy])
        for policy in policies
    }
    as_summary = policy_summaries["avellaneda_stoikov"]
    ppo_summary = policy_summaries["ppo"]
    comparisons = {
        metric: _percent_change(
            float(as_summary[metric]["mean"]),
            float(ppo_summary[metric]["mean"]),
        )
        for metric in (
            "net_pnl",
            "max_abs_inventory",
            "negative_markout_loss",
            "gross_spread_capture_pnl",
        )
    }
    summary = {
        "scope": "seeded synthetic holdout sessions only; no historical or live-market performance",
        "training": {
            "algorithm": "PPO",
            "library": "stable-baselines3",
            "timesteps": args.timesteps,
            "seed": training_seed,
            "episode_steps": args.episode_steps,
            "training_replays": "randomized across seeds and four regimes",
            "observation_timing": "public history before the next aggressive order",
        },
        "evaluation": {
            "session_count": args.eval_sessions,
            "seed_range": [args.eval_seed_start, args.eval_seed_start + args.eval_sessions - 1],
            "regimes": [regime.name for regime in REGIMES],
        },
        "policies": policy_summaries,
        "ppo_percent_change_vs_avellaneda_stoikov": comparisons,
    }
    (output_dir / "ppo_training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(output_dir / "ppo_training_summary.json")


if __name__ == "__main__":
    main()
