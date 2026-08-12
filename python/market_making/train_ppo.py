"""Train and evaluate a small PPO policy on the synthetic Gymnasium environment.

This is intentionally a lightweight engineering smoke test. Its outputs apply
only to the documented synthetic environment, not to real markets.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from market_making.environment import MarketMakingEnv
from market_making.ppo import evaluate_actions, summarize_episodes
from market_making.simulator import BacktestConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=4_096)
    parser.add_argument("--eval-seeds", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if args.timesteps <= 0 or args.eval_seeds <= 0:
        parser.error("--timesteps and --eval-seeds must be positive")

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover - exercised only without the optional dependency.
        raise SystemExit(
            "stable-baselines3 is optional. Install it with: python3 -m pip install --user stable-baselines3"
        ) from exc

    config = BacktestConfig(steps=200)
    environment = MarketMakingEnv(config)
    model = PPO(
        "MlpPolicy",
        environment,
        seed=20260812,
        n_steps=128,
        batch_size=64,
        n_epochs=5,
        learning_rate=3e-4,
        verbose=0,
    )
    model.learn(total_timesteps=args.timesteps)

    seeds = range(1, args.eval_seeds + 1)
    ppo_rows = evaluate_actions(config, seeds, lambda observation: int(model.predict(observation, deterministic=True)[0]))
    fixed_rows = evaluate_actions(config, seeds, lambda observation: 0)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "ppo_evaluation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["policy", *ppo_rows[0].keys()])
        writer.writeheader()
        for policy, rows in (("fixed_skew", fixed_rows), ("ppo", ppo_rows)):
            for row in rows:
                writer.writerow({"policy": policy, **row})

    summary = {
        "scope": "small PPO smoke test on the synthetic Gymnasium environment only; not a live-market result",
        "training": {
            "algorithm": "PPO (stable-baselines3)",
            "timesteps": args.timesteps,
            "seed": 20260812,
            "environment_steps_per_episode": config.steps,
        },
        "evaluation_seed_count": args.eval_seeds,
        "fixed_skew": summarize_episodes(fixed_rows),
        "ppo": summarize_episodes(ppo_rows),
    }
    summary_path = output_dir / "ppo_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "summary": str(summary_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
