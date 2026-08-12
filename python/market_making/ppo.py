"""Evaluation helpers shared by the optional PPO scripts."""

from __future__ import annotations

from statistics import mean, pstdev
from typing import Callable, Iterable

from market_making.environment import MarketMakingEnv
from market_making.simulator import BacktestConfig


ActionSelector = Callable[[object], int]


def run_episode(config: BacktestConfig, seed: int, select_action: ActionSelector) -> dict[str, float | int]:
    env = MarketMakingEnv(config)
    observation, _ = env.reset(seed=seed)
    total_reward = 0.0
    fills = 0
    while True:
        action = select_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        fills += int(info["filled_quantity"])
        if terminated or truncated:
            return {
                "seed": seed,
                "total_reward": round(total_reward, 6),
                "mark_to_market_pnl": int(info["mark_to_market_pnl"]),
                "final_inventory": int(info["inventory"]),
                "fills": fills,
            }


def evaluate_actions(
    config: BacktestConfig, seeds: Iterable[int], select_action: ActionSelector
) -> list[dict[str, float | int]]:
    return [run_episode(config, seed, select_action) for seed in seeds]


def summarize_episodes(rows: list[dict[str, float | int]]) -> dict[str, dict[str, float]]:
    if not rows:
        raise ValueError("cannot summarize an empty evaluation")
    return {
        key: {
            "mean": round(mean(float(row[key]) for row in rows), 4),
            "population_stddev": round(pstdev(float(row[key]) for row in rows), 4),
            "minimum": min(float(row[key]) for row in rows),
            "maximum": max(float(row[key]) for row in rows),
        }
        for key in ("total_reward", "mark_to_market_pnl", "final_inventory", "fills")
    }
