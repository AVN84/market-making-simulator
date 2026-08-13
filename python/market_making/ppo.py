"""Evaluation helpers for learned and rule-based quoting policies."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from statistics import mean, pstdev

from market_making.environment import MarketMakingEnv
from market_making.research import SessionConfig

ActionSelector = Callable[[object], int]
EPISODE_METRICS = (
    "total_reward",
    "net_pnl",
    "mark_to_market_pnl",
    "final_inventory",
    "max_abs_inventory",
    "mean_abs_inventory",
    "gross_spread_capture_pnl",
    "transaction_cost_pnl",
    "post_fill_markout_pnl",
    "negative_markout_quantity",
    "negative_markout_loss",
    "fills",
)


def run_episode(
    config: SessionConfig,
    select_action: ActionSelector,
) -> dict[str, float | int | str]:
    env = MarketMakingEnv(config)
    observation, reset_info = env.reset(seed=config.seed)
    total_reward = 0.0
    fills = 0
    action_counts: dict[int, int] = {}
    while True:
        action = int(select_action(observation))
        action_counts[action] = action_counts.get(action, 0) + 1
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        fills += int(info["filled_quantity"])
        if terminated or truncated:
            return {
                "seed": config.seed,
                "regime": str(reset_info["regime"]),
                "policy": "ppo",
                "total_reward": round(total_reward, 6),
                "net_pnl": round(float(info["net_pnl"]), 6),
                "mark_to_market_pnl": round(float(info["mark_to_market_pnl"]), 6),
                "final_inventory": int(info["inventory"]),
                "max_abs_inventory": int(info["max_abs_inventory"]),
                "mean_abs_inventory": round(float(info["mean_abs_inventory"]), 6),
                "gross_spread_capture_pnl": round(float(info["gross_spread_capture_pnl"]), 6),
                "transaction_cost_pnl": round(float(info["transaction_cost_pnl"]), 6),
                "post_fill_markout_pnl": round(float(info["post_fill_markout_pnl"]), 6),
                "negative_markout_quantity": int(info["negative_markout_quantity"]),
                "negative_markout_loss": round(float(info["negative_markout_loss"]), 6),
                "fills": fills,
                "action_counts": ";".join(
                    f"{action}:{count}" for action, count in sorted(action_counts.items())
                ),
            }


def evaluate_actions(
    configs: Iterable[SessionConfig], select_action: ActionSelector
) -> list[dict[str, float | int | str]]:
    return [run_episode(config, select_action) for config in configs]


def summarize_episodes(
    rows: list[dict[str, float | int | str]],
) -> dict[str, dict[str, float] | float]:
    if not rows:
        raise ValueError("cannot summarize an empty evaluation")
    summary: dict[str, dict[str, float] | float] = {
        key: {
            "mean": round(mean(float(row[key]) for row in rows), 6),
            "population_stddev": round(pstdev(float(row[key]) for row in rows), 6),
            "minimum": min(float(row[key]) for row in rows),
            "maximum": max(float(row[key]) for row in rows),
        }
        for key in EPISODE_METRICS
    }
    pnl_values = [float(row["net_pnl"]) for row in rows]
    pnl_stddev = pstdev(pnl_values)
    summary["session_pnl_sharpe"] = round(mean(pnl_values) / pnl_stddev, 6) if pnl_stddev else 0.0
    return summary
