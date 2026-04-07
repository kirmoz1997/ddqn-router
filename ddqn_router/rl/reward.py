"""Reward functions for the routing MDP."""

from __future__ import annotations

import random


def jaccard_similarity(selected: set[int], target: set[int]) -> float:
    if not selected and not target:
        return 1.0
    if not selected or not target:
        return 0.0
    return len(selected & target) / len(selected | target)


def compute_reward(
    selected_agents: set[int],
    target_agents: set[int],
    step_cost: float,
    mode: str = "jaccard",
    is_terminal: bool = False,
) -> float:
    """Compute step or terminal reward.

    In jaccard mode:
      - Each non-terminal step incurs -step_cost.
      - Terminal step receives Jaccard similarity as reward.
    In stochastic mode:
      - Terminal reward is 1.0 if a randomly sampled target agent is in selected, else 0.0.
      - Step cost still applies.
    """
    if not is_terminal:
        return -step_cost

    if mode == "jaccard":
        return jaccard_similarity(selected_agents, target_agents)
    elif mode == "stochastic":
        if not target_agents:
            return 0.0
        sampled = random.choice(list(target_agents))
        return 1.0 if sampled in selected_agents else 0.0
    else:
        raise ValueError(f"Unknown reward mode: {mode}")
