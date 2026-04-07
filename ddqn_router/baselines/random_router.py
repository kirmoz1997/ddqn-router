"""Random baseline: selects a random subset of agents."""

from __future__ import annotations

import random

from ddqn_router.agents import AgentRegistry
from ddqn_router.dataset.dataset import Task


def random_route(
    tasks: list[Task],
    registry: AgentRegistry,
    seed: int = 42,
) -> list[set[int]]:
    rng = random.Random(seed)
    ids = registry.ids()
    results: list[set[int]] = []
    for task in tasks:
        target_size = len(task["required_agents"])
        k = min(max(1, rng.randint(1, target_size + 1)), len(ids))
        selected = set(rng.sample(ids, k))
        results.append(selected)
    return results
