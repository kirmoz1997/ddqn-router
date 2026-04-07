"""Rule-based baseline: keyword matching on agent descriptions."""

from __future__ import annotations

from ddqn_router.agents import AgentRegistry
from ddqn_router.dataset.dataset import Task


def rule_route(
    tasks: list[Task],
    registry: AgentRegistry,
    min_agents: int = 1,
) -> list[set[int]]:
    agent_keywords: dict[int, set[str]] = {}
    for agent in registry.all_agents():
        words = set(agent.description.lower().split())
        agent_keywords[agent.id] = {w for w in words if len(w) > 3}

    results: list[set[int]] = []
    for task in tasks:
        text_lower = task["text"].lower()
        text_words = set(text_lower.split())
        scores: list[tuple[int, int]] = []
        for aid, keywords in agent_keywords.items():
            overlap = len(keywords & text_words)
            # Also count substring matches for multi-word phrases
            substring_hits = sum(1 for kw in keywords if kw in text_lower)
            scores.append((aid, overlap + substring_hits))

        scores.sort(key=lambda x: x[1], reverse=True)
        selected: set[int] = set()
        for aid, score in scores:
            if score > 0:
                selected.add(aid)
        if len(selected) < min_agents:
            for aid, _ in scores[:min_agents]:
                selected.add(aid)
        results.append(selected)
    return results
