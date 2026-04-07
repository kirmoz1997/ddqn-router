"""LLM baseline: uses the same labeler infrastructure for routing."""

from __future__ import annotations

from ddqn_router.agents import AgentRegistry
from ddqn_router.config import LabelerConfig
from ddqn_router.dataset.dataset import Task
from ddqn_router.labeler.labeler import LLMLabeler


def llm_route(
    tasks: list[Task],
    registry: AgentRegistry,
    labeler_config: LabelerConfig,
) -> list[set[int]]:
    labeler = LLMLabeler(labeler_config, registry)
    results: list[set[int]] = []
    try:
        for task in tasks:
            result = labeler.label_one(task["text"])
            if result is not None:
                results.append(set(result["required_agents"]))
            else:
                results.append(set())
    finally:
        labeler.close()
    return results
