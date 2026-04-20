"""Agent registry — loads agent definitions, provides lookup and STOP action ID."""

from __future__ import annotations

from ddqn_router.config import AgentDef


class AgentRegistry:
    def __init__(self, agents: list[AgentDef]) -> None:
        if not agents:
            raise ValueError("At least one agent must be defined")

        seen_ids: set[int] = set()
        seen_names: set[str] = set()
        for a in agents:
            if a.id in seen_ids:
                raise ValueError(f"Duplicate agent id: {a.id}")
            if a.name in seen_names:
                raise ValueError(f"Duplicate agent name: {a.name!r}")
            seen_ids.add(a.id)
            seen_names.add(a.name)

        sorted_ids = sorted(seen_ids)
        expected = list(range(sorted_ids[0], sorted_ids[0] + len(sorted_ids)))
        if sorted_ids != expected:
            missing = sorted(set(expected) - seen_ids)
            raise ValueError(f"Agent ids must be contiguous without gaps; missing ids: {missing}")

        self._agents = {a.id: a for a in agents}
        self._by_name = {a.name: a for a in agents}

    @property
    def num_agents(self) -> int:
        return len(self._agents)

    @property
    def stop_action(self) -> int:
        """STOP action ID = N (number of agents)."""
        return self.num_agents

    @property
    def num_actions(self) -> int:
        """Total action space size: N agents + 1 STOP."""
        return self.num_agents + 1

    def get_by_id(self, agent_id: int) -> AgentDef:
        return self._agents[agent_id]

    def get_by_name(self, name: str) -> AgentDef:
        return self._by_name[name]

    def all_agents(self) -> list[AgentDef]:
        return list(self._agents.values())

    def ids(self) -> list[int]:
        return list(self._agents.keys())

    def names(self) -> list[str]:
        return [a.name for a in self._agents.values()]

    def descriptions(self) -> list[str]:
        return [a.description for a in self._agents.values()]
