"""Tests for AgentRegistry."""

from __future__ import annotations

import pytest

from ddqn_router.agents import AgentRegistry
from ddqn_router.config import AgentDef


def test_registry_round_trip(sample_agents: list[AgentDef]) -> None:
    reg = AgentRegistry(sample_agents)
    assert reg.num_agents == 3
    assert reg.ids() == [0, 1, 2]
    assert reg.names() == ["Billing", "Technical", "Account"]
    assert reg.stop_action == 3
    assert reg.num_actions == 4


def test_lookup_by_id_and_name(sample_agents: list[AgentDef]) -> None:
    reg = AgentRegistry(sample_agents)
    assert reg.get_by_id(1).name == "Technical"
    assert reg.get_by_name("Account").id == 2


def test_duplicate_id_rejected() -> None:
    agents = [
        AgentDef(id=0, name="A", description="a"),
        AgentDef(id=0, name="B", description="b"),
    ]
    with pytest.raises(ValueError, match="Duplicate agent id"):
        AgentRegistry(agents)


def test_id_gap_rejected() -> None:
    agents = [
        AgentDef(id=0, name="A", description="a"),
        AgentDef(id=2, name="C", description="c"),
    ]
    with pytest.raises(ValueError, match="contiguous|gap|missing"):
        AgentRegistry(agents)


def test_empty_registry_rejected() -> None:
    with pytest.raises(ValueError):
        AgentRegistry([])
