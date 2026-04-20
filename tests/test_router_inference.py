"""Tests for DDQNRouter inference API."""

from __future__ import annotations

from pathlib import Path

import pytest

from ddqn_router.inference.router import (
    DDQNRouter,
    RouterNotTrainedError,
    StepTrace,
)


def test_load_from_trained_artifacts(trained_artifacts: Path) -> None:
    router = DDQNRouter.load(trained_artifacts)
    assert len(router.agents) == 3


def test_route_returns_valid_subset(trained_artifacts: Path) -> None:
    router = DDQNRouter.load(trained_artifacts)
    result = router.route("billing invoice payment")
    valid_ids = {a["id"] for a in router.agents}
    assert set(result.agents).issubset(valid_ids)
    assert len(result.agents) == len(set(result.agents))
    assert 0.0 <= result.confidence <= 1.0
    assert result.steps >= 1


def test_route_batch_equivalence(trained_artifacts: Path) -> None:
    router = DDQNRouter.load(trained_artifacts)
    queries = ["billing invoice", "api bug", "account password"]
    batch = router.route_batch(queries)
    individual = [router.route(q) for q in queries]
    for b, i in zip(batch, individual):
        assert b.agents == i.agents
        assert b.agent_names == i.agent_names


def test_route_verbose_populates_trace(trained_artifacts: Path) -> None:
    router = DDQNRouter.load(trained_artifacts)
    result = router.route_verbose("billing invoice payment")
    assert result.steps_trace is not None
    assert len(result.steps_trace) == result.steps
    first = result.steps_trace[0]
    assert isinstance(first, StepTrace)
    assert first.step_index == 0
    # Agents returned by route match selections in trace
    selected_from_trace = [s.selected_action for s in result.steps_trace if not s.stop_selected]
    assert selected_from_trace == result.agents


def test_missing_artifacts_raises(tmp_path: Path) -> None:
    with pytest.raises(RouterNotTrainedError) as exc_info:
        DDQNRouter.load(tmp_path / "nonexistent")
    msg = str(exc_info.value)
    assert "train" in msg.lower() or "label" in msg.lower()
