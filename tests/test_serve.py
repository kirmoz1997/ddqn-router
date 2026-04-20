"""Tests for the optional FastAPI serve module."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from ddqn_router.serve.app import create_app  # noqa: E402


@pytest.fixture
def client(trained_artifacts: Path) -> TestClient:
    app = create_app(str(trained_artifacts))
    return TestClient(app)


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_agents_listed(client: TestClient) -> None:
    r = client.get("/agents")
    assert r.status_code == 200
    payload = r.json()
    assert "agents" in payload
    assert len(payload["agents"]) == 3


def test_route(client: TestClient) -> None:
    r = client.post("/route", json={"query": "billing invoice payment"})
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"agents", "agent_names", "confidence", "steps"}
    assert isinstance(body["agents"], list)
    assert 0.0 <= body["confidence"] <= 1.0


def test_route_batch(client: TestClient) -> None:
    r = client.post(
        "/route/batch",
        json={"queries": ["billing invoice", "api bug"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert len(body) == 2
